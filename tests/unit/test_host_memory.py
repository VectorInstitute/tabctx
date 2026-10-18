"""Unit tests for process-wide host-memory admission (memory/host.py)."""

import threading
import time

import pytest

from tabctx.backends.fake import FakeBackend
from tabctx.cache.manager import ContextCacheManager
from tabctx.engine import TabctxEngine
from tabctx.memory import (
    A100_40GB_TABICL_CALIBRATION,
    AdaptiveMemoryEstimator,
    PowerLawMemoryEstimator,
)
from tabctx.memory.host import GiB, HostMemoryGuard


def test_slots_from_budget():
    assert HostMemoryGuard(176 * GiB, large_fit_bytes=48 * GiB).slots == 3
    assert HostMemoryGuard(10 * GiB, large_fit_bytes=48 * GiB).slots == 1  # never 0


def test_calls_estimated_at_zero_never_wait():
    guard = HostMemoryGuard(GiB, large_fit_bytes=GiB)
    with guard.reserve(0), guard.reserve(0):
        assert guard.stats()["calls_active"] == 0


def _hold(guard, nbytes, inside, release):
    def run():
        with guard.reserve(nbytes):
            inside.release()
            release.wait(5)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def test_calls_are_admitted_while_their_bytes_fit_the_budget():
    guard = HostMemoryGuard(10 * GiB, large_fit_bytes=GiB)
    inside, release = threading.Semaphore(0), threading.Event()
    threads = [_hold(guard, 4 * GiB, inside, release) for _ in range(3)]
    assert inside.acquire(timeout=5) and inside.acquire(timeout=5)
    assert not inside.acquire(timeout=0.3)  # 12 GiB > 10: the third waits
    stats = guard.stats()
    assert (stats["reserved_bytes"], stats["calls_active"], stats["calls_waiting"]) == (
        8 * GiB,
        2,
        1,
    )
    release.set()
    for t in threads:
        t.join(5)
    assert guard.stats()["reserved_bytes"] == 0


def test_many_modest_calls_run_together():
    """What the old slot guard got wrong: every TabPFN fit over 3M cells
    took a whole 48 GiB slot though it uses a few GiB."""
    guard = HostMemoryGuard(100 * GiB)
    inside, release = threading.Semaphore(0), threading.Event()
    threads = [_hold(guard, 5 * GiB, inside, release) for _ in range(8)]
    assert all(inside.acquire(timeout=5) for _ in range(8))
    release.set()
    for t in threads:
        t.join(5)


def test_a_call_bigger_than_the_budget_runs_alone():
    guard = HostMemoryGuard(10 * GiB, large_fit_bytes=GiB)
    inside, release = threading.Semaphore(0), threading.Event()
    big = _hold(guard, 50 * GiB, inside, release)
    assert inside.acquire(timeout=5)  # admitted: nothing else was running
    small = _hold(guard, GiB, inside, release)
    assert not inside.acquire(timeout=0.3)  # waits for the big one
    release.set()
    for t in (big, small):
        t.join(5)
    assert inside.acquire(timeout=1)


def test_a_multiprocessing_guard_reserves_and_reports():
    import multiprocessing as mp

    guard = HostMemoryGuard(10 * GiB, mp_context=mp.get_context("spawn"))
    with guard.reserve(3 * GiB):
        assert guard.stats()["reserved_bytes"] == 3 * GiB
    assert guard.stats()["calls_active"] == 0


def test_calls_under_the_floor_never_wait_even_beside_a_call_running_alone():
    """An over-budget call runs alone, but small calls (inside the 8 GiB
    reserve) must keep flowing: they bypass the guard entirely."""
    guard = HostMemoryGuard(10 * GiB, large_fit_bytes=GiB)
    inside, release = threading.Semaphore(0), threading.Event()
    big = _hold(guard, 50 * GiB, inside, release)
    assert inside.acquire(timeout=5)
    with guard.reserve(GiB // 2):  # immediate
        assert guard.stats()["calls_active"] == 1  # not even counted
    release.set()
    big.join(5)


def test_admission_is_first_come_first_served():
    """A large call queued behind running ones isn't overtaken by smaller
    ones that would fit now: without this, steady modest traffic starves
    it forever."""
    guard = HostMemoryGuard(10 * GiB, large_fit_bytes=GiB)
    inside, release = threading.Semaphore(0), threading.Event()
    running = _hold(guard, 4 * GiB, inside, release)
    assert inside.acquire(timeout=5)
    order: list[str] = []
    big_release = threading.Event()

    def call(name, nbytes, done):
        with guard.reserve(nbytes):
            order.append(name)
            done.wait(5)

    big = threading.Thread(target=call, args=("big", 8 * GiB, big_release), daemon=True)
    big.start()
    _wait_for(lambda: guard.stats()["calls_waiting"] == 1)
    small = threading.Thread(target=call, args=("small", 3 * GiB, release), daemon=True)
    small.start()  # 4 + 3 <= 10 would fit, but "big" was first
    _wait_for(lambda: guard.stats()["calls_waiting"] == 2)
    assert order == []
    release.set()  # the running call ends: big goes first (8 + 3 > 10)
    _wait_for(lambda: order == ["big"])
    assert not _settles(lambda: len(order) > 1)
    big_release.set()
    for t in (running, big, small):
        t.join(5)
    assert order == ["big", "small"]


def test_offloading_calls_are_capped_at_slots():
    """Each offloading call budgets 1/slots of free host memory for its
    staging, so no more than `slots` may run at once even if bytes fit."""
    guard = HostMemoryGuard(100 * GiB, large_fit_bytes=50 * GiB)  # 2 slots
    held = threading.Event()
    entered = threading.Semaphore(0)

    def offloading():
        with guard.reserve(10 * GiB, offloads=True):
            entered.release()
            held.wait(5)

    threads = [threading.Thread(target=offloading, daemon=True) for _ in range(3)]
    for t in threads:
        t.start()
    assert entered.acquire(timeout=5) and entered.acquire(timeout=5)
    assert not entered.acquire(timeout=0.3)
    assert guard.stats()["calls_offloading"] == 2
    held.set()
    assert entered.acquire(timeout=5)
    for t in threads:
        t.join(5)


def test_a_none_estimate_counts_as_nothing():
    guard = HostMemoryGuard(GiB)
    with guard.reserve(None):
        assert guard.stats()["calls_active"] == 0


def test_an_interrupted_waiter_leaves_the_queue():
    guard = HostMemoryGuard(10 * GiB, large_fit_bytes=GiB)
    inside, release = threading.Semaphore(0), threading.Event()
    running = _hold(guard, 8 * GiB, inside, release)
    assert inside.acquire(timeout=5)
    real_wait = guard._cond.wait

    def interrupted(timeout=None):
        guard._cond.wait = real_wait
        raise KeyboardInterrupt

    guard._cond.wait = interrupted
    try:
        with guard.reserve(4 * GiB):
            raise AssertionError("must not be admitted")
    except KeyboardInterrupt:
        pass
    stats = guard.stats()
    assert (stats["calls_waiting"], stats["calls_active"]) == (0, 1)
    release.set()
    running.join(5)


def test_a_stuck_lock_fails_retryably_instead_of_hanging():
    from tabctx.errors import HostMemoryUnavailableError

    guard = HostMemoryGuard(10 * GiB, large_fit_bytes=GiB, lock_timeout_s=0.2)
    holding, done = threading.Event(), threading.Event()

    def hold_lock():  # stands in for a worker killed holding it
        with guard._cond:
            holding.set()
            done.wait(5)

    t = threading.Thread(target=hold_lock, daemon=True)
    t.start()
    assert holding.wait(5)
    with pytest.raises(HostMemoryUnavailableError), guard.reserve(2 * GiB):
        pass
    with pytest.raises(HostMemoryUnavailableError):
        guard.stats()
    done.set()
    t.join(5)


def _reserve_in_child(guard, owner, nbytes, ready):
    guard.bind(owner)
    with guard.reserve(nbytes):
        ready.set()
        threading.Event().wait(60)  # until killed


def test_reservations_are_node_wide_and_a_dead_owner_is_released():
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    guard = HostMemoryGuard(10 * GiB, large_fit_bytes=GiB, mp_context=ctx, owners=2)
    ready = ctx.Event()
    child = ctx.Process(
        target=_reserve_in_child, args=(guard, 1, 8 * GiB, ready), daemon=True
    )
    child.start()
    try:
        assert ready.wait(60)
        assert guard.stats()["reserved_bytes"] == 8 * GiB  # seen by the parent
        admitted = threading.Event()

        def parent_call():
            with guard.reserve(4 * GiB):
                admitted.set()

        t = threading.Thread(target=parent_call, daemon=True)
        t.start()
        assert not admitted.wait(0.5)  # the child's bytes block it
        child.kill()  # OOM-killed: its finally never runs
        child.join(10)
        assert not admitted.wait(0.5)  # leaked until released...
        guard.release_owner(1)
        assert admitted.wait(5)  # ...then returned
        t.join(5)
        assert guard.stats()["reserved_bytes"] == 0
    finally:
        if child.is_alive():
            child.kill()


def _wait_in_child(guard, owner, nbytes, queued):
    guard.bind(owner)
    queued.set()
    with guard.reserve(nbytes):
        pass


def test_a_process_killed_while_waiting_does_not_wedge_the_guard():
    """With mp.Condition, a process killed in wait() left its sleeper count
    behind and the next notify_all blocked forever, holding the lock."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    guard = HostMemoryGuard(10 * GiB, mp_context=ctx, owners=2, lock_timeout_s=5)
    queued = ctx.Event()
    done = threading.Event()
    with guard.reserve(8 * GiB):  # blocks the child's 4 GiB
        child = ctx.Process(
            target=_wait_in_child, args=(guard, 1, 4 * GiB, queued), daemon=True
        )
        child.start()
        assert queued.wait(60)
        _wait_for(lambda: guard.stats()["calls_waiting"] == 1, timeout=30)
        child.kill()
        child.join(10)

        def after():
            guard.release_owner(1)
            done.set()

        threading.Thread(target=after, daemon=True).start()
        assert done.wait(10), "release_owner hung after a waiter was killed"
        assert guard.stats()["calls_waiting"] == 0
    with guard.reserve(9 * GiB):  # and the guard still admits
        assert guard.stats()["reserved_bytes"] == 9 * GiB


def test_a_stuck_lock_at_release_does_not_fail_the_call(caplog):
    guard = HostMemoryGuard(10 * GiB, large_fit_bytes=GiB, lock_timeout_s=0.2)
    holding, done = threading.Event(), threading.Event()

    def hold_lock():
        with guard._cond:
            holding.set()
            done.wait(5)

    with caplog.at_level("ERROR"), guard.reserve(2 * GiB):  # the call succeeds...
        t = threading.Thread(target=hold_lock, daemon=True)
        t.start()
        assert holding.wait(5)
        # ...and its release, unable to get the lock, logs instead of raising
    assert "not returned" in caplog.text
    done.set()
    t.join(5)


def test_bind_rejects_unknown_owners():
    with pytest.raises(ValueError):
        HostMemoryGuard(GiB, owners=2).bind(2)


def _settles(predicate, within=0.3):
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.01)


class GatedFit(FakeBackend):
    def __init__(self) -> None:
        super().__init__(bytes_hint=10)
        self.gate = threading.Event()
        self.entered = threading.Event()

    def fit(self, X, y, task):
        if len(X) >= 100:
            self.entered.set()
            assert self.gate.wait(5)
        return super().fit(X, y, task)


def test_a_waiting_large_fit_does_not_block_predicts_on_its_device():
    guard = HostMemoryGuard(GiB, large_fit_cells=100, large_fit_bytes=GiB)  # 1 fit
    backends = [GatedFit(), GatedFit()]
    est = AdaptiveMemoryEstimator(
        fallback=PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION)
    )
    engines = [
        TabctxEngine(
            backend=b,
            cache=ContextCacheManager(est.ceiling_bytes()),
            estimator=est,
            host_guard=guard,
        )
        for b in backends
    ]
    engines[1].fit([[1.0]] * 3, ["a"] * 3, dataset_id="small")
    big = ([[1.0]] * 100, ["a"] * 100)
    t0 = threading.Thread(target=engines[0].fit, args=big, daemon=True)
    t0.start()  # takes the only slot, then blocks inside the backend
    assert backends[0].entered.wait(5)
    t1 = threading.Thread(target=engines[1].fit, args=big, daemon=True)
    t1.start()  # waits for the host slot
    # Device 1 still serves predicts while its large fit waits for host memory.
    done = threading.Event()
    threading.Thread(
        target=lambda: (engines[1].predict("small", [[1.0]]), done.set()), daemon=True
    ).start()
    assert done.wait(3), "a fit waiting on host memory blocked its device"
    assert guard.stats()["calls_waiting"] == 1
    backends[0].gate.set()
    backends[1].gate.set()
    t0.join(5)
    t1.join(5)


def test_budget_from_cgroup_v2_minus_baseline_rss(tmp_path):
    from tabctx.memory.host import detect_host_budget_bytes

    (tmp_path / "memory.max").write_text(f"{64 * GiB}\n")
    (tmp_path / "status").write_text("Name: python\nVmRSS:\t 4194304 kB\n")  # 4 GiB
    budget = detect_host_budget_bytes(
        fraction=0.85,
        reserve_bytes=8 * GiB,
        limit_paths=(str(tmp_path / "memory.max"),),
        meminfo_path=str(tmp_path / "none"),
        status_path=str(tmp_path / "status"),
    )
    assert budget == int(64 * GiB * 0.85) - 4 * GiB - 8 * GiB


def test_unlimited_cgroup_falls_back_to_physical_memory(tmp_path):
    from tabctx.memory.host import detect_host_budget_bytes

    (tmp_path / "v2").write_text("max\n")
    (tmp_path / "v1").write_text("9223372036854771712\n")  # v1 "unlimited"
    (tmp_path / "meminfo").write_text("MemTotal:       16777216 kB\n")  # 16 GiB
    budget = detect_host_budget_bytes(
        fraction=1.0,
        reserve_bytes=0,
        limit_paths=(str(tmp_path / "v2"), str(tmp_path / "v1")),
        meminfo_path=str(tmp_path / "meminfo"),
        status_path=str(tmp_path / "none"),
    )
    assert budget == 16 * GiB


def test_unreadable_limit_means_no_budget(tmp_path):
    from tabctx.memory.host import detect_host_budget_bytes

    assert (
        detect_host_budget_bytes(
            limit_paths=(str(tmp_path / "x"),), meminfo_path=str(tmp_path / "y")
        )
        is None
    )


def test_budget_below_one_large_fit_warns(caplog):
    with caplog.at_level("WARNING"):
        assert HostMemoryGuard(20 * GiB, large_fit_bytes=48 * GiB).slots == 1
    assert "below one large fit" in caplog.text


class _Offloading(FakeBackend):
    """A backend that says its calls offload (like TabICL on a busy card)."""

    def __init__(self, offloads: bool) -> None:
        super().__init__(bytes_hint=10)
        self.offloads = offloads
        self.queries: list[tuple[int, int]] = []

    def may_offload(self, n_rows, n_features):
        self.queries.append((n_rows, n_features))
        return self.offloads


def _spy(guard):
    seen = []
    real = guard.reserve
    guard.reserve = lambda nbytes, offloads=False: (
        seen.append((nbytes, offloads)),
        real(nbytes, offloads=offloads),
    )[1]
    return seen


def _engine(backend, guard):
    est = AdaptiveMemoryEstimator(
        fallback=PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION)
    )
    return TabctxEngine(
        backend=backend,
        cache=ContextCacheManager(est.ceiling_bytes()),
        estimator=est,
        host_guard=guard,
    )


def test_backend_hint_decides_and_predicts_are_guarded_too():
    guard = HostMemoryGuard(GiB, large_fit_cells=10**12, large_fit_bytes=GiB)
    backend = _Offloading(offloads=True)
    engine = _engine(backend, guard)
    seen = _spy(guard)
    engine.fit([[1.0, 2.0]] * 3, ["a"] * 3, dataset_id="d")
    engine.predict("d", [[1.0, 2.0]] * 5)
    # The hint, not the (huge) cell threshold, decided -- for fit AND predict.
    assert seen == [(GiB, True), (GiB, True)]
    assert backend.queries == [(3, 2), (5, 2)]


def test_without_a_hint_the_cell_threshold_decides():
    guard = HostMemoryGuard(GiB, large_fit_cells=6, large_fit_bytes=GiB)
    engine = _engine(FakeBackend(bytes_hint=10), guard)
    seen = _spy(guard)
    engine.fit([[1.0, 2.0]] * 3, ["a"] * 3, dataset_id="d")  # 6 cells: large
    engine.predict("d", [[1.0, 2.0]])  # 2 cells: not
    assert seen == [(GiB, True), (0, False)]


def test_a_backend_estimate_wins_over_the_fallbacks():
    class Estimating(_Offloading):
        def host_bytes_estimate(self, n_rows, n_features, *, offloads=False):
            return 7 * n_rows * n_features * (2 if offloads else 1)

    guard = HostMemoryGuard(GiB, large_fit_cells=1, large_fit_bytes=GiB)
    engine = _engine(Estimating(offloads=True), guard)
    seen = _spy(guard)
    engine.fit([[1.0, 2.0]] * 3, ["a"] * 3, dataset_id="d")
    assert seen == [(84, True)]


def test_a_two_argument_host_estimate_still_works():
    """Backends written before the engine passed `offloads=`."""

    class OldStyle(FakeBackend):
        def host_bytes_estimate(self, n_rows, n_features):
            return 5 * n_rows * n_features

    guard = HostMemoryGuard(GiB, large_fit_bytes=GiB, min_guarded_bytes=0)
    engine = _engine(OldStyle(bytes_hint=10), guard)
    seen = _spy(guard)
    engine.fit([[1.0, 2.0]] * 3, ["a"] * 3, dataset_id="d")
    assert seen == [(30, False)]  # 6 cells: under the large-fit threshold
