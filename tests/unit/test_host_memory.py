"""Unit tests for process-wide host-memory admission (memory/host.py)."""

import threading

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
    seen = []
    real = guard.reserve
    guard.reserve = lambda nbytes: (seen.append(nbytes), real(nbytes))[1]
    engine.fit([[1.0, 2.0]] * 3, ["a"] * 3, dataset_id="d")
    engine.predict("d", [[1.0, 2.0]] * 5)
    # The hint, not the (huge) cell threshold, decided -- for fit AND predict.
    assert seen == [GiB, GiB]
    assert backend.queries == [(3, 2), (5, 2)]


def test_without_a_hint_the_cell_threshold_decides():
    guard = HostMemoryGuard(GiB, large_fit_cells=6, large_fit_bytes=GiB)
    engine = _engine(FakeBackend(bytes_hint=10), guard)
    seen = []
    real = guard.reserve
    guard.reserve = lambda nbytes: (seen.append(nbytes), real(nbytes))[1]
    engine.fit([[1.0, 2.0]] * 3, ["a"] * 3, dataset_id="d")  # 6 cells: large
    engine.predict("d", [[1.0, 2.0]])  # 2 cells: not
    assert seen == [GiB, 0]


def test_a_backend_estimate_wins_over_the_fallbacks():
    class Estimating(_Offloading):
        def host_bytes_estimate(self, n_rows, n_features):
            return 7 * n_rows * n_features

    guard = HostMemoryGuard(GiB, large_fit_cells=1, large_fit_bytes=GiB)
    engine = _engine(Estimating(offloads=True), guard)
    seen = []
    real = guard.reserve
    guard.reserve = lambda nbytes: (seen.append(nbytes), real(nbytes))[1]
    engine.fit([[1.0, 2.0]] * 3, ["a"] * 3, dataset_id="d")
    assert seen == [42]
