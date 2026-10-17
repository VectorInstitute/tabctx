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


def test_small_fits_never_wait():
    guard = HostMemoryGuard(GiB, large_fit_cells=100, large_fit_bytes=GiB)
    with guard.reserve(50, 1), guard.reserve(50, 1):  # both small: no slot used
        assert guard.stats()["large_fits_active"] == 0


def test_large_fits_are_limited_to_the_slots():
    guard = HostMemoryGuard(
        2 * GiB, large_fit_cells=100, large_fit_bytes=GiB
    )  # 2 slots
    inside, release = threading.Semaphore(0), threading.Event()

    def large():
        with guard.reserve(100, 1):
            inside.release()
            release.wait(5)

    threads = [threading.Thread(target=large, daemon=True) for _ in range(3)]
    for t in threads:
        t.start()
    assert inside.acquire(timeout=5) and inside.acquire(timeout=5)
    assert not inside.acquire(timeout=0.3)  # the third waits for a slot
    assert guard.stats() == {
        "large_fit_slots": 2,
        "large_fits_active": 2,
        "large_fits_waiting": 1,
    }
    release.set()
    for t in threads:
        t.join(5)
    assert guard.stats()["large_fits_active"] == 0


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
    guard = HostMemoryGuard(GiB, large_fit_cells=100, large_fit_bytes=GiB)  # 1 slot
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
    assert guard.stats()["large_fits_waiting"] == 1
    backends[0].gate.set()
    backends[1].gate.set()
    t0.join(5)
    t1.join(5)
