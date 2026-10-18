"""Per-device scheduling in TabctxEngine: light predicts overlap a running
fit; large predicts, restores and fits serialize; evict-ahead never takes a
context a predict is using. Interleavings are forced with gates, never
timing assertions."""

import threading

import pytest

from tabctx.backends.fake import FakeBackend
from tabctx.cache.manager import ContextCacheManager
from tabctx.cache.spill import DiskSpillStore
from tabctx.engine import LIGHT_PREDICT_ROWS, TabctxEngine
from tabctx.errors import BackendComputeError

BYTES_PER_CELL = 10
CAPACITY = 10_000


class CellEstimator:
    """Need = resident = cells x BYTES_PER_CELL; headroom = what the cache
    doesn't hold."""

    def estimate_bytes(self, n_train, n_test, n_features):
        return (n_train + n_test) * n_features * BYTES_PER_CELL

    def admit(self, n_train, n_test, n_features):
        return self.estimate_bytes(n_train, n_test, n_features) <= CAPACITY

    def ceiling_bytes(self):
        return CAPACITY

    def admission_headroom_bytes(self, used_bytes):
        return max(0, CAPACITY - used_bytes)

    def confidence(self):
        return "test"

    def record_observation(self, n_train, n_features, real_bytes):
        pass


class GatedFake(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.fit_gate: threading.Event | None = None
        self.predict_gate: threading.Event | None = None
        self.fit_entered = threading.Event()
        self.predict_entered = threading.Event()
        self.fail_next_fits = 0
        self._last = None

    def fit(self, X, y, task):
        self._last = len(X) * len(X[0]) * BYTES_PER_CELL
        if self.fit_gate is not None:
            self.fit_entered.set()
            assert self.fit_gate.wait(5), "fit gate never released"
        if self.fail_next_fits:
            self.fail_next_fits -= 1
            raise BackendComputeError("CUDA OOM during fit()")
        return super().fit(X, y, task)

    def predict(self, payload, X_test, return_proba=False):
        if self.predict_gate is not None:
            self.predict_entered.set()
            assert self.predict_gate.wait(5), "predict gate never released"
        return super().predict(payload, X_test, return_proba)

    def context_bytes_hint(self, n_train, n_features):
        return self._last


def build(tmp_path=None):
    backend = GatedFake()
    spill = DiskSpillStore(tmp_path) if tmp_path else None
    engine = TabctxEngine(
        backend=backend,
        estimator=CellEstimator(),
        cache=ContextCacheManager(CAPACITY, spill_store=spill),
    )
    return engine, backend


def table(rows, label="a"):
    return [[float(i)] for i in range(rows)], [label] * rows


def start(fn, *args, **kw):
    box: dict = {}

    def run():
        try:
            box["result"] = fn(*args, **kw)
        except Exception as e:  # noqa: BLE001
            box["error"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, box


def test_light_predict_runs_while_a_fit_is_in_progress():
    engine, backend = build()
    engine.fit(*table(10, "x"), dataset_id="small")
    backend.fit_gate = threading.Event()
    fitting, _ = start(engine.fit, *table(100), dataset_id="big")
    assert backend.fit_entered.wait(5)
    # The fit is mid-flight; a small predict doesn't wait for it.
    assert engine.predict("small", [[0.0]]).predictions == ["x"]
    backend.fit_gate.set()
    fitting.join(5)
    assert engine._cache.holds("big")


def test_large_predict_waits_for_the_fit():
    engine, backend = build()
    engine.fit(*table(10, "x"), dataset_id="small")
    backend.fit_gate = threading.Event()
    fitting, _ = start(engine.fit, *table(100), dataset_id="big")
    assert backend.fit_entered.wait(5)
    rows = [[0.0]] * (LIGHT_PREDICT_ROWS + 1)
    predicting, box = start(engine.predict, "small", rows)
    predicting.join(0.2)
    assert predicting.is_alive() and backend.predict_calls == 0
    backend.fit_gate.set()
    fitting.join(5)
    predicting.join(5)
    assert len(box["result"].predictions) == LIGHT_PREDICT_ROWS + 1


def test_restoring_a_spilled_context_waits_for_the_fit(tmp_path):
    engine, backend = build(tmp_path)
    engine.fit(*table(10, "x"), dataset_id="cold")
    engine._cache.make_room(CAPACITY)  # spill it
    assert not engine._cache.holds("cold") and engine._cache.has("cold")
    backend.fit_gate = threading.Event()
    fitting, _ = start(engine.fit, *table(100), dataset_id="big")
    assert backend.fit_entered.wait(5)
    predicting, box = start(engine.predict, "cold", [[0.0]])
    predicting.join(0.2)
    assert predicting.is_alive()  # a restore takes device memory
    backend.fit_gate.set()
    fitting.join(5)
    predicting.join(5)
    assert box["result"].predictions == ["x"]


def test_evict_ahead_skips_a_context_in_use_then_takes_it_once_released(tmp_path):
    engine, backend = build(tmp_path)
    engine.fit(*table(600, "busy"), dataset_id="busy")  # 6,000 of 10,000 bytes
    backend.predict_gate = threading.Event()
    predicting, box = start(engine.predict, "busy", [[0.0]])
    assert backend.predict_entered.wait(5)
    # Needs 6,000 bytes: only evicting "busy" makes room, and it is pinned.
    fitting, _ = start(engine.fit, *table(600), dataset_id="next")
    fitting.join(0.2)
    assert fitting.is_alive() and engine._cache.holds("busy")
    backend.predict_gate.set()
    predicting.join(5)
    fitting.join(5)
    assert box["result"].predictions == ["busy"]
    assert engine._cache.holds("next") and not engine._cache.holds("busy")
    assert engine._cache.has("busy")  # spilled, not lost


def test_evict_ahead_prefers_unpinned_victims(tmp_path):
    engine, backend = build(tmp_path)
    engine.fit(*table(300, "busy"), dataset_id="busy")
    engine.fit(*table(300, "idle"), dataset_id="idle")
    backend.predict_gate = threading.Event()
    predicting, _ = start(engine.predict, "busy", [[0.0]])
    assert backend.predict_entered.wait(5)
    # LRU would pick "busy" (older), but it's in use.
    engine.fit(*table(500), dataset_id="new")
    assert engine._cache.holds("busy") and not engine._cache.holds("idle")
    backend.predict_gate.set()
    predicting.join(5)


def test_a_fit_that_runs_out_of_memory_retries_alone():
    engine, backend = build()
    backend.fail_next_fits = 1
    engine.fit(*table(10, "x"), dataset_id="d")
    assert backend.fit_calls == 1  # the failed attempt raised before counting
    assert engine.predict("d", [[0.0]]).predictions == ["x"]


def test_a_fit_failing_twice_raises():
    engine, backend = build()
    backend.fail_next_fits = 2
    with pytest.raises(BackendComputeError):
        engine.fit(*table(10), dataset_id="d")
    assert not engine._cache.has("d")


def test_stats_never_wait_for_a_fit():
    engine, backend = build()
    backend.fit_gate = threading.Event()
    fitting, _ = start(engine.fit, *table(100), dataset_id="big")
    assert backend.fit_entered.wait(5)
    assert engine.stats().n_cached_contexts == 0  # the cache lock is free
    backend.fit_gate.set()
    fitting.join(5)


# ---- regressions from the review of 2e801ae --------------------------------


def test_large_fit_and_large_predict_with_one_host_slot_do_not_deadlock():
    """fit takes host slot -> _fit_lock; a large predict used to take
    _fit_lock -> host slot. Forced: a small (slot-free) fit holds the
    device while a large predict queues for it and then a large fit takes
    the only host slot."""
    from tabctx.memory.host import HostMemoryGuard

    backend = GatedFake()
    guard = HostMemoryGuard(48 * 1024**3, large_fit_cells=50)  # 1 slot
    engine = TabctxEngine(
        backend=backend,
        estimator=CellEstimator(),
        cache=ContextCacheManager(CAPACITY),
        host_guard=guard,
    )
    engine.fit(*table(100, "x"), dataset_id="d")
    backend.fit_gate = threading.Event()
    small, _ = start(engine.fit, *table(10), dataset_id="small")  # 10 cells
    assert backend.fit_entered.wait(5)
    rows = [[0.0]] * (LIGHT_PREDICT_ROWS + 1)
    predicting, pbox = start(engine.predict, "d", rows)
    predicting.join(0.2)
    fitting, fbox = start(engine.fit, *table(100), dataset_id="big")
    fitting.join(0.2)
    backend.fit_gate.set()
    for t in (small, predicting, fitting):
        t.join(10)
    assert not predicting.is_alive() and not fitting.is_alive(), "deadlocked"
    assert "error" not in pbox and "error" not in fbox


def test_a_busy_context_is_evicted_when_nothing_else_makes_room(tmp_path):
    """Back-to-back predicts kept a hot context pinned forever; the fit
    went ahead without room and the cache went over capacity."""
    engine, backend = build(tmp_path)
    engine.fit(*table(600, "hot"), dataset_id="hot")
    stop = threading.Event()
    errors: list = []

    def hammer():
        while not stop.is_set():
            try:
                assert engine.predict("hot", [[0.0]]).predictions == ["hot"]
            except Exception as e:  # noqa: BLE001
                errors.append(e)
                return

    hammers = [threading.Thread(target=hammer, daemon=True) for _ in range(3)]
    for t in hammers:
        t.start()
    engine.fit(*table(600), dataset_id="new")  # needs hot's room
    stats = engine.stats()
    assert stats.used_bytes <= stats.capacity_bytes
    assert engine._cache.holds("new")
    stop.set()
    for t in hammers:
        t.join(10)
    assert errors == []  # predicts on "hot" restored it as needed


def test_contexts_evicted_during_a_fit_are_freed_after_it_measures():
    engine, backend = build()
    engine.fit(*table(10), dataset_id="old")
    backend.fit_gate = threading.Event()
    fitting, _ = start(engine.fit, *table(100), dataset_id="big")
    assert backend.fit_entered.wait(5)
    engine.evict("old")
    assert not engine._cache.holds("old")
    assert [c.dataset_id for c in engine._parked] == ["old"]  # not freed yet
    backend.fit_gate.set()
    fitting.join(5)
    assert engine._parked == []


def test_a_restore_that_runs_out_of_memory_keeps_the_spilled_copy(tmp_path):
    engine, backend = build(tmp_path)
    engine.fit(*table(10, "x"), dataset_id="cold")
    engine._cache.make_room(CAPACITY)
    spill = engine._cache._spill
    real_load, calls = spill.load, []

    def oom_once(dataset_id):
        calls.append(dataset_id)
        if len(calls) == 1:
            meta_path, payload_path = spill._paths(dataset_id)
            assert meta_path.exists()
            raise BackendComputeError("CUDA OOM while deserializing")
        return real_load(dataset_id)

    spill.load = oom_once
    assert engine.predict("cold", [[0.0]]).predictions == ["x"]
    assert len(calls) == 2  # retried; the file survived the first OOM


class PreparingFake(GatedFake):
    """A backend that splits its fit into a CPU prepare and a GPU half."""

    def __init__(self) -> None:
        super().__init__()
        self.prepared_during: list[bool] = []
        self.engine: TabctxEngine | None = None

    def prepare_fit(self, X, y, task):
        # Record whether some fit held the device while this prepared.
        self.prepared_during.append(self.engine._fit_lock.locked())
        return (X, y, task)

    def fit_prepared(self, prepared):
        return GatedFake.fit(self, *prepared)


def test_a_fit_prepares_while_another_fit_holds_the_device():
    backend = PreparingFake()
    engine = TabctxEngine(
        backend=backend, estimator=CellEstimator(), cache=ContextCacheManager(CAPACITY)
    )
    backend.engine = engine
    backend.fit_gate = threading.Event()
    first, _ = start(engine.fit, *table(100), dataset_id="a")
    assert backend.fit_entered.wait(5)
    second, _ = start(engine.fit, *table(100), dataset_id="b")
    for _ in range(200):
        if len(backend.prepared_during) == 2:
            break
        threading.Event().wait(0.01)
    assert backend.prepared_during == [False, True]  # b prepared during a's fit
    backend.fit_gate.set()
    first.join(5)
    second.join(5)
    assert engine._cache.holds("a") and engine._cache.holds("b")


def test_an_infeasible_fit_is_rejected_before_preparing():
    from tabctx.errors import AdmissionRejected

    backend = PreparingFake()
    engine = TabctxEngine(
        backend=backend, estimator=CellEstimator(), cache=ContextCacheManager(CAPACITY)
    )
    backend.engine = engine
    with pytest.raises(AdmissionRejected):
        engine.fit(*table(2000), dataset_id="huge")  # 20,000 > 10,000 bytes
    assert backend.prepared_during == []
