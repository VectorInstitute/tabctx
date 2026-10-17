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
