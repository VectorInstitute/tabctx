import pytest

from tabctx.backends.fake import FakeBackend
from tabctx.cache.manager import ContextCacheManager
from tabctx.engine import TabctxEngine
from tabctx.errors import AdmissionRejected, DatasetNotFoundError
from tabctx.memory.calibration_data import A100_40GB_TABICL_CALIBRATION
from tabctx.memory.estimator import PowerLawMemoryEstimator

TRAIN_X = [[float(i), float(i) * 2] for i in range(20)]
TRAIN_Y = ["a" if i % 2 == 0 else "b" for i in range(20)]
TEST_X = [[1.0, 2.0], [3.0, 4.0]]


def make_engine(backend=None, capacity_bytes=200_000_000):
    # NB: the power-law estimator, calibrated on 6,000-5,200,000 cells,
    # overestimates badly for tiny inputs like this test's 20x2 table (far
    # below the calibration range) -- see PowerLawMemoryEstimator's
    # docstring and the "small-input extrapolation" gap in the README.
    # 200MB comfortably accommodates that overestimate for these tests.
    backend = backend or FakeBackend()
    cache = ContextCacheManager(capacity_bytes=capacity_bytes)
    estimator = PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION)
    return TabctxEngine(backend=backend, cache=cache, estimator=estimator), backend


def test_fit_then_predict_reuses_cached_context_without_refitting():
    engine, backend = make_engine()
    dataset_id = engine.fit(TRAIN_X, TRAIN_Y, task="classification")
    assert backend.fit_calls == 1

    engine.predict(dataset_id, TEST_X)
    engine.predict(dataset_id, TEST_X)  # different call, same cached context

    assert backend.fit_calls == 1, "predict() must not re-fit an already-cached context"
    assert backend.predict_calls == 2


def test_predict_unknown_dataset_id_raises():
    engine, _ = make_engine()
    with pytest.raises(DatasetNotFoundError):
        engine.predict("never-fit", TEST_X)


def test_return_proba_is_a_single_backend_call_not_two():
    engine, backend = make_engine()
    dataset_id = engine.fit(TRAIN_X, TRAIN_Y, task="classification")
    backend.predict_calls = 0
    result = engine.predict(dataset_id, TEST_X, return_proba=True)
    assert backend.predict_calls == 1, (
        "return_proba=True must not cost a second inference pass -- this is "
        "exactly the double predict()/predict_proba() bug tabctx replaces"
    )
    assert result.probabilities is not None
    assert result.classes is not None


def test_fit_predict_evicts_after_one_shot():
    engine, _ = make_engine()
    result = engine.fit_predict(TRAIN_X, TRAIN_Y, TEST_X)
    assert len(result.predictions) == len(TEST_X)
    assert engine.stats().n_cached_contexts == 0


def test_context_bytes_hint_is_queried_after_fit_not_before():
    # Regression test: cache-accounting size must come from the backend
    # AFTER fit() runs (so a real backend can report an actual measurement
    # of the fit that just happened), not before. Found to matter a lot in
    # practice -- the pre-fit formula-based estimate ran ~14x higher than
    # real measured GPU memory for a realistic shape on real hardware.
    calls = []

    class OrderTrackingBackend(FakeBackend):
        def fit(self, X, y, task):
            calls.append("fit")
            return super().fit(X, y, task)

        def context_bytes_hint(self, n_train, n_features):
            calls.append("context_bytes_hint")
            return super().context_bytes_hint(n_train, n_features)

    engine, _ = make_engine(backend=OrderTrackingBackend(bytes_hint=1234))
    engine.fit(TRAIN_X, TRAIN_Y)
    assert calls == ["fit", "context_bytes_hint"], (
        f"expected fit() before context_bytes_hint(), got order {calls}"
    )


def test_fit_predict_evicts_even_on_predict_failure():
    class FailingBackend(FakeBackend):
        def predict(self, payload, X_test, return_proba=False):
            del payload, X_test, return_proba
            raise RuntimeError("boom")

    engine, _ = make_engine(backend=FailingBackend())
    with pytest.raises(RuntimeError):
        engine.fit_predict(TRAIN_X, TRAIN_Y, TEST_X)
    assert engine.stats().n_cached_contexts == 0


def test_fit_rejects_oversized_training_shape_before_calling_backend():
    engine, backend = make_engine()
    huge_X = [[0.0] * 200 for _ in range(90_000)]
    huge_y = ["a"] * 90_000
    with pytest.raises(AdmissionRejected):
        engine.fit(huge_X, huge_y)
    assert backend.fit_calls == 0, "admission control must reject before touching the backend"


def test_predict_chunks_large_test_sets_transparently():
    engine, _ = make_engine()
    dataset_id = engine.fit(TRAIN_X, TRAIN_Y, task="classification")
    big_test = [[float(i), float(i)] for i in range(5_000)]
    result = engine.predict(dataset_id, big_test)
    assert len(result.predictions) == 5_000
    # Small training context shouldn't need chunking at all for 5k rows in
    # this estimator, but the important invariant is correctness regardless:
    # every row gets exactly one prediction, in order.


def test_stats_reflect_cached_contexts():
    engine, _ = make_engine()
    assert engine.stats().n_cached_contexts == 0
    dataset_id = engine.fit(TRAIN_X, TRAIN_Y)
    assert engine.stats().n_cached_contexts == 1
    engine.evict(dataset_id)
    assert engine.stats().n_cached_contexts == 0
