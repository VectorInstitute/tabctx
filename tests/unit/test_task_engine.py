"""Survival and causal fits through the engine: a fit goes to a model
that serves its task, and a predict's quantiles and survival times are
checked against the context's task (fake backends: no torch)."""

import pytest

from tabctx.backends.fake import FakeBackend
from tabctx.batching import CoalescingPredictor
from tabctx.cache.manager import ContextCacheManager
from tabctx.engine import TabctxEngine
from tabctx.errors import InvalidInputError
from tabctx.memory import (
    A100_40GB_TABICL_CALIBRATION,
    AdaptiveMemoryEstimator,
    PowerLawMemoryEstimator,
)
from tabctx.tasks import PREDICTIVE

X = [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]]
SURVIVAL_Y = [[1.0, 1], [2.0, 0], [3.0, 1], [4.0, 1]]
CAUSAL_Y = [[1.0, 0], [2.0, 1], [1.5, 0], [3.5, 1]]


class PredictiveFake(FakeBackend):
    """A model that serves classification and regression only."""

    tasks = PREDICTIVE


def _estimator():
    return AdaptiveMemoryEstimator(
        fallback=PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION)
    )


@pytest.fixture
def engine():
    predictive = PredictiveFake(bytes_hint=100, name="predictive")
    every = FakeBackend(bytes_hint=100, name="every")
    est = _estimator()
    eng = TabctxEngine(
        backends={"predictive": predictive, "every": every},
        estimators={"predictive": est, "every": _estimator()},
        cache=ContextCacheManager(capacity_bytes=est.ceiling_bytes()),
        default_backend="predictive",
    )
    return eng, predictive, every


class TestModelChoice:
    def test_lists_each_models_tasks(self, engine):
        eng, _, _ = engine
        assert eng.backend_tasks["predictive"] == PREDICTIVE
        assert "survival" in eng.backend_tasks["every"]

    def test_a_survival_fit_goes_to_a_model_that_serves_it(self, engine):
        eng, predictive, every = engine
        eng.fit(X, SURVIVAL_Y, task="survival", dataset_id="s")
        assert (predictive.fit_calls, every.fit_calls) == (0, 1)
        (info,) = eng.contexts()
        assert (info.model, info.task) == ("every", "survival")
        assert eng.resolve_backend_name(None, "survival") == "every"
        assert eng.resolve_backend_name(None, "regression") == "predictive"

    def test_a_named_model_that_does_not_serve_the_task(self, engine):
        eng, predictive, _ = engine
        with pytest.raises(InvalidInputError, match="use model 'every'"):
            eng.fit(X, SURVIVAL_Y, task="survival", backend="predictive")
        assert predictive.fit_calls == 0

    def test_the_targets_are_checked_before_any_fit(self, engine):
        eng, _, every = engine
        with pytest.raises(InvalidInputError, match="at least one event"):
            eng.fit(X, [[1.0, 0]] * 4, task="survival")
        assert every.fit_calls == 0


class TestPredictOutputs:
    def test_survival_times_and_quantiles(self, engine):
        eng, _, _ = engine
        eng.fit(X, SURVIVAL_Y, task="survival", dataset_id="s")
        out = eng.predict("s", X[:2], quantiles=[0.5], times=[0.0, 2.5, 10.0])
        assert out.predictions == [2.5, 2.5]  # the fake: the times' median
        assert out.quantiles == [[2.5], [2.5]]
        assert out.survival == [[1.0, 0.5, 0.0]] * 2

    def test_rmst_and_potential_outcomes(self, engine):
        eng, _, _ = engine
        eng.fit(X, SURVIVAL_Y, task="survival", dataset_id="s")
        # The fake: the mean of the training times capped at the horizon.
        assert eng.predict("s", X[:1], rmst_horizon=3.0).rmst == [2.25]
        eng.fit(X, CAUSAL_Y, task="causal", dataset_id="c")
        out = eng.predict("c", X[:2], potential_outcomes=True)
        assert out.potential_outcomes == [[0.0, pytest.approx(1.5)]] * 2

    def test_no_class_probabilities_for_survival_or_causal(self, engine):
        eng, _, _ = engine
        eng.fit(X, SURVIVAL_Y, task="survival", dataset_id="s")
        with pytest.raises(
            InvalidInputError, match="return_proba is for classification"
        ):
            eng.predict("s", X, return_proba=True)

    def test_quantiles_stay_positional(self, engine):
        eng, _, _ = engine
        eng.fit(X, SURVIVAL_Y, task="survival", dataset_id="s")
        assert eng.predict("s", X[:1], False, [0.5]).quantiles == [[2.5]]

    def test_times_are_for_survival_contexts(self, engine):
        eng, _, _ = engine
        eng.fit(X, CAUSAL_Y, task="causal", dataset_id="c")
        with pytest.raises(InvalidInputError, match="times are for survival"):
            eng.predict("c", X, times=[1.0])

    def test_causal_contexts_return_quantiles(self, engine):
        eng, _, _ = engine
        eng.fit(X, CAUSAL_Y, task="causal", dataset_id="c")
        out = eng.predict("c", X[:1], quantiles=[0.1, 0.9])
        assert out.predictions == [pytest.approx(1.5)]  # 2.75 - 1.25
        assert len(out.quantiles[0]) == 2

    def test_classification_returns_no_quantiles(self, engine):
        eng, _, _ = engine
        eng.fit(X, ["a", "b", "a", "b"], task="classification", dataset_id="k")
        with pytest.raises(InvalidInputError, match="class probabilities"):
            eng.predict("k", X, quantiles=[0.5])

    def test_a_model_whose_predict_takes_no_times(self):
        class NoTimes(FakeBackend):
            def predict(self, payload, X_test, return_proba=False, quantiles=None):
                return super().predict(payload, X_test, return_proba, quantiles)

        backend = NoTimes(bytes_hint=100)
        est = _estimator()
        eng = TabctxEngine(
            backend=backend,
            estimator=est,
            cache=ContextCacheManager(capacity_bytes=est.ceiling_bytes()),
        )
        eng.fit(X, SURVIVAL_Y, task="survival", dataset_id="s")
        with pytest.raises(
            InvalidInputError, match="can't return survival probabilities"
        ):
            eng.predict("s", X, times=[1.0])
        assert eng.predict("s", X).predictions  # every other predict works

    @pytest.mark.parametrize("times", [[], [-1.0], [float("inf")], "30"])
    def test_bad_times(self, engine, times):
        eng, _, _ = engine
        eng.fit(X, SURVIVAL_Y, task="survival", dataset_id="s")
        with pytest.raises(InvalidInputError, match="times"):
            eng.predict("s", X, times=times)


def test_coalescing_keeps_requests_for_different_times_apart(engine):
    eng, _, _ = engine
    eng.fit(X, SURVIVAL_Y, task="survival", dataset_id="s")
    predictor = CoalescingPredictor(eng, window_s=0)
    a = predictor.predict("s", X[:1], times=[0.0])
    b = predictor.predict("s", X[:1], times=[10.0])
    assert (a.survival, b.survival) == ([[1.0]], [[0.0]])
