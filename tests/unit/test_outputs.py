"""tabctx.outputs: what a predict can ask for besides its predictions,
validated once and checked against the context's task and model."""

import pytest

from tabctx.backends.fake import FakeBackend
from tabctx.errors import InvalidInputError
from tabctx.outputs import PredictOutputs


def test_validates_and_keeps_only_what_was_asked():
    asked = PredictOutputs.of(quantiles=[0.9, 0.1], rmst_horizon=30)
    assert asked.asked() == {"quantiles": (0.9, 0.1), "rmst_horizon": 30.0}
    assert PredictOutputs.of().asked() == {}
    # Hashable and equal by value: the coalescer keys batches by it.
    assert hash(asked) == hash(
        PredictOutputs.of(quantiles=(0.9, 0.1), rmst_horizon=30.0)
    )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"rmst_horizon": 0}, "finite and positive"),
        ({"rmst_horizon": float("inf")}, "finite and positive"),
        ({"rmst_horizon": "soon"}, "must be a number"),
        ({"times": [-1]}, "non-negative"),
        ({"quantiles": [1.5]}, "between 0 and 1"),
    ],
)
def test_bad_values(kwargs, message):
    with pytest.raises(InvalidInputError, match=message):
        PredictOutputs.of(**kwargs)


def test_an_unknown_output_is_a_type_error():
    with pytest.raises(TypeError):
        PredictOutputs.of(hazard=[1.0])


@pytest.mark.parametrize(
    "kwargs, task, message",
    [
        ({"quantiles": [0.5]}, "classification", "class probabilities"),
        ({"times": [1.0]}, "causal", "times are for survival"),
        ({"rmst_horizon": 5.0}, "regression", "rmst_horizon is for survival"),
        ({"potential_outcomes": True}, "survival", "potential_outcomes are for causal"),
    ],
)
def test_each_output_is_for_its_tasks(kwargs, task, message):
    with pytest.raises(InvalidInputError, match=message):
        PredictOutputs.of(**kwargs).check(task, FakeBackend())


def test_a_model_that_cannot_give_an_output():
    class PointOnly(FakeBackend):
        def predict(self, payload, X_test, return_proba=False):
            return super().predict(payload, X_test, return_proba)

    with pytest.raises(InvalidInputError, match="can't return rmst horizon"):
        PredictOutputs.of(rmst_horizon=1.0).check("survival", PointOnly())
    PredictOutputs.of().check("survival", PointOnly())  # nothing asked: fine
