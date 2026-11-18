"""tabctx.tasks: which model fits a task, and a survival or causal fit's
two target columns."""

import numpy as np
import pytest

from tabctx.backends.fake import FakeBackend
from tabctx.errors import InvalidInputError
from tabctx.labels import resolve_task
from tabctx.tasks import (
    PREDICTIVE,
    choose_model,
    infer_task_with_second,
    pack_targets,
    split_targets,
    tasks_of,
    validate_targets,
)

SERVED = {
    "tabicl-v2": PREDICTIVE,
    "survivalpfn-v0.1": frozenset({"survival"}),
    "causalpfn-v0": frozenset({"causal"}),
}


class TestChooseModel:
    def test_the_default_when_it_fits_the_task(self):
        assert choose_model("regression", None, SERVED, "tabicl-v2") == "tabicl-v2"

    def test_the_first_model_that_fits_when_the_default_does_not(self):
        assert choose_model("survival", None, SERVED, "tabicl-v2") == "survivalpfn-v0.1"
        assert choose_model("causal", None, SERVED, "tabicl-v2") == "causalpfn-v0"

    def test_a_requested_model_that_fits(self):
        assert (
            choose_model("survival", "survivalpfn-v0.1", SERVED, "tabicl-v2")
            == "survivalpfn-v0.1"
        )

    def test_a_requested_model_that_does_not_fit_names_one_that_does(self):
        with pytest.raises(InvalidInputError, match="use model 'survivalpfn-v0.1'"):
            choose_model("survival", "tabicl-v2", SERVED, "tabicl-v2")
        with pytest.raises(InvalidInputError, match="serves survival, not regression"):
            choose_model("regression", "survivalpfn-v0.1", SERVED, "tabicl-v2")

    def test_an_unknown_model(self):
        with pytest.raises(InvalidInputError, match="unknown backend 'nope'"):
            choose_model("regression", "nope", SERVED, "tabicl-v2")

    def test_no_model_fits(self):
        only = {"tabicl-v2": PREDICTIVE}
        with pytest.raises(InvalidInputError, match="no model .* fits causal"):
            choose_model("causal", None, only, "tabicl-v2")


def test_a_backend_without_tasks_serves_classification_and_regression():
    class Old:
        name = "old"

    assert tasks_of(Old()) == PREDICTIVE
    assert tasks_of(FakeBackend()) >= {"survival", "causal"}


class TestPackTargets:
    def test_classification_and_regression_keep_y(self):
        y = ["a", "b"]
        assert pack_targets("classification", y, None) is y

    def test_survival_and_causal_stack_two_columns(self):
        y = pack_targets("survival", [3.0, 5.0], [1, 0])
        assert np.asarray(y).tolist() == [[3.0, 1.0], [5.0, 0.0]]
        times, events = split_targets(y)
        assert times.tolist() == [3.0, 5.0] and events.tolist() == [1.0, 0.0]

    def test_the_second_column_is_required(self):
        with pytest.raises(InvalidInputError, match="needs its event column"):
            pack_targets("survival", [1.0], None)
        with pytest.raises(InvalidInputError, match="needs its treatment column"):
            pack_targets("causal", [1.0], None)

    def test_a_stray_second_column(self):
        with pytest.raises(InvalidInputError, match="not regression"):
            pack_targets("regression", [1.0], [1])

    def test_lengths_must_match(self):
        with pytest.raises(InvalidInputError, match="2 values but event has 1"):
            pack_targets("survival", [1.0, 2.0], [1])

    def test_numbers_only(self):
        with pytest.raises(InvalidInputError, match="must be numbers"):
            pack_targets("causal", [1.0], ["yes"])


class TestValidateTargets:
    def test_accepts_usable_targets(self):
        validate_targets([[1.0, 1], [2.0, 0]], "survival")
        validate_targets([[1.0, 1], [2.0, 0]], "causal")

    @pytest.mark.parametrize(
        ("y", "task", "message"),
        [
            ([1.0, 2.0], "survival", "two values per training row"),
            ([[1.0, 2], [2.0, 0]], "survival", "event must be 0 or 1"),
            ([[1.0, 1], [float("nan"), 0]], "causal", "missing or infinite"),
            ([[-1.0, 1], [2.0, 0]], "survival", "can't be negative"),
            ([[1.0, 0], [2.0, 0]], "survival", "at least one event"),
            ([[1.0, 1], [2.0, 1]], "causal", "both treated"),
            ([[1.0, 0], [2.0, 0]], "causal", "both treated"),
        ],
    )
    def test_rejects(self, y, task, message):
        with pytest.raises(InvalidInputError, match=message):
            validate_targets(y, task)


class TestResolveTask:
    def test_two_columns_need_a_named_task(self):
        with pytest.raises(InvalidInputError, match="name the task"):
            resolve_task([[1.0, 1], [2.0, 0]], None)
        with pytest.raises(InvalidInputError, match="name the task"):
            resolve_task(np.array([[1.0, 1], [2.0, 0]]), None)

    def test_a_named_survival_task_is_validated(self):
        assert resolve_task([[1.0, 1], [2.0, 0]], "survival") == "survival"
        with pytest.raises(InvalidInputError, match="at least one event"):
            resolve_task([[1.0, 0], [2.0, 0]], "survival")

    def test_two_columns_are_not_a_regression(self):
        with pytest.raises(InvalidInputError, match="one label per row"):
            resolve_task([[1.0, 1], [2.0, 0]], "regression")


class TestInferTaskWithSecond:
    def test_the_column_sent_implies_the_task(self):
        assert infer_task_with_second(event=True, treatment=False) == "survival"
        assert infer_task_with_second(event=False, treatment=True) == "causal"
        assert infer_task_with_second(event=False, treatment=False) is None

    def test_not_both(self):
        with pytest.raises(InvalidInputError, match="not both"):
            infer_task_with_second(event=True, treatment=True)
