"""Label checks that depend on the task (tabctx/labels.py), and their
wiring into the engine and both backends' rejection messages."""

import numpy as np
import pytest

from tabctx.errors import InvalidInputError
from tabctx.labels import infer_task, regression_hint, resolve_task, validate_labels


class TestValidateLabels:
    @pytest.mark.parametrize(
        "y",
        [
            ["a", "b", "a"],  # text classes
            [0, 1, 2, 1],  # integer-coded classes
            [0.0, 1.0, 1.0],  # integer-valued floats (JSON numbers)
            ["0", "1", "1"],  # CSV labels arrive as text
            ["yes", "1.5"],  # mixed text: not numeric, left to the backend
            ["250.01", "401.9"],  # numeric-looking text is still categories
            [True, False],
            [10**400, 1],  # beyond float range: whole, not checked further
        ],
    )
    def test_accepts_discrete_classes(self, y):
        validate_labels(y, "classification")

    @pytest.mark.parametrize(
        "y",
        [
            [212500.5, 180000.0, 99000.0],  # a price
            np.array([0.1, 0.2]),
            np.array([[0.1], [0.2]]),  # a column vector
        ],
    )
    def test_rejects_non_integer_numbers_with_the_fix_in_the_message(self, y):
        with pytest.raises(InvalidInputError, match="task='regression'"):
            validate_labels(y, "classification")

    def test_regression_takes_any_numbers(self):
        validate_labels([212500.5, 0.1], "regression")

    @pytest.mark.parametrize("task", ["classification", "regression", None])
    @pytest.mark.parametrize("y", [[1.0, float("nan")], np.array([0.5, np.inf])])
    def test_missing_labels_are_rejected_whatever_the_task(self, y, task):
        with pytest.raises(InvalidInputError, match="NaN/inf"):
            resolve_task(y, task)


class TestRegressionHint:
    def test_hints_for_numeric_classification_labels(self):
        assert "task='regression'" in regression_hint([1, 2, 3], "classification")

    def test_silent_for_text_labels_or_regression(self):
        assert regression_hint(["a", "b"], "classification") == ""
        assert regression_hint([1.5, 2.5], "regression") == ""


def test_engine_rejects_continuous_labels_before_any_backend_work():
    from tabctx.backends.fake import FakeBackend
    from tabctx.cache.manager import ContextCacheManager
    from tabctx.engine import TabctxEngine
    from tabctx.memory.calibration_data import A100_40GB_TABICL_CALIBRATION
    from tabctx.memory.estimator import PowerLawMemoryEstimator

    backend = FakeBackend()
    engine = TabctxEngine(
        backend=backend,
        cache=ContextCacheManager(capacity_bytes=200_000_000),
        estimator=PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION),
    )
    X = [[float(i), 1.0] for i in range(10)]
    with pytest.raises(InvalidInputError, match="task='regression'"):
        engine.fit(X, [i * 1.5 for i in range(10)], task="classification")
    assert backend.fit_calls == 0
    # The same table fits as regression.
    engine.fit(X, [i * 1.5 for i in range(10)], task="regression")


def test_tabicl_input_errors_are_invalid_input_not_500(monkeypatch):
    pytest.importorskip("torch")
    from tabctx.backends.tabicl import TabICLBackend

    class Rejecting:
        kv_cache = True

        def fit(self, X, y):
            raise ValueError("Unknown label type: continuous")

    backend = TabICLBackend.__new__(TabICLBackend)  # no weights needed
    backend._device = "cpu"
    backend._fast_preprocessing = False
    monkeypatch.setattr(
        backend, "_make_estimator", lambda task: Rejecting(), raising=False
    )
    with pytest.raises(InvalidInputError, match="tabicl rejected.*task='regression'"):
        backend.prepare_fit([[1.0], [2.0]], [1, 2], "classification")


class TestInferTask:
    @pytest.mark.parametrize(
        "y, task",
        [
            ([212500.5, 180000.0], "regression"),  # a continuous target
            (["4.526", "3.585"], "classification"),  # text: categories
            (["a", "b"], "classification"),
            ([0, 1, 1], "classification"),  # integer class codes stay classes
            ([1.0, 2.0], "classification"),  # whole numbers too
            (["0", "1"], "classification"),
        ],
    )
    def test_from_the_labels(self, y, task):
        assert infer_task(y) == task

    def test_resolve_infers_only_when_the_caller_did_not_say(self):
        assert resolve_task([0.5, 1.5], None) == "regression"
        assert resolve_task([0, 1], "regression") == "regression"
        with pytest.raises(InvalidInputError, match="task='regression'"):
            resolve_task([0.5, 1.5], "classification")


def test_engine_fits_the_inferred_task_when_none_is_given():
    from tabctx.backends.fake import FakeBackend
    from tabctx.cache.manager import ContextCacheManager
    from tabctx.engine import TabctxEngine
    from tabctx.memory.calibration_data import A100_40GB_TABICL_CALIBRATION
    from tabctx.memory.estimator import PowerLawMemoryEstimator

    engine = TabctxEngine(
        backend=FakeBackend(),
        cache=ContextCacheManager(capacity_bytes=200_000_000),
        estimator=PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION),
    )
    X = [[float(i), 1.0] for i in range(10)]
    reg = engine.fit(X, [i * 1.5 for i in range(10)])
    assert engine.predict(reg, X[:2], return_proba=True).classes is None  # regression
    clf = engine.fit(X, [i % 2 for i in range(10)])
    assert engine.predict(clf, X[:2], return_proba=True).classes == ["0", "1"]


def test_whole_number_quantities_are_inferred_as_regression():
    from tabctx.labels import MAX_INFERRED_CLASSES

    prices = [208500, 181500, 223500, 140000] * 10 + list(range(100, 200))
    assert infer_task(prices) == "regression"
    codes = list(range(MAX_INFERRED_CLASSES)) * 3  # many classes, still codes
    assert infer_task(codes) == "classification"
    states = [i % 50 for i in range(1000)]  # 50 codes, each seen ~20 times
    assert infer_task(states) == "classification"
    assert infer_task(list(range(MAX_INFERRED_CLASSES + 1))) == "regression"
    # An explicit classification of whole numbers is the caller's call.
    assert resolve_task(prices, "classification") == "classification"


def test_tabicl_reports_its_class_limit_instead_of_failing_the_cache_build(
    monkeypatch,
):
    pytest.importorskip("torch")
    from types import SimpleNamespace

    from tabctx.backends.tabicl import TabICLBackend

    class ManyClasses:
        kv_cache = True
        n_classes_ = 573
        model_ = SimpleNamespace(max_classes=10)

        def fit(self, X, y):
            return self

    backend = TabICLBackend.__new__(TabICLBackend)
    backend._device = "cpu"
    backend._fast_preprocessing = False
    monkeypatch.setattr(
        backend, "_make_estimator", lambda task: ManyClasses(), raising=False
    )
    with pytest.raises(InvalidInputError, match="at most 10 classes.*TabPFN: 160"):
        backend.prepare_fit([[1.0]] * 3, ["a", "b", "c"], "classification")


def test_tabicl_rejects_too_many_classes_before_preprocessing(monkeypatch):
    pytest.importorskip("torch")
    from types import SimpleNamespace

    from tabctx.backends.tabicl import TabICLBackend

    class Loaded:
        kv_cache = True
        model_ = SimpleNamespace(max_classes=10)

        def fit(self, X, y):
            raise AssertionError("preprocessing ran for a table it must reject")

    backend = TabICLBackend.__new__(TabICLBackend)
    backend._device = "cpu"
    backend._fast_preprocessing = False
    monkeypatch.setattr(
        backend, "_make_estimator", lambda task: Loaded(), raising=False
    )
    labels = [f"c{i}" for i in range(15)]
    with pytest.raises(
        InvalidInputError, match="at most 10 classes and this table has 15"
    ):
        backend.prepare_fit([[1.0]] * 15, labels, "classification")
