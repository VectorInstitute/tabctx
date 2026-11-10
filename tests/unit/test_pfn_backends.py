"""The SurvivalPFN and CausalPFN backends' parts that need neither torch nor
the models: missing values, JSON-safe outputs, the spill format."""

import numpy as np
import pytest

from tabctx.backends._pfn_common import array_bytes, dumps_context, loads_context
from tabctx.backends.causalpfn import _column_medians, _fill
from tabctx.backends.survivalpfn import _finite_or_none


def test_missing_values_are_filled_with_each_columns_median():
    X = np.array([[1.0, np.nan, np.nan], [3.0, 4.0, np.nan], [np.nan, 6.0, np.nan]])
    medians = _column_medians(X.astype(np.float32))
    assert medians.tolist() == [2.0, 5.0, 0.0]  # 0 for a column with none
    filled = _fill(X.astype(np.float32), medians)
    assert filled.tolist() == [[1.0, 5.0, 0.0], [3.0, 4.0, 0.0], [2.0, 6.0, 0.0]]
    whole = np.ones((2, 3), dtype=np.float32)
    assert _fill(whole, medians) is whole


def test_times_past_follow_up_are_none():
    assert _finite_or_none(np.array([1.5, np.inf])) == [1.5, None]
    assert _finite_or_none(np.array([[1.0, np.inf], [2.0, 3.0]])) == [
        [1.0, None],
        [2.0, 3.0],
    ]


class _Estimator:
    def __init__(self) -> None:
        self.icl_model = object()  # stands in for the network
        self.device = "cuda:0"
        self.X_train = np.zeros((3, 2), dtype=np.float32)


class _Context:
    def __init__(self) -> None:
        self.estimator = _Estimator()


def test_a_context_spills_without_its_network():
    context = _Context()
    network = context.estimator.icl_model
    data = dumps_context(context, b"T1:")
    assert context.estimator.icl_model is network  # put back
    other = object()
    back = loads_context(data, b"T1:", other, "cuda:1")
    assert back.estimator.icl_model is other
    assert back.estimator.device == "cuda:1"
    with pytest.raises(ValueError, match="model version"):
        loads_context(data, b"T2:", other, "cpu")


def test_array_bytes():
    assert array_bytes(np.zeros((3, 2), dtype=np.float32), None) == 24
