"""Inline JSON tables -> arrays (serve/tables.py)."""

import math

import numpy as np
import pytest

from tabctx.errors import InvalidInputError
from tabctx.schema import TableSchema
from tabctx.serve.tables import encode_rows, learn_rows, rows_to_array


def test_null_cells_become_nan():
    arr = rows_to_array([[1.0, None], [None, 2.0]])
    assert isinstance(arr, np.ndarray) and arr.dtype == np.float32
    assert math.isnan(arr[0, 1]) and math.isnan(arr[1, 0])
    assert arr[0, 0] == 1.0 and arr[1, 1] == 2.0


def test_numeric_tables_learn_an_all_numeric_schema():
    X, schema = learn_rows([[1.0, 2.0], [3.0, None]])
    assert X.dtype == np.float32 and X.shape == (2, 2)
    assert schema == TableSchema((None, None))


def test_text_cells_make_text_columns_encoded_the_same_at_predict():
    X, schema = learn_rows([[1.0, "gold"], [2.0, "basic"], [None, None]])
    assert schema.categories == (None, ("basic", "gold"))
    assert X[:2].tolist() == [[1.0, 1.0], [2.0, 0.0]]
    Xt = encode_rows([[5.0, "basic"], [6.0, "platinum"]], schema)
    assert Xt[0].tolist() == [5.0, 0.0]
    assert math.isnan(Xt[1, 1])  # unseen category -> missing


def test_numbers_for_a_text_column_are_matched_as_text():
    _, schema = learn_rows([["1"], ["2"], ["x"]])
    assert encode_rows([[2], [1]], schema).tolist() == [[1.0], [0.0]]


def test_text_where_training_had_numbers_is_rejected():
    _, schema = learn_rows([[1.0], [2.0]])
    with pytest.raises(InvalidInputError, match="held numbers"):
        encode_rows([["x"]], schema)
    # Contexts from before schemas existed: numbers only, same error.
    with pytest.raises(InvalidInputError, match="held numbers"):
        encode_rows([["x"]], None)


def test_non_rectangles_pass_through_for_the_engine_to_report():
    ragged = [[1.0, "a"], [3.0]]
    assert learn_rows(ragged) == (ragged, None)
    assert encode_rows(ragged, None) is ragged
    assert learn_rows([]) == ([], None)


def test_engine_fits_and_predicts_tables_with_missing_values():
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
    X, schema = learn_rows([[float(i), None if i % 3 else "a"] for i in range(10)])
    ds = engine.fit(X, ["a", "b"] * 5, schema=schema)
    out = engine.predict(ds, encode_rows([[None, "a"], [2.0, None]], schema))
    assert len(out.predictions) == 2


def test_json_rows_get_the_offsets_of_a_file_fit_context():
    # A time column fit from a file is stored from its training minimum;
    # JSON rows (float64 until encoded) must be shifted the same way.
    schema, _ = TableSchema.learn(
        [np.array([1577872800.5, 1577872801.0]), np.array([1.0, 2.0], np.float32)]
    )
    assert schema.offsets == (1577872800.5, None)
    assert encode_rows([[1577872801.5, 1.0]], schema).tolist() == [[1.0, 1.0]]
