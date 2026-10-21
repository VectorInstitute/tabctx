"""Inline JSON tables -> arrays (serve/tables.py)."""

import math

import numpy as np

from tabctx.serve.tables import rows_to_array


def test_null_cells_become_nan():
    arr = rows_to_array([[1.0, None], [None, 2.0]])
    assert isinstance(arr, np.ndarray) and arr.dtype == float
    assert math.isnan(arr[0, 1]) and math.isnan(arr[1, 0])
    assert arr[0, 0] == 1.0 and arr[1, 1] == 2.0


def test_non_rectangles_pass_through_for_the_engine_to_report():
    ragged = [[1.0, 2.0], [3.0]]
    assert rows_to_array(ragged) is ragged
    assert rows_to_array([]) == []


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
    X = rows_to_array([[float(i), None if i % 3 else 1.0] for i in range(10)])
    ds = engine.fit(X, ["a", "b"] * 5)
    out = engine.predict(ds, rows_to_array([[None, 1.0], [2.0, None]]))
    assert len(out.predictions) == 2
