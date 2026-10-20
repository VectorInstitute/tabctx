import pytest

from tabctx.chunking import choose_chunk_rows, split_rows
from tabctx.memory.calibration_data import A100_40GB_TABICL_CALIBRATION
from tabctx.memory.estimator import PowerLawMemoryEstimator


def test_split_rows_preserves_order_and_all_elements():
    data = list(range(10))
    chunks = split_rows(data, chunk_rows=3)
    assert chunks == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert sum(chunks, []) == data


def test_split_rows_rejects_non_positive_chunk_size():
    with pytest.raises(ValueError):
        split_rows([1, 2, 3], chunk_rows=0)


def test_choose_chunk_rows_smaller_for_larger_training_context():
    est = PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION)
    # A budget the large context fits at 64 test rows (a halving of 2,048).
    budget = est.estimate_bytes(90_000, 64, 200)
    small_context_chunk = choose_chunk_rows(
        est, n_train=500, n_features=10, remaining_budget_bytes=budget
    )
    large_context_chunk = choose_chunk_rows(
        est, n_train=90_000, n_features=200, remaining_budget_bytes=budget
    )
    assert large_context_chunk == 64 < small_context_chunk
