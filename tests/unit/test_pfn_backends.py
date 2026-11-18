"""The SurvivalPFN and CausalPFN backends' parts that need neither torch nor
the models: missing values, JSON-safe outputs, the spill format."""

import numpy as np
import pytest

from tabctx.backends._pfn_common import (
    array_bytes,
    dumps_context,
    loads_context,
    strata,
    subsample,
    tagged_dumps,
    tagged_loads,
)
from tabctx.backends.causalpfn import _column_medians, _fill
from tabctx.backends.survivalpfn import _within


def test_missing_values_are_filled_with_each_columns_median():
    X = np.array([[1.0, np.nan, np.nan], [3.0, 4.0, np.nan], [np.nan, 6.0, np.nan]])
    medians = _column_medians(X.astype(np.float32))
    assert medians.tolist() == [2.0, 5.0, 0.0]  # 0 for a column with none
    filled = _fill(X.astype(np.float32), medians)
    assert filled.tolist() == [[1.0, 5.0, 0.0], [3.0, 4.0, 0.0], [2.0, 6.0, 0.0]]
    whole = np.ones((2, 3), dtype=np.float32)
    assert _fill(whole, medians) is whole


def test_answers_past_follow_up_or_in_the_last_bin_are_none():
    assert _within(np.array([1.5, np.inf, 4.0]), limit=3.0) == [1.5, None, None]
    assert _within(np.array([[1.0, np.inf], [2.0, 3.0]]), limit=3.0) == [
        [1.0, None],
        [2.0, 3.0],
    ]


class TestSubsample:
    def test_every_row_when_few_enough(self):
        assert subsample(np.array([0, 1, 1]), 3, seed=0).tolist() == [0, 1, 2]

    def test_keeps_each_groups_share_and_repeats(self):
        groups = np.array([1] * 900 + [0] * 100)
        keep = subsample(groups, 100, seed=0)
        assert len(keep) == 100 and (np.diff(keep) > 0).all()
        assert (groups[keep] == 0).sum() == 10
        assert keep.tolist() == subsample(groups, 100, seed=0).tolist()

    def test_a_rare_group_keeps_a_row(self):
        groups = np.array([1] * 999 + [0])
        keep = subsample(groups, 10, seed=0)
        assert len(keep) == 10 and 999 in keep


class TestStrata:
    def test_equal_count_bands_of_values_within_each_flag(self):
        flags = np.array([0, 1] * 50)
        values = np.arange(100.0)[::-1]
        groups = strata(flags, values, size=50, bands=5)
        for flag, offset in ((0, 0), (1, 5)):
            rows = flags == flag
            assert sorted(set(groups[rows])) == list(range(offset, offset + 5))
            assert np.bincount(groups[rows] - offset).tolist() == [10] * 5
            # a band holds a contiguous range of values: higher band, larger values
            order = np.argsort(values[rows])
            assert (np.diff(groups[rows][order]) >= 0).all()

    def test_ties_are_banded_in_row_order(self):
        groups = strata(np.ones(6), np.zeros(6), size=3, bands=3)
        assert groups.tolist() == [0, 0, 1, 1, 2, 2]

    def test_a_flag_with_fewer_rows_than_bands(self):
        flags = np.array([1, 1, 1] + [0] * 997)
        groups = strata(flags, np.arange(1000.0), size=1000)
        assert len(set(groups[flags == 1])) == 3  # one band per row
        assert len(set(groups[flags == 0])) == 10

    @pytest.mark.parametrize("treated", [0.5, 0.25, 0.05, 0.002])
    def test_every_size_keeps_both_flags_and_exactly_size_rows(self, treated):
        rng = np.random.default_rng(0)
        flags = (rng.random(1000) < treated).astype(float)
        flags[:2] = [0.0, 1.0]  # both flags present
        values = rng.normal(size=1000)
        for size in range(2, 40):
            keep = subsample(strata(flags, values, size), size, seed=0)
            assert len(keep) == size
            assert set(flags[keep]) == {0.0, 1.0}

    def test_a_rare_flag_keeps_its_share(self):
        flags = np.zeros(200_000)
        flags[:110] = 1  # 0.055%
        keep = subsample(strata(flags, np.arange(200_000.0), 4096), 4096, seed=0)
        assert 1 <= flags[keep].sum() <= 3  # 0.055% of 4,096 is 2.3

    def test_a_sample_keeps_the_flags_share_and_the_spread_of_values(self):
        rng = np.random.default_rng(0)
        flags = (rng.random(20_000) < 0.3).astype(float)
        values = rng.lognormal(sigma=2.0, size=20_000)  # long right tail
        keep = subsample(strata(flags, values, 1_000), 1_000, seed=0)
        assert abs(flags[keep].mean() - flags.mean()) < 0.005
        for flag in (0.0, 1.0):
            full = values[flags == flag]
            kept = values[keep][flags[keep] == flag]
            levels = np.linspace(0.05, 0.95, 10)
            # every decile of the kept values sits within the full table's
            # neighbouring deciles
            ranks = np.searchsorted(np.sort(full), np.quantile(kept, levels)) / len(
                full
            )
            assert np.abs(ranks - levels).max() < 0.03


def test_tagged_pickles():
    data = tagged_dumps({"a": 1}, b"T1:")
    assert tagged_loads(data, b"T1:") == {"a": 1}
    with pytest.raises(ValueError, match="model version"):
        tagged_loads(data, b"T2:")


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
