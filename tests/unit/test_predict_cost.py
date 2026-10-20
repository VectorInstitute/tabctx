"""PredictCostModel (memory/predict.py) and how predict chunking uses it (#47)."""

import itertools

import pytest

from tabctx.chunking import choose_chunk_rows
from tabctx.memory.adaptive import AdaptiveMemoryEstimator
from tabctx.memory.calibration_data import A100_40GB_TABICL_CALIBRATION
from tabctx.memory.estimator import PowerLawMemoryEstimator
from tabctx.memory.predict import PredictCostModel, PredictPoint
from tabctx.serve.factory import ServeSettings, build_estimator

GB = 10**9
A100 = 42_404_806_656


def _true_peak(n_train: int, n_test: int, n_features: int) -> int:
    """A ground truth of the measured form: a fixed part plus, per test
    row, a fixed part and a per-feature part, plus a cost per training row."""
    per_row = 200_000 + 8_000 * n_features
    return 50_000_000 + n_test * per_row + 7_000 * n_train


def _measure(widths, n_trains=(1_000, 20_000), n_tests=(100, 1_000)):
    return [
        PredictPoint(n, t, f, _true_peak(n, t, f))
        for f, n, t in itertools.product(widths, n_trains, n_tests)
    ]


class TestPredictCostModel:
    def test_bounds_every_measurement(self):
        points = _measure((10, 100, 500))
        model = PredictCostModel(points, margin=1.0)
        for p in points:
            assert model.bytes(p.n_train, p.n_test, p.n_features) >= p.peak_bytes

    @pytest.mark.parametrize("n_features", [20, 150, 499, 1_000, 5_000])
    @pytest.mark.parametrize("n_train", [500, 50_000, 300_000])
    def test_never_underestimates_unmeasured_shapes(self, n_features, n_train):
        """Between measured widths (next wider one's cost) and past the
        widest (proportional scaling), for any context size and test rows."""
        model = PredictCostModel(_measure((10, 100, 500)), margin=1.0)
        for n_test in (1, 100, 2_000):
            truth = _true_peak(n_train, n_test, n_features)
            assert model.bytes(n_train, n_test, n_features) >= truth

    def test_cost_follows_test_rows_not_the_training_table(self):
        """The bug this model fixes: a predict's cost was sized with the
        fit model, which grows with the training table."""
        model = PredictCostModel(_measure((10, 100, 500)))
        small = model.bytes(1_000, 100, 500)
        large_context = model.bytes(20_000, 100, 500)
        assert large_context - small == pytest.approx(
            19_000 * model.train_row_bytes * 1.1, rel=1e-6
        )
        assert model.bytes(20_000, 1_000, 500) > 5 * model.bytes(20_000, 100, 500)

    def test_monotone_in_every_dimension(self):
        model = PredictCostModel(_measure((10, 100, 500)))
        for base in [(5_000, 100, 50), (5_000, 100, 700)]:
            n_train, n_test, n_features = base
            cost = model.bytes(*base)
            assert model.bytes(n_train * 2, n_test, n_features) >= cost
            assert model.bytes(n_train, n_test * 2, n_features) >= cost
            assert model.bytes(n_train, n_test, n_features * 2) >= cost

    def test_a_narrower_width_never_costs_more_per_row(self):
        """Non-decreasing in width even when a narrow measurement is costlier
        (tabicl's fp32 path for small contexts measured so)."""
        points = _measure((10, 100, 500))
        points += [
            PredictPoint(1_000, 100, 50, 10 * GB),
            PredictPoint(1_000, 1_000, 50, 90 * GB),
        ]
        model = PredictCostModel(points)
        assert model.row_bytes(100) >= model.row_bytes(50)

    def test_margin_scales_the_bound(self):
        points = _measure((10, 100))
        assert PredictCostModel(points, margin=1.5).bytes(
            1_000, 10, 10
        ) == pytest.approx(
            1.5 * PredictCostModel(points, margin=1.0).bytes(1_000, 10, 10), rel=1e-6
        )

    def test_rejects_calibration_without_two_test_sizes(self):
        one_size = [PredictPoint(1_000, 100, f, GB) for f in (10, 100)]
        with pytest.raises(ValueError):
            PredictCostModel(one_size)


class TestShippedCalibration:
    @pytest.mark.parametrize("kind", ["tabicl", "tabpfn"])
    def test_bounds_every_measured_predict(self, kind):
        from tabctx.memory import calibration_predict_a100 as grid

        points = getattr(grid, f"A100_40GB_{kind.upper()}_PREDICT_POINTS")
        model = PredictCostModel(points)
        for p in points:
            assert model.bytes(p.n_train, p.n_test, p.n_features) >= p.peak_bytes


class TestRaggedCalibration:
    def test_a_width_measured_at_one_test_size_still_bounds_its_points(self):
        """No per-row slope at that width (one size), but its point still
        lies under the bound: the width takes the next wider row cost, and
        what's left becomes its fixed call cost."""
        points = _measure((10, 500)) + [PredictPoint(1_000, 100, 100, 5 * GB)]
        model = PredictCostModel(points, margin=1.0)
        for p in points:
            assert model.bytes(p.n_train, p.n_test, p.n_features) >= p.peak_bytes

    def test_negative_slopes_are_noise_not_savings(self):
        points = _measure((10, 100)) + [
            PredictPoint(1_000, 100, 200, 3 * GB),
            PredictPoint(1_000, 1_000, 200, 2 * GB),
        ]
        model = PredictCostModel(points, margin=1.0)
        assert model.row_bytes(200) >= model.row_bytes(100) >= 0
        for p in points:
            assert model.bytes(p.n_train, p.n_test, p.n_features) >= p.peak_bytes

    def test_narrower_than_any_measurement_takes_the_narrowest_cost(self):
        model = PredictCostModel(_measure((10, 100)))
        assert model.row_bytes(1) == model.row_bytes(10)
        assert model.call_bytes(1) == model.call_bytes(10)

    def test_steepest_consecutive_slope_bounds_a_convex_curve(self):
        """Three test sizes on a convex curve: the end-to-end slope would
        undershoot the middle-to-end one."""
        peaks = {100: 1 * GB, 1_000: 2 * GB, 2_000: 6 * GB}
        points = [PredictPoint(1_000, t, 50, b) for t, b in peaks.items()]
        points += [PredictPoint(1_000, t, 10, b // 5) for t, b in peaks.items()]
        model = PredictCostModel(points, margin=1.0)
        assert model.row_bytes(50) == pytest.approx(4 * GB / 1_000)


class TestChunking:
    class _Linear:
        """estimate = fixed + per_row x rows, for exercising the sizing rule."""

        def __init__(self, fixed: int, per_row: int) -> None:
            self.fixed, self.per_row = fixed, per_row

        def estimate_bytes(self, n_train, n_test, n_features):
            return self.fixed + self.per_row * n_test

    def test_largest_halving_that_fits(self):
        est = self._Linear(fixed=0, per_row=1_000)
        assert choose_chunk_rows(est, 1, 1, remaining_budget_bytes=600_000) == 512
        assert choose_chunk_rows(est, 1, 1, remaining_budget_bytes=10**12) == 2_048

    def test_a_fixed_cost_over_budget_is_not_split_away(self):
        """#47: splitting can't shrink a call's fixed cost, so only the rows
        above it are sized against the budget, not split to one row each."""
        est = self._Linear(fixed=10 * GB, per_row=1_000_000)
        # 1 GB of budget above the smallest call fits ~1,000 more rows.
        assert choose_chunk_rows(est, 1, 1, remaining_budget_bytes=GB) == 512

    def test_no_headroom_means_the_smallest_chunk(self):
        """A full device (say, beside a fit at its peak): nothing beyond the
        smallest call is added, slow but safe."""
        est = self._Linear(fixed=10 * GB, per_row=1_000)
        assert choose_chunk_rows(est, 1, 1, remaining_budget_bytes=0) == 1

    def test_rejects_a_zero_floor(self):
        with pytest.raises(ValueError):
            choose_chunk_rows(self._Linear(0, 1), 1, 1, 10, min_chunk_rows=0)

    def test_predict_queries_use_the_cost_model_and_fits_do_not(self):
        model = PredictCostModel(_measure((10, 100, 500)))
        fallback = PowerLawMemoryEstimator(
            A100_40GB_TABICL_CALIBRATION, gpu_capacity_bytes=A100
        )
        with_model = AdaptiveMemoryEstimator(fallback=fallback, predict_cost=model)
        without = AdaptiveMemoryEstimator(fallback=fallback)
        assert with_model.estimate_bytes(3_500, 100, 500) == model.bytes(
            3_500, 100, 500
        )
        assert with_model.estimate_bytes(3_500, 0, 500) == without.estimate_bytes(
            3_500, 0, 500
        )

    @pytest.mark.parametrize("kind", ["tabicl", "tabpfn"])
    @pytest.mark.parametrize("n_train", [3_500, 10_000, 100_000])
    def test_wide_context_predict_fits_in_one_call(self, kind, n_train):
        """Regression (#47): a 100-row predict against a 3,500 x 500 context
        ran as 100 one-row calls (14 s instead of 0.1 s on an A100), because
        its memory was estimated with the fit model, which charges for the
        whole training table and exceeded the device's headroom."""
        settings = ServeSettings(backends=(kind,))
        estimator = build_estimator(settings, kind, gpu_capacity_bytes=A100)
        headroom = estimator.admission_headroom_bytes(0)
        assert estimator.estimate_bytes(n_train, 100, 500) <= headroom
        assert choose_chunk_rows(estimator, n_train, 500, headroom) >= 100
        # What it was: the same query on the fit model alone.
        fit_model = build_estimator(
            ServeSettings(backends=(kind,), kv_cache="repr"), kind, A100
        )
        assert fit_model.estimate_bytes(n_train, 100, 500) > headroom

    def test_factory_installs_the_model_for_its_calibrated_configuration(self):
        def cost(**kw):
            settings = ServeSettings(backends=("tabicl",), **kw)
            return build_estimator(settings, "tabicl", A100)._predict_cost  # noqa: SLF001

        assert cost() is not None  # serving defaults: kv, int8, compiled
        assert cost(kv_cache="repr") is None
        assert cost(kv_precision="native") is None
        assert cost(compile_models=False) is None
