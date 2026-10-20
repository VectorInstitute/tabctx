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


class TestChunking:
    def _estimator(self, predict_cost=None):
        fallback = PowerLawMemoryEstimator(
            A100_40GB_TABICL_CALIBRATION, gpu_capacity_bytes=A100
        )
        return AdaptiveMemoryEstimator(fallback=fallback, predict_cost=predict_cost)

    def test_predict_queries_use_the_cost_model_and_fits_do_not(self):
        model = PredictCostModel(_measure((10, 100, 500)))
        estimator = self._estimator(model)
        assert estimator.estimate_bytes(3_500, 100, 500) == model.bytes(3_500, 100, 500)
        assert estimator.estimate_bytes(
            3_500, 0, 500
        ) == self._estimator().estimate_bytes(3_500, 0, 500)

    @pytest.mark.parametrize("kind", ["tabicl", "tabpfn"])
    def test_wide_context_predicts_in_one_call(self, kind):
        """Regression (#47): a 100-row predict against a 3,500 x 500 context
        chunked to 1 row -- 100 sequential model calls, 14 s instead of
        0.1 s on an A100 -- because chunks were sized with the fit model."""
        estimator = build_estimator(
            ServeSettings(backends=(kind,)), kind, gpu_capacity_bytes=A100
        )
        headroom = estimator.admission_headroom_bytes(0)
        for n_train in (3_500, 10_000, 100_000):
            chunk = choose_chunk_rows(estimator, n_train, 500, headroom)
            assert chunk >= 100, (n_train, chunk)

    def test_factory_installs_the_model_for_kv_contexts_only(self):
        kv = build_estimator(ServeSettings(backends=("tabicl",)), "tabicl", A100)
        repr_mode = build_estimator(
            ServeSettings(backends=("tabicl",), kv_cache="repr"), "tabicl", A100
        )
        assert kv._predict_cost is not None  # noqa: SLF001
        assert repr_mode._predict_cost is None  # noqa: SLF001
