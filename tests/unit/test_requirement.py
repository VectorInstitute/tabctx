"""Unit tests for requirement-based admission (memory/requirement.py):
fits are learned as resident + working floor, not unconstrained peaks."""

import math

import pytest

from tabctx.backends.fake import FakeBackend
from tabctx.cache.manager import CachedContext, ContextCacheManager
from tabctx.engine import TabctxEngine
from tabctx.memory import (
    A100_40GB_TABICL_CALIBRATION,
    AdaptiveMemoryEstimator,
    PowerLawMemoryEstimator,
)
from tabctx.memory.adaptive import Observation
from tabctx.memory.requirement import FloorPoint, WorkingFloorModel

GiB = 1024**3
POINTS = (
    FloorPoint(2_000, 10, int(0.3 * GiB)),
    FloorPoint(20_000, 100, int(1.6 * GiB)),
    FloorPoint(80_000, 10, int(9.5 * GiB)),
)


class TestWorkingFloorModel:
    def test_needs_measurements(self):
        with pytest.raises(ValueError):
            WorkingFloorModel(())

    def test_dominated_shape_uses_the_tightest_measurement(self):
        m = WorkingFloorModel(POINTS, margin=1.25)
        # dominated by (20000,100) and (80000,10)? only (20000,100) has >=50 features
        assert m.bytes(10_000, 50) == pytest.approx(1.6 * GiB * 1.25, rel=1e-6)
        assert m.bytes(1_000, 5) == pytest.approx(0.3 * GiB * 1.25, rel=1e-6)

    def test_row_extrapolation_never_undershoots_a_long_fit(self):
        m = WorkingFloorModel(POINTS, margin=1.25)
        for p in POINTS:
            if p.n_train >= 10_000:  # long fits set the slope
                # more rows than any measurement at these features
                assert m.bytes(10 * p.n_train, p.n_features) >= p.working_bytes * 1.25

    def test_wider_than_any_measurement_is_unknown(self):
        assert WorkingFloorModel(POINTS).bytes(100, 5_000) is None

    def test_extrapolation_grows_with_rows(self):
        m = WorkingFloorModel(POINTS)
        assert m.bytes(200_000, 10) > m.bytes(100_000, 10) > m.bytes(90_000, 10)


def _estimator(floor=None, capacity=40 * GiB, preloaded=()):
    fallback = PowerLawMemoryEstimator(
        A100_40GB_TABICL_CALIBRATION,
        hard_ceiling_bytes=int(capacity * 0.6),
        gpu_capacity_bytes=capacity,
    )
    return AdaptiveMemoryEstimator(
        fallback=fallback, working_floor=floor, preloaded=preloaded
    )


class TestRecordFit:
    def test_with_a_floor_model_learns_resident_plus_floor(self):
        floor = WorkingFloorModel(POINTS, margin=1.0)
        est = _estimator(floor)
        est.record_fit(20_000, 100, resident_bytes=4 * GiB, peak_bytes=25 * GiB)
        expected = (4 + 1.6) * GiB * 1.5  # runtime safety margin
        assert est.estimate_bytes(20_000, 0, 100) == pytest.approx(expected, rel=1e-3)

    def test_not_capped_at_a_peak_measured_under_pressure(self):
        # A peak below resident + floor only says how little was free; the
        # floor is a measured minimum, so the requirement keeps it.
        est = _estimator(WorkingFloorModel(POINTS, margin=1.0))
        est.record_fit(20_000, 100, resident_bytes=4 * GiB, peak_bytes=int(4.5 * GiB))
        assert est.estimate_bytes(20_000, 0, 100) == pytest.approx(
            5.6 * GiB * 1.5, rel=1e-3
        )

    def test_without_a_floor_model_learns_the_peak(self):
        est = _estimator()
        est.record_fit(20_000, 100, resident_bytes=4 * GiB, peak_bytes=25 * GiB)
        assert est.estimate_bytes(20_000, 0, 100) == pytest.approx(
            25 * GiB * 1.5, rel=1e-3
        )
        est2 = _estimator()
        est2.record_fit(1_000, 10, resident_bytes=GiB, peak_bytes=None)
        assert est2.estimate_bytes(1_000, 0, 10) == pytest.approx(GiB * 1.5, rel=1e-3)


class _AdaptiveFake(FakeBackend):
    """Reports a resident context and an unconstrained peak that fills the
    card, like TabICL with plenty free."""

    def __init__(self, resident: int, peak: int) -> None:
        super().__init__(bytes_hint=resident, peak_bytes_hint=peak)


class TestAdmissionEffect:
    def test_wide_table_is_admitted_without_evicting_other_tenants(self):
        """The motivating case. Both estimators start from the same measured
        PEAK (25 GB for 20k x 100, as the real peak grid has); the only
        difference is the working-floor model. With it, the first fit is
        learned as resident + floor (4 + 2 GB), so the second fit beside five
        4 GB tenants needs ~9 GB and evicts nobody; without it, the learned
        peak makes the fit evict tenants for room it doesn't need."""

        def run(floor):
            est = _estimator(floor, preloaded=(Observation(20_000, 100, 25 * GiB),))
            backend = _AdaptiveFake(resident=4 * GiB, peak=25 * GiB)
            cache = ContextCacheManager(capacity_bytes=34 * GiB)
            engine = TabctxEngine(backend=backend, cache=cache, estimator=est)
            X = [[0.0] * 100] * 20_000
            y = ["a"] * 20_000
            engine.fit(X, y, dataset_id="first")  # learn from one fit
            for i in range(5):  # other tenants: 5 x 4 GB resident
                cache.put(
                    CachedContext(
                        f"t{i}", "fake", "classification", 10, 1, None, 4 * GiB
                    )
                )
            engine.fit(X, y, dataset_id="second")
            return [f"t{i}" for i in range(5) if cache.holds(f"t{i}")]

        assert len(run(WorkingFloorModel(POINTS))) == 5  # nobody evicted
        assert len(run(None)) < 5  # peak semantics evict tenants


class TestResidentRateModel:
    CAL = (
        FloorPoint(20_000, 10, GiB, resident_bytes=4 * GiB),
        FloorPoint(80_000, 10, GiB, resident_bytes=16 * GiB),
        FloorPoint(2_000, 1_000, GiB, resident_bytes=2 * GiB),
    )

    def test_extrapolates_linearly_from_the_nearest_anchor(self):
        from tabctx.memory.requirement import ResidentRateModel

        m = ResidentRateModel(self.CAL)
        slope = 16 * GiB / 80_000  # steepest long fit
        assert m.estimate(100_000, 10) == pytest.approx(
            16 * GiB + slope * 20_000, rel=1e-6
        )

    def test_short_wide_fits_do_not_set_the_slope(self):
        from tabctx.memory.requirement import ResidentRateModel

        m = ResidentRateModel(self.CAL)
        # 2000x1000 has 1 MiB/row (column cache) but is too short to set it.
        assert m.estimate(100_000, 10) < 25 * GiB

    def test_wider_query_uses_only_anchors_at_least_as_wide(self):
        from tabctx.memory.requirement import ResidentRateModel

        m = ResidentRateModel(self.CAL)
        slope = 16 * GiB / 80_000
        assert m.estimate(10_000, 500) == pytest.approx(
            2 * GiB + slope * 8_000, rel=1e-6
        )
        assert m.estimate(10, 5_000) is None

    def test_runtime_long_fits_supersede_the_calibration_slope(self):
        from tabctx.memory.requirement import ResidentRateModel

        m = ResidentRateModel(self.CAL)
        m.record(50_000, 10, 5 * GiB)  # a replica fit: 0.1 MiB/row
        assert m.estimate(100_000, 10) == pytest.approx(5 * GiB + 5 * GiB, rel=1e-6)

    def test_runtime_history_is_bounded(self):
        from tabctx.memory.requirement import ResidentRateModel

        m = ResidentRateModel()
        for i in range(700):
            m.record(10_000 + i, 10, GiB)
        assert len(m._runtime) == 500


class TestExtrapolatedRequirement:
    def test_beyond_the_grid_uses_resident_plus_floor_not_the_formula(self):
        from tabctx.memory.requirement import ResidentRateModel

        floor = WorkingFloorModel(POINTS)
        rate = ResidentRateModel(TestResidentRateModel.CAL)
        est = AdaptiveMemoryEstimator(
            fallback=PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION),
            working_floor=floor,
            resident_rate=rate,
            resident_margin=1.2,
        )
        n, d = 120_000, 10
        expected = math.ceil(rate.estimate(n, d) * 1.2) + floor.bytes(n, d)
        assert est.estimate_bytes(n, 0, d) == expected

    def test_wider_than_any_measurement_falls_back(self):
        from tabctx.memory.requirement import ResidentRateModel

        est = AdaptiveMemoryEstimator(
            fallback=PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION),
            working_floor=WorkingFloorModel(POINTS),
            resident_rate=ResidentRateModel(TestResidentRateModel.CAL),
        )
        assert est.estimate_bytes(100, 0, 5_000) == est._fallback.estimate_bytes(
            100, 0, 5_000
        )


def test_floor_model_is_monotone_in_rows():
    m = WorkingFloorModel(
        (
            FloorPoint(2_000, 1_000, int(1.15 * GiB)),
            FloorPoint(20_000, 100, int(1.6 * GiB)),
        )
    )
    assert m.bytes(3_000, 1_000) >= m.bytes(2_000, 1_000)


def test_factory_wires_floor_and_resident_models_for_tabicl_kv_only():
    from tabctx.serve.factory import ServeSettings, build_estimator

    kv = build_estimator(ServeSettings(backends=("tabicl",)), "tabicl", 40 * GiB)
    assert kv._working_floor is not None and kv._resident_rate is not None
    for settings, kind in (
        (ServeSettings(backends=("tabicl",), kv_cache="repr"), "tabicl"),
        (ServeSettings(backends=("tabpfn",)), "tabpfn"),
    ):
        est = build_estimator(settings, kind, 40 * GiB)
        assert est._working_floor is None and est._resident_rate is None
