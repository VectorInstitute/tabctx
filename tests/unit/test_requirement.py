"""Unit tests for requirement-based admission (memory/requirement.py):
fits are learned as resident + working floor, not unconstrained peaks."""

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

    def test_extrapolation_never_undershoots_a_measurement(self):
        m = WorkingFloorModel(POINTS, margin=1.25)
        for p in POINTS:
            # a slightly larger shape than any measurement: extrapolated
            assert m.bytes(p.n_train + 1, 10_000) >= p.working_bytes * 1.25

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

    def test_never_more_than_the_measured_peak(self):
        est = _estimator(WorkingFloorModel(POINTS, margin=1.0))
        est.record_fit(20_000, 100, resident_bytes=4 * GiB, peak_bytes=int(4.5 * GiB))
        assert est.estimate_bytes(20_000, 0, 100) == pytest.approx(
            4.5 * GiB * 1.5, rel=1e-3
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
        """The motivating case: a 20k x 100 fit's unconstrained peak is 25 GB,
        but it needs only resident + floor (5.6 GB). Each estimator is
        calibrated as the factory does (a measured requirement vs a measured
        peak). Beside a warm cache of five 4 GB tenants, peak semantics evict
        tenants to make room the fit doesn't need; requirement semantics
        admit it untouched."""

        def run(floor):
            measured = (5.6 if floor else 25) * GiB
            est = _estimator(
                floor, preloaded=(Observation(20_000, 100, int(measured)),)
            )
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
