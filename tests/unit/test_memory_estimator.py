import pytest

from tabctx.memory.calibration_data import A100_40GB_TABICL_CALIBRATION
from tabctx.memory.estimator import CalibrationPoint, PowerLawMemoryEstimator


def test_requires_at_least_two_ok_points():
    with pytest.raises(ValueError):
        PowerLawMemoryEstimator([CalibrationPoint(100, 10, 5, 1000, 2000, "ok")])


def test_estimate_grows_with_shape():
    est = PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION)
    small = est.estimate_bytes(500, 100, 10)
    large = est.estimate_bytes(50_000, 2_000, 100)
    assert 0 < small < large


def test_admit_rejects_known_oom_shape():
    est = PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION)
    # The exact shape that crashed the replica in real testing.
    assert est.admit(90_000, 2_000, 200) is False


def test_admit_rejects_zero_headroom_shape_by_design():
    est = PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION)
    # This shape "succeeded" empirically but reserved 40.77GB on a ~40.5GB
    # card -- essentially zero headroom. The hard ceiling is deliberately
    # set below this point on purpose (see estimator.py docstring), so it
    # must be rejected, not just the strictly-larger OOM shape.
    assert est.admit(50_000, 2_000, 100) is False


def test_admit_accepts_small_typical_shape():
    est = PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION)
    assert est.admit(500, 100, 10) is True


def test_ceiling_bytes_is_stricter_of_hard_ceiling_and_gpu_capacity():
    est = PowerLawMemoryEstimator(
        A100_40GB_TABICL_CALIBRATION,
        hard_ceiling_bytes=10,
        gpu_capacity_bytes=1_000_000,
    )
    assert est.ceiling_bytes() == 10

    est2 = PowerLawMemoryEstimator(
        A100_40GB_TABICL_CALIBRATION,
        hard_ceiling_bytes=1_000_000,
        gpu_capacity_bytes=10,
    )
    assert est2.ceiling_bytes() == 10


def test_confidence_names_the_oom_boundary_window():
    est = PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION)
    msg = est.confidence()
    assert "LOW" in msg
    assert "5,200,000" in msg
    assert "18,400,000" in msg


def test_zero_cells_estimate_is_zero():
    # (n_train + n_test) * n_features == 0 -- nothing to encode, so the
    # power-law fit (undefined at cells=0) is skipped entirely.
    est = PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION)
    assert est.estimate_bytes(0, 0, 0) == 0


def test_higher_safety_margin_reduces_admitted_shapes():
    lax = PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION, safety_margin=1.0)
    strict = PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION, safety_margin=5.0)
    n_train, n_test, n_features = 20_000, 1_000, 80
    assert strict.estimate_bytes(n_train, n_test, n_features) > lax.estimate_bytes(
        n_train, n_test, n_features
    )
