"""Real calibration points, measured on a single A100-40GB GPU running
TabICLv2 via a hand-rolled Ray Serve wrapper deployed to GKE
(VectorInstitute/inference-platform, branch test/tabicl-gke-onboard,
tests/gke-tabicl-test/{app.py,probe.py}), 2026-08-28.

This is the ONLY calibration data tabctx's default estimator has. See
PowerLawMemoryEstimator's docstring and confidence() for exactly how little
that is and what it does and doesn't tell you. Recalibrate (more shapes,
other GPU types, other backends) before trusting this anywhere else.
"""

from __future__ import annotations

from tabctx.memory.estimator import CalibrationPoint

A100_40GB_TABICL_CALIBRATION: list[CalibrationPoint] = [
    CalibrationPoint(
        n_train=500, n_test=100, n_features=10,
        allocated_bytes=316_500_000, reserved_bytes=446_700_000,
        outcome="ok",
    ),
    CalibrationPoint(
        n_train=2_000, n_test=500, n_features=20,
        allocated_bytes=929_900_000, reserved_bytes=1_247_800_000,
        outcome="ok",
    ),
    CalibrationPoint(
        n_train=10_000, n_test=1_000, n_features=50,
        allocated_bytes=8_042_300_000, reserved_bytes=11_775_500_000,
        outcome="ok",
    ),
    CalibrationPoint(
        n_train=50_000, n_test=2_000, n_features=100,
        allocated_bytes=32_264_600_000, reserved_bytes=40_766_500_000,
        outcome="ok",
    ),
    # No allocated/reserved reading exists for this one -- it crashed the
    # replica with a hard CUDA OOM (non-JSON error response) before any
    # memory stats could be captured. Recorded with outcome="oom" and
    # bytes=0 so the estimator can still use it as a known-bad upper bound
    # without pretending to know its actual memory footprint.
    CalibrationPoint(
        n_train=90_000, n_test=2_000, n_features=200,
        allocated_bytes=0, reserved_bytes=0,
        outcome="oom",
    ),
]  # fmt: skip
