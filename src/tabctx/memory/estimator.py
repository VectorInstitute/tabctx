"""Memory estimator used for admission control and eviction sizing.

Derived from tabctx/memory/calibration_data.py -- 4 successful measurements
plus 1 known-OOM boundary, all on a single A100-40GB GPU running TabICLv2.
Read confidence() before trusting this anywhere else; it is deliberately
blunt about how little data this is calibrated from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class CalibrationPoint:
    n_train: int
    n_test: int
    n_features: int
    allocated_bytes: int
    reserved_bytes: int
    outcome: Literal["ok", "oom"] = "ok"

    @property
    def cells(self) -> int:
        return (self.n_train + self.n_test) * self.n_features


class MemoryEstimator(Protocol):
    def estimate_bytes(self, n_train: int, n_test: int, n_features: int) -> int: ...

    def admit(self, n_train: int, n_test: int, n_features: int) -> bool: ...

    def ceiling_bytes(self) -> int: ...

    def confidence(self) -> str: ...

    def record_observation(self, n_train: int, n_features: int, real_bytes: int) -> None:
        """Called by the engine after a successful fit() whenever the
        backend reported a real measured cost (see backends/tabicl.py).
        Implementations that can't learn from this (like this module's
        static PowerLawMemoryEstimator) should no-op; AdaptiveMemoryEstimator
        (adaptive.py) is what actually uses it."""
        ...


# Defaults for PowerLawMemoryEstimator's two ceilings, exposed as module
# constants so a deployment that needs to scale them (e.g. two replicas
# sharing one physical GPU each budget a fraction; see serve/app.py's
# TABCTX_GPU_MEMORY_FRACTION) can derive from the same numbers instead of
# hardcoding copies.
DEFAULT_HARD_CEILING_BYTES = 24 * 1024**3
DEFAULT_GPU_CAPACITY_BYTES = 40536 * 1024**2  # A100-40GB, MiB-reported


class PowerLawMemoryEstimator:
    """estimate_bytes ~= a * cells^b, fit by OLS in log-log space over the
    "ok" calibration points, with a multiplicative safety_margin and an
    independent hard_ceiling_bytes as a backstop.

    Both guards matter for different reasons:
      - safety_margin compensates for the fit itself being a poor physical
        model -- the calibration data's local log-log slope is NOT constant
        (~0.51, then ~0.90, then ~0.62 between consecutive points), so this
        margin is covering known model misspecification, not just measurement
        noise. A future version with more calibration points should split
        the row-count and feature-count scaling terms instead of leaning
        harder on this margin.
      - hard_ceiling_bytes is independent of the fit entirely, and is set
        BELOW the largest known-good point on purpose: that point (50,000
        train + 2,000 test rows x 100 features) reserved 40.77GB on a
        ~40.5GB card -- essentially the entire device, with zero headroom.
        A request that barely fits with nothing left over is not a request
        this estimator should admit again; the ceiling exists precisely to
        stay clear of that edge, not to reproduce it.

    KNOWN LIMITATION (found via testing, not just theorized): the
    calibration range only covers 6,000-5,200,000 cells. Below that range
    the fit tends to OVERESTIMATE substantially -- e.g. a 20-row, 2-feature
    training set (40 cells) estimates to ~23MB, far more than such a tiny
    table plausibly needs. This is consistent with the true cost curve
    having a fixed baseline overhead (model weights, CUDA context, etc.)
    that a pure power law with no intercept can't represent at small scale.
    Net effect: v1 is safe (never under-estimates in a way that risks OOM)
    but is unnecessarily conservative for small, typical requests -- exactly
    the case tabctx's multi-tenancy is supposed to make cheap. Worth a
    piecewise or additive-intercept model once more calibration data exists.

    NOTE (v0.4.0): this class is normally used as the fallback prior inside
    an AdaptiveMemoryEstimator (see adaptive.py) rather than directly, so
    real operational measurements can progressively override its
    conservative guesses for shapes the service has actually seen.
    """

    def __init__(
        self,
        calibration: Sequence[CalibrationPoint],
        safety_margin: float = 2.0,
        hard_ceiling_bytes: int = 24 * 1024**3,
        gpu_capacity_bytes: int = 40536 * 1024**2,  # A100-40GB, MiB-reported
    ) -> None:
        ok_points = [p for p in calibration if p.outcome == "ok"]
        if len(ok_points) < 2:
            raise ValueError("need at least 2 'ok' calibration points to fit a curve")
        self._ok_points = ok_points
        self._oom_points = [p for p in calibration if p.outcome == "oom"]
        self._safety_margin = safety_margin
        self._hard_ceiling_bytes = hard_ceiling_bytes
        self._gpu_capacity_bytes = gpu_capacity_bytes
        self._effective_ceiling = min(hard_ceiling_bytes, gpu_capacity_bytes)

        log_cells = np.log10([p.cells for p in ok_points])
        log_reserved = np.log10([p.reserved_bytes for p in ok_points])
        b, log_a = np.polyfit(log_cells, log_reserved, deg=1)
        self._exponent = float(b)
        self._coefficient = float(10**log_a)

    def _raw_estimate_bytes(self, n_train: int, n_test: int, n_features: int) -> float:
        cells = (n_train + n_test) * n_features
        if cells <= 0:
            return 0.0
        return self._coefficient * (cells**self._exponent)

    def estimate_bytes(self, n_train: int, n_test: int, n_features: int) -> int:
        return math.ceil(
            self._raw_estimate_bytes(n_train, n_test, n_features) * self._safety_margin
        )

    def admit(self, n_train: int, n_test: int, n_features: int) -> bool:
        return self.estimate_bytes(n_train, n_test, n_features) <= self._effective_ceiling

    def ceiling_bytes(self) -> int:
        return self._effective_ceiling

    def record_observation(self, n_train: int, n_features: int, real_bytes: int) -> None:
        # Static/fixed-calibration estimator: intentionally does not learn
        # from live traffic. See AdaptiveMemoryEstimator for that.
        del n_train, n_features, real_bytes

    def confidence(self) -> str:
        oom_note = ""
        if self._oom_points:
            lo = max(p.cells for p in self._ok_points)
            hi = min(p.cells for p in self._oom_points)
            oom_note = (
                f" The known OOM boundary is only bounded between "
                f"{lo:,} cells (last known-good) and {hi:,} cells "
                f"(first known-crash) -- anything in between is unverified."
            )
        return (
            f"LOW confidence: fitted from {len(self._ok_points)} successful "
            f"calibration points"
            + (f" + {len(self._oom_points)} known-OOM boundary" if self._oom_points else "")
            + ", single A100-40GB card, single backend (TabICL). The fit's "
            "local log-log slope is not constant across the calibration "
            "range, so this power-law fit is a smoothing convenience, not a "
            f"physical model; safety_margin={self._safety_margin}x is "
            "compensating for that model misspecification, not measurement "
            "noise." + oom_note + " Recalibrate before trusting this on a "
            "different GPU type, model, or backend."
        )
