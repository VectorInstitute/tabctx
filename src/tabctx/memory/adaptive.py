"""AdaptiveMemoryEstimator: learns from real fit() measurements over time so
admission decisions become progressively less conservative for shapes the
service has actually seen in production, without ever trusting a
measurement for a LARGER shape than the one it was taken on.

Motivation (v0.3.0/v0.4.0, found via extensive multi-tenant load testing on
a real A100-40GB): the static PowerLawMemoryEstimator is calibrated on only
5 points and is known to be badly overestimating in places (see its
docstring). Cache-accounting already moved to real per-fit measurement
(engine.py calls backend.context_bytes_hint() after fit()) -- this class
closes the loop by feeding those same real measurements back into the
PRE-FIT admission gate too, which structurally cannot use a real
measurement for the request it's currently deciding on (nothing has run
yet) but CAN reuse a real measurement from a past request, as long as
that's done safely.

Safety argument: memory cost is assumed monotonically non-decreasing in
both n_train and n_features (a larger table needs at least as much memory
as a smaller one -- true for attention-style architectures processing more
rows/features). So a real measurement taken on shape (train=A, features=B)
is a valid (if imperfect) upper bound for ANY query shape (train<=A,
features<=B) -- never for a larger one. This is why estimate_bytes() only
ever uses an observation that *dominates* the query in both dimensions,
picking the tightest (lowest-cost) dominating observation available, and
still applies a safety_margin on top. Queries outside anything observed so
far fall back to the static estimator unchanged -- this class never makes
things less safe than the fallback, only tighter where real data allows it.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

from tabctx.memory.estimator import MemoryEstimator


@dataclass(frozen=True)
class Observation:
    n_train: int
    n_features: int
    real_bytes: int

    @property
    def cells(self) -> int:
        return self.n_train * self.n_features

    def dominates(self, n_train: int, n_features: int) -> bool:
        return self.n_train >= n_train and self.n_features >= n_features


class AdaptiveMemoryEstimator:
    def __init__(
        self,
        fallback: MemoryEstimator,
        safety_margin: float = 1.5,
        max_observations: int = 500,
        preloaded: tuple[Observation, ...] = (),
        preloaded_margin: float = 1.1,
        transient_capacity_fraction: float = 0.9,
    ) -> None:
        """preloaded: factory-installed observations (e.g. a measured
        calibration grid -- see memory/calibration_data.py), consulted
        exactly like runtime observations but immutable and exempt from
        the runtime FIFO cap, so a long-running replica can never evict
        its own calibration. As of v0.9.0 observations are PEAK fit
        bytes (the admission-relevant quantity), not resident context
        size -- see engine.fit() and backends/base.py.

        preloaded_margin: margin applied when the tightest dominating
        observation is a calibration point (a direct measurement of
        exactly the bounded quantity on this hardware, so it earns a
        smaller margin than a runtime observation, which keeps
        safety_margin). Calibration points near device capacity would
        otherwise be pushed over the admission line by the 1.5x runtime
        margin and re-forbid fits that measurably succeed.

        transient_capacity_fraction: admission_headroom_bytes() allows a
        fit's estimated peak to use up to this fraction of the
        (fraction-scaled) device capacity MINUS what the cache already
        holds resident -- peak-plus-resident is what actually OOMs."""
        self._fallback = fallback
        self._safety_margin = safety_margin
        self._max_observations = max_observations
        self._preloaded: tuple[Observation, ...] = tuple(preloaded)
        self._preloaded_margin = preloaded_margin
        self._transient_capacity_fraction = transient_capacity_fraction
        self._observations: list[Observation] = []
        self._lock = threading.Lock()

    def record_observation(self, n_train: int, n_features: int, real_bytes: int) -> None:
        with self._lock:
            self._observations.append(Observation(n_train, n_features, real_bytes))
            # Bounded FIFO -- simple, avoids unbounded growth over a long-
            # running replica's lifetime. A future version could prune to a
            # Pareto frontier of dominating points instead of dropping
            # oldest-first, to retain coverage rather than just recency.
            if len(self._observations) > self._max_observations:
                self._observations.pop(0)

    def _best_dominating_observation(
        self, n_train: int, n_features: int
    ) -> tuple[Observation, bool] | None:
        """Returns (observation, is_preloaded) for the tightest
        dominating measurement, or None."""
        with self._lock:
            runtime = [o for o in self._observations if o.dominates(n_train, n_features)]
        preloaded = [o for o in self._preloaded if o.dominates(n_train, n_features)]
        candidates = [(o, False) for o in runtime] + [(o, True) for o in preloaded]
        if not candidates:
            return None
        return min(candidates, key=lambda pair: pair[0].cells)

    def estimate_bytes(self, n_train: int, n_test: int, n_features: int) -> int:
        # Only fit-time queries (n_test == 0) can use a learned
        # observation -- predict()-time chunking (n_test > 0) needs to
        # account for active test-row memory we have no measurement of
        # yet, so it always uses the (conservative) fallback unchanged.
        if n_test == 0:
            best = self._best_dominating_observation(n_train, n_features)
            if best is not None:
                obs, is_preloaded = best
                margin = self._preloaded_margin if is_preloaded else self._safety_margin
                return math.ceil(obs.real_bytes * margin)
        return self._fallback.estimate_bytes(n_train, n_test, n_features)

    def admit(self, n_train: int, n_test: int, n_features: int) -> bool:
        return self.estimate_bytes(n_train, n_test, n_features) <= self.ceiling_bytes()

    def ceiling_bytes(self) -> int:
        return self._fallback.ceiling_bytes()

    def admission_headroom_bytes(self, used_bytes: int) -> int:
        """Usage-aware: a fit's transient peak coexists with everything
        the cache holds resident, so the peak budget is (a fraction of)
        device capacity minus current usage. Falls back to the static
        ceiling when the fallback doesn't expose real capacity."""
        capacity = getattr(self._fallback, "gpu_capacity_bytes", None)
        if capacity is None:
            return self._fallback.admission_headroom_bytes(used_bytes)
        return max(
            0, int(capacity * self._transient_capacity_fraction) - used_bytes
        )

    def confidence(self) -> str:
        with self._lock:
            n_obs = len(self._observations)
        return (
            f"{self._fallback.confidence()} Additionally backed by "
            f"{len(self._preloaded)} preloaded calibration measurement(s) and "
            f"{n_obs} real operational fit() measurement(s): admission "
            "decisions for a requested shape use a real measurement instead "
            "of the formula above whenever a past fit at least as large in "
            "both rows and features has been observed (with a "
            f"{self._safety_margin}x margin on top); smaller or novel "
            "shapes still fall back to the formula's conservative estimate."
        )
