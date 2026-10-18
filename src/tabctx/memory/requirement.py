"""What a fit actually NEEDS on the device: resident context + working floor.

Measured on an A100 (benchmarks/bench_memory_pressure.py --search;
benchmarks/baselines/minfree-tabicl-kv-a100.jsonl): TabICL sizes its
stage batches from whatever GPU memory is free, so an unconstrained fit's
transient peak grows to fill the card. It does not reflect a requirement.
With only a fixed amount left free, the same fits succeed using a small
*working floor* on top of the resident context they build:

    5,000 x 500    peak 25.3 GB   needs 1.7 GB resident + 1.2 GB working
    20,000 x 100   peak 24.9 GB   needs 3.8 GB resident + 1.6 GB working
    80,000 x 10    peak 18.6 GB   needs 11.2 GB resident + 9.5 GB working

at 1-1.5x the fit time. So admission should bound resident + floor, not
the peak: bounding the peak rejects tables that fit and evicts other
tenants' contexts to make room the fit doesn't need. Within that room,
the model still uses whatever extra memory is free.

``WorkingFloorModel`` turns measured floors into a conservative estimate
for any shape: the smallest margin-scaled floor among measurements that
dominate the shape (at least as many rows and features -- the floor grows
with both), else a per-row extrapolation fit as an upper envelope of the
measurements.

``ResidentRateModel`` extrapolates the resident context to shapes no
measurement dominates. TabICL's resident context is a kv cache with one
key/value per training row (plus a column cache independent of rows), so
it is linear in rows: resident(n, d) <= n * max rate observed at >= d
features. Rates are per-row bytes from real fits, which take precedence
over calibration points. Residents are fit-time (pre-compression) sizes:
a compressed context is smaller at rest, but the fit builds it at full
precision first.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class FloorPoint:
    """A measured working floor: the least memory beyond the resident
    context with which a fit of this shape succeeded (and that resident
    context's size, when recorded)."""

    n_train: int
    n_features: int
    working_bytes: int
    resident_bytes: int | None = None

    def dominates(self, n_train: int, n_features: int) -> bool:
        return self.n_train >= n_train and self.n_features >= n_features


class WorkingFloorModel:
    def __init__(
        self,
        points: Iterable[FloorPoint],
        margin: float = 1.25,
        base_bytes: int = 1 << 30,
        slope_min_rows: int = 10_000,
    ) -> None:
        """margin: applied to measured floors (the binary search has finite
        precision, and fragmentation varies run to run). base_bytes: the
        floor's intercept for row extrapolation. slope_min_rows: only fits
        at least this long set the per-row slope -- shorter ones are
        dominated by the intercept."""
        self._points = tuple(points)
        if not self._points:
            raise ValueError("a working floor model needs measurements")
        self._margin = margin
        self._base = base_bytes
        self._max_features = max(p.n_features for p in self._points)
        long_fits = [p for p in self._points if p.n_train >= slope_min_rows]
        # Upper envelope through the intercept: the steepest per-row slope
        # a long fit needs, so row extrapolation never undershoots one.
        self._per_row = max(
            max(0.0, p.working_bytes * margin - base_bytes) / p.n_train
            for p in (long_fits or self._points)
        )

    def bytes(self, n_train: int, n_features: int) -> int | None:
        """Conservative working floor, or None for shapes wider than any
        measurement (the floor's growth with features isn't extrapolated)."""
        dominating = [p for p in self._points if p.dominates(n_train, n_features)]
        if dominating:
            return math.ceil(min(p.working_bytes for p in dominating) * self._margin)
        if n_features > self._max_features:
            return None
        # Monotone in rows: never below a measured floor at >= these
        # features and fewer rows (the per-row line ignores features).
        shorter = [
            p.working_bytes * self._margin
            for p in self._points
            if p.n_features >= n_features and p.n_train <= n_train
        ]
        return math.ceil(max([self._base + self._per_row * n_train, *shorter]))


class ResidentRateModel:
    """Extrapolates a kv context's resident size to longer tables.

    For any measured fit (n0 rows, d0 >= d features, resident r0),
    resident(n, d) <= r0 + rate * (n - n0) for n >= n0; the estimate is
    the smallest such bound. What bounds the per-row rate depends on how
    the model's cache scales:

    - Width-independent (TabICL): a kv context is a column cache that
      depends on features but not rows, plus an ICL cache with one
      key/value per training row whose size doesn't depend on features.
      The rate is the steepest among long fits (>= slope_min_rows; short
      fits are dominated by the column cache), preferring the narrowest
      measured width >= d.
    - Width-dependent (TabPFN, ``width_dependent=True``): the per-row
      cache grows with features, so only fits at least as wide bound the
      rate. It comes from the narrowest measured width >= d -- long fits
      there if any, else short ones (whose fixed overhead overstates the
      rate: conservative) -- and shapes wider than every measurement get
      no estimate at all.

    Residents are fit-time sizes (before any post-fit compression -- what
    the fit needs while it runs), the same quantity the calibration
    measured. This replica's own fits supersede the calibration's rate
    once any qualify."""

    def __init__(
        self,
        calibration: Iterable[FloorPoint] = (),
        slope_min_rows: int = 10_000,
        width_dependent: bool = False,
    ) -> None:
        self._min_rows = slope_min_rows
        self._width_dependent = width_dependent
        self._calibration = [
            (p.n_train, p.n_features, p.resident_bytes)
            for p in calibration
            if p.resident_bytes
        ]
        self._runtime: list[tuple[int, int, int]] = []
        self._lock = threading.Lock()

    def record(self, n_train: int, n_features: int, resident_bytes: int) -> None:
        if n_train > 0 and resident_bytes > 0:
            with self._lock:
                self._runtime.append((n_train, n_features, resident_bytes))
                del self._runtime[:-500]  # bounded, newest kept

    def _slope(
        self, points: list[tuple[int, int, int]], n_features: int
    ) -> float | None:
        """Width-independent rate: steepest long fit at the narrowest
        measured width >= n_features, else among all long fits."""
        long_fits = [(d, res / n) for n, d, res in points if n >= self._min_rows]
        if not long_fits:
            return None
        widths = [d for d, _ in long_fits if d >= n_features]
        if not widths:
            return max(rate for _, rate in long_fits)
        return max(rate for d, rate in long_fits if d == min(widths))

    def _width_dependent_slope(
        self,
        runtime: list[tuple[int, int, int]],
        n_features: int,
    ) -> float | None:
        """Rate from fits at least as wide as the query only: long fits at the
        narrowest width >= n_features that has any (this replica's own
        superseding the calibration's), else -- only when no long fit that
        wide exists anywhere -- short fits at the narrowest such width,
        whose fixed overhead overstates the rate (conservative)."""

        def long_rate(points: list[tuple[int, int, int]]) -> float | None:
            wide = [
                (d, res / n)
                for n, d, res in points
                if n >= self._min_rows and d >= n_features
            ]
            if not wide:
                return None
            at = min(d for d, _ in wide)
            return max(r for d, r in wide if d == at)

        rate = long_rate(runtime)
        if rate is None:
            rate = long_rate(self._calibration)
        if rate is not None:
            return rate
        short = [
            (d, res / n) for n, d, res in runtime + self._calibration if d >= n_features
        ]
        if not short:
            return None
        at = min(d for d, _ in short)
        return max(r for d, r in short if d == at)

    def estimate(self, n_train: int, n_features: int) -> int | None:
        with self._lock:
            runtime = list(self._runtime)
        if self._width_dependent:
            slope = self._width_dependent_slope(runtime, n_features)
        else:
            slope = self._slope(runtime, n_features) or self._slope(
                self._calibration, n_features
            )
        if slope is None:
            return None
        bounds = [
            res + slope * max(0, n_train - n)
            for n, d, res in runtime + self._calibration
            if d >= n_features
        ]
        return math.ceil(min(bounds)) if bounds else None
