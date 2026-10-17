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
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class FloorPoint:
    """A measured working floor: the least memory beyond the resident
    context with which a fit of this shape succeeded."""

    n_train: int
    n_features: int
    working_bytes: int

    def dominates(self, n_train: int, n_features: int) -> bool:
        return self.n_train >= n_train and self.n_features >= n_features


class WorkingFloorModel:
    def __init__(
        self,
        points: Iterable[FloorPoint],
        margin: float = 1.25,
        base_bytes: int = 1 << 30,
    ) -> None:
        """margin: applied to measured floors (the binary search has finite
        precision, and fragmentation varies run to run). base_bytes: the
        floor's intercept for the extrapolation."""
        self._points = tuple(points)
        if not self._points:
            raise ValueError("a working floor model needs measurements")
        self._margin = margin
        self._base = base_bytes
        # Upper envelope through the intercept: the steepest per-row slope
        # any measurement needs, so extrapolation never undershoots one.
        self._per_row = max(
            max(0, p.working_bytes * margin - base_bytes) / p.n_train
            for p in self._points
        )

    def bytes(self, n_train: int, n_features: int) -> int:
        dominating = [p for p in self._points if p.dominates(n_train, n_features)]
        if dominating:
            return math.ceil(min(p.working_bytes for p in dominating) * self._margin)
        return math.ceil(self._base + self._per_row * n_train)
