"""What a predict against a resident kv context needs on the device.

A predict against a kv-cached context doesn't re-encode the training
table: the test rows run through the model and attend to cached keys and
values. Measured under the serving configuration (A100, compiled TabICL,
int8 kv; benchmarks/calibrate_predict.py), its peak has three parts:

- a cost per test row that grows with the table's width (TabICLv2: ~1.1 MB
  at 100 features, ~4.1 MB at 500, ~8.2 MB at 1,000), whatever the size of
  the training table. TabPFN-3.5 batches test rows itself, so its peak at
  1,000 test rows stops growing past ~200 features (~6.7 GB);
- a fixed cost per call that also grows with width (TabPFN-3.5: ~3 GB at
  1,000 features for even 100 test rows);
- the cached context read back for attention: ~7 KB (TabICLv2) and ~17 KB
  (TabPFN-3.5) per training row, visible when tables are narrow (an int8
  cache is dequantized a layer at a time).

Sizing a predict's chunk with the fit model instead -- which scales with the
whole training table -- is wrong in the direction that hurts most: past some
table size no chunk "fits", chunking falls to one row, and a 100-row request
becomes 100 sequential model calls (14 s instead of 0.1 s on an A100, #47).

``PredictCostModel`` is built from measured marginal costs, not a fitted
surface: fitted surfaces and fitted slopes underestimated held-out widths by
up to a third (tabicl's fp32 path for small contexts and TabPFN's own test
batching bend the curve), which a bound must not do.

- ``row_bytes(f)``: per measured width, the extra peak per extra test row
  (largest across context sizes).
- ``train_row_bytes``: the extra peak per extra training row, largest across
  measured widths and test sizes.
- ``call_bytes(f)``: per measured width, the largest remainder of a measured
  peak after the two above -- the fixed cost of a call at that width.

Both width-dependent costs are made non-decreasing in width; a width between
measurements takes the next wider one's cost (as fit admission's dominance
rule does), and past the widest the widest's cost scales in proportion to
width: if a cost is a + b x f with a >= 0, then cost(w) x f / w >= a + b x f
for every f >= w, so it can't underestimate.

bytes = margin x (call_bytes(f) + n_test x row_bytes(f) + n_train x train_row_bytes).
"""

from __future__ import annotations

import bisect
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class PredictPoint:
    """One measured predict: its shape and its peak allocation above what
    was allocated before the call."""

    n_train: int
    n_test: int
    n_features: int
    peak_bytes: int


class PredictCostModel:
    """Upper bound on a kv-context predict's peak memory, from measurements.

    Needs predicts of two or more test sizes against the same context;
    otherwise raises ValueError, so a deployment without a usable calibration
    gets no model rather than a guess.
    """

    def __init__(self, points: Iterable[PredictPoint], margin: float = 1.1) -> None:
        pts = [p for p in points if p.n_test > 0 and p.n_features > 0]
        per_row = _largest(_marginals(pts, "n_test", ("n_features", "n_train")))
        if not per_row:
            raise ValueError(
                "PredictCostModel needs predicts of two test sizes against one context"
            )
        self._row = _WidthCurve(per_row)
        by_train = _marginals(pts, "n_train", ("n_features", "n_test"))
        self._train_row_bytes = max(
            (max(0.0, r) for rates in by_train.values() for r in rates), default=0.0
        )
        call: dict[int, float] = defaultdict(float)
        for p in pts:
            rest = p.peak_bytes - self._variable(p.n_train, p.n_test, p.n_features)
            call[p.n_features] = max(call[p.n_features], rest)
        self._call = _WidthCurve(call)
        self._margin = margin

    @property
    def train_row_bytes(self) -> float:
        return self._train_row_bytes

    def row_bytes(self, n_features: int) -> float:
        """Peak per test row at this width."""
        return self._row(n_features)

    def call_bytes(self, n_features: int) -> float:
        """Fixed peak of one call at this width, whatever its rows."""
        return self._call(n_features)

    def bytes(self, n_train: int, n_test: int, n_features: int) -> int:
        """Bound on the peak of a predict of n_test rows x n_features against
        a context of n_train training rows."""
        raw = self._call(n_features) + self._variable(n_train, n_test, n_features)
        return math.ceil(raw * self._margin)

    def _variable(self, n_train: int, n_test: int, n_features: int) -> float:
        return (
            max(0, n_test) * self._row(n_features)
            + max(0, n_train) * self._train_row_bytes
        )


class _WidthCurve:
    """A measured cost per width as a bound for any width: non-decreasing,
    the next wider measurement's cost between measurements, and scaled in
    proportion to width past the widest."""

    def __init__(self, by_width: Mapping[int, float]) -> None:
        self._widths = sorted(by_width)
        self._costs: list[float] = []
        for w in self._widths:
            self._costs.append(max(0.0, by_width[w], *self._costs[-1:]))

    def __call__(self, n_features: int) -> float:
        i = bisect.bisect_left(self._widths, n_features)
        if i < len(self._widths):
            return self._costs[i]
        return self._costs[-1] * n_features / self._widths[-1]


def _marginals(
    pts: list[PredictPoint], vary: str, fixed: tuple[str, ...]
) -> dict[tuple, list[float]]:
    """Extra peak per unit of `vary`, between the smallest and largest
    measurement of each group that holds `fixed` constant."""
    groups: dict[tuple, list[PredictPoint]] = defaultdict(list)
    for p in pts:
        groups[tuple(getattr(p, f) for f in fixed)].append(p)
    out: dict[tuple, list[float]] = defaultdict(list)
    for key, group in groups.items():
        lo = min(group, key=lambda p: getattr(p, vary))
        hi = max(group, key=lambda p: getattr(p, vary))
        span = getattr(hi, vary) - getattr(lo, vary)
        if span > 0:
            out[key].append((hi.peak_bytes - lo.peak_bytes) / span)
    return out


def _largest(by_group: dict[tuple, list[float]]) -> dict[int, float]:
    """Per width (each group key's first element), the largest rate."""
    per_width: dict[int, float] = {}
    for (n_features, *_), rates in by_group.items():
        per_width[n_features] = max(per_width.get(n_features, 0.0), *rates)
    return per_width
