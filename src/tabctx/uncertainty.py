"""Predictive uncertainty: which quantiles of a regression's predicted
distribution a caller may ask for.

TabICL and TabPFN both predict a distribution over the target, not just a
number, and return its quantiles from the same forward pass as the mean.
A predict asks for levels in (0, 1) -- e.g. (0.05, 0.95) for a 90%
interval -- and gets each test row's value at each level.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from tabctx.errors import InvalidInputError

MAX_QUANTILES = 99


def check_quantiles(levels: Sequence[float] | None) -> tuple[float, ...] | None:
    """Validated levels, in the caller's order (None: none were asked for).
    InvalidInputError for an empty list, a level outside (0, 1), a repeat,
    or more than MAX_QUANTILES."""
    if levels is None:
        return None
    try:
        values = tuple(float(q) for q in levels)
    except (TypeError, ValueError) as e:
        raise InvalidInputError(f"quantiles must be numbers: {e}") from e
    if not values:
        raise InvalidInputError("quantiles: ask for at least one level, or none")
    if len(values) > MAX_QUANTILES:
        raise InvalidInputError(
            f"quantiles: at most {MAX_QUANTILES} levels, got {len(values)}"
        )
    bad = [q for q in values if not (math.isfinite(q) and 0.0 < q < 1.0)]
    if bad:
        raise InvalidInputError(
            f"quantiles must be strictly between 0 and 1, got {bad}"
        )
    if len(set(values)) != len(values):
        raise InvalidInputError("quantiles: each level at most once")
    return values
