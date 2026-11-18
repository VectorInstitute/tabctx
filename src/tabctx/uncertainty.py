"""Predictive uncertainty: which quantiles of a predicted distribution,
and which survival times, a caller may ask for.

The regression models predict a distribution over the target, not just a
number, and return its quantiles from the same forward pass as the mean;
so do the survival model (over the event time) and the causal model (over
the treatment effect). A predict asks for levels in (0, 1) -- e.g. (0.05,
0.95) for a 90% interval -- and gets each test row's value at each level.
A survival predict can also ask for times, and gets each row's
probability of no event by each.
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
    if isinstance(levels, (str, bytes)) or not hasattr(levels, "__iter__"):
        raise InvalidInputError(
            f"quantiles must be a list of levels, e.g. [0.05, 0.95]; got {levels!r}"
        )
    try:
        values = tuple(float(q) for q in levels)
    except (TypeError, ValueError) as e:
        raise InvalidInputError(f"quantiles must be numbers: {e}") from e
    if not values:
        raise InvalidInputError("quantiles: give at least one level")
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


# Survival: the probability of no event by each of up to MAX_TIMES times.
MAX_TIMES = 1000


def check_times(times: Sequence[float] | None) -> tuple[float, ...] | None:
    """Validated survival times, in the caller's order (None: none were
    asked for). InvalidInputError for an empty list, a negative or
    non-finite time, or more than MAX_TIMES. Repeats are allowed (a
    survival curve's grid is the caller's to choose)."""
    if times is None:
        return None
    if isinstance(times, (str, bytes)) or not hasattr(times, "__iter__"):
        raise InvalidInputError(
            f"times must be a list of times, e.g. [30, 90, 365]; got {times!r}"
        )
    try:
        values = tuple(float(t) for t in times)
    except (TypeError, ValueError) as e:
        raise InvalidInputError(f"times must be numbers: {e}") from e
    if not values:
        raise InvalidInputError("times: give at least one time")
    if len(values) > MAX_TIMES:
        raise InvalidInputError(f"times: at most {MAX_TIMES}, got {len(values)}")
    bad = [t for t in values if not (math.isfinite(t) and t >= 0.0)]
    if bad:
        raise InvalidInputError(f"times must be finite and non-negative, got {bad}")
    return values
