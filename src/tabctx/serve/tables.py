"""Inline JSON tables -> arrays for the engine.

JSON has no NaN, so a missing cell arrives as ``null`` (None). Both
backends handle missing values natively, so a null becomes NaN here and
the table goes on as a float array. Converting once at the edge also
spares the engine its per-row Python checks for large inline tables.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def rows_to_array(rows: Sequence[Sequence[float | None]]) -> np.ndarray | Sequence:
    """A float array with null -> NaN, or the rows unchanged when they
    aren't a rectangle (ragged, empty), so the engine's own validation
    reports the problem in its usual terms."""
    try:
        arr = np.asarray(rows, dtype=float)
    except (TypeError, ValueError):
        return rows
    return arr if arr.ndim == 2 else rows
