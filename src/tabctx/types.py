"""Shared value types for tabctx."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

Task = Literal["classification", "regression"]

# Accept anything array-like (list-of-lists, numpy array, etc.) at the API
# boundary; backends are responsible for coercing to whatever they need.
ArrayLike = Sequence[Sequence[float]] | Sequence[float] | Any


@dataclass(frozen=True)
class PredictOutcome:
    """Result of a single predict() call.

    predictions/probabilities/classes come from ONE inference pass -- a
    backend must never call predict() and predict_proba() separately for the
    same request (see backends/tabicl.py for why this matters).
    """

    predictions: list
    probabilities: list[list[float]] | None = None
    classes: list[str] | None = None


@dataclass(frozen=True)
class EngineStats:
    n_cached_contexts: int
    used_bytes: int
    free_bytes: int
    capacity_bytes: int
