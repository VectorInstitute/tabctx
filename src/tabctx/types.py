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


# Where a cached context is: in the cache's primary tier (device memory),
# or spilled to the disk tier (restored on its next predict).
Location = Literal["resident", "spilled"]


@dataclass(frozen=True)
class ContextInfo:
    """What a listing shows about one cached context: its metadata, never
    its contents."""

    dataset_id: str
    model: str
    task: Task
    n_train: int
    n_features: int
    # Its cache-accounting size: the device memory it occupies when
    # resident (and will again once a spilled copy is restored).
    bytes: int
    location: Location
    # Unix time of the fit; None for a context spilled by tabctx < 0.19,
    # which didn't record it.
    fitted_at: float | None
    # Set by a multi-device engine; None on a single device.
    device: str | None = None
