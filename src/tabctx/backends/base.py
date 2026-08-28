"""Backend protocol: what tabctx needs from a tabular ICL model implementation.

This is the seam that lets tabctx add a second backend (e.g. TabPFN) later
without touching the cache manager, memory estimator, or engine -- none of
those import a specific backend, they only depend on this protocol.
"""

from __future__ import annotations

from typing import Any, Protocol

from tabctx.types import ArrayLike, PredictOutcome, Task


class TabularICLBackend(Protocol):
    """A tabular in-context-learning model, wrapped for tabctx.

    Implementations MUST construct a fresh underlying estimator/model
    instance inside every fit() call. fit() mutates instance state (that's
    the whole point -- it's where the training context gets encoded), so
    reusing one shared instance across dataset_ids would silently corrupt
    every cached context that isn't the most recent one.
    """

    name: str

    def fit(self, X: ArrayLike, y: ArrayLike, task: Task) -> Any:
        """Encode the training context. Returns an opaque, backend-owned
        payload that the cache manager retains standalone (the engine never
        inspects it) and later passes back into predict()."""
        ...

    def predict(
        self, payload: Any, X_test: ArrayLike, return_proba: bool = False
    ) -> PredictOutcome:
        """Run inference against a cached context. Must be a SINGLE
        inference pass regardless of return_proba -- derive predictions from
        probabilities (e.g. argmax) rather than calling predict() and
        predict_proba() separately, since each call is a full forward pass
        over the cached context."""
        ...

    def context_bytes_hint(self, n_train: int, n_features: int) -> int | None:
        """Cost of the context most recently returned by fit(), for cache
        capacity/eviction accounting -- called by the engine AFTER fit(),
        not before, so an implementation can report a real measurement
        (e.g. actual device memory delta) instead of a guess. Return None
        if the backend has no opinion; the engine falls back to its own
        (much more conservative, pre-fit-only) MemoryEstimator."""
        ...
