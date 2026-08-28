"""TabctxEngine: orchestrates a backend, a context cache, and a memory
estimator behind a small fit/predict/fit_predict API.

This is the one class most callers (including serve/app.py) actually use.
"""

from __future__ import annotations

import uuid

from tabctx.backends.base import TabularICLBackend
from tabctx.cache.manager import CachedContext, ContextCacheManager
from tabctx.chunking import choose_chunk_rows, split_rows
from tabctx.errors import AdmissionRejected, DatasetNotFoundError
from tabctx.memory.estimator import MemoryEstimator
from tabctx.types import ArrayLike, EngineStats, PredictOutcome, Task


class TabctxEngine:
    def __init__(
        self,
        backend: TabularICLBackend,
        cache: ContextCacheManager,
        estimator: MemoryEstimator,
    ) -> None:
        self._backend = backend
        self._cache = cache
        self._estimator = estimator

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        task: Task = "classification",
        dataset_id: str | None = None,
    ) -> str:
        """Encode and cache a training context. Returns a dataset_id to pass
        to predict() -- generated automatically unless the caller supplies
        one (e.g. to reuse a stable, caller-known identifier).

        Raises AdmissionRejected before any backend/GPU work if the training
        shape alone is estimated to exceed the configured memory ceiling.
        """
        n_train = len(X)
        n_features = len(X[0]) if n_train else 0

        if not self._estimator.admit(n_train, 0, n_features):
            raise AdmissionRejected(
                f"training shape ({n_train} rows x {n_features} features) "
                f"is estimated to need "
                f"{self._estimator.estimate_bytes(n_train, 0, n_features)} bytes, "
                f"exceeding the {self._estimator.ceiling_bytes()} byte ceiling"
            )

        est_bytes = self._backend.context_bytes_hint(n_train, n_features)
        if est_bytes is None:
            est_bytes = self._estimator.estimate_bytes(n_train, 0, n_features)

        resolved_id = dataset_id or str(uuid.uuid4())
        with self._cache.lock:
            payload = self._backend.fit(X, y, task)
            context = CachedContext(
                dataset_id=resolved_id,
                backend_name=self._backend.name,
                task=task,
                n_train=n_train,
                n_features=n_features,
                payload=payload,
                est_bytes=est_bytes,
            )
            self._cache.put(context)
        return resolved_id

    def predict(
        self, dataset_id: str, X_test: ArrayLike, return_proba: bool = False
    ) -> PredictOutcome:
        """Predict against a previously-fit context, reusing its cached
        training-context encoding. Large test sets are automatically chunked
        (see chunking.py) against the memory ceiling -- this is the one
        thing that must never let a single request take down the replica,
        which the naive one-shot wrapper this library replaces did not do.
        """
        context = self._cache.get(dataset_id)
        if context is None:
            raise DatasetNotFoundError(
                f"no cached context for dataset_id={dataset_id!r} -- it was "
                "never fit, was evicted, or was lost to a replica restart "
                "(the cache has no durability across process restarts)"
            )

        n_test = len(X_test)
        chunk_rows = choose_chunk_rows(
            self._estimator,
            context.n_train,
            context.n_features,
            self._estimator.ceiling_bytes(),
        )
        chunks = split_rows(X_test, chunk_rows) if chunk_rows < n_test else [X_test]

        all_predictions: list = []
        all_probabilities: list[list[float]] = []
        classes: list[str] | None = None
        with self._cache.lock:
            self._cache.touch(dataset_id)
            for chunk in chunks:
                outcome = self._backend.predict(
                    context.payload, chunk, return_proba=return_proba
                )
                all_predictions.extend(outcome.predictions)
                if return_proba:
                    all_probabilities.extend(outcome.probabilities or [])
                    classes = outcome.classes or classes

        return PredictOutcome(
            predictions=all_predictions,
            probabilities=all_probabilities if return_proba else None,
            classes=classes,
        )

    def fit_predict(
        self,
        X: ArrayLike,
        y: ArrayLike,
        X_test: ArrayLike,
        task: Task = "classification",
        return_proba: bool = False,
    ) -> PredictOutcome:
        """Convenience matching the one-shot semantics of the wrapper this
        library replaces: fit, predict once, evict. Prefer fit()+predict()
        directly when the same training set will be queried more than
        once -- that's the entire point of tabctx's caching."""
        dataset_id = self.fit(X, y, task)
        try:
            return self.predict(dataset_id, X_test, return_proba=return_proba)
        finally:
            self.evict(dataset_id)

    def evict(self, dataset_id: str) -> None:
        self._cache.evict(dataset_id)

    def stats(self) -> EngineStats:
        return self._cache.stats()
