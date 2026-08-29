"""TabctxEngine: orchestrates a backend, a context cache, and a memory
estimator behind a small fit/predict/fit_predict API.

This is the one class most callers (including serve/app.py) actually use.
"""

from __future__ import annotations

import uuid

from tabctx.backends.base import TabularICLBackend
from tabctx.cache.manager import CachedContext, ContextCacheManager
from tabctx.chunking import choose_chunk_rows, split_rows
from tabctx.errors import AdmissionRejected, DatasetNotFoundError, InvalidInputError
from tabctx.memory.estimator import MemoryEstimator
from tabctx.types import ArrayLike, EngineStats, PredictOutcome, Task


def _rect_shape(X: ArrayLike) -> tuple[int, int] | None:
    """(n_rows, n_cols) when X is a 2D array-with-shape (e.g. numpy) --
    rectangular by construction, so the per-row Python loop below would
    be pure waste (it matters: fit/predict-by-upload feeds 10^5-10^6-row
    numpy arrays through here). None for plain nested sequences."""
    shape = getattr(X, "shape", None)
    if shape is not None and len(shape) == 2:
        return int(shape[0]), int(shape[1])
    return None


def _row_lengths(X: ArrayLike) -> list[int]:
    return [len(row) for row in X]


def _validate_fit_input(X: ArrayLike, y: ArrayLike) -> int:
    """Returns n_features. Raises InvalidInputError for anything that would
    otherwise reach the backend as a malformed array and blow up as an
    unhandled, untranslated exception (numpy/sklearn shape errors) --
    found via load testing to surface as a bare 500 with no useful detail."""
    n_train = len(X)
    if n_train == 0:
        raise InvalidInputError("train_X must be non-empty")
    if len(y) != n_train:
        raise InvalidInputError(
            f"train_X has {n_train} rows but train_y has {len(y)} labels"
        )
    rect = _rect_shape(X)
    if rect is not None:
        n_features = rect[1]
        if n_features == 0:
            raise InvalidInputError("training rows must have at least one feature")
        return n_features
    lengths = _row_lengths(X)
    n_features = lengths[0]
    if n_features == 0:
        raise InvalidInputError("training rows must have at least one feature")
    if any(length != n_features for length in lengths):
        raise InvalidInputError(
            "all training rows must have the same number of features "
            f"(saw lengths ranging {min(lengths)}-{max(lengths)})"
        )
    return n_features


def _validate_predict_input(X_test: ArrayLike, n_features: int) -> None:
    if len(X_test) == 0:
        raise InvalidInputError("test_X must be non-empty")
    rect = _rect_shape(X_test)
    if rect is not None:
        if rect[1] != n_features:
            raise InvalidInputError(
                f"test_X rows must have {n_features} features (matching the "
                f"cached training context), got {rect[1]}"
            )
        return
    lengths = _row_lengths(X_test)
    if any(length != n_features for length in lengths):
        raise InvalidInputError(
            f"test_X rows must have {n_features} features (matching the "
            f"cached training context), saw lengths ranging "
            f"{min(lengths)}-{max(lengths)}"
        )


class TabctxEngine:
    """Orchestrates one or more backends over a single shared context
    cache and GPU budget. Multi-backend serving (e.g. TabICL and TabPFN
    behind one endpoint) works because every cached context remembers
    which backend fit it (CachedContext.backend_name): fit() picks a
    backend, predict() dispatches to whichever one owns the context.
    Memory estimates are per-backend (models peak differently for the
    same shape) while admission headroom is global (they share the
    device)."""

    def __init__(
        self,
        backend: TabularICLBackend | None = None,
        cache: ContextCacheManager | None = None,
        estimator: MemoryEstimator | None = None,
        backends: dict[str, TabularICLBackend] | None = None,
        estimators: dict[str, MemoryEstimator] | None = None,
        default_backend: str | None = None,
    ) -> None:
        """Single-backend form: TabctxEngine(backend=, cache=, estimator=).
        Multi-backend form: TabctxEngine(backends={name: b}, cache=,
        estimators={name: e}, default_backend=name)."""
        if cache is None:
            raise ValueError("cache is required")
        if backends is None:
            if backend is None or estimator is None:
                raise ValueError("provide backend+estimator, or backends+estimators")
            backends = {backend.name: backend}
            estimators = {backend.name: estimator}
            default_backend = backend.name
        if estimators is None or default_backend not in backends:
            raise ValueError(
                "multi-backend form needs estimators and a default_backend "
                "that is one of backends"
            )
        if set(estimators) != set(backends):
            raise ValueError("backends and estimators must share keys")
        self._backends = backends
        self._estimators = estimators
        self._default_backend = default_backend
        self._cache = cache

    @property
    def backend_names(self) -> list[str]:
        return list(self._backends)

    @property
    def default_backend(self) -> str:
        return self._default_backend

    def estimator_for(self, backend: str | None = None) -> MemoryEstimator:
        return self._estimators[backend or self._default_backend]

    def _resolve_backend(self, backend: str | None) -> tuple[str, TabularICLBackend]:
        name = backend or self._default_backend
        b = self._backends.get(name)
        if b is None:
            raise InvalidInputError(
                f"unknown backend {name!r}; this deployment serves "
                f"{sorted(self._backends)}"
            )
        return name, b

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        task: Task = "classification",
        dataset_id: str | None = None,
        backend: str | None = None,
    ) -> str:
        """Encode and cache a training context. Returns a dataset_id to pass
        to predict() -- generated automatically unless the caller supplies
        one (e.g. to reuse a stable, caller-known identifier).

        Raises InvalidInputError for malformed input (mismatched X/y
        lengths, empty tables, ragged rows) before anything else -- checked
        first so a careless client's bad request can't reach the backend as
        an unhandled exception.

        Raises AdmissionRejected before any backend/GPU work if the training
        shape alone is estimated to exceed the configured memory ceiling
        (using the conservative formula-based estimator -- this pre-fit gate
        must stay conservative since nothing has run yet to measure).

        The context's cache-accounting size (used for capacity/eviction
        bookkeeping, not for this admission gate) is queried from the
        backend AFTER fit() completes, not before: a real backend can
        measure actual post-fit GPU memory now that the fit has happened,
        which is far more accurate than any pre-fit estimate. Confirmed
        empirically this mattered -- the formula-based estimate ran ~14x
        higher than real measured GPU memory for a realistic multi-tenant
        shape, needlessly throttling how many contexts actually fit.
        """
        n_train = len(X)
        n_features = _validate_fit_input(X, y)
        backend_name, chosen = self._resolve_backend(backend)
        estimator = self._estimators[backend_name]

        resolved_id = dataset_id or str(uuid.uuid4())
        with self._cache.lock:
            # Usage-aware admission (v0.9.0): what OOMs is the fit's
            # transient PEAK on top of everything resident, so the
            # estimated peak must fit the headroom given current cache
            # usage -- checked under the cache lock so two concurrent
            # fits can't both pass against the same snapshot.
            #
            # Evict-ahead-of-fit: when the cache is warm, cold contexts
            # are evicted (spilled, when a spill tier is attached) until
            # the fit's transient need fits -- without this, a filling
            # replica would reject fits that are perfectly safe once a
            # cold context steps aside. Proven necessary on real
            # hardware: a fourth ~13GB-peak fit OOMed an A100 with three
            # ~8GB contexts resident, exactly the state this drains.
            estimated = estimator.estimate_bytes(n_train, 0, n_features)
            # Feasibility FIRST, against an EMPTY cache's headroom: a fit
            # that can never be admitted must be rejected before evicting
            # anything -- otherwise one oversized request drains every
            # tenant's context on its way to a 413 (observed live: a
            # 200k x 60 request spilled 30 contexts and then failed).
            if estimated > estimator.admission_headroom_bytes(0):
                raise AdmissionRejected(
                    f"training shape ({n_train} rows x {n_features} "
                    f"features) is estimated to need {estimated} bytes at "
                    f"peak, exceeding this replica's "
                    f"{estimator.admission_headroom_bytes(0)} byte maximum "
                    "admission headroom (even with an empty cache)"
                )
            while True:
                headroom = estimator.admission_headroom_bytes(
                    self._cache.stats().used_bytes
                )
                if estimated <= headroom:
                    break
                # Feasible but blocked by warm contexts: evict-ahead.
                # Guaranteed to terminate at the feasibility bound above.
                self._cache.evict_one()
            payload = chosen.fit(X, y, task)
            est_bytes = chosen.context_bytes_hint(n_train, n_features)
            if est_bytes is None:
                est_bytes = estimator.estimate_bytes(n_train, 0, n_features)
            else:
                # Feed a real measurement back so the PRE-FIT admission
                # gate can use it (safely, as a bound for smaller/equal
                # future shapes -- see memory/adaptive.py). What admission
                # must bound is the fit's transient PEAK (that's what
                # OOMs), so prefer the peak hint when the backend reports
                # one; the resident size is only a (low) fallback for
                # backends without peak measurement.
                peak_hint = getattr(chosen, "fit_peak_bytes_hint", None)
                peak_bytes = peak_hint() if callable(peak_hint) else None
                estimator.record_observation(
                    n_train, n_features, peak_bytes if peak_bytes else est_bytes
                )
            context = CachedContext(
                dataset_id=resolved_id,
                backend_name=backend_name,
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
        _validate_predict_input(X_test, context.n_features)
        # Dispatch to whichever backend fit this context (multi-backend
        # deployments serve several models over one cache).
        backend = self._backends.get(context.backend_name)
        if backend is None:
            raise DatasetNotFoundError(
                f"context for {dataset_id!r} was fit by backend "
                f"{context.backend_name!r}, which this deployment no longer "
                "serves -- re-fit with one of "
                f"{sorted(self._backends)}"
            )
        estimator = self._estimators[context.backend_name]

        n_test = len(X_test)
        # Chunk against the usage-aware transient headroom (what the
        # device can actually take on top of resident contexts), not the
        # cache-capacity ceiling -- the same quantity fit admission uses.
        chunk_rows = choose_chunk_rows(
            estimator,
            context.n_train,
            context.n_features,
            estimator.admission_headroom_bytes(self._cache.stats().used_bytes),
        )
        chunks = split_rows(X_test, chunk_rows) if chunk_rows < n_test else [X_test]

        all_predictions: list = []
        all_probabilities: list[list[float]] = []
        classes: list[str] | None = None
        with self._cache.lock:
            self._cache.touch(dataset_id)
            for chunk in chunks:
                outcome = backend.predict(
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
