"""TabctxEngine: orchestrates a backend, a context cache, and a memory
estimator behind a small fit/predict/fit_predict API.

This is the one class most callers (including serve/app.py) actually use.
"""

from __future__ import annotations

import contextlib
import gc
import threading
import uuid
from collections.abc import Callable, Iterator

from tabctx import timing
from tabctx.backends._cuda import release_cached_device_memory
from tabctx.backends.base import TabularICLBackend
from tabctx.cache.manager import CachedContext, ContextCacheManager
from tabctx.chunking import MAX_CHUNK_ROWS, choose_chunk_rows, split_rows
from tabctx.errors import (
    AdmissionRejected,
    BackendComputeError,
    DatasetNotFoundError,
    InvalidInputError,
)
from tabctx.labels import resolve_task
from tabctx.memory.estimator import MemoryEstimator
from tabctx.memory.host import HostMemoryGuard
from tabctx.schema import TableSchema
from tabctx.types import ArrayLike, EngineStats, PredictOutcome, Task

# Predicts of at most this many test rows against a resident context run
# beside a fit on the same device instead of waiting for it (head-of-line,
# #39): measured on an A100, a 100-row predict took ~0.1 s p50 during a
# 40k x 100 fit (vs the fit's full 11 s), slowing the fit ~15%. Larger
# predicts, and predicts that must first restore a spilled context, need
# device memory a running fit may be using, so they wait their turn. One
# chunk's worth (chunking.MAX_CHUNK_ROWS), so a light predict is one call.
LIGHT_PREDICT_ROWS = MAX_CHUNK_ROWS
# A predict whose context is evicted again (under memory pressure) between
# its restore and getting the device tries this many times.
_PREDICT_ATTEMPTS = 3
# Large fits prepared (CPU half done) and waiting for the device at once;
# smaller fits' prepared copies are too small to limit.
_PREPARED_AHEAD = 1
GATED_FIT_CELLS = 1_000_000


def _not_found(dataset_id: str) -> DatasetNotFoundError:
    return DatasetNotFoundError(
        f"no cached context for dataset_id={dataset_id!r} -- it was "
        "never fit, was evicted, or was lost to a replica restart "
        "(the cache has no durability across process restarts)"
    )


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


def _predict_splitting_on_oom(
    backend: TabularICLBackend, payload: object, X_test: ArrayLike, return_proba: bool
) -> list[PredictOutcome]:
    """Predict, halving the chunk on a CUDA OOM until it fits (or one row
    still doesn't). Chunk sizing estimates can't see everything on a nearly
    full device -- e.g. a compressed kv cache's per-layer decompression --
    and a model may pick a single batch with no OOM retry of its own. The
    halves run after the except block, whose traceback would otherwise keep
    the failed attempt's tensors alive."""
    try:
        return [backend.predict(payload, X_test, return_proba=return_proba)]
    except BackendComputeError:
        if len(X_test) <= 1:
            raise
    gc.collect()
    half = len(X_test) // 2
    return _predict_splitting_on_oom(
        backend, payload, X_test[:half], return_proba
    ) + _predict_splitting_on_oom(backend, payload, X_test[half:], return_proba)


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
        host_guard: HostMemoryGuard | None = None,
        device_baseline_bytes: int = 0,
        device_available: Callable[[], int | None] | None = None,
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
        # Device scheduling. _fit_lock serializes everything that grows
        # residency on this device: fits, restores of spilled contexts and
        # large predicts. _predict_lock runs one GPU predict at a time, so
        # a light predict (see LIGHT_PREDICT_ROWS) overlaps a running fit
        # rather than queueing behind it. The cache's own lock only guards
        # bookkeeping. Lock order: prepare gate (fits only), host slot,
        # _fit_lock, _predict_lock (the cache lock and _release_lock are
        # leaves).
        self._fit_lock = threading.Lock()
        self._prepare_gate = threading.BoundedSemaphore(_PREPARED_AHEAD)
        # Re-entrant: a fit retrying alone holds it while the backend's
        # measurement takes it again (see _cuda.QuiescedMeasurement).
        self._predict_lock = threading.RLock()
        # Contexts removed while a fit measures itself, freed after (see
        # _measuring); the release lock orders frees against measurement.
        self._release_lock = threading.Lock()
        self._fitting = False
        self._parked: list[CachedContext] = []
        for b in backends.values():
            lend = getattr(b, "set_measurement_quiesce", None)
            if callable(lend):
                lend(self._predict_lock)
        # Process-wide host-memory admission, shared by every device's
        # engine (see memory/host.py); None = unguarded.
        self._host_guard = host_guard
        # Device memory in use before any context: CUDA context, model
        # weights (TabPFN's shared models ~1.7 GB, TabICL's backbone),
        # allocator overhead -- invisible to the cache's accounting.
        # Measured after warmup; counted against every headroom check (on a
        # 22 GB L4 the 15% transient reserve doesn't absorb it: fits the
        # books said were safe OOMed).
        self._baseline_bytes = device_baseline_bytes
        # The device's real free memory (see _cuda.device_available_bytes):
        # making room also checks it, since memory outside the books grows
        # after warmup (allocator fragmentation, compiled kernels'
        # workspaces) -- on a 22 GB L4 an admitted fit OOMed beside 18 GB
        # the books didn't fully see.
        self._device_available = device_available

    @property
    def backend_names(self) -> list[str]:
        return list(self._backends)

    @property
    def default_backend(self) -> str:
        return self._default_backend

    def backend_for(self, backend: str | None = None) -> TabularICLBackend:
        """The backend instance serving a model (the default when None)."""
        return self._resolve_backend(backend)[1]

    def estimator_for(self, backend: str | None = None) -> MemoryEstimator:
        return self._estimators[backend or self._default_backend]

    def admission_headroom_bytes(self, backend: str | None = None) -> int:
        """Transient headroom a fit could use right now, given what the
        cache holds resident (the quantity fit admission bounds).
        Lock-free snapshot: never waits on this engine's GPU work."""
        return self._headroom(
            self.estimator_for(backend), self._cache.stats_nowait().used_bytes
        )

    def _headroom(self, estimator: MemoryEstimator, used_bytes: int) -> int:
        return estimator.admission_headroom_bytes(used_bytes + self._baseline_bytes)

    def holds(self, dataset_id: str) -> bool:
        """Resident in this engine's cache (lock-free snapshot; a
        spilled copy doesn't count)."""
        return self._cache.holds(dataset_id)

    def has(self, dataset_id: str) -> bool:
        """Resident or spilled here (lock-free snapshot)."""
        return self._cache.has(dataset_id)

    def residency_nowait(self) -> tuple[frozenset[str], frozenset[str]]:
        """(resident ids, resident-or-spilled ids), lock-free."""
        return self._cache.residency_nowait()

    def estimate_fit_bytes(
        self, n_train: int, n_features: int, backend: str | None = None
    ) -> int:
        """What admission would charge a fit of this shape (placement uses
        it to pick a device; the fit itself re-checks under the lock)."""
        return self.estimator_for(backend).estimate_bytes(n_train, 0, n_features)

    def resolve_backend_name(self, backend: str | None) -> str:
        """The model a fit would use; InvalidInputError if unknown."""
        return self._resolve_backend(backend)[0]

    def stats_nowait(self) -> EngineStats:
        return self._cache.stats_nowait()

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
        task: Task | None = None,
        dataset_id: str | None = None,
        backend: str | None = None,
        schema: TableSchema | None = None,
    ) -> str:
        """Encode and cache a training context. Returns a dataset_id to pass
        to predict() -- generated automatically unless the caller supplies
        one (e.g. to reuse a stable, caller-known identifier).

        schema (optional) records how the training table's columns were
        encoded (tabctx.schema) on the cached context, for callers that
        encode test tables the same way; the engine itself only ever
        checks feature *count*.

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
        task = resolve_task(y, task)
        backend_name, chosen = self._resolve_backend(backend)
        estimator = self._estimators[backend_name]

        resolved_id = dataset_id or str(uuid.uuid4())
        estimated = estimator.estimate_bytes(n_train, 0, n_features)
        self._check_feasible(estimator, estimated, n_train, n_features)
        # The fit's CPU half (backends that split it: TabICL's preprocessing,
        # ~1/3 of a fit) runs before taking the device, overlapping the GPU
        # half of whatever fit holds it now.
        # At most _PREPARED_AHEAD large fits hold prepared (host-resident)
        # data while waiting for the device, so queued large tables don't
        # each keep their preprocessed copies. The permit is returned once
        # the fit has the device, and small fits skip the gate, so neither
        # a running fit nor large fits waiting for host memory hold up a
        # small fit on this device.
        # Lock order everywhere: prepare gate, host slot, _fit_lock,
        # _predict_lock. Host memory before the device, so a large fit
        # waiting for host memory doesn't hold this device meanwhile.
        with timing.counting_compiles(), contextlib.ExitStack() as gate:
            if n_train * n_features >= GATED_FIT_CELLS:
                with timing.phase("wait_prepare_gate"):
                    self._prepare_gate.acquire()
                gate.callback(self._prepare_gate.release)
            prepare = getattr(chosen, "prepare_fit", None)
            with timing.phase("prepare"):
                prepared = prepare(X, y, task) if callable(prepare) else None
            with contextlib.ExitStack() as device:
                with timing.phase("wait_host_memory"):
                    device.enter_context(self._host_slot(chosen, n_train, n_features))
                with timing.phase("wait_device"):
                    device.enter_context(self._fit_lock)
                gate.close()  # the device is ours: let the next fit prepare
                with timing.phase("make_room"):
                    self._make_room(estimator, estimated)
                with self._measuring(), timing.phase("fit_device"):
                    payload = self._fit_backend(
                        chosen, X, y, task, estimator, estimated, prepared
                    )
                    est_bytes = chosen.context_bytes_hint(n_train, n_features)
                    if est_bytes is None:
                        est_bytes = estimated
                    else:
                        self._learn(chosen, estimator, n_train, n_features, est_bytes)
                context = CachedContext(
                    dataset_id=resolved_id,
                    backend_name=backend_name,
                    task=task,
                    n_train=n_train,
                    n_features=n_features,
                    payload=payload,
                    est_bytes=est_bytes,
                    schema=schema,
                )
                self._cache.put(context)
        return resolved_id

    def _check_feasible(
        self,
        estimator: MemoryEstimator,
        estimated: int,
        n_train: int,
        n_features: int,
    ) -> None:
        """Reject a fit that can never be admitted, before any work.

        Usage-aware admission (v0.9.0): what OOMs is the fit's need on top
        of everything resident, so under _fit_lock the fit evicts ahead
        (_make_room) until it fits -- without that, a filling replica would
        reject fits that are perfectly safe once a cold context steps aside
        (proven on real hardware: a fourth ~13GB-peak fit OOMed an A100 with
        three ~8GB contexts resident). Feasibility comes FIRST, against an
        EMPTY cache's headroom: otherwise one oversized request drains every
        tenant's context on its way to a 413 (observed live: a 200k x 60
        request spilled 30 contexts and then failed)."""
        max_headroom = self._headroom(estimator, 0)
        if estimated > max_headroom:
            raise AdmissionRejected(
                f"training shape ({n_train} rows x {n_features} "
                f"features) is estimated to need {estimated} bytes at "
                f"peak, exceeding this replica's "
                f"{max_headroom} byte maximum "
                "admission headroom (even with an empty cache)"
            )

    def _make_room(self, estimator: MemoryEstimator, needed: int) -> None:
        """Evict until `needed` bytes of headroom are free, or nothing is
        left to evict. Caller holds _fit_lock (and no cache or predict
        lock). Contexts a predict is using are pinned and evicted last:
        when only they stand in the way, predicts are held off (no predict
        is then running, and queued ones re-check residency once they get
        the device) and those contexts go too."""

        def short() -> bool:
            if needed > self._headroom(estimator, self._cache.stats().used_bytes):
                return True
            available = self._device_available() if self._device_available else None
            return available is not None and needed > available

        while short():
            if self._cache.evict_one() is None:
                break
        if short() and self._cache.has_pins():
            with self._predict_lock:
                while short() and self._cache.evict_one(ignore_pins=True) is not None:
                    pass

    @contextlib.contextmanager
    def _measuring(self) -> Iterator[None]:
        """While a fit measures its footprint as an allocator delta, no
        other context's memory may be freed on this device: evicted
        payloads are parked (see _release) and dropped afterwards."""
        with self._release_lock:
            self._fitting = True
        try:
            yield
        finally:
            with self._release_lock:
                self._fitting = False
                parked, self._parked = self._parked, []
            del parked

    def _drop_parked(self) -> None:
        with self._release_lock:
            parked, self._parked = self._parked, []
        del parked

    def _release(self, context: CachedContext | None) -> None:
        """Drop the caller's last reference to a removed context now, or
        park it until the running fit has measured itself. The caller must
        not keep another reference."""
        with self._release_lock:
            if context is not None and self._fitting:
                self._parked.append(context)
            context = None

    def _fit_backend(
        self,
        backend: TabularICLBackend,
        X: ArrayLike,
        y: ArrayLike,
        task: Task,
        estimator: MemoryEstimator,
        estimated: int,
        prepared: object | None = None,
    ) -> object:
        """Fit (its GPU half, when prepared), retrying once if it ran out of
        device memory while sharing the device with a predict: after the
        failed attempt's tensors are freed (outside the except block, whose
        traceback would keep them alive), with predicts held off and room
        made again."""

        def attempt() -> object:
            if prepared is not None:
                return backend.fit_prepared(prepared)  # type: ignore[attr-defined]
            return backend.fit(X, y, task)

        try:
            return attempt()
        except BackendComputeError:
            pass
        with self._predict_lock:
            self._make_room(estimator, estimated)
            # Contexts removed during the failed attempt (explicit evicts,
            # re-fits) were parked so as not to skew its measurement
            # (_measuring); that measurement is void and the retry measures
            # afresh, so free them now and give the retry their memory.
            self._drop_parked()
            gc.collect()
            release_cached_device_memory()
            return attempt()

    def _learn(
        self,
        backend: TabularICLBackend,
        estimator: MemoryEstimator,
        n_train: int,
        n_features: int,
        est_bytes: int,
    ) -> None:
        """Feed a fit's real measurements back so the PRE-FIT admission gate
        can use them (safely, as a bound for smaller/equal future shapes --
        see memory/adaptive.py). The estimator decides what the fit needed:
        resident + working floor for models whose working memory adapts to
        what is free (their peak only says how much was free), else the
        transient peak."""
        peak_hint = getattr(backend, "fit_peak_bytes_hint", None)
        peak_bytes = peak_hint() if callable(peak_hint) else None
        # What the fit needed resident while it ran: backends that compress
        # the context afterwards (TabICL's int8 kv cache) report the
        # pre-compression size; est_bytes, the compressed size, stays the
        # cache-accounting quantity.
        fit_hint = getattr(backend, "fit_resident_bytes_hint", None)
        fit_resident = fit_hint() if callable(fit_hint) else None
        record_fit = getattr(estimator, "record_fit", None)
        if callable(record_fit):
            record_fit(n_train, n_features, fit_resident or est_bytes, peak_bytes)
        else:
            estimator.record_observation(
                n_train, n_features, peak_bytes if peak_bytes else est_bytes
            )

    def _host_slot(self, backend: TabularICLBackend, n_rows: int, n_features: int):
        """Reserve this call's estimated host memory against the process-
        wide (or node-wide) budget. Whether it offloads comes from the
        backend's may_offload (else the guard's cell threshold); its bytes
        from host_bytes_estimate, else one large fit's worth when it
        offloads. A no-op without a guard."""
        guard = self._host_guard
        if guard is None:
            return contextlib.nullcontext()
        may_offload = getattr(backend, "may_offload", None)
        offloads = bool(
            may_offload(n_rows, n_features)
            if callable(may_offload)
            else guard.is_large(n_rows, n_features)
        )
        estimate = getattr(backend, "host_bytes_estimate", None)
        if callable(estimate):
            try:
                nbytes = estimate(n_rows, n_features, offloads=offloads)
            except TypeError:  # an estimate from before `offloads` existed
                nbytes = estimate(n_rows, n_features)
        else:
            nbytes = guard.large_fit_bytes if offloads else 0
        return guard.reserve(nbytes, offloads=offloads)

    def _restore(self, dataset_id: str) -> None:
        """Bring a spilled context back onto the device (DatasetNotFoundError
        if it is neither resident nor spilled). Restores grow residency, so
        they take _fit_lock and make room for the context first; one that
        runs out of memory beside a predict retries with predicts held off
        (the spilled copy survives a device OOM)."""
        with self._fit_lock:
            # A context mid-spill is in neither tier until the write lands.
            self._cache.wait_settled(dataset_id)
            if self._cache.peek(dataset_id) is not None:
                return
            size = self._cache.spilled_size(dataset_id)
            if size is None:
                raise _not_found(dataset_id)
            estimator = self._estimators[self._default_backend]
            self._make_room(estimator, size)
            try:
                restored = self._cache.get(dataset_id)
            except BackendComputeError:
                restored = None
                retry = True
            else:
                retry = False
            if retry:
                gc.collect()
                release_cached_device_memory()
                with self._predict_lock:
                    restored = self._cache.get(dataset_id)
            if restored is None:
                raise _not_found(dataset_id)

    def schema(self, dataset_id: str) -> TableSchema | None:
        """The schema recorded at fit() for a cached context, or None
        when the context is unknown or was fit without one. Reloads a
        spilled context as a side effect, exactly as predict() would."""
        if self._cache.peek(dataset_id) is None:
            try:
                self._restore(dataset_id)
            except DatasetNotFoundError:
                return None
        context = self._cache.peek(dataset_id)
        return context.schema if context is not None else None

    def predict(
        self, dataset_id: str, X_test: ArrayLike, return_proba: bool = False
    ) -> PredictOutcome:
        """Predict against a previously-fit context, reusing its cached
        training-context encoding. Large test sets are automatically chunked
        (see chunking.py) against the memory ceiling -- this is the one
        thing that must never let a single request take down the replica,
        which the naive one-shot wrapper this library replaces did not do.

        A light predict (a resident context, at most LIGHT_PREDICT_ROWS test
        rows) runs beside a fit on this device; larger ones wait for the
        device's fits, and a spilled context is restored first.
        """
        exclusive = len(X_test) > LIGHT_PREDICT_ROWS
        with self._cache.pinned(dataset_id), timing.counting_compiles():
            for _ in range(_PREDICT_ATTEMPTS):
                if self._cache.peek(dataset_id) is None:
                    with timing.phase("restore"):
                        self._restore(dataset_id)
                    timing.count("restores")
                outcome = self._predict_resident(
                    dataset_id, X_test, return_proba, exclusive
                )
                if outcome is not None:
                    return outcome
        raise BackendComputeError(
            f"context {dataset_id!r} was evicted again before its predict "
            f"could run ({_PREDICT_ATTEMPTS} attempts); the device is under "
            "memory pressure -- retry"
        )

    def _predict_resident(
        self, dataset_id: str, X_test: ArrayLike, return_proba: bool, exclusive: bool
    ) -> PredictOutcome | None:
        """Run a predict if the context is resident once this call has the
        device (None if it was evicted meanwhile). Holds no reference to the
        context while waiting for locks, so an eviction can really free it."""
        context = self._cache.peek(dataset_id)
        if context is None:
            return None
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
        n_features = context.n_features
        # Chunk against the usage-aware transient headroom (what the
        # device can actually take on top of resident contexts), not the
        # cache-capacity ceiling -- the same quantity fit admission uses.
        chunk_rows = choose_chunk_rows(
            estimator,
            context.n_train,
            n_features,
            self._headroom(estimator, self._cache.stats_nowait().used_bytes),
        )
        context = None
        n_test = len(X_test)
        chunks = split_rows(X_test, chunk_rows) if chunk_rows < n_test else [X_test]
        largest_chunk = max(len(c) for c in chunks)

        all_predictions: list = []
        all_probabilities: list[list[float]] = []
        classes: list[str] | None = None
        with contextlib.ExitStack() as device:
            with timing.phase("wait_host_memory"):
                device.enter_context(
                    self._host_slot(backend, largest_chunk, n_features)
                )
            with timing.phase("wait_device"):
                if exclusive:
                    device.enter_context(self._fit_lock)
                device.enter_context(self._predict_lock)
            context = self._cache.peek(dataset_id)
            if context is None:
                return None
            self._cache.touch(dataset_id)
            timing.count("predict_chunks", len(chunks))
            try:
                with timing.phase("predict_device"):
                    outcomes = [
                        outcome
                        for chunk in chunks
                        for outcome in _predict_splitting_on_oom(
                            backend, context.payload, chunk, return_proba
                        )
                    ]
                for outcome in outcomes:
                    all_predictions.extend(outcome.predictions)
                    if return_proba:
                        all_probabilities.extend(outcome.probabilities or [])
                        classes = outcome.classes or classes
            finally:
                # Still under the predict lock: if the context was evicted
                # while this ran, its memory is freed now -- atomic with
                # respect to a fit's measurements -- or parked.
                if self._cache.peek(dataset_id) is not context:
                    self._release(context)
                context = None

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
        task: Task | None = None,
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
        # Under the release lock, so the memory is freed either before a
        # fit starts measuring or after it is done (see _measuring).
        with self._release_lock:
            removed = self._cache.evict(dataset_id)
            if removed is not None and self._fitting:
                self._parked.append(removed)
            removed = None

    def stats(self) -> EngineStats:
        return self._cache.stats()
