"""MultiDeviceEngine: one replica serving from several GPUs.

Neither TabICL nor TabPFN can split one training context across GPUs
(TabICL is single-device; TabPFN's device lists only parallelize ensemble
members, and not in its kv-cache fit mode). So a multi-GPU replica runs
one complete TabctxEngine per device -- its own backend instances bound
to that device, its own cache, lock and spill tier -- and this class
routes between them:

- fit() places each context where a PlacementPolicy says (placement.py;
  the default, LeastActiveThatFits, prefers devices where the table fits
  without eviction, the least active among them, else the one with most
  headroom). Each device's own usage-aware admission gate then has the
  final say, exactly as on a single GPU. A table's ceiling is therefore
  one device's memory; what more devices buy is N x the resident cache
  and concurrency.
- predict() runs on the device that holds the context, found from each
  device's residency and spill index (so a context spilled before a
  restart is found on, and restored to, its own device).

Concurrency. Each device's engine schedules its own GPU work (one fit at a
time, light predicts beside it; see engine.py), and this router never
waits on a device to make a decision: placement and lookup read lock-free snapshots (used bytes,
residency, spill index). Work on different devices therefore runs
concurrently, fits included. Per-dataset_id locks serialize fits and
evicts of the SAME id, so two concurrent fits of one id end with a single
copy. A re-fit happens on the chosen device first; only after it succeeds
are copies on other devices removed -- a failed re-fit (413, OOM) leaves
the previous context serving, as on a single GPU. A predict racing a
re-fit is answered by whichever copy it reaches (old or new), never 404.

Placement is a snapshot: two concurrent fits of different ids may pick
the same device, in which case the second evicts-ahead there like any
single-GPU fit would. Serializing placement would cost the concurrency
this class exists for.

Residency events (events.py) report each placement and removal, so a
platform scheduler can follow where contexts live without polling.

Same public surface as TabctxEngine, so the serving layer and batching
use either interchangeably.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from tabctx import timing
from tabctx.engine import _validate_fit_input
from tabctx.errors import CacheUnavailableError, DatasetNotFoundError
from tabctx.events import EventBus, EventKind, ResidencyEvent
from tabctx.keyed_locks import KeyedLocks
from tabctx.labels import resolve_task
from tabctx.placement import (
    DeviceView,
    FitRequest,
    LeastActiveThatFits,
    PlacementPolicy,
)
from tabctx.schema import TableSchema
from tabctx.types import ArrayLike, ContextInfo, EngineStats, PredictOutcome, Task

logger = logging.getLogger(__name__)

# How long a started call counts toward a device's recent activity.
ACTIVITY_WINDOW_S = 30.0


class Shard(Protocol):
    """One device's engine as the router sees it: a TabctxEngine in this
    process, or a ProcessShard proxying one in a worker process
    (serve/workers.py). The *_nowait-style reads (holds, has,
    stats_nowait, admission_headroom_bytes) must not wait on the device's
    GPU work."""

    @property
    def backend_names(self) -> list[str]: ...

    @property
    def default_backend(self) -> str: ...

    def resolve_backend_name(self, backend: str | None) -> str: ...

    def estimate_fit_bytes(
        self, n_train: int, n_features: int, backend: str | None = None
    ) -> int: ...

    def admission_headroom_bytes(self, backend: str | None = None) -> int: ...

    def holds(self, dataset_id: str) -> bool: ...

    def residency_nowait(self) -> tuple[frozenset[str], frozenset[str]]: ...

    def has(self, dataset_id: str) -> bool: ...

    def stats_nowait(self) -> EngineStats: ...

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        task: Task | None = None,
        dataset_id: str | None = None,
        backend: str | None = None,
        schema: TableSchema | None = None,
        expires_at: float | None = None,
    ) -> str: ...

    def predict(
        self,
        dataset_id: str,
        X_test: ArrayLike,
        return_proba: bool = False,
        quantiles: Sequence[float] | None = None,
    ) -> PredictOutcome: ...

    def schema(self, dataset_id: str) -> TableSchema | None: ...

    def evict(self, dataset_id: str) -> bool: ...

    def contexts(self) -> list[ContextInfo]: ...


def _alive(shard: Shard) -> bool:
    """A worker process's liveness; an in-process engine is always alive."""
    return getattr(shard, "alive", True)


def _not_found(dataset_id: str) -> DatasetNotFoundError:
    return DatasetNotFoundError(
        f"no cached context for dataset_id={dataset_id!r} -- it was never "
        "fit, was evicted, or was lost to a replica restart"
    )


class MultiDeviceEngine:
    def __init__(
        self,
        shards: list[Shard],
        devices: list[str],
        activity_window_s: float = ACTIVITY_WINDOW_S,
        clock: Callable[[], float] = time.monotonic,
        placement: PlacementPolicy | None = None,
        events: EventBus | None = None,
    ) -> None:
        if len(shards) < 2 or len(shards) != len(devices):
            raise ValueError("need one engine per device, and at least two devices")
        names = {tuple(s.backend_names) for s in shards}
        defaults = {s.default_backend for s in shards}
        if len(names) != 1 or len(defaults) != 1:
            raise ValueError("every device must serve the same models")
        self._shards = shards
        self._devices = devices
        self._id_locks = KeyedLocks()
        self._window_s = activity_window_s
        self._clock = clock
        self._inflight = [0] * len(shards)
        self._recent: list[deque[float]] = [deque() for _ in shards]
        self._activity_lock = threading.Lock()
        self._placement: PlacementPolicy = placement or LeastActiveThatFits()
        self.events = events or EventBus()
        # Calls that go to every device at once (deletes, listings).
        self._fanout = ThreadPoolExecutor(len(shards), thread_name_prefix="fanout")

    @property
    def devices(self) -> list[str]:
        return list(self._devices)

    @property
    def backend_names(self) -> list[str]:
        return self._shards[0].backend_names

    @property
    def default_backend(self) -> str:
        return self._shards[0].default_backend

    def admission_headroom_bytes(self, backend: str | None = None) -> int:
        """The best any one device offers: a fit lands on a single device."""
        return max(self.device_headroom_bytes(backend))

    def device_headroom_bytes(self, backend: str | None = None) -> list[int]:
        return [s.admission_headroom_bytes(backend) for s in self._shards]

    # ---- activity --------------------------------------------------

    def _prune(self, recent: deque[float], cutoff: float) -> None:
        while recent and recent[0] < cutoff:
            recent.popleft()

    @contextlib.contextmanager
    def _active(self, index: int) -> Iterator[None]:
        now = self._clock()
        with self._activity_lock:
            self._inflight[index] += 1
            self._recent[index].append(now)
            self._prune(self._recent[index], now - self._window_s)
        try:
            yield
        finally:
            with self._activity_lock:
                self._inflight[index] -= 1

    def _activity(self) -> list[tuple[int, int]]:
        """(calls in flight, calls started within the window) per device."""
        cutoff = self._clock() - self._window_s
        with self._activity_lock:
            for recent in self._recent:
                self._prune(recent, cutoff)
            return [(self._inflight[i], len(r)) for i, r in enumerate(self._recent)]

    # ---- placement and lookup (lock-free snapshots) ----------------

    def _place(self, request: FitRequest) -> int:
        """Ask the policy, from lock-free snapshots of every device."""
        headrooms = self.device_headroom_bytes(request.backend)
        activity = self._activity()
        views = [
            DeviceView(
                index=i,
                device=self._devices[i],
                headroom_bytes=headrooms[i],
                # Each shard prices the fit with its own estimator (worker
                # processes learn separately; in-process shards share one).
                need_bytes=s.estimate_fit_bytes(
                    request.n_train, request.n_features, request.backend
                ),
                in_flight=activity[i][0],
                recent_calls=activity[i][1],
                alive=getattr(s, "alive", True),
            )
            for i, s in enumerate(self._shards)
        ]
        index = self._placement.choose(request, views)
        if not 0 <= index < len(self._shards):
            raise ValueError(f"placement policy chose device {index} of {len(views)}")
        return index

    def _locate(self, dataset_id: str) -> list[int]:
        """Devices holding the id: resident first, then spilled copies."""
        resident = [i for i, s in enumerate(self._shards) if s.holds(dataset_id)]
        spilled = [
            i
            for i, s in enumerate(self._shards)
            if i not in resident and s.has(dataset_id)
        ]
        return resident + spilled

    # ---- API -------------------------------------------------------

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        task: Task | None = None,
        dataset_id: str | None = None,
        backend: str | None = None,
        schema: TableSchema | None = None,
        expires_at: float | None = None,
    ) -> str:
        return self._fit(
            X,
            y,
            task,
            dataset_id,
            backend,
            schema,
            announce=True,
            expires_at=expires_at,
        )

    def _fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        task: Task | None,
        dataset_id: str | None,
        backend: str | None,
        schema: TableSchema | None,
        announce: bool,
        expires_at: float | None = None,
    ) -> str:
        """fit(); `announce` emits residency events (not for fit_predict's
        throwaway contexts, which would flood subscribers)."""
        n_features = _validate_fit_input(X, y)
        task = resolve_task(y, task)
        name = self._shards[0].resolve_backend_name(backend)
        request = FitRequest(len(X), n_features, name, dataset_id)
        if dataset_id is None:
            with timing.phase("place"):
                index = self._place(request)
            resolved = self._fit_on(index, X, y, task, None, name, schema, expires_at)
            if announce:
                self._emit("placed", resolved, index, name)
            return resolved
        with contextlib.ExitStack() as held:
            with timing.phase("wait_dataset_lock"):
                held.enter_context(self._id_locks.hold(dataset_id))
            with timing.phase("place"):
                index = self._place(request)
            resolved = self._fit_on(
                index, X, y, task, dataset_id, name, schema, expires_at
            )
            if announce:
                self._emit("placed", resolved, index, name)
            # Only now that the new context is in place: drop copies on
            # other devices (a failed fit above leaves the old one serving).
            others = [i for i in self._locate(dataset_id) if i != index]
            self._evict_from(dataset_id, others, announce)
            return resolved

    def _evict_from(
        self,
        dataset_id: str,
        indices: list[int],
        announce: bool,
        strict: bool = False,
    ) -> bool:
        """Evict the id from each device at once, reporting each removal
        that happened; True if any device had it. A device that fails (a
        dead worker) is logged and skipped, so the others are still cleaned
        up; `strict` then raises CacheUnavailableError once all were tried."""
        removed = False
        failed: list[str] = []
        for i, had, error in self._on_each(indices, lambda s: s.evict(dataset_id)):
            if error is not None:
                logger.error(
                    "evicting %s from %s failed: %r",
                    dataset_id,
                    self._devices[i],
                    error,
                )
                failed.append(f"{self._devices[i]}: {type(error).__name__}")
                continue
            removed = removed or had
            if announce and had:
                self._emit("removed", dataset_id, i)
        if strict and failed:
            raise CacheUnavailableError(
                f"deleting failed on {', '.join(failed)}; other GPUs may already "
                "be cleared, retry"
            )
        return removed

    def _on_each(
        self, indices: list[int], call: Callable[[Shard], object]
    ) -> list[tuple[int, object, Exception | None]]:
        """call(shard) on each device concurrently: (index, result, error)
        per device, in order. Concurrently, since each device answers from
        its own queue and a slow one mustn't add its wait to the others'."""
        futures = [(i, self._fanout.submit(call, self._shards[i])) for i in indices]
        out = []
        for i, future in futures:
            try:
                out.append((i, future.result(), None))
            except Exception as e:  # noqa: BLE001 -- reported to the caller
                out.append((i, None, e))
        return out

    def _require_all_alive(self) -> list[int]:
        """Every device's index, or CacheUnavailableError if one is dead:
        its spilled copies stay on disk, unlisted and undeletable, and come
        back when the replica restarts it (/readyz and /livez report it
        down, so the replica is restarted)."""
        dead = [
            d for d, s in zip(self._devices, self._shards, strict=True) if not _alive(s)
        ]
        if dead:
            raise CacheUnavailableError(
                f"GPU worker {', '.join(dead)} is down until the replica restarts; "
                "its contexts can't be listed or deleted until then, retry"
            )
        return list(range(len(self._shards)))

    def _emit(
        self, kind: EventKind, dataset_id: str, index: int, backend: str | None = None
    ) -> None:
        self.events.emit(
            ResidencyEvent(kind, dataset_id, self._devices[index], backend)
        )

    def _fit_on(
        self, index, X, y, task, dataset_id, backend, schema, expires_at=None
    ) -> str:
        with self._active(index):
            return self._shards[index].fit(
                X,
                y,
                task=task,
                dataset_id=dataset_id,
                backend=backend,
                schema=schema,
                expires_at=expires_at,
            )

    def predict(
        self,
        dataset_id: str,
        X_test: ArrayLike,
        return_proba: bool = False,
        quantiles: Sequence[float] | None = None,
    ) -> PredictOutcome:
        tried: set[int] = set()
        # Two passes: a re-fit can move the id to another device between
        # the first snapshot and the calls (it lands there, then the old
        # copy is evicted), so a miss everywhere looks once more.
        for _ in range(2):
            for index in self._locate(dataset_id):
                if index in tried:
                    continue
                tried.add(index)
                try:
                    with self._active(index):
                        return self._shards[index].predict(
                            dataset_id,
                            X_test,
                            return_proba=return_proba,
                            quantiles=quantiles,
                        )
                except DatasetNotFoundError:
                    # Moved or evicted between the snapshot and the call.
                    continue
        raise _not_found(dataset_id)

    def schema(self, dataset_id: str) -> TableSchema | None:
        for index in self._locate(dataset_id):
            try:
                return self._shards[index].schema(dataset_id)
            except DatasetNotFoundError:
                continue  # moved or evicted since the snapshot
        raise _not_found(dataset_id)

    def fit_predict(
        self,
        X: ArrayLike,
        y: ArrayLike,
        X_test: ArrayLike,
        task: Task | None = None,
        return_proba: bool = False,
    ) -> PredictOutcome:
        dataset_id = self._fit(X, y, task, None, None, None, announce=False)
        try:
            return self.predict(dataset_id, X_test, return_proba=return_proba)
        finally:
            with self._id_locks.hold(dataset_id):
                self._evict_from(dataset_id, self._locate(dataset_id), announce=False)

    def evict(self, dataset_id: str) -> bool:
        """Remove the id from every device (resident and spilled copies);
        True if any had it. Asks every device, not just those the residency
        snapshots list (they may lag), and raises CacheUnavailableError if
        one can't answer, before deleting anything if a worker is dead: a
        delete must not report success while a copy may survive."""
        devices = self._require_all_alive()
        with self._id_locks.hold(dataset_id):
            return self._evict_from(dataset_id, devices, announce=True, strict=True)

    def contexts(self) -> list[ContextInfo]:
        """Every context on every device, each tagged with its device.
        During a re-fit that moves an id, both copies may be listed.
        CacheUnavailableError if a device is dead or doesn't answer: a
        listing that silently leaves a GPU out would look complete."""
        out: list[ContextInfo] = []
        for i, infos, error in self._on_each(
            self._require_all_alive(), lambda s: s.contexts()
        ):
            if error is not None:
                raise CacheUnavailableError(
                    f"{self._devices[i]} didn't list its contexts "
                    f"({type(error).__name__}); retry"
                ) from error
            out += [dataclasses.replace(c, device=self._devices[i]) for c in infos]
        return out

    def residency(self) -> dict[str, dict[str, list[str]]]:
        """Where every context is now, per device (lock-free snapshots):
        what event subscribers reconcile against, since capacity eviction
        without a spill tier drops contexts without an event."""
        out = {}
        for device, shard in zip(self._devices, self._shards, strict=True):
            resident, known = shard.residency_nowait()
            out[device] = {
                "resident": sorted(resident),
                "spilled": sorted(known - resident),
            }
        return out

    def stats(self) -> EngineStats:
        per_device = self.device_stats()
        return EngineStats(
            n_cached_contexts=sum(s.n_cached_contexts for s in per_device),
            used_bytes=sum(s.used_bytes for s in per_device),
            free_bytes=sum(s.free_bytes for s in per_device),
            capacity_bytes=sum(s.capacity_bytes for s in per_device),
        )

    def stats_nowait(self) -> EngineStats:
        """stats(): already lock-free (every device's counters are)."""
        return self.stats()

    def device_stats(self) -> list[EngineStats]:
        """Lock-free: monitoring never waits on a device's GPU work."""
        return [s.stats_nowait() for s in self._shards]

    def device_activity(self) -> list[dict[str, int]]:
        return [
            {"in_flight": inflight, "recent_calls": recent}
            for inflight, recent in self._activity()
        ]
