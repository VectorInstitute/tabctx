"""MultiDeviceEngine: one replica serving from several GPUs.

Neither TabICL nor TabPFN can split one training context across GPUs
(TabICL is single-device; TabPFN's device lists only parallelize ensemble
members, and not in its kv-cache fit mode). So a multi-GPU replica runs
one complete TabctxEngine per device -- its own backend instances bound
to that device, its own cache, lock and spill tier -- and this class
routes between them:

- fit() places each context with a rule aimed at the usual workload --
  a few concurrent users, each fitting a table (small or large) and then
  predicting against it repeatedly:
    1. Devices where the table's estimated peak fits WITHOUT evicting
       anything are preferred; among them, the least active one wins
       (fewest calls in flight, then fewest started recently), ties to
       the most headroom. Each device serializes its own GPU work, so
       spreading active users across devices is what lets them run
       concurrently.
    2. If no device can take it without eviction (a large table, or a
       full replica), it goes to the device with the most headroom,
       evicting the fewest contexts.
  Each device's own usage-aware admission gate then has the final say,
  exactly as on a single GPU. A table's ceiling is therefore one
  device's memory; what more devices buy is N x the resident cache and
  concurrency.
- predict() runs on the device that holds the context, found from each
  device's residency and spill index (so a context spilled before a
  restart is found on, and restored to, its own device).

Concurrency. Each device's engine holds its cache lock across its whole
GPU call, so this router never takes another device's lock to make a
decision: placement and lookup read lock-free snapshots (used bytes,
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

Same public surface as TabctxEngine, so the serving layer and batching
use either interchangeably.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator

from tabctx.engine import TabctxEngine, _validate_fit_input
from tabctx.errors import DatasetNotFoundError
from tabctx.memory.estimator import MemoryEstimator
from tabctx.types import ArrayLike, EngineStats, PredictOutcome, Task

# How long a started call counts toward a device's recent activity.
ACTIVITY_WINDOW_S = 30.0


class _KeyedLocks:
    """One lock per key, created on demand and dropped when unused."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locks: dict[str, list] = {}  # key -> [lock, users]

    @contextlib.contextmanager
    def hold(self, key: str) -> Iterator[None]:
        with self._guard:
            entry = self._locks.setdefault(key, [threading.Lock(), 0])
            entry[1] += 1
        try:
            with entry[0]:
                yield
        finally:
            with self._guard:
                entry[1] -= 1
                if entry[1] == 0:
                    del self._locks[key]


def _not_found(dataset_id: str) -> DatasetNotFoundError:
    return DatasetNotFoundError(
        f"no cached context for dataset_id={dataset_id!r} -- it was never "
        "fit, was evicted, or was lost to a replica restart"
    )


class MultiDeviceEngine:
    def __init__(
        self,
        shards: list[TabctxEngine],
        devices: list[str],
        activity_window_s: float = ACTIVITY_WINDOW_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if len(shards) < 2 or len(shards) != len(devices):
            raise ValueError("need one engine per device, and at least two devices")
        names = {tuple(s.backend_names) for s in shards}
        defaults = {s.default_backend for s in shards}
        if len(names) != 1 or len(defaults) != 1:
            raise ValueError("every device must serve the same models")
        self._shards = shards
        self._devices = devices
        self._id_locks = _KeyedLocks()
        self._window_s = activity_window_s
        self._clock = clock
        self._inflight = [0] * len(shards)
        self._recent: list[deque[float]] = [deque() for _ in shards]
        self._activity_lock = threading.Lock()

    @property
    def devices(self) -> list[str]:
        return list(self._devices)

    @property
    def backend_names(self) -> list[str]:
        return self._shards[0].backend_names

    @property
    def default_backend(self) -> str:
        return self._shards[0].default_backend

    def estimator_for(self, backend: str | None = None) -> MemoryEstimator:
        return self._shards[0].estimator_for(backend)

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

    def _place(self, estimated_bytes: int, backend: str) -> int:
        headrooms = self.device_headroom_bytes(backend)
        activity = self._activity()
        indices = range(len(self._shards))
        fits = [i for i in indices if headrooms[i] >= estimated_bytes]
        if fits:
            return min(fits, key=lambda i: (activity[i], -headrooms[i], i))
        return max(indices, key=lambda i: (headrooms[i], -i))

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
        task: Task = "classification",
        dataset_id: str | None = None,
        backend: str | None = None,
        feature_names: list[str] | None = None,
    ) -> str:
        n_features = _validate_fit_input(X, y)
        name = self._shards[0].resolve_backend_name(backend)
        estimated = self.estimator_for(name).estimate_bytes(len(X), 0, n_features)
        if dataset_id is None:
            return self._fit_on(
                self._place(estimated, name), X, y, task, None, name, feature_names
            )
        with self._id_locks.hold(dataset_id):
            index = self._place(estimated, name)
            resolved = self._fit_on(index, X, y, task, dataset_id, name, feature_names)
            # Only now that the new context is in place: drop copies on
            # other devices (a failed fit above leaves the old one serving).
            for i in self._locate(dataset_id):
                if i != index:
                    self._shards[i].evict(dataset_id)
            return resolved

    def _fit_on(self, index, X, y, task, dataset_id, backend, feature_names) -> str:
        with self._active(index):
            return self._shards[index].fit(
                X,
                y,
                task=task,
                dataset_id=dataset_id,
                backend=backend,
                feature_names=feature_names,
            )

    def predict(
        self, dataset_id: str, X_test: ArrayLike, return_proba: bool = False
    ) -> PredictOutcome:
        for index in self._locate(dataset_id):
            try:
                with self._active(index):
                    return self._shards[index].predict(
                        dataset_id, X_test, return_proba=return_proba
                    )
            except DatasetNotFoundError:
                # Moved or evicted between the snapshot and the call.
                continue
        raise _not_found(dataset_id)

    def feature_names(self, dataset_id: str) -> list[str] | None:
        for index in self._locate(dataset_id):
            names = self._shards[index].feature_names(dataset_id)
            if names is not None:
                return names
        return None

    def fit_predict(
        self,
        X: ArrayLike,
        y: ArrayLike,
        X_test: ArrayLike,
        task: Task = "classification",
        return_proba: bool = False,
    ) -> PredictOutcome:
        dataset_id = self.fit(X, y, task)
        try:
            return self.predict(dataset_id, X_test, return_proba=return_proba)
        finally:
            self.evict(dataset_id)

    def evict(self, dataset_id: str) -> None:
        with self._id_locks.hold(dataset_id):
            for index in self._locate(dataset_id):
                self._shards[index].evict(dataset_id)

    def stats(self) -> EngineStats:
        per_device = self.device_stats()
        return EngineStats(
            n_cached_contexts=sum(s.n_cached_contexts for s in per_device),
            used_bytes=sum(s.used_bytes for s in per_device),
            free_bytes=sum(s.free_bytes for s in per_device),
            capacity_bytes=sum(s.capacity_bytes for s in per_device),
        )

    def device_stats(self) -> list[EngineStats]:
        """Lock-free: monitoring never waits on a device's GPU work."""
        return [s.stats_nowait() for s in self._shards]

    def device_activity(self) -> list[dict[str, int]]:
        return [
            {"in_flight": inflight, "recent_calls": recent}
            for inflight, recent in self._activity()
        ]
