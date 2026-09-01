"""Multi-tenant, evictable cache of encoded training contexts.

This is the core thing tabctx exists to add: TabICL and TabPFN both cache
the encoded training context inside a single fitted estimator object
(fit-once, predict-many), but that cache lives in exactly one process with
no notion of capacity, eviction, or sharing across concurrent training sets.
ContextCacheManager makes that a real multi-tenant resource -- the direct
analog of what PagedAttention did for per-request LLM KV caches, scoped here
to per-training-set tabular ICL contexts.

v1 scaling limit (see tabctx repo README/roadmap): one RLock guards the
entire cache, and the engine holds it across backend calls too, so all GPU
work on a replica is effectively serialized. That's the right tradeoff on a
single GPU under real memory pressure -- two concurrent fits/predicts against
the same device would just contend for the same memory anyway -- but it caps
throughput at one in-flight backend call per replica until a future version
does finer-grained locking or real multi-replica routing.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from tabctx.cache.policies import EvictionPolicy, LRUEvictionPolicy
from tabctx.errors import CacheCapacityError
from tabctx.types import EngineStats, Task


@dataclass
class CachedContext:
    dataset_id: str
    backend_name: str
    task: Task
    n_train: int
    n_features: int
    payload: Any
    est_bytes: int
    # Training-table column names when the context was fit from a CSV
    # (serve/csv_io.py); None for inline tables. Travels with the
    # context through the spill tier so a restored context keeps its
    # schema check.
    feature_names: list[str] | None = None
    created_at: float = field(default_factory=time.monotonic)
    last_accessed_at: float = field(default_factory=time.monotonic)


class ContextCacheManager:
    def __init__(
        self,
        capacity_bytes: int,
        policy: EvictionPolicy | None = None,
        spill_store: Any | None = None,
    ) -> None:
        """spill_store: optional disk tier (cache/spill.py). When set,
        capacity-pressure evictions spill instead of dropping, and get()
        transparently reloads spilled contexts. Explicit evict() (caller
        intent, re-fit overwrite) never spills -- and also drops any
        spilled copy, so an evicted context can't resurrect."""
        self._capacity_bytes = capacity_bytes
        self._policy = policy or LRUEvictionPolicy()
        self._entries: dict[str, CachedContext] = {}
        # Maintained incrementally so stats()/free_bytes stay O(1) --
        # make_room() polls free_bytes once per eviction.
        self._used_bytes = 0
        self._spill = spill_store
        self._lock = threading.RLock()

    def get(self, dataset_id: str) -> CachedContext | None:
        with self._lock:
            entry = self._entries.get(dataset_id)
            if entry is not None or self._spill is None:
                return entry
            # Miss with a spill tier: try reloading. Re-admitting evicts
            # via the normal policy to make room (RLock, so the nested
            # put() is fine); what it displaces may itself spill.
            restored = self._spill.load(dataset_id)
            if restored is None:
                return None
            restored.last_accessed_at = time.monotonic()
            try:
                self.put(restored)
            except CacheCapacityError:
                # Spilled under a larger budget than this process has
                # (e.g. TABCTX_GPU_MEMORY_FRACTION lowered across a
                # restart). Nothing can re-admit it; treat as a miss so
                # the caller re-fits instead of the request 500ing.
                return None
            return restored

    def touch(self, dataset_id: str) -> None:
        with self._lock:
            entry = self._entries.get(dataset_id)
            if entry is not None:
                entry.last_accessed_at = time.monotonic()

    def _remove(self, dataset_id: str) -> CachedContext | None:
        """Drop an entry from the primary tier (no spill). Caller holds
        the lock."""
        entry = self._entries.pop(dataset_id, None)
        if entry is not None:
            self._used_bytes -= entry.est_bytes
        return entry

    def _evict_victim(self) -> str | None:
        """Policy-chosen eviction, spilling when a tier is attached.
        Caller holds the lock."""
        victim_id = self._policy.select_victim(list(self._entries.values()))
        if victim_id is None:
            return None
        victim = self._remove(victim_id)
        if self._spill is not None:
            self._spill.spill(victim)  # best-effort by contract
        return victim_id

    def make_room(self, needed_bytes: int) -> list[str]:
        """Evict entries (via the configured policy) until at least
        needed_bytes is free, or there's nothing left to evict. Does not
        raise -- callers must check free_bytes afterward if they need a
        guarantee (put() does this for you)."""
        evicted: list[str] = []
        with self._lock:
            while self.free_bytes < needed_bytes:
                victim_id = self._evict_victim()
                if victim_id is None:
                    break
                evicted.append(victim_id)
        return evicted

    def evict_one(self) -> str | None:
        """Evict a single victim chosen by the policy (spilling it when a
        spill tier is attached). Returns the evicted dataset_id, or None
        if the cache is empty. Used by the engine's evict-ahead-of-fit
        path to convert resident bytes into transient fit headroom."""
        with self._lock:
            return self._evict_victim()

    def put(self, context: CachedContext) -> list[str]:
        """Insert a context, evicting via the policy to make room first.
        Returns the list of evicted dataset_ids. Raises CacheCapacityError
        if context.est_bytes alone exceeds total capacity -- no amount of
        eviction would ever make room for it. Overwriting an existing
        dataset_id replaces it in both tiers (a stale spilled copy must
        never outlive the context that superseded it)."""
        if context.est_bytes > self._capacity_bytes:
            raise CacheCapacityError(
                f"context for {context.dataset_id!r} needs "
                f"{context.est_bytes} bytes but cache capacity is only "
                f"{self._capacity_bytes} bytes"
            )
        with self._lock:
            self._remove(context.dataset_id)
            if self._spill is not None:
                self._spill.delete(context.dataset_id)
            evicted = self.make_room(context.est_bytes)
            self._entries[context.dataset_id] = context
            self._used_bytes += context.est_bytes
            return evicted

    def evict(self, dataset_id: str) -> None:
        """Drop a context from every tier. Never spills: this is caller
        intent (fit_predict's one-shot cleanup, an explicit delete), and
        a spilled copy left behind would silently resurrect on the next
        get()."""
        with self._lock:
            self._remove(dataset_id)
            if self._spill is not None:
                self._spill.delete(dataset_id)

    def dataset_ids(self) -> list[str]:
        """Ids currently resident in the primary tier (not spilled)."""
        with self._lock:
            return list(self._entries)

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return self._used_bytes

    @property
    def free_bytes(self) -> int:
        return max(0, self._capacity_bytes - self.used_bytes)

    def stats(self) -> EngineStats:
        with self._lock:
            return EngineStats(
                n_cached_contexts=len(self._entries),
                used_bytes=self._used_bytes,
                free_bytes=self.free_bytes,
                capacity_bytes=self._capacity_bytes,
            )

    @property
    def lock(self) -> threading.RLock:
        """Exposed so the engine can hold one lock across the full
        check-evict-fit-insert sequence (see module docstring) rather than
        risking two requests both passing admission control against the
        same free_bytes snapshot and then both calling into the backend."""
        return self._lock
