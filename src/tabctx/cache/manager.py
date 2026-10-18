"""Multi-tenant, evictable cache of encoded training contexts.

This is the core thing tabctx exists to add: TabICL and TabPFN both cache
the encoded training context inside a single fitted estimator object
(fit-once, predict-many), but that cache lives in exactly one process with
no notion of capacity, eviction, or sharing across concurrent training sets.
ContextCacheManager makes that a real multi-tenant resource -- the direct
analog of what PagedAttention did for per-request LLM KV caches, scoped here
to per-training-set tabular ICL contexts.

Locking. The cache's lock guards only its bookkeeping (entries, usage,
pins, ids in transit); spill reads and writes run outside it. Scheduling
GPU work -- one fit at a time per device, light predicts beside it -- is
the engine's job (engine.py), so the cache never waits on a backend call.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from tabctx.cache.policies import EvictionPolicy, LRUEvictionPolicy
from tabctx.errors import CacheCapacityError
from tabctx.types import EngineStats, Task

logger = logging.getLogger(__name__)


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
        # Ids moving between tiers (being spilled, restored, or replaced by a
        # re-fit), which neither tier lists until the move finishes; counted,
        # since moves can nest (a restore's put). Lock-free readers (has)
        # must still see them: a multi-device router that saw neither would
        # answer 404 for a context that is merely moving. Spill I/O runs
        # outside the lock, so bookkeeping (and light predicts) never wait
        # on a multi-GB disk write.
        self._in_transit: dict[str, int] = {}
        self._settled = threading.Condition(self._lock)
        # Contexts a call is using right now (see pinned()): never chosen as
        # policy eviction victims, so a fit's evict-ahead can't spill a
        # context out from under a predict running beside it -- the predict
        # would keep the GPU memory alive while the books count it free.
        self._pins: dict[str, int] = {}

    # ---- transit bookkeeping (caller holds the lock) -----------------

    def _transit_add(self, dataset_id: str) -> None:
        self._in_transit[dataset_id] = self._in_transit.get(dataset_id, 0) + 1

    def _transit_discard(self, dataset_id: str) -> None:
        left = self._in_transit.get(dataset_id, 0) - 1
        if left > 0:
            self._in_transit[dataset_id] = left
        else:
            self._in_transit.pop(dataset_id, None)
            self._settled.notify_all()

    def wait_settled(self, dataset_id: str, timeout_s: float = 600.0) -> bool:
        """Wait until an id is no longer moving between tiers (a spill
        write, restore or replacement in progress). False on timeout."""
        with self._settled:
            return self._settled.wait_for(
                lambda: dataset_id not in self._in_transit, timeout_s
            )

    def get(self, dataset_id: str) -> CachedContext | None:
        """The resident context, else a spilled one reloaded and re-admitted
        (its displacements may spill in turn). The reload runs outside the
        lock; callers serialize restores of one device (the engine does so
        under its fit lock)."""
        with self._lock:
            entry = self._entries.get(dataset_id)
            if entry is not None or self._spill is None:
                return entry
            self._transit_add(dataset_id)
        try:
            restored = self._spill.load(dataset_id)
            if restored is None:
                return None
            restored.last_accessed_at = time.monotonic()
            try:
                self.put(restored)
            except CacheCapacityError:
                # Spilled under a larger budget than this process has (e.g.
                # TABCTX_GPU_MEMORY_FRACTION lowered across a restart).
                # Nothing can re-admit it; treat as a miss so the caller
                # re-fits instead of the request 500ing.
                return None
            return restored
        finally:
            with self._lock:
                self._transit_discard(dataset_id)

    def spilled_size(self, dataset_id: str) -> int | None:
        """A spilled context's est_bytes, without loading it (None if not
        spilled): what restoring it will occupy."""
        if self._spill is None:
            return None
        return self._spill.est_bytes(dataset_id)

    @contextlib.contextmanager
    def pinned(self, dataset_id: str) -> Iterator[None]:
        """Keep a context out of policy eviction's reach for the duration
        (explicit evict(), re-fits and evict_one(ignore_pins=True) still
        take it: caller intent)."""
        with self._lock:
            self._pins[dataset_id] = self._pins.get(dataset_id, 0) + 1
        try:
            yield
        finally:
            with self._lock:
                left = self._pins[dataset_id] - 1
                if left:
                    self._pins[dataset_id] = left
                else:
                    del self._pins[dataset_id]

    def has_pins(self) -> bool:
        return bool(self._pins)

    def peek(self, dataset_id: str) -> CachedContext | None:
        """The resident context, or None -- never reloads a spilled one (a
        restore needs device memory, which only an exclusive call may take)."""
        with self._lock:
            return self._entries.get(dataset_id)

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

    def evict_one(self, ignore_pins: bool = False) -> str | None:
        """Evict a single victim chosen by the policy (spilling it when a
        spill tier is attached). Returns the evicted dataset_id, or None if
        nothing is evictable. Used by the engine's evict-ahead-of-fit path
        to convert resident bytes into transient fit headroom; ignore_pins
        is for callers that have stopped every predict first."""
        with self._lock:
            candidates = [
                e
                for e in self._entries.values()
                if ignore_pins or e.dataset_id not in self._pins
            ]
            victim_id = self._policy.select_victim(candidates)
            if victim_id is None:
                return None
            victim = self._remove(victim_id)
            if self._spill is None:
                return victim_id
            self._transit_add(victim_id)
        try:
            self._spill.spill(victim)  # best-effort by contract; no lock held
        finally:
            with self._lock:
                self._transit_discard(victim_id)
        return victim_id

    def make_room(self, needed_bytes: int) -> list[str]:
        """Evict entries (via the configured policy) until at least
        needed_bytes is free, or there's nothing left to evict. Does not
        raise -- callers must check free_bytes afterward if they need a
        guarantee."""
        evicted: list[str] = []
        while self.free_bytes < needed_bytes:
            victim_id = self.evict_one()
            if victim_id is None:
                break
            evicted.append(victim_id)
        return evicted

    def put(self, context: CachedContext) -> list[str]:
        """Insert a context, evicting via the policy to make room first.
        Returns the list of evicted dataset_ids. Raises CacheCapacityError
        if context.est_bytes alone exceeds total capacity -- no amount of
        eviction would ever make room for it. Overwriting an existing
        dataset_id replaces it in both tiers (a stale spilled copy must
        never outlive the context that superseded it); the id stays
        visible to has() throughout."""
        if context.est_bytes > self._capacity_bytes:
            raise CacheCapacityError(
                f"context for {context.dataset_id!r} needs "
                f"{context.est_bytes} bytes but cache capacity is only "
                f"{self._capacity_bytes} bytes"
            )
        with self._lock:
            self._transit_add(context.dataset_id)
            self._remove(context.dataset_id)
        try:
            if self._spill is not None:
                self._spill.delete(context.dataset_id)
            evicted = self.make_room(context.est_bytes)
            with self._lock:
                self._entries[context.dataset_id] = context
                self._used_bytes += context.est_bytes
                if self._used_bytes > self._capacity_bytes:
                    # Only pinned contexts were left to evict: the books
                    # stay truthful (the memory is in use), but say so.
                    logger.warning(
                        "cache over capacity after inserting %r: %d of %d bytes "
                        "(contexts in use could not be evicted)",
                        context.dataset_id,
                        self._used_bytes,
                        self._capacity_bytes,
                    )
            return evicted
        finally:
            with self._lock:
                self._transit_discard(context.dataset_id)

    def evict(self, dataset_id: str) -> CachedContext | None:
        """Drop a context from every tier and return the resident one (the
        caller decides when its memory may be released). Never spills: this
        is caller intent (fit_predict's one-shot cleanup, an explicit
        delete), and a spilled copy left behind would silently resurrect on
        the next get()."""
        with self._lock:
            removed = self._remove(dataset_id)
        if self._spill is not None:
            self._spill.delete(dataset_id)
        return removed

    # ---- lock-free snapshots ---------------------------------------
    # Readers that only need bookkeeping (a multi-device router choosing a
    # device, /readyz) never take the lock. Single int / dict-membership
    # reads are atomic under the GIL; these are snapshots, and every
    # mutation stays locked.

    def holds(self, dataset_id: str) -> bool:
        """Resident in the primary tier (snapshot, lock-free)."""
        return dataset_id in self._entries

    def has(self, dataset_id: str) -> bool:
        """Resident, spilled, or moving between the two (snapshot,
        lock-free). An id in transit is reported held: the caller's
        locked get() waits for the move and finds it (or, if the spill
        failed, misses then)."""
        return (
            dataset_id in self._entries
            or dataset_id in self._in_transit
            or (self._spill is not None and self._spill.contains(dataset_id))
        )

    def residency_nowait(self) -> tuple[frozenset[str], frozenset[str]]:
        """(resident ids, every id has() reports: resident, in transit or
        spilled) as lock-free snapshots -- what a router in another
        process mirrors to answer holds()/has() without a round trip."""
        resident = frozenset(list(self._entries))
        known = resident | frozenset(list(self._in_transit))
        if self._spill is not None:
            known |= self._spill.ids()
        return resident, known

    def stats_nowait(self) -> EngineStats:
        """stats() without the lock: fields may be from slightly different
        instants, which is fine for placement and monitoring."""
        used = self._used_bytes
        return EngineStats(
            n_cached_contexts=len(self._entries),
            used_bytes=used,
            free_bytes=max(0, self._capacity_bytes - used),
            capacity_bytes=self._capacity_bytes,
        )

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
