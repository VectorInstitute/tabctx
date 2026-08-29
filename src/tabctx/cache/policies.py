"""Eviction policies for ContextCacheManager.

v1 ships LRU only. Kept as a Protocol so a future policy (e.g. informed by
"Bounded Context Management for Tabular Foundation Models on Stream
Learning", arXiv 2606.18677) can be dropped in without touching the manager.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from tabctx.cache.manager import CachedContext


class EvictionPolicy(Protocol):
    def select_victim(self, entries: list[CachedContext]) -> str | None:
        """Return the dataset_id to evict next, or None if entries is empty."""
        ...


class LRUEvictionPolicy:
    """Evicts the least-recently-accessed context."""

    def select_victim(self, entries: list[CachedContext]) -> str | None:
        if not entries:
            return None
        oldest = min(entries, key=lambda c: c.last_accessed_at)
        return oldest.dataset_id
