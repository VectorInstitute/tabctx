"""Unit tests for eviction policies (cache/policies.py)."""

from tabctx.cache.manager import CachedContext
from tabctx.cache.policies import LRUEvictionPolicy


def _ctx(dataset_id: str, last_accessed_at: float) -> CachedContext:
    ctx = CachedContext(
        dataset_id=dataset_id,
        backend_name="fake",
        task="classification",
        n_train=1,
        n_features=1,
        payload=None,
        est_bytes=1,
    )
    ctx.last_accessed_at = last_accessed_at
    return ctx


def test_select_victim_on_empty_entries_returns_none():
    assert LRUEvictionPolicy().select_victim([]) is None


def test_select_victim_picks_oldest_last_accessed():
    entries = [_ctx("a", 3.0), _ctx("b", 1.0), _ctx("c", 2.0)]
    assert LRUEvictionPolicy().select_victim(entries) == "b"
