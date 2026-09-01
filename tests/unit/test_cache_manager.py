import pytest

from tabctx.cache.manager import CachedContext, ContextCacheManager
from tabctx.errors import CacheCapacityError


def make_context(
    dataset_id: str, est_bytes: int, last_accessed_at: float | None = None
):
    ctx = CachedContext(
        dataset_id=dataset_id,
        backend_name="fake",
        task="classification",
        n_train=10,
        n_features=2,
        payload=object(),
        est_bytes=est_bytes,
    )
    if last_accessed_at is not None:
        ctx.last_accessed_at = last_accessed_at
    return ctx


def test_put_and_get_roundtrip():
    cache = ContextCacheManager(capacity_bytes=1000)
    ctx = make_context("a", 100)
    cache.put(ctx)
    assert cache.get("a") is ctx
    assert cache.get("missing") is None


def test_stats_report_usage():
    cache = ContextCacheManager(capacity_bytes=1000)
    cache.put(make_context("a", 100))
    cache.put(make_context("b", 200))
    stats = cache.stats()
    assert stats.n_cached_contexts == 2
    assert stats.used_bytes == 300
    assert stats.free_bytes == 700
    assert stats.capacity_bytes == 1000


def test_put_evicts_lru_to_make_room():
    cache = ContextCacheManager(capacity_bytes=250)
    cache.put(make_context("old", 100, last_accessed_at=1.0))
    cache.put(make_context("newer", 100, last_accessed_at=2.0))
    # Inserting a 100-byte context needs 300 total; only "old" needs evicting
    # to free enough room (100 -> 200 used, +100 new = 300 > 250, so evict
    # the least-recently-accessed entry, "old").
    evicted = cache.put(make_context("newest", 100, last_accessed_at=3.0))
    assert evicted == ["old"]
    assert cache.get("old") is None
    assert cache.get("newer") is not None
    assert cache.get("newest") is not None


def test_put_raises_when_context_alone_exceeds_capacity():
    cache = ContextCacheManager(capacity_bytes=100)
    with pytest.raises(CacheCapacityError):
        cache.put(make_context("too_big", 200))
    # Rejected insert must not partially land in the cache.
    assert cache.get("too_big") is None


def test_touch_updates_recency_and_changes_eviction_order():
    cache = ContextCacheManager(capacity_bytes=200)
    cache.put(make_context("a", 100, last_accessed_at=1.0))
    cache.put(make_context("b", 100, last_accessed_at=2.0))
    # Without a touch, "a" (last_accessed_at=1.0) is older and would be
    # evicted first. Touching "a" now makes it the most-recently-accessed,
    # so "b" becomes the eviction victim instead.
    cache.touch("a")
    evicted = cache.make_room(100)
    assert evicted == ["b"]


def test_evict_removes_entry():
    cache = ContextCacheManager(capacity_bytes=1000)
    cache.put(make_context("a", 100))
    cache.evict("a")
    assert cache.get("a") is None
    assert cache.stats().used_bytes == 0


def test_evict_missing_id_is_a_noop():
    cache = ContextCacheManager(capacity_bytes=1000)
    cache.evict("does-not-exist")  # must not raise


def test_make_room_stops_when_nothing_left_to_evict():
    # Calling make_room directly (bypassing put()'s own-size guard) with a
    # target larger than total capacity must stop cleanly once the cache is
    # empty, rather than looping forever.
    cache = ContextCacheManager(capacity_bytes=100)
    cache.put(make_context("a", 100))
    evicted = cache.make_room(1000)
    assert evicted == ["a"]
    assert cache.stats().n_cached_contexts == 0


def test_evict_one_on_empty_cache_returns_none():
    cache = ContextCacheManager(capacity_bytes=1000)
    assert cache.evict_one() is None


def test_evict_one_evicts_lru_victim():
    cache = ContextCacheManager(capacity_bytes=1000)
    cache.put(make_context("old", 100, last_accessed_at=1.0))
    cache.put(make_context("new", 100, last_accessed_at=2.0))
    victim = cache.evict_one()
    assert victim == "old"
    assert cache.get("old") is None
    assert cache.get("new") is not None


def test_evict_one_spills_when_a_spill_tier_is_attached(tmp_path):
    from tabctx.cache.spill import DiskSpillStore

    spill = DiskSpillStore(tmp_path)
    cache = ContextCacheManager(capacity_bytes=1000, spill_store=spill)
    cache.put(make_context("a", 100))
    assert cache.evict_one() == "a"
    assert spill.stats()["n_spilled_contexts"] == 1


def test_used_bytes_tracks_put_evict_and_overwrite():
    cache = ContextCacheManager(capacity_bytes=1000)
    cache.put(make_context("a", 100))
    cache.put(make_context("b", 200))
    assert cache.used_bytes == 300
    # Overwriting an id replaces its bytes rather than double-counting.
    cache.put(make_context("a", 150))
    assert cache.used_bytes == 350
    cache.evict("b")
    assert cache.used_bytes == 150
    cache.evict("b")  # repeated evict of a gone id changes nothing
    assert cache.used_bytes == 150
    assert cache.dataset_ids() == ["a"]


def test_put_overwrite_does_not_evict_others_needlessly():
    # Replacing "a" (100 -> 100) in a full cache must not evict "b":
    # the old "a" is released before room is measured.
    cache = ContextCacheManager(capacity_bytes=200)
    cache.put(make_context("a", 100, last_accessed_at=1.0))
    cache.put(make_context("b", 100, last_accessed_at=2.0))
    assert cache.put(make_context("a", 100, last_accessed_at=3.0)) == []
    assert cache.get("b") is not None
