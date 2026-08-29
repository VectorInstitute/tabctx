"""Unit tests for the disk spillover tier (cache/spill.py + its
ContextCacheManager integration)."""

import pytest

from tabctx.backends.fake import FakeBackend
from tabctx.cache.manager import CachedContext, ContextCacheManager
from tabctx.cache.spill import DiskSpillStore
from tabctx.engine import TabctxEngine
from tabctx.memory import (
    A100_40GB_TABICL_CALIBRATION,
    AdaptiveMemoryEstimator,
    PowerLawMemoryEstimator,
)


def _ctx(dataset_id: str, est_bytes: int = 100, payload=None) -> CachedContext:
    return CachedContext(
        dataset_id=dataset_id,
        backend_name="fake",
        task="classification",
        n_train=3,
        n_features=2,
        payload=payload if payload is not None else {"data": dataset_id},
        est_bytes=est_bytes,
    )


class TestDiskSpillStore:
    def test_spill_load_roundtrip(self, tmp_path):
        store = DiskSpillStore(tmp_path)
        assert store.spill(_ctx("ds-1", payload={"k": [1, 2, 3]}))
        restored = store.load("ds-1")
        assert restored is not None
        assert restored.payload == {"k": [1, 2, 3]}
        assert restored.est_bytes == 100
        # Load removes from the tier (it's back in the primary cache).
        assert store.load("ds-1") is None

    def test_tenant_scoped_ids_are_filesystem_safe(self, tmp_path):
        store = DiskSpillStore(tmp_path)
        assert store.spill(_ctx("t:acme:my/ds:1"))
        assert store.load("t:acme:my/ds:1").dataset_id == "t:acme:my/ds:1"

    def test_unpicklable_payload_downgrades_to_plain_eviction(self, tmp_path):
        store = DiskSpillStore(tmp_path)
        assert store.spill(_ctx("bad", payload=lambda: None)) is False
        assert list(tmp_path.iterdir()) == []

    def test_capacity_lru(self, tmp_path):
        store = DiskSpillStore(tmp_path, capacity_bytes=100)
        store.spill(_ctx("old", payload=b"x" * 40))
        store.spill(_ctx("newer", payload=b"y" * 40))
        # A third spill must displace the oldest for real.
        store.spill(_ctx("newest", payload=b"z" * 40))
        assert store.load("old") is None
        assert store.load("newest") is not None

    def test_warm_restart_reload(self, tmp_path):
        DiskSpillStore(tmp_path).spill(_ctx("survivor"))
        # A NEW store over the same directory (fresh process) can load it.
        assert DiskSpillStore(tmp_path).load("survivor") is not None


class TestCacheManagerIntegration:
    def test_pressure_eviction_spills_and_get_restores(self, tmp_path):
        spill = DiskSpillStore(tmp_path)
        cache = ContextCacheManager(capacity_bytes=250, spill_store=spill)
        cache.put(_ctx("a", est_bytes=100))
        cache.put(_ctx("b", est_bytes=100))
        evicted = cache.put(_ctx("c", est_bytes=100))
        assert evicted == ["a"]
        assert spill.stats()["n_spilled_contexts"] == 1
        # Transparent restore: get() reloads "a" (displacing LRU "b",
        # which spills in turn).
        restored = cache.get("a")
        assert restored is not None and restored.payload == {"data": "a"}
        assert cache.get("b").payload == {"data": "b"}

    def test_explicit_evict_never_spills(self, tmp_path):
        spill = DiskSpillStore(tmp_path)
        cache = ContextCacheManager(capacity_bytes=1000, spill_store=spill)
        cache.put(_ctx("gone"))
        cache.evict("gone")
        assert cache.get("gone") is None
        assert spill.stats()["n_spilled_contexts"] == 0

    def test_no_spill_store_behaves_as_before(self):
        cache = ContextCacheManager(capacity_bytes=150)
        cache.put(_ctx("a", est_bytes=100))
        cache.put(_ctx("b", est_bytes=100))
        assert cache.get("a") is None

    def test_end_to_end_predictions_survive_eviction(self, tmp_path):
        """The user-visible promise: an evicted-then-restored context
        still predicts, with no re-fit."""
        estimator = AdaptiveMemoryEstimator(
            fallback=PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION)
        )
        backend = FakeBackend(bytes_hint=100)
        cache = ContextCacheManager(
            capacity_bytes=250, spill_store=DiskSpillStore(tmp_path)
        )
        engine = TabctxEngine(backend=backend, cache=cache, estimator=estimator)

        engine.fit([[1.0, 2.0]] * 3, ["hot"] * 3, dataset_id="ds-a")
        engine.fit([[1.0, 2.0]] * 3, ["cold"] * 3, dataset_id="ds-b")
        engine.fit([[1.0, 2.0]] * 3, ["warm"] * 3, dataset_id="ds-c")  # evicts ds-a
        fits_before = backend.fit_calls

        outcome = engine.predict("ds-a", [[9.0, 9.0]])
        assert outcome.predictions == ["hot"]
        assert backend.fit_calls == fits_before, "restore must not re-fit"