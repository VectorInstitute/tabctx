"""Unit tests for the disk spillover tier (cache/spill.py + its
ContextCacheManager integration)."""

from pathlib import Path

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

    def test_write_failure_downgrades_to_plain_eviction(self, tmp_path, monkeypatch):
        store = DiskSpillStore(tmp_path)

        def boom(self, data):
            raise OSError("disk full")

        monkeypatch.setattr(Path, "write_bytes", boom)
        assert store.spill(_ctx("x")) is False
        # No partial files left behind.
        assert list(tmp_path.iterdir()) == []

    def test_make_room_gives_up_when_disk_has_untracked_files(self, tmp_path):
        # A payload file present on disk but absent from the in-memory
        # index (e.g. left by a process that crashed mid-spill) must not
        # make _make_room loop forever -- it gives up rather than evicting
        # something it has no record of.
        store = DiskSpillStore(tmp_path, capacity_bytes=10)
        (tmp_path / "stray.payload").write_bytes(b"x" * 100)
        assert store._index == {}
        # spill() still succeeds (capacity is soft, not enforced on write).
        assert store.spill(_ctx("y", payload=b"z" * 5))
        assert store.load("y") is not None


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

    def test_explicit_evict_also_drops_a_spilled_copy(self, tmp_path):
        # Regression: a context spilled under pressure, then explicitly
        # evicted, used to resurrect from disk on the next get().
        spill = DiskSpillStore(tmp_path)
        cache = ContextCacheManager(capacity_bytes=100, spill_store=spill)
        cache.put(_ctx("a", est_bytes=100))
        cache.put(_ctx("b", est_bytes=100))  # spills "a"
        assert spill.stats()["n_spilled_contexts"] == 1
        cache.evict("a")
        assert spill.stats()["n_spilled_contexts"] == 0
        assert cache.get("a") is None

    def test_refit_overwrite_drops_the_stale_spilled_copy(self, tmp_path):
        # "a" v1 spills; the caller re-fits "a" (v2). If v2 is later
        # explicitly evicted, v1 must not come back from disk.
        spill = DiskSpillStore(tmp_path)
        cache = ContextCacheManager(capacity_bytes=100, spill_store=spill)
        cache.put(_ctx("a", est_bytes=100, payload={"v": 1}))
        cache.put(_ctx("b", est_bytes=100))  # spills a-v1
        cache.put(_ctx("a", est_bytes=100, payload={"v": 2}))  # spills b
        assert spill.stats()["n_spilled_contexts"] == 1  # just "b"
        cache.evict("a")
        assert cache.get("a") is None

    def test_restore_that_no_longer_fits_capacity_is_a_miss(self, tmp_path):
        # Spilled under a larger budget (e.g. GPU fraction lowered across
        # a restart): must read as a clean miss, not a CacheCapacityError.
        DiskSpillStore(tmp_path).spill(_ctx("big", est_bytes=500))
        spill = DiskSpillStore(tmp_path)
        cache = ContextCacheManager(capacity_bytes=100, spill_store=spill)
        assert cache.get("big") is None

    def test_feature_names_survive_the_spill_tier(self, tmp_path):
        spill = DiskSpillStore(tmp_path)
        ctx = _ctx("csv")
        ctx.feature_names = ["age", "bmi"]
        assert spill.spill(ctx)
        assert spill.load("csv").feature_names == ["age", "bmi"]

    def test_meta_without_feature_names_loads_as_none(self, tmp_path):
        # Files written by a version before feature_names existed.
        import json

        spill = DiskSpillStore(tmp_path)
        assert spill.spill(_ctx("old"))
        meta_path, _ = spill._paths("old")
        meta = json.loads(meta_path.read_text())
        del meta["feature_names"]
        meta_path.write_text(json.dumps(meta))
        assert spill.load("old").feature_names is None

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
