"""Unit tests for the disk spillover tier (cache/spill.py + its
ContextCacheManager integration)."""

import threading
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
from tabctx.schema import TableSchema


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

    def test_schema_survives_the_spill_tier(self, tmp_path):
        spill = DiskSpillStore(tmp_path)
        ctx = _ctx("csv")
        ctx.schema = TableSchema((None, ("f", "m")), ("age", "sex"))
        assert spill.spill(ctx)
        assert spill.load("csv").schema == ctx.schema

    def _rewrite_meta(self, spill, dataset_id, edit):
        import json

        meta_path, _ = spill._paths(dataset_id)
        meta = json.loads(meta_path.read_text())
        edit(meta)
        meta_path.write_text(json.dumps(meta))

    def test_meta_from_before_schemas_loads_its_names(self, tmp_path):
        # 0.14 and earlier recorded only a CSV's column names.
        spill = DiskSpillStore(tmp_path)
        assert spill.spill(_ctx("old"))

        def old_format(meta):
            del meta["schema"]
            meta["feature_names"] = ["age", "bmi"]

        self._rewrite_meta(spill, "old", old_format)
        assert spill.load("old").schema == TableSchema((None, None), ("age", "bmi"))

    def test_meta_without_any_schema_loads_as_none(self, tmp_path):
        spill = DiskSpillStore(tmp_path)
        assert spill.spill(_ctx("older"))
        self._rewrite_meta(spill, "older", lambda meta: meta.pop("schema"))
        assert spill.load("older").schema is None

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


def test_warm_restart_indexes_by_dataset_id(tmp_path):
    """Regression: the restart index was keyed by the hex file stem, so
    contains()/delete()/disk-pressure cleanup missed every entry a
    predecessor process spilled."""
    from tabctx.cache.manager import CachedContext
    from tabctx.cache.spill import DiskSpillStore

    first = DiskSpillStore(tmp_path, capacity_bytes=10**6)
    ctx = CachedContext("t:alice:ds-1", "fake", "classification", 3, 2, {"p": 1}, 10)
    assert first.spill(ctx)
    restarted = DiskSpillStore(tmp_path, capacity_bytes=10**6)
    assert restarted.contains("t:alice:ds-1")
    restarted.delete("t:alice:ds-1")
    assert list(tmp_path.iterdir()) == []


class GatedSpillStore(DiskSpillStore):
    """Blocks inside spill()/load() until released, to observe the cache
    while a context is between tiers."""

    def __init__(self, directory) -> None:
        super().__init__(directory)
        self.gate = threading.Event()
        self.entered = threading.Event()

    def spill(self, context):
        self.entered.set()
        assert self.gate.wait(5), "spill gate never released"
        return super().spill(context)

    def load(self, dataset_id):
        self.entered.set()
        assert self.gate.wait(5), "load gate never released"
        return super().load(dataset_id)


class TestInTransitVisibility:
    """A context being spilled or restored is listed by neither tier until
    the I/O finishes; lock-free has() must still report it (a multi-device
    router that didn't would answer 404 for a context that is moving)."""

    def test_has_during_spill(self, tmp_path):
        store = GatedSpillStore(tmp_path)
        cache = ContextCacheManager(1000, spill_store=store)
        cache.put(_ctx("moving"))
        t = threading.Thread(target=cache.evict_one)
        t.start()
        assert store.entered.wait(5)
        assert not cache.holds("moving")
        assert cache.has("moving")
        store.gate.set()
        t.join(5)
        assert cache.has("moving") and store.contains("moving")

    def test_has_during_restore(self, tmp_path):
        store = GatedSpillStore(tmp_path)
        store.gate.set()
        cache = ContextCacheManager(1000, spill_store=store)
        cache.put(_ctx("moving"))
        cache.evict_one()
        store.gate.clear()
        store.entered.clear()
        t = threading.Thread(target=cache.get, args=("moving",))
        t.start()
        assert store.entered.wait(5)
        assert cache.has("moving")
        store.gate.set()
        t.join(5)
        assert cache.holds("moving")

    def test_failed_spill_is_a_real_miss_afterwards(self, tmp_path):
        cache = ContextCacheManager(1000, spill_store=DiskSpillStore(tmp_path))
        cache.put(_ctx("bad", payload=lambda: None))
        cache.evict_one()
        assert not cache.has("bad")


def test_put_keeps_a_replaced_id_visible_while_making_room(tmp_path):
    """A same-device re-fit removed the old entry, then spilled others to
    make room, then inserted: has() said 'nowhere' in between."""
    store = GatedSpillStore(tmp_path)
    store.gate.set()
    cache = ContextCacheManager(200, spill_store=store)
    cache.put(_ctx("refit", est_bytes=100))
    cache.put(_ctx("other", est_bytes=100))
    store.gate.clear()
    store.entered.clear()
    t = threading.Thread(target=cache.put, args=(_ctx("refit", est_bytes=200),))
    t.start()
    assert store.entered.wait(5)  # spilling "other" to make room
    assert cache.has("refit")
    store.gate.set()
    t.join(5)
    assert cache.holds("refit")
