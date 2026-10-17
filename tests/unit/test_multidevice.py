"""Unit tests for MultiDeviceEngine: one replica, several GPUs.

Placement is tested against a deterministic estimator (bytes = cells x
BYTES_PER_CELL) and a fake backend whose resident size follows the same
rule, so every headroom number in these tests is exact.
"""

import threading
import time

import pytest

from tabctx.backends.fake import FakeBackend
from tabctx.cache.manager import ContextCacheManager
from tabctx.cache.spill import DiskSpillStore
from tabctx.engine import TabctxEngine
from tabctx.errors import AdmissionRejected, DatasetNotFoundError, InvalidInputError
from tabctx.multidevice import MultiDeviceEngine

BYTES_PER_CELL = 10
CAPACITY = 10_000  # per device: 1,000 cells resident or in flight


class CellEstimator:
    """Peak and resident cost are both cells x BYTES_PER_CELL; headroom is
    whatever the device's cache doesn't hold."""

    def __init__(self, capacity: int = CAPACITY) -> None:
        self.capacity = capacity
        self.observations = 0

    def estimate_bytes(self, n_train: int, n_test: int, n_features: int) -> int:
        return (n_train + n_test) * n_features * BYTES_PER_CELL

    def admit(self, n_train: int, n_test: int, n_features: int) -> bool:
        return self.estimate_bytes(n_train, n_test, n_features) <= self.capacity

    def ceiling_bytes(self) -> int:
        return self.capacity

    def admission_headroom_bytes(self, used_bytes: int) -> int:
        return max(0, self.capacity - used_bytes)

    def confidence(self) -> str:
        return "test"

    def record_observation(
        self, n_train: int, n_features: int, real_bytes: int
    ) -> None:
        self.observations += 1


class SizedFake(FakeBackend):
    """Resident size = cells x BYTES_PER_CELL, like the estimator."""

    def __init__(self, device: str, **kw) -> None:
        super().__init__(**kw)
        self.device = device
        self._last_bytes = None

    def fit(self, X, y, task):
        self._last_bytes = len(X) * len(X[0]) * BYTES_PER_CELL
        return super().fit(X, y, task)

    def context_bytes_hint(self, n_train, n_features):
        return self._last_bytes


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def table(rows: int, features: int = 1, label: str = "a"):
    return [[float(i)] * features for i in range(rows)], [label] * rows


def build(n_devices=2, spill_root=None, clock=None, **fake_kw):
    devices = [f"cuda:{i}" for i in range(n_devices)]
    estimator = CellEstimator()
    shards, fakes = [], []
    for device in devices:
        fake = SizedFake(device, name="m", **fake_kw)
        spill = (
            DiskSpillStore(str(spill_root / device.replace(":", "_")), 10**9)
            if spill_root
            else None
        )
        shards.append(
            TabctxEngine(
                backends={"m": fake},
                estimators={"m": estimator},
                cache=ContextCacheManager(CAPACITY, spill_store=spill),
                default_backend="m",
            )
        )
        fakes.append(fake)
    engine = MultiDeviceEngine(shards, devices, clock=clock or FakeClock())
    return engine, shards, fakes


def where(shards, dataset_id):
    return [i for i, s in enumerate(shards) if s.holds(dataset_id)]


class TestConstruction:
    def test_needs_two_or_more_devices(self):
        _, shards, _ = build(2)
        with pytest.raises(ValueError):
            MultiDeviceEngine(shards[:1], ["cuda:0"])
        with pytest.raises(ValueError):
            MultiDeviceEngine(shards, ["cuda:0"])

    def test_devices_must_serve_the_same_models(self):
        _, shards, _ = build(2)
        other = TabctxEngine(
            backends={"x": FakeBackend(name="x")},
            estimators={"x": CellEstimator()},
            cache=ContextCacheManager(CAPACITY),
            default_backend="x",
        )
        with pytest.raises(ValueError, match="same models"):
            MultiDeviceEngine([shards[0], other], ["cuda:0", "cuda:1"])

    def test_exposes_the_single_engine_surface(self):
        engine, _, _ = build(3)
        assert engine.devices == ["cuda:0", "cuda:1", "cuda:2"]
        assert engine.backend_names == ["m"]
        assert engine.default_backend == "m"
        assert engine.admission_headroom_bytes() == CAPACITY


class TestPlacementSmallTables:
    def test_concurrent_users_spread_across_devices(self):
        engine, shards, _ = build(4)
        for user in range(4):
            engine.fit(*table(10), dataset_id=f"user{user}")
        placed = [where(shards, f"user{u}")[0] for u in range(4)]
        assert sorted(placed) == [0, 1, 2, 3]

    def test_recently_active_device_is_avoided(self):
        clock = FakeClock()
        engine, shards, _ = build(2, clock=clock)
        engine.fit(*table(10), dataset_id="busy")
        busy = where(shards, "busy")[0]
        for _ in range(5):
            engine.predict("busy", [[0.0]])
        engine.fit(*table(10), dataset_id="newcomer")
        assert where(shards, "newcomer") == [1 - busy]

    def test_activity_expires_after_the_window(self):
        clock = FakeClock()
        engine, shards, _ = build(2, clock=clock)
        engine.fit(*table(10), dataset_id="old")
        first = where(shards, "old")[0]
        clock.now += 31
        # Activity has expired, so the tie breaks on headroom: the empty
        # device wins over the one holding "old".
        engine.fit(*table(10), dataset_id="new")
        assert where(shards, "new") == [1 - first]

    def test_in_flight_work_outranks_recent_history(self):
        clock = FakeClock()
        engine, shards, fakes = build(2, clock=clock)
        # Device 1 has more recent calls, device 0 has one in flight.
        engine._recent[1].extend([clock.now] * 5)
        engine._inflight[0] = 1
        engine.fit(*table(10), dataset_id="d")
        assert where(shards, "d") == [1]


class TestPlacementLargeTables:
    def test_large_table_goes_where_it_fits_without_eviction(self):
        clock = FakeClock()
        engine, shards, _ = build(2, clock=clock)
        engine.fit(*table(600), dataset_id="big0")  # 6,000 B on one device
        clock.now += 60
        engine.fit(*table(500), dataset_id="big1")  # only fits on the other
        assert where(shards, "big0") != where(shards, "big1")
        assert shards[0].stats().n_cached_contexts == 1
        assert shards[1].stats().n_cached_contexts == 1

    def test_when_nothing_fits_it_evicts_on_the_emptiest_device_only(self):
        clock = FakeClock()
        engine, shards, _ = build(2, clock=clock)
        engine.fit(*table(700), dataset_id="a")  # device X: 7,000 used
        clock.now += 60
        engine.fit(*table(400), dataset_id="b")  # device Y: 4,000 used
        clock.now += 60
        # 800 cells fits nowhere without eviction; Y has more headroom,
        # so only "b" is evicted and "a" is untouched.
        engine.fit(*table(800), dataset_id="c")
        assert where(shards, "a") != [] and where(shards, "b") == []
        assert where(shards, "c") != where(shards, "a")

    def test_table_larger_than_any_device_is_rejected_without_evictions(self):
        engine, shards, _ = build(2)
        engine.fit(*table(100), dataset_id="keep0")
        engine.fit(*table(100), dataset_id="keep1")
        with pytest.raises(AdmissionRejected):
            engine.fit(*table(2000), dataset_id="huge")
        assert where(shards, "keep0") and where(shards, "keep1")

    def test_small_tables_do_not_block_a_later_large_table(self):
        clock = FakeClock()
        engine, shards, _ = build(4, clock=clock)
        for u in range(8):
            engine.fit(*table(50), dataset_id=f"small{u}")
            clock.now += 1
        clock.now += 60
        engine.fit(*table(900), dataset_id="large")
        # 8 small contexts spread 2 per device leave 9,000 B on every
        # device, so the large table needs no eviction anywhere.
        assert all(where(shards, f"small{u}") for u in range(8))
        assert len(where(shards, "large")) == 1


class TestRouting:
    def test_predict_runs_on_the_owning_device(self):
        engine, shards, fakes = build(2)
        engine.fit(*table(10, label="x"), dataset_id="d")
        owner = where(shards, "d")[0]
        assert engine.predict("d", [[0.0]]).predictions == ["x"]
        assert fakes[owner].predict_calls == 1
        assert fakes[1 - owner].predict_calls == 0

    def test_unknown_dataset_is_not_found(self):
        engine, _, _ = build(2)
        with pytest.raises(DatasetNotFoundError):
            engine.predict("nope", [[0.0]])

    def test_refit_moves_the_context_and_leaves_no_stale_copy(self):
        clock = FakeClock()
        engine, shards, _ = build(2, clock=clock)
        engine.fit(*table(10, label="v1"), dataset_id="d")
        first = where(shards, "d")[0]
        # Make the first device look busy so the re-fit lands elsewhere.
        engine._recent[first].extend([clock.now] * 10)
        engine.fit(*table(10, label="v2"), dataset_id="d")
        assert where(shards, "d") == [1 - first]
        assert engine.predict("d", [[0.0]]).predictions == ["v2"]

    def test_evict_drops_every_copy(self):
        engine, shards, _ = build(2)
        engine.fit(*table(10), dataset_id="d")
        engine.evict("d")
        assert where(shards, "d") == []
        with pytest.raises(DatasetNotFoundError):
            engine.predict("d", [[0.0]])

    def test_generated_dataset_ids_are_routed(self):
        engine, _, _ = build(2)
        dataset_id = engine.fit(*table(10, label="g"))
        assert engine.predict(dataset_id, [[0.0]]).predictions == ["g"]

    def test_feature_names_follow_the_context(self):
        engine, _, _ = build(2)
        engine.fit(*table(5, features=2), dataset_id="d", feature_names=["p", "q"])
        assert engine.feature_names("d") == ["p", "q"]
        assert engine.feature_names("missing") is None

    def test_fit_predict_leaves_nothing_cached(self):
        engine, shards, _ = build(2)
        out = engine.fit_predict(*table(10, label="z"), [[0.0]])
        assert out.predictions == ["z"]
        assert engine.stats().n_cached_contexts == 0

    def test_invalid_input_is_rejected_before_placement(self):
        engine, _, _ = build(2)
        with pytest.raises(InvalidInputError):
            engine.fit([[1.0], [2.0]], ["a"], dataset_id="bad")
        assert engine.device_activity() == [
            {"in_flight": 0, "recent_calls": 0},
            {"in_flight": 0, "recent_calls": 0},
        ]


class TestSpill:
    def test_capacity_eviction_spills_to_the_device_and_restores_there(self, tmp_path):
        clock = FakeClock()
        engine, shards, fakes = build(2, spill_root=tmp_path, clock=clock)
        engine.fit(*table(600, label="cold"), dataset_id="cold")  # idle tie: dev 0
        clock.now += 60
        engine.fit(*table(600), dataset_id="other")  # fits only on dev 1
        assert (where(shards, "cold"), where(shards, "other")) == ([0], [1])
        clock.now += 60
        # 900 cells fits nowhere without eviction; headroom ties, so dev 0
        # takes it and spills "cold" to dev 0's own spill tier.
        engine.fit(*table(900), dataset_id="big")
        assert where(shards, "big") == [0] and where(shards, "cold") == []
        assert (tmp_path / "cuda_0").exists()
        # Restoring "cold" brings it back onto dev 0 (spilling "big" there)
        # and never disturbs dev 1.
        assert engine.predict("cold", [[0.0]]).predictions == ["cold"]
        assert where(shards, "cold") == [0]
        assert where(shards, "other") == [1]
        assert fakes[1].predict_calls == 0

    def test_after_a_restart_spilled_contexts_are_found_on_their_device(self, tmp_path):
        engine, shards, _ = build(2, spill_root=tmp_path)
        engine.fit(*table(10, label="warm"), dataset_id="d")
        owner = where(shards, "d")[0]
        shards[owner]._cache.make_room(CAPACITY)  # spill everything there
        assert where(shards, "d") == []
        # Fresh engine over the same spill dirs = a restarted replica.
        restarted, new_shards, new_fakes = build(2, spill_root=tmp_path)
        assert restarted.predict("d", [[0.0]]).predictions == ["warm"]
        assert where(new_shards, "d") == [owner]
        assert new_fakes[owner].predict_calls == 1


class TestStats:
    def test_stats_sum_over_devices(self):
        engine, _, _ = build(3)
        engine.fit(*table(10), dataset_id="a")
        engine.fit(*table(20), dataset_id="b")
        total = engine.stats()
        per = engine.device_stats()
        assert total.n_cached_contexts == 2
        assert total.used_bytes == 300 == sum(s.used_bytes for s in per)
        assert total.capacity_bytes == 3 * CAPACITY

    def test_headroom_is_the_best_single_device(self):
        engine, _, _ = build(2)
        engine.fit(*table(300), dataset_id="a")
        assert engine.admission_headroom_bytes() == CAPACITY


class TestConcurrency:
    def test_work_on_different_devices_runs_in_parallel(self):
        engine, shards, fakes = build(2, predict_delay_s=0.3)
        engine.fit(*table(10), dataset_id="a")
        engine.fit(*table(10), dataset_id="b")
        assert where(shards, "a") != where(shards, "b")
        threads = [
            threading.Thread(target=engine.predict, args=(d, [[0.0]])) for d in "ab"
        ]
        start = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert time.monotonic() - start < 0.5  # ~0.3s, not ~0.6s serialized

    def test_concurrent_fits_spread_and_all_succeed(self):
        engine, shards, _ = build(4, fit_delay_s=0.05)
        errors = []

        def fit(user):
            try:
                engine.fit(*table(20), dataset_id=f"u{user}")
            except Exception as e:  # pragma: no cover - surfaced below
                errors.append(e)

        threads = [threading.Thread(target=fit, args=(u,)) for u in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        counts = [s.stats().n_cached_contexts for s in shards]
        assert sum(counts) == 16
        assert min(counts) >= 2  # spread, not piled on one device
        for u in range(16):
            assert engine.predict(f"u{u}", [[0.0]]).predictions == ["a"]
