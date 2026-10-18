"""Pluggable placement (placement.py) and residency events (events.py)."""

import pytest

from tabctx.events import EventBus, RecentEvents, ResidencyEvent
from tabctx.multidevice import MultiDeviceEngine
from tabctx.placement import DeviceView, FitRequest, LeastActiveThatFits
from tests.unit.test_multidevice import build, table, where

REQ = FitRequest(n_train=10, n_features=1, backend="m")


def view(index, headroom, need=100, in_flight=0, recent=0, alive=True):
    return DeviceView(index, f"cuda:{index}", headroom, need, in_flight, recent, alive)


class TestDefaultPolicy:
    def test_least_active_device_that_fits(self):
        views = [view(0, 1000, in_flight=1), view(1, 500), view(2, 900, recent=3)]
        assert LeastActiveThatFits().choose(REQ, views) == 1

    def test_ties_go_to_most_headroom(self):
        assert LeastActiveThatFits().choose(REQ, [view(0, 500), view(1, 900)]) == 1

    def test_nothing_fits_takes_most_headroom_left(self):
        views = [view(0, 50, need=200), view(1, 80, need=100)]
        assert LeastActiveThatFits().choose(REQ, views) == 1

    def test_dead_devices_are_skipped(self):
        views = [view(0, 10**9, alive=False), view(1, 10, need=100)]
        assert LeastActiveThatFits().choose(REQ, views) == 1


class Pinned:
    """A platform policy: keep tabular work off device 0."""

    def __init__(self):
        self.seen: list[tuple[FitRequest, list[DeviceView]]] = []

    def choose(self, request, devices):
        self.seen.append((request, list(devices)))
        return len(devices) - 1


def test_a_custom_policy_decides_placement():
    engine, shards, _ = build(n_devices=3)
    policy = Pinned()
    router = MultiDeviceEngine(shards, engine.devices, placement=policy)
    router.fit(*table(10), dataset_id="d")
    assert where(shards, "d") == [2]
    request, views = policy.seen[0]
    assert (request.n_train, request.n_features, request.dataset_id) == (10, 1, "d")
    assert [v.device for v in views] == ["cuda:0", "cuda:1", "cuda:2"]
    assert all(v.need_bytes > 0 and v.headroom_bytes > 0 for v in views)


def test_a_policy_choosing_a_bad_index_is_an_error():
    engine, shards, _ = build(n_devices=2)

    class Broken:
        def choose(self, request, devices):
            return 7

    router = MultiDeviceEngine(shards, engine.devices, placement=Broken())
    with pytest.raises(ValueError, match="placement policy"):
        router.fit(*table(10), dataset_id="d")


def test_placements_and_removals_are_reported():
    engine, shards, _ = build(n_devices=2)
    log = RecentEvents()
    engine.events.subscribe(log)
    engine.fit(*table(10), dataset_id="d")
    first = where(shards, "d")[0]
    # Re-fit on the other device (a policy that picks it), then evict.
    other = 1 - first
    engine._placement = type("P", (), {"choose": lambda self, r, d: other})()
    engine.fit(*table(10), dataset_id="d")
    engine.evict("d")
    kinds = [(e["kind"], e["device"]) for e in log.snapshot()]
    assert kinds == [
        ("placed", f"cuda:{first}"),
        ("placed", f"cuda:{other}"),
        ("removed", f"cuda:{first}"),
        ("removed", f"cuda:{other}"),
    ]
    assert log.snapshot()[0]["backend"] == "m"


def test_a_failing_sink_never_fails_the_request(caplog):
    bus = EventBus()
    seen = []

    def broken(event):
        raise RuntimeError("sink down")

    bus.subscribe(broken)
    bus.subscribe(seen.append)
    with caplog.at_level("ERROR"):
        bus.emit(ResidencyEvent("placed", "d", "cuda:0"))
    assert len(seen) == 1 and "sink failed" in caplog.text


def test_recent_events_are_bounded():
    log = RecentEvents(maxlen=3)
    for i in range(5):
        log(ResidencyEvent("placed", f"d{i}", "cuda:0"))
    assert [e["dataset_id"] for e in log.snapshot()] == ["d2", "d3", "d4"]
