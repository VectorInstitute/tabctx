"""PeriodicReport: health measurements read without waiting (#51)."""

import threading

from tabctx.serve.reporting import PeriodicReport


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def wait_for(predicate, timeout=5.0):
    done = threading.Event()
    for _ in range(int(timeout / 0.01)):
        if predicate():
            return
        done.wait(0.01)
    raise AssertionError("condition not reached")


def test_latest_and_its_age():
    clock = Clock()
    report = PeriodicReport(lambda: {"x": 1}, interval_s=60, clock=clock)
    try:
        wait_for(lambda: report.latest()[0] is not None)
        clock.now += 7
        assert report.latest() == ({"x": 1}, 7)
    finally:
        report.stop()


def test_reading_never_waits_on_a_measurement():
    """A measurement stuck behind something (torch's allocator lock, a
    worker's queue) leaves the last value, getting older."""
    calls, stuck, release = [], threading.Event(), threading.Event()

    def measure():
        calls.append(1)
        if len(calls) > 1:
            stuck.set()
            release.wait(5)
        return len(calls)

    report = PeriodicReport(measure, interval_s=0.01)
    try:
        assert stuck.wait(5)
        assert report.latest()[0] == 1  # answered at once, the last value
    finally:
        release.set()
        report.stop()


def test_a_failing_measurement_keeps_the_last_value():
    values = iter([1])

    def measure():
        return next(values)  # StopIteration from the second call on

    report = PeriodicReport(measure, interval_s=0.01)
    try:
        wait_for(lambda: report.latest()[0] == 1)
        threading.Event().wait(0.05)
        assert report.latest()[0] == 1
    finally:
        report.stop()


def test_publish_receives_each_measurement():
    published = []
    report = PeriodicReport(lambda: "r", interval_s=0.01, publish=published.append)
    try:
        wait_for(lambda: len(published) >= 3)
    finally:
        report.stop()
    assert set(published) == {"r"}
