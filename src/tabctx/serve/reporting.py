"""Measurements a health probe can read without waiting.

/readyz reports what only a slow or blocking call can measure: the GPU
memory torch holds (torch's allocator lock waits behind a fit freeing
memory), a spill directory's size (a walk of the directory), a worker
process's view of both (a round trip behind its queued calls). A probe
that waits on those makes a busy replica look hung (#51).

So each is measured on its own thread, on a timer, and the probe reads
the latest result and its age: never a call that can wait. A measurement
that stops arriving is itself the signal: its age keeps growing.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

REPORT_INTERVAL_S = 5.0
# A report this old means its measuring thread (or process) is stuck.
REPORT_STALE_S = 600.0


class PeriodicReport:
    """Calls `measure` every `interval_s` on a daemon thread and keeps the
    latest result. `publish`, if given, also receives each result (a
    worker process sends it to the serving process). `measure` returning
    PeriodicReport.STOP ends the thread (e.g. its owner is gone)."""

    STOP = object()

    def __init__(
        self,
        measure: Callable[[], Any],
        interval_s: float | None = None,
        name: str = "report",
        publish: Callable[[Any], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._measure = measure
        # Read at construction, not definition, so tests can shorten it.
        self._interval_s = REPORT_INTERVAL_S if interval_s is None else interval_s
        self._publish = publish
        self._clock = clock
        self._latest: tuple[Any, float] | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                value = self._measure()
            except Exception:  # noqa: BLE001 -- a failed measurement goes stale
                logger.exception("periodic report failed")
            else:
                if value is PeriodicReport.STOP:
                    return
                self._latest = (value, self._clock())
                if self._publish is not None:
                    try:
                        self._publish(value)
                    except Exception:  # noqa: BLE001 -- e.g. the pipe is gone
                        logger.debug("could not publish a report", exc_info=True)
            self._stop.wait(self._interval_s)

    def latest(self) -> tuple[Any, float | None]:
        """(the latest measurement, its age in seconds), or (None, None)
        before the first. Never waits."""
        latest = self._latest
        if latest is None:
            return None, None
        value, at = latest
        return value, self._clock() - at

    def stop(self) -> None:
        self._stop.set()


def spill_stats(store: Any) -> dict | None:
    """A spill tier's stats for /readyz, or an error in their place: a disk
    that can't be read is reported, not mistaken for a stuck worker."""
    if store is None:
        return None
    try:
        return store.stats()
    except OSError as e:
        return {"error": f"{type(e).__name__}: {e}"}
