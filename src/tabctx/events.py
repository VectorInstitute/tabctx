"""Residency events: where contexts live, pushed to whoever schedules (#39).

A platform that co-schedules LLMs and tabular models needs to know which
GPU holds which tenant's context without polling every replica. The
multi-device router emits a ``ResidencyEvent`` when a fit places a context
on a device and when a copy is removed (re-fit elsewhere, explicit evict).
Sinks are plain callables, so any transport plugs in: a log, a metrics
counter, a Ray actor, a Dynamo event plane publisher.

Events report the router's decisions, not every cache-internal move:
capacity eviction to the spill tier and restores stay on their device,
so a context's device doesn't change until the router says so.

Sinks run on the request's thread after the change, so they must be
fast and must not raise (one that does is logged and skipped).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

EventKind = Literal["placed", "removed"]


@dataclass(frozen=True)
class ResidencyEvent:
    kind: EventKind
    dataset_id: str
    device: str
    backend: str | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


EventSink = Callable[[ResidencyEvent], None]


class EventBus:
    """Fans events out to sinks; a failing sink never fails the request."""

    def __init__(self, sinks: list[EventSink] | None = None) -> None:
        self._sinks: list[EventSink] = list(sinks or [])
        self._lock = threading.Lock()

    def subscribe(self, sink: EventSink) -> None:
        with self._lock:
            self._sinks.append(sink)

    def emit(self, event: ResidencyEvent) -> None:
        with self._lock:
            sinks = list(self._sinks)
        for sink in sinks:
            try:
                sink(event)
            except Exception:  # noqa: BLE001 -- observers must not break serving
                logger.exception("residency event sink failed on %s", event)


class RecentEvents:
    """A bounded in-memory log of the latest events (a sink)."""

    def __init__(self, maxlen: int = 256) -> None:
        self._events: deque[ResidencyEvent] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def __call__(self, event: ResidencyEvent) -> None:
        with self._lock:
            self._events.append(event)

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [e.to_dict() for e in self._events]
