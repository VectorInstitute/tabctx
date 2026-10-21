"""Where a request's time went: a per-request breakdown into phases.

A slow fit or predict can be waiting (for another fit to release the
device, for host memory, for placement to hear back from a busy worker),
moving data (spilling contexts to make room, restoring a spilled one), or
computing (CPU preprocessing, the GPU itself, compiling a new graph).
Those look identical from outside, so each request carries a Timings
collector in a ContextVar and the code that does each of those things
records its phase into whatever collector is current. The serving layer
returns the breakdown with the response (``timings_ms``), so it can be
read wherever the response is -- including through a gateway, without
log access.

Recording is a no-op when no collector is current (library use, warmup),
and phases nest freely: each records its own wall time. Worker processes
collect their own and send them back with the reply (serve/workers.py),
which merges them into the caller's.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar


class Timings:
    """Milliseconds per phase, plus counters (e.g. graphs compiled)."""

    def __init__(self) -> None:
        self.ms: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, phase: str, seconds: float) -> None:
        self.ms[phase] = self.ms.get(phase, 0.0) + seconds * 1000

    def count(self, name: str, n: int = 1) -> None:
        if n:
            self.counts[name] = self.counts.get(name, 0) + n

    def merge(self, other: Mapping[str, float]) -> None:
        """Fold in a breakdown as as_dict() produced it (from a worker)."""
        for key, value in other.items():
            if key.startswith("n_"):
                self.count(key[2:], int(value))
            else:
                self.ms[key] = self.ms.get(key, 0.0) + float(value)

    def as_dict(self) -> dict[str, float]:
        """Phases in ms (rounded) and counters as ``n_<name>``."""
        out: dict[str, float] = {k: round(v, 1) for k, v in self.ms.items()}
        out.update({f"n_{k}": v for k, v in self.counts.items()})
        return out


_current: ContextVar[Timings | None] = ContextVar("tabctx_timings", default=None)


def current() -> Timings | None:
    return _current.get()


@contextmanager
def collect() -> Iterator[Timings]:
    """Make a fresh collector current for the duration of the block."""
    timings = Timings()
    token = _current.set(timings)
    try:
        yield timings
    finally:
        _current.reset(token)


@contextmanager
def phase(name: str) -> Iterator[None]:
    """Record the block's wall time under ``name`` (no-op without a collector)."""
    timings = _current.get()
    if timings is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        timings.add(name, time.perf_counter() - start)


def count(name: str, n: int = 1) -> None:
    timings = _current.get()
    if timings is not None:
        timings.count(name, n)


def compiled_graphs() -> int:
    """Graphs torch.compile has built in this process so far (0 without
    torch). Process-wide, so concurrent calls in one process share it."""
    try:
        from torch._dynamo.utils import counters
    except ImportError:
        return 0
    return int(counters["stats"]["unique_graphs"])


@contextmanager
def counting_compiles() -> Iterator[None]:
    """Count the graphs compiled while the block ran (``n_graphs_compiled``)."""
    if _current.get() is None:
        yield
        return
    before = compiled_graphs()
    try:
        yield
    finally:
        count("graphs_compiled", compiled_graphs() - before)
