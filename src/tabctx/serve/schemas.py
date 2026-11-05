"""The schemas of a replica's recently fitted contexts, in its own process.

A predict encodes its rows with the training table's schema
(tabctx.schema). The engine keeps the schema with the context, but on a
multi-GPU replica asking for it is a round trip to a worker process
(and a large text column makes the schema large), so the serving layer
keeps the schemas it recorded at fit. Session affinity routes a
dataset's fits and predicts to the same replica, so this copy is current
for every context fit since the replica started; a miss (after a
restart, or past the bounds) falls back to the engine. A cached schema
outliving its context is harmless: the predict still gets the engine's
404.

Fits of one dataset id are recorded one at a time (fitting()): the
engine serializes them too, so the schema recorded last is the resident
context's. A failed fit leaves the engine's previous context serving and
no cached schema, so predicts ask the engine for the one that matches.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator

from tabctx.keyed_locks import KeyedLocks
from tabctx.schema import TableSchema

DEFAULT_MAX_ENTRIES = 1024
DEFAULT_MAX_BYTES = 64 * 1024**2


class SchemaCache:
    """A thread-safe LRU of dataset id -> TableSchema, bounded in entries
    and in (approximate) bytes; a schema over the byte bound isn't kept."""

    def __init__(
        self, max_entries: int = DEFAULT_MAX_ENTRIES, max_bytes: int = DEFAULT_MAX_BYTES
    ) -> None:
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._entries: OrderedDict[str, TableSchema] = OrderedDict()
        # dataset_id -> the retention deadline of its context (unix time):
        # the schema holds the training table's text categories, so it goes
        # when the context does.
        self._deadlines: dict[str, float] = {}
        self._bytes = 0
        self._lock = threading.Lock()
        self._fits = KeyedLocks()

    @contextlib.contextmanager
    def fitting(
        self, dataset_id: str
    ) -> Iterator[Callable[[TableSchema, float | None], None]]:
        """Around a fit of dataset_id: yields the function that records its
        schema (and its context's retention deadline, if any) once the fit
        succeeds."""
        with self._fits.hold(dataset_id):
            self._drop(dataset_id)
            yield lambda schema, deadline=None: self._put(dataset_id, schema, deadline)

    def get(self, dataset_id: str) -> TableSchema | None:
        with self._lock:
            deadline = self._deadlines.get(dataset_id)
            if deadline is not None and deadline <= time.time():
                self._drop_locked(dataset_id)
                return None
            schema = self._entries.get(dataset_id)
            if schema is not None:
                self._entries.move_to_end(dataset_id)
            return schema

    def purge_expired(self, now: float | None = None) -> int:
        """Drop the schemas of contexts past their retention deadline."""
        now = time.time() if now is None else now
        with self._lock:
            due = [i for i, at in self._deadlines.items() if at <= now]
            for dataset_id in due:
                self._drop_locked(dataset_id)
        return len(due)

    def _put(
        self, dataset_id: str, schema: TableSchema, deadline: float | None = None
    ) -> None:
        size = schema.approx_bytes
        if size > self._max_bytes:
            return
        with self._lock:
            self._drop_locked(dataset_id)
            self._entries[dataset_id] = schema
            self._bytes += size
            if deadline is not None:
                self._deadlines[dataset_id] = deadline
            while (
                len(self._entries) > self._max_entries or self._bytes > self._max_bytes
            ):
                oldest = next(iter(self._entries))
                self._drop_locked(oldest)

    @contextlib.contextmanager
    def deleting(self, dataset_id: str) -> Iterator[None]:
        """Around a delete of dataset_id: one at a time with its fits, and
        its schema (which holds the training table's text categories) is
        dropped. A fit that finished first recorded its schema already."""
        with self._fits.hold(dataset_id):
            yield
            self._drop(dataset_id)

    def _drop(self, dataset_id: str) -> None:
        with self._lock:
            self._drop_locked(dataset_id)

    def _drop_locked(self, dataset_id: str) -> None:
        old = self._entries.pop(dataset_id, None)
        if old is not None:
            self._bytes -= old.approx_bytes
        self._deadlines.pop(dataset_id, None)
