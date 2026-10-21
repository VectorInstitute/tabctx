"""The schemas of a replica's recently fitted contexts, in its own process.

A predict encodes its rows with the training table's schema
(tabctx.schema). The engine keeps the schema with the context, but on a
multi-GPU replica asking for it is a round trip to a worker process
(and a large text column makes the schema large), so the serving layer
keeps the schemas it recorded at fit. Session affinity routes a
dataset's fits and predicts to the same replica, so this copy is current
for every context fit since the replica started; a miss (after a
restart, or past the bound) falls back to the engine. A cached schema
outliving its context is harmless: the predict still gets the engine's
404.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from tabctx.schema import TableSchema

DEFAULT_MAX_ENTRIES = 1024


class SchemaCache:
    """A thread-safe LRU of dataset id -> TableSchema."""

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self._max = max_entries
        self._entries: OrderedDict[str, TableSchema] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, dataset_id: str, schema: TableSchema) -> None:
        with self._lock:
            self._entries[dataset_id] = schema
            self._entries.move_to_end(dataset_id)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)

    def get(self, dataset_id: str) -> TableSchema | None:
        with self._lock:
            schema = self._entries.get(dataset_id)
            if schema is not None:
                self._entries.move_to_end(dataset_id)
            return schema
