"""One lock per key (a dataset id), created on demand and dropped when
unused, so calls for one key run one at a time while others proceed."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator


class KeyedLocks:
    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locks: dict[str, list] = {}  # key -> [lock, users]

    @contextlib.contextmanager
    def hold(self, key: str) -> Iterator[None]:
        with self._guard:
            entry = self._locks.setdefault(key, [threading.Lock(), 0])
            entry[1] += 1
        try:
            with entry[0]:
                yield
        finally:
            with self._guard:
                entry[1] -= 1
                if entry[1] == 0:
                    del self._locks[key]
