"""Disk spillover tier for evicted contexts (ROADMAP "cache durability").

When GPU-budget pressure evicts a context, the caller-visible effect used
to be a later 404 and a full re-fit. With a spill store attached
(TABCTX_SPILL_DIR), capacity evictions instead serialize the context to
local disk, and a later get() for it transparently reloads and re-admits
it -- the same trade LLM engines make when they swap KV blocks to host
memory. Reloading a multi-GB context from local SSD is typically ~2x
faster than re-fitting it, and unlike a re-fit it needs no training data
from the caller.

Deliberately best-effort and bounded:

- The disk tier has its own capacity and LRU (file access order); when
  it fills, the oldest spilled context is dropped for real (then the
  caller re-fits, exactly as before spillover existed).
- Serialization failures downgrade to plain eviction -- a payload that
  can't be pickled must never turn a working eviction into an error.
- Spilled files live under a directory that does NOT survive intentional
  deletion or node replacement; this is a warm-restart/eviction cushion,
  not durable storage. (A restarted replica CAN reload contexts spilled
  by its predecessor on the same node if pointed at the same directory.)

Backends may customize serialization by exposing ``dumps_payload(payload)
-> bytes`` / ``loads_payload(data) -> payload`` (probed with getattr;
default is pickle). TabICL's implementation strips the shared pretrained
backbone before pickling and re-attaches it on load, so a spilled
context costs its own tensors only.
"""

from __future__ import annotations

import json
import pickle
import threading
import time
from collections.abc import Callable
from pathlib import Path

from tabctx.cache.manager import CachedContext

DEFAULT_SPILL_CAPACITY_BYTES = 50 * 1024**3


class DiskSpillStore:
    def __init__(
        self,
        directory: str | Path,
        capacity_bytes: int = DEFAULT_SPILL_CAPACITY_BYTES,
        serializers: dict[
            str, tuple[Callable[[object], bytes], Callable[[bytes], object]]
        ]
        | None = None,
    ) -> None:
        """serializers: per-backend-name (dumps, loads) pairs; backends
        absent from the map use pickle. Dispatch happens on the
        context's backend_name (multi-model deployments spill each
        model's contexts with that model's serializer)."""
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._capacity_bytes = capacity_bytes
        self._serializers = serializers or {}
        self._lock = threading.Lock()
        # dataset_id -> (meta, last_used monotonic) for entries this
        # process knows about; rebuilt lazily from disk for entries a
        # predecessor process spilled (warm restart).
        self._index: dict[str, float] = {}
        for meta_path in self._dir.glob("*.meta.json"):
            self._index[meta_path.name[: -len(".meta.json")]] = 0.0

    def _paths(self, dataset_id: str) -> tuple[Path, Path]:
        # dataset_ids may contain ':' (tenant scoping) -- hex-encode for
        # a filesystem-safe, collision-free name.
        safe = dataset_id.encode().hex()
        return (self._dir / f"{safe}.meta.json", self._dir / f"{safe}.payload")

    def spill(self, context: CachedContext) -> bool:
        """Serialize an evicted context to disk. Returns False (and
        leaves no partial files) on any failure -- best-effort."""
        meta_path, payload_path = self._paths(context.dataset_id)
        dumps, _ = self._serializers.get(
            context.backend_name, (pickle.dumps, pickle.loads)
        )
        try:
            blob = dumps(context.payload)
        except Exception:  # noqa: BLE001 -- unpicklable payloads downgrade
            return False
        try:
            self._make_room(len(blob))
            payload_path.write_bytes(blob)
            meta_path.write_text(
                json.dumps(
                    {
                        "dataset_id": context.dataset_id,
                        "backend_name": context.backend_name,
                        "task": context.task,
                        "n_train": context.n_train,
                        "n_features": context.n_features,
                        "est_bytes": context.est_bytes,
                    }
                )
            )
        except OSError:
            payload_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            return False
        with self._lock:
            self._index[context.dataset_id] = time.monotonic()
        return True

    def load(self, dataset_id: str) -> CachedContext | None:
        """Reload a spilled context (removing it from the spill tier --
        it's going back into the primary cache). None if absent or
        unreadable."""
        meta_path, payload_path = self._paths(dataset_id)
        try:
            meta = json.loads(meta_path.read_text())
            _, loads = self._serializers.get(
                meta["backend_name"], (pickle.dumps, pickle.loads)
            )
            payload = loads(payload_path.read_bytes())
        except Exception:  # noqa: BLE001 -- absent or corrupt == miss
            return None
        finally:
            # Success: the context moves back to the primary cache.
            # Failure: drop the unreadable files rather than retrying
            # them forever.
            self.delete(dataset_id)
        return CachedContext(
            dataset_id=meta["dataset_id"],
            backend_name=meta["backend_name"],
            task=meta["task"],
            n_train=meta["n_train"],
            n_features=meta["n_features"],
            payload=payload,
            est_bytes=meta["est_bytes"],
        )

    def delete(self, dataset_id: str) -> None:
        meta_path, payload_path = self._paths(dataset_id)
        payload_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        with self._lock:
            self._index.pop(dataset_id, None)

    def _make_room(self, needed: int) -> None:
        while self.used_bytes + needed > self._capacity_bytes:
            with self._lock:
                if not self._index:
                    return
                oldest = min(self._index, key=self._index.__getitem__)
            self.delete(oldest)

    @property
    def used_bytes(self) -> int:
        return sum(p.stat().st_size for p in self._dir.glob("*.payload") if p.exists())

    def stats(self) -> dict:
        with self._lock:
            n = len(self._index)
        return {
            "n_spilled_contexts": n,
            "spill_used_bytes": self.used_bytes,
            "spill_capacity_bytes": self._capacity_bytes,
        }
