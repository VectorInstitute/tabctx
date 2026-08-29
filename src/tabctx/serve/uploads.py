"""Replica-local upload store: the self-hosted answer to signed-URL flows.

Large tables can't travel as inline JSON (a 1M-row x 200-feature table is
multi-GB of payload and parse time), so the serving layer splits data
upload from orchestration, the same lifecycle PriorLabs' hosted API uses
(prepare-upload -> PUT -> fit-by-reference) minus the object store:
tabctx streams the CSV straight to disk on the replica and hands back an
upload id that fit()/predict() consume by reference.

Correctness in a multi-replica deployment rests on the SAME two
contracts the rest of the serving layer already enforces:

- **Affinity**: uploads are replica-local, so the upload request must
  carry the `x-session-id: <dataset_id>` header, which routes it to the
  replica that the subsequent fit/predict for that dataset_id will also
  reach (consistent-hash routing, see serve/affinity.py). An upload made
  without the header lands on an arbitrary replica and the later fit
  surfaces a clear UploadNotFoundError naming this cause.
- **Tenancy**: upload ids are scoped by tenant exactly like dataset ids
  (see serve/tenancy.py) -- and they are unguessable UUIDs regardless,
  so one tenant can never reference another's upload even in unscoped
  dev deployments.

Lifecycle: uploads are SINGLE-USE (consumed by the fit/predict that
references them, then deleted -- the parsed table lives on in the
context cache, so keeping the CSV would double storage) and expire after
a TTL otherwise. Expiry sweeps run opportunistically on store activity;
there is no background thread to leak.

Thread-safety: the registry dict is lock-guarded; file writes happen
outside the lock (each upload writes to its own unique path).
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable, Iterator

from tabctx.errors import UploadNotFoundError, UploadTooLargeError

DEFAULT_TTL_S = 3600.0
DEFAULT_MAX_UPLOAD_BYTES = 4 * 1024**3


@dataclass(frozen=True)
class UploadRecord:
    upload_id: str
    tenant_id: str | None
    path: Path
    n_bytes: int
    created_at: float


class UploadStore:
    def __init__(
        self,
        directory: str | Path | None = None,
        ttl_s: float = DEFAULT_TTL_S,
        max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    ) -> None:
        if directory is None:
            import tempfile

            directory = Path(tempfile.mkdtemp(prefix="tabctx-uploads-"))
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._ttl_s = ttl_s
        self._max_upload_bytes = max_upload_bytes
        self._records: dict[str, UploadRecord] = {}
        self._lock = threading.Lock()

    @property
    def max_upload_bytes(self) -> int:
        return self._max_upload_bytes

    def begin(self) -> "_UploadWriter":
        """Start a streamed upload. The caller feeds chunks with
        .write() (sync or from an async loop -- writes are plain file
        appends) and finishes with .commit(tenant_id) or .abort().
        The size cap is enforced as bytes arrive; the excess never
        reaches disk."""
        self.sweep_expired()
        upload_id = uuid.uuid4().hex
        return _UploadWriter(self, upload_id, self._dir / upload_id)

    def put(self, chunks: Iterable[bytes], tenant_id: str | None = None) -> UploadRecord:
        """Convenience over begin(): stream an iterable of chunks."""
        writer = self.begin()
        try:
            for chunk in chunks:
                writer.write(chunk)
        except BaseException:
            writer.abort()
            raise
        return writer.commit(tenant_id)

    def _register(self, record: UploadRecord) -> None:
        with self._lock:
            self._records[record.upload_id] = record

    def consume(self, upload_id: str, tenant_id: str | None = None) -> Path:
        """Claim an upload for use: removes it from the registry and
        returns its file path. The caller reads the file and then calls
        discard(path). Single-use by design -- a second consume of the
        same id raises UploadNotFoundError."""
        self.sweep_expired()
        with self._lock:
            record = self._records.get(upload_id)
            # Tenant check inside the lock so a mismatch can't race a
            # legitimate consume. A mismatched tenant gets the same
            # "not found" as a missing id -- never confirmation that the
            # id exists under someone else's tenant.
            if record is not None and record.tenant_id == tenant_id:
                del self._records[upload_id]
            else:
                record = None
        if record is None:
            raise UploadNotFoundError(
                f"no upload {upload_id!r} on this replica -- it was never "
                "uploaded, expired, was already used (uploads are single-"
                "use), belongs to a different tenant, or was uploaded "
                "without the session-affinity header and landed on a "
                "different replica (multi-replica deployments require "
                "x-session-id on the upload request too)"
            )
        return record.path

    @staticmethod
    def discard(path: Path) -> None:
        path.unlink(missing_ok=True)

    def sweep_expired(self) -> int:
        now = time.monotonic()
        with self._lock:
            expired = [
                r for r in self._records.values() if now - r.created_at > self._ttl_s
            ]
            for r in expired:
                del self._records[r.upload_id]
        for r in expired:
            r.path.unlink(missing_ok=True)
        return len(expired)

    def stats(self) -> dict:
        with self._lock:
            records = list(self._records.values())
        return {
            "n_pending_uploads": len(records),
            "pending_bytes": sum(r.n_bytes for r in records),
            "ttl_s": self._ttl_s,
            "max_upload_bytes": self._max_upload_bytes,
        }

    def __iter__(self) -> Iterator[UploadRecord]:  # for tests/debugging
        with self._lock:
            return iter(list(self._records.values()))


class _UploadWriter:
    def __init__(self, store: UploadStore, upload_id: str, path: Path) -> None:
        self._store = store
        self.upload_id = upload_id
        self._path = path
        self._file = open(path, "wb")
        self._n_bytes = 0
        self._open = True

    def write(self, chunk: bytes) -> None:
        assert self._open, "writer already committed/aborted"
        self._n_bytes += len(chunk)
        if self._n_bytes > self._store.max_upload_bytes:
            self.abort()
            raise UploadTooLargeError(
                f"upload exceeds the {self._store.max_upload_bytes}-byte cap "
                "(TABCTX_MAX_UPLOAD_BYTES)"
            )
        self._file.write(chunk)

    def commit(self, tenant_id: str | None = None) -> UploadRecord:
        assert self._open, "writer already committed/aborted"
        self._open = False
        self._file.close()
        record = UploadRecord(
            upload_id=self.upload_id,
            tenant_id=tenant_id,
            path=self._path,
            n_bytes=self._n_bytes,
            created_at=time.monotonic(),
        )
        self._store._register(record)
        return record

    def abort(self) -> None:
        if self._open:
            self._open = False
            self._file.close()
            self._path.unlink(missing_ok=True)
