"""Per-GPU worker processes: one TabctxEngine per device, each in its own
process, routed by a MultiDeviceEngine in the serving process.

Why processes. TabICL and TabPFN fits are host-bound: the Python thread
driving a fit is busy for ~90% of its wall time (kernel launches,
preprocessing). Threads of one process driving several GPUs therefore
contend for the GIL -- measured on 4x A100 (benchmarks/bench_scaling.py),
four concurrent 10k x 50 fits took 3.6 s each from threads vs 1.9 s
alone, and 1.8 s each from a process per GPU (4.3x aggregate vs 2.1x).

Shape. ``ProcessShard`` implements the router's ``Shard`` protocol
(multidevice.py) by proxying calls to a worker process that owns one
device's complete engine (built by factory.build_shard: backends, cache,
spill tier, its own estimator). The serving layer, the router and its
placement policy are unchanged, and none of this depends on Ray: the
same core runs under any serving framework that hosts one Python process
per node (Ray Serve today, Dynamo tomorrow).

Protocol. One duplex pipe per worker; requests ``(id, method, args,
kwargs)`` are multiplexed (a worker serves several at once from a thread
pool, so predicts queue behind a fit exactly as in-process), and every
response carries a ``Snapshot`` of the worker's residency, stats and
admission headroom. The router's lock-free reads (holds, has,
stats_nowait, admission_headroom_bytes) answer from the latest snapshot,
never a round trip. Snapshots are sequence-numbered so a late response
can't roll the view back. The view may lag the worker's truth, safely:
a context the view still lists but the worker evicted makes a predict
miss there and the router look elsewhere (a real 404 only if nowhere);
a new context is listed before its fit's response returns to the caller.

Failure. If a worker dies (host OOM kill, a CUDA fatal error), its
pending calls fail with WorkerDiedError (a BackendComputeError, so the
API answers 500, not a hang), the shard reports itself dead, placement
skips it, and /readyz reports the replica unhealthy so the serving
framework restarts it.
"""

from __future__ import annotations

import itertools
import logging
import pickle
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from tabctx.errors import BackendComputeError
from tabctx.types import ArrayLike, EngineStats, PredictOutcome, Task

logger = logging.getLogger(__name__)

# Calls the worker serves (the Shard surface's blocking part, plus reports).
_METHODS = frozenset(
    {
        "fit",
        "predict",
        "evict",
        "feature_names",
        "estimate_fit_bytes",
        "report",
    }
)
# Concurrent calls per worker: its engine schedules GPU work itself (see
# engine.py); these only need to queue and answer cheap calls promptly.
_WORKER_THREADS = 16


class WorkerDiedError(BackendComputeError):
    """A worker process exited with calls in flight."""


@dataclass(frozen=True)
class Snapshot:
    seq: int
    resident: frozenset[str]
    known: frozenset[str]  # resident, in transit or spilled
    stats: EngineStats
    headroom: dict[str, int]  # backend -> admission headroom bytes


@dataclass(frozen=True)
class _Ready:
    backend_names: list[str]
    default_backend: str
    snapshot: Snapshot


@dataclass
class _Failure:
    error: BaseException


def _portable(error: BaseException) -> BaseException:
    """The exception itself if it pickles (tabctx errors do), else a
    RuntimeError carrying its type and message."""
    try:
        pickle.dumps(error)
        return error
    except Exception:  # noqa: BLE001
        return RuntimeError(f"{type(error).__name__}: {error}")


# ---- worker process --------------------------------------------------------


def _snapshot(engine: Any, seq: int) -> Snapshot:
    resident, known = engine.residency_nowait()
    return Snapshot(
        seq=seq,
        resident=resident,
        known=known,
        stats=engine.stats_nowait(),
        headroom={b: engine.admission_headroom_bytes(b) for b in engine.backend_names},
    )


def _torch_if_cuda() -> Any:
    try:
        import torch
    except ImportError:
        return None
    return torch if torch.cuda.is_available() else None


def _report(shard: Any, device: str) -> dict:
    """What /readyz shows per device that only the owning process can
    measure: its spill tier and the GPU memory torch really holds."""
    gpu = None
    torch = _torch_if_cuda() if device.startswith("cuda") else None
    if torch is not None:
        gpu = {
            "allocated_mb": torch.cuda.memory_allocated(device) / 1e6,
            "reserved_mb": torch.cuda.memory_reserved(device) / 1e6,
        }
    return {
        "spill": shard.spill_store.stats() if shard.spill_store else None,
        "gpu_memory": gpu,
    }


def _worker_main(
    conn: Any,
    settings: Any,
    device: str,
    n_devices: int,
    gpu_capacity: int | None,
    host_guard: Any,
) -> None:
    """Entry point of a worker process: build the device's engine, then
    serve calls until the pipe closes."""
    logging.basicConfig(level=logging.INFO)
    from tabctx.serve.factory import build_shard

    try:
        shard = build_shard(
            settings,
            device,
            n_devices=n_devices,
            gpu_capacity=gpu_capacity,
            host_guard=host_guard,
        )
    except BaseException as e:  # noqa: BLE001 -- reported to the parent
        conn.send(_Failure(_portable(e)))
        return
    engine = shard.engine
    seq = itertools.count(1)
    seq_lock = threading.Lock()
    send_lock = threading.Lock()

    def snapshot() -> Snapshot:
        # Numbered and taken under one lock, so a higher seq is never an
        # older view.
        with seq_lock:
            return _snapshot(engine, next(seq))

    def handle(req_id: int, method: str, args: tuple, kwargs: dict) -> None:
        try:
            if method == "report":
                result: Any = _report(shard, device)
            else:
                result = getattr(engine, method)(*args, **kwargs)
            reply = (req_id, True, result, snapshot())
        except BaseException as e:  # noqa: BLE001 -- returned to the caller
            reply = (req_id, False, _portable(e), snapshot())
        with send_lock:
            conn.send(reply)

    conn.send(_Ready(list(engine.backend_names), engine.default_backend, snapshot()))
    with ThreadPoolExecutor(_WORKER_THREADS, thread_name_prefix=f"w-{device}") as pool:
        while True:
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                break
            if msg is None:  # orderly shutdown
                break
            req_id, method, args, kwargs = msg
            if method not in _METHODS:
                with send_lock:
                    conn.send(
                        (
                            req_id,
                            False,
                            ValueError(f"unknown call {method!r}"),
                            snapshot(),
                        )
                    )
                continue
            pool.submit(handle, req_id, method, args, kwargs)


# ---- serving process -------------------------------------------------------


class ProcessShard:
    """The router's Shard for one device whose engine runs in a worker
    process. Thread-safe: any number of serving threads may call it."""

    def __init__(
        self,
        device: str,
        settings: Any,
        *,
        n_devices: int,
        gpu_capacity: int | None,
        host_guard: Any,
        mp_context: Any,
    ) -> None:
        self.device = device
        parent, child = mp_context.Pipe()
        self._conn = parent
        self._proc = mp_context.Process(
            target=_worker_main,
            args=(child, settings, device, n_devices, gpu_capacity, host_guard),
            name=f"tabctx-worker-{device}",
            daemon=True,
        )
        self._proc.start()
        child.close()
        self._send_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._pending: dict[int, Future] = {}
        self._ids = itertools.count(1)
        self._dead: str | None = None
        self._snapshot: Snapshot | None = None
        self._backend_names: list[str] = []
        self._default_backend = ""
        self._reader: threading.Thread | None = None

    # ---- lifecycle ---------------------------------------------------

    def wait_ready(self, timeout_s: float) -> None:
        """Block until the worker built its engine (loading model weights);
        raise its error if it failed."""
        if not self._conn.poll(timeout_s):
            self.close()
            raise TimeoutError(f"worker for {self.device} not ready in {timeout_s}s")
        try:
            first = self._conn.recv()
        except (EOFError, OSError) as e:
            raise WorkerDiedError(
                f"worker for {self.device} exited during startup "
                f"(exit code {self._proc.exitcode})"
            ) from e
        if isinstance(first, _Failure):
            self._proc.join(5)
            raise first.error
        self._backend_names = first.backend_names
        self._default_backend = first.default_backend
        self._snapshot = first.snapshot
        self._reader = threading.Thread(
            target=self._read_loop, name=f"tabctx-reader-{self.device}", daemon=True
        )
        self._reader.start()

    @classmethod
    def start_all(
        cls,
        settings: Any,
        devices: tuple[str, ...],
        *,
        gpu_capacity: int | None,
        host_guard: Any,
        mp_context: Any,
        ready_timeout_s: float = 900.0,
    ) -> list[ProcessShard]:
        """Start one worker per device in parallel (each loads its own model
        weights) and wait for all of them."""
        shards: list[ProcessShard] = [
            cls(
                d,
                settings,
                n_devices=len(devices),
                gpu_capacity=gpu_capacity,
                host_guard=host_guard,
                mp_context=mp_context,
            )
            for d in devices
        ]
        try:
            deadline = time.monotonic() + ready_timeout_s
            for s in shards:
                s.wait_ready(max(1.0, deadline - time.monotonic()))
        except BaseException:
            for s in shards:
                s.close()
            raise
        return shards

    def close(self) -> None:
        with self._state_lock:
            if self._dead is None:
                self._dead = f"worker for {self.device} was shut down"
        with self._send_lock:
            try:
                self._conn.send(None)
            except (OSError, ValueError):
                pass
        self._proc.join(10)
        if self._proc.is_alive():
            self._proc.kill()
            self._proc.join(5)
        self._conn.close()
        self._fail_pending(f"worker for {self.device} was shut down")

    @property
    def alive(self) -> bool:
        return self._dead is None and self._proc.is_alive()

    @property
    def death_reason(self) -> str | None:
        if self._dead is None and not self._proc.is_alive():
            return f"worker for {self.device} exited (code {self._proc.exitcode})"
        return self._dead

    # ---- transport ---------------------------------------------------

    def _read_loop(self) -> None:
        while True:
            try:
                req_id, ok, payload, snap = self._conn.recv()
            except (EOFError, OSError):
                break
            self._apply(snap)
            with self._state_lock:
                fut = self._pending.pop(req_id, None)
            if fut is None:
                continue
            if ok:
                fut.set_result(payload)
            else:
                fut.set_exception(payload)
        self._proc.join(1)  # reap it, for a meaningful exit code
        reason = (
            f"worker for {self.device} exited (code {self._proc.exitcode}); "
            "its contexts are lost"
        )
        if self._dead is None:
            logger.error(reason)
        self._fail_pending(reason)

    def _fail_pending(self, reason: str) -> None:
        with self._state_lock:
            if self._dead is None:
                self._dead = reason
            pending, self._pending = self._pending, {}
        for fut in pending.values():
            if not fut.done():
                fut.set_exception(WorkerDiedError(reason))

    def _apply(self, snap: Snapshot) -> None:
        with self._state_lock:
            if self._snapshot is None or snap.seq > self._snapshot.seq:
                self._snapshot = snap

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        fut: Future = Future()
        with self._state_lock:
            if self._dead is not None:
                raise WorkerDiedError(self._dead)
            req_id = next(self._ids)
            self._pending[req_id] = fut
        try:
            with self._send_lock:
                self._conn.send((req_id, method, args, kwargs))
        except (OSError, ValueError) as e:
            with self._state_lock:
                self._pending.pop(req_id, None)
            raise WorkerDiedError(f"worker for {self.device} is gone: {e}") from e
        return fut.result()

    # ---- Shard: lock-free reads from the latest snapshot -------------

    def _snap(self) -> Snapshot:
        snap = self._snapshot
        assert snap is not None, "wait_ready() first"
        return snap

    @property
    def backend_names(self) -> list[str]:
        return list(self._backend_names)

    @property
    def default_backend(self) -> str:
        return self._default_backend

    def resolve_backend_name(self, backend: str | None) -> str:
        from tabctx.errors import InvalidInputError

        name = backend or self._default_backend
        if name not in self._backend_names:
            raise InvalidInputError(
                f"unknown backend {name!r}; this deployment serves "
                f"{', '.join(self._backend_names)}"
            )
        return name

    def holds(self, dataset_id: str) -> bool:
        return self.alive and dataset_id in self._snap().resident

    def has(self, dataset_id: str) -> bool:
        return self.alive and dataset_id in self._snap().known

    def stats_nowait(self) -> EngineStats:
        return self._snap().stats

    def admission_headroom_bytes(self, backend: str | None = None) -> int:
        # A dead worker offers nothing, so placement never picks it.
        if not self.alive:
            return 0
        return self._snap().headroom[backend or self._default_backend]

    # ---- Shard: calls ------------------------------------------------

    def estimate_fit_bytes(
        self, n_train: int, n_features: int, backend: str | None = None
    ) -> int:
        if not self.alive:
            return 1 << 62  # never fits: placement skips a dead worker
        return self._call("estimate_fit_bytes", n_train, n_features, backend)

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        task: Task = "classification",
        dataset_id: str | None = None,
        backend: str | None = None,
        feature_names: list[str] | None = None,
    ) -> str:
        return self._call(
            "fit",
            X,
            y,
            task=task,
            dataset_id=dataset_id,
            backend=backend,
            feature_names=feature_names,
        )

    def predict(
        self, dataset_id: str, X_test: ArrayLike, return_proba: bool = False
    ) -> PredictOutcome:
        return self._call("predict", dataset_id, X_test, return_proba=return_proba)

    def feature_names(self, dataset_id: str) -> list[str] | None:
        return self._call("feature_names", dataset_id)

    def evict(self, dataset_id: str) -> None:
        self._call("evict", dataset_id)

    def report(self) -> dict:
        """The worker's spill stats and real GPU memory (see _report)."""
        return self._call("report")
