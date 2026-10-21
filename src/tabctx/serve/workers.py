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

import atexit
import itertools
import logging
import multiprocessing.util  # noqa: F401 -- registers its exit handler first
import os
import pickle
import threading
import time
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from multiprocessing.connection import wait as wait_for_handles
from pathlib import Path
from typing import Any

from tabctx import timing
from tabctx.errors import BackendComputeError
from tabctx.schema import TableSchema
from tabctx.types import ArrayLike, EngineStats, PredictOutcome, Task

logger = logging.getLogger(__name__)

# Calls the worker serves (the Shard surface's blocking part, plus reports).
_METHODS = frozenset(
    {
        "fit",
        "predict",
        "evict",
        "schema",
        "estimate_fit_bytes",
        "report",
    }
)
# Concurrent calls per worker: its engine schedules GPU work itself (see
# engine.py); these only need to queue and answer cheap calls promptly.
_WORKER_THREADS = 16


_NEVER_FITS = 1 << 62


# Live shards, closed at interpreter exit -- before multiprocessing's own
# exit handler terminates the daemon workers, which each reader thread
# would otherwise report as a death. atexit runs handlers in reverse, and
# importing multiprocessing.util above registers mp's first, so ours runs
# before it whatever imported this module.
_LIVE_SHARDS: weakref.WeakSet = weakref.WeakSet()
# How long a worker gets to exit after its pipe closes, before it is killed.
_CLOSE_GRACE_S = 10.0


@atexit.register
def _close_live_shards() -> None:
    ProcessShard.close_all(list(_LIVE_SHARDS))


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
    """The exception itself if it survives a pickle round trip (tabctx
    errors do), else a RuntimeError carrying its type and message. Checking
    only dumps() isn't enough: an exception whose __init__ takes more than
    its message pickles fine and then fails to unpickle in the reader."""
    try:
        pickle.loads(pickle.dumps(error))
        return error
    except Exception:  # noqa: BLE001
        return RuntimeError(f"{type(error).__name__}: {error}")


# Bounded waits for calls that should answer promptly (fits and predicts
# legitimately run for many minutes, so theirs are unbounded).
QUICK_CALL_TIMEOUT_S = 30.0


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


def available_cpus() -> int:
    """CPUs this container may use: its cgroup CPU quota (v2 cpu.max) if
    set, else the CPUs it is scheduled on."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        return os.cpu_count() or 1


def _share_cpus(n_workers: int) -> None:
    """Size this worker's torch.compile pool to its share of the pod's CPUs
    (Inductor sizes it from the host's cores, not the container's): four
    workers compiling at once at startup would otherwise oversubscribe the
    pod and stretch warmup. A deployment's own setting wins."""
    if "TORCHINDUCTOR_COMPILE_THREADS" not in os.environ:
        share = max(1, available_cpus() // max(1, n_workers))
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(share)


def _worker_main(
    conn: Any,
    settings: Any,
    device: str,
    n_devices: int,
    gpu_capacity: int | None,
    host_guard: Any,
    index: int,
) -> None:
    """Entry point of a worker process: build the device's engine, then
    serve calls until the pipe closes."""
    logging.basicConfig(level=logging.INFO)
    from tabctx.serve.factory import build_shard, finish_warmup

    if host_guard is not None:
        host_guard.bind(index)  # so the parent can release it if we die
    _share_cpus(n_devices)
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
    if set(settings.backends) != {"fake"}:
        finish_warmup(settings)  # this process's only shard is warm
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
        # The call's timing breakdown travels back with its reply, so the
        # serving process can report where a request's time went.
        with timing.collect() as timings:
            try:
                if method == "report":
                    result: Any = _report(shard, device)
                else:
                    result = getattr(engine, method)(*args, **kwargs)
                ok, payload = True, result
            except BaseException as e:  # noqa: BLE001 -- returned to the caller
                ok, payload = False, _portable(e)
        reply = (req_id, ok, payload, snapshot(), timings.as_dict())
        with send_lock:
            try:
                conn.send(reply)
            except Exception as e:  # noqa: BLE001 -- e.g. an unpicklable result
                conn.send(
                    (
                        req_id,
                        False,
                        RuntimeError(f"unsendable reply: {e}"),
                        snapshot(),
                        {},
                    )
                )

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
                            {},
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
        index: int = 0,
    ) -> None:
        self.device = device
        self._index = index
        self._host_guard = host_guard
        parent, child = mp_context.Pipe()
        self._conn = parent
        self._proc = mp_context.Process(
            target=_worker_main,
            args=(child, settings, device, n_devices, gpu_capacity, host_guard, index),
            name=f"tabctx-worker-{device}",
            daemon=True,
        )
        self._proc.start()
        child.close()
        _LIVE_SHARDS.add(self)
        self._send_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._reap_lock = threading.Lock()
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
            self._exited(5)  # reap it, for a meaningful exit code
            raise WorkerDiedError(
                f"worker for {self.device} exited during startup "
                f"(exit code {self._exit_code})"
            ) from e
        if isinstance(first, _Failure):
            self._exited(5)
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
                index=i,
            )
            for i, d in enumerate(devices)
        ]
        try:
            deadline = time.monotonic() + ready_timeout_s
            for s in shards:
                s.wait_ready(max(1.0, deadline - time.monotonic()))
        except BaseException:
            cls.close_all(shards)
            raise
        return shards

    @staticmethod
    def close_all(shards: list[ProcessShard]) -> None:
        """Shut workers down together: ask them all to exit, then wait with
        one shared deadline (closing them one by one could take 10 s each,
        past a pod's termination grace period)."""
        for shard in shards:
            shard._request_exit()
        deadline = time.monotonic() + _CLOSE_GRACE_S
        for shard in shards:
            shard._finish_close(max(0.0, deadline - time.monotonic()))

    def close(self) -> None:
        self._request_exit()
        self._finish_close(_CLOSE_GRACE_S)

    def _request_exit(self) -> None:
        with self._state_lock:
            if self._dead is None:
                self._dead = f"worker for {self.device} was shut down"
        with self._send_lock:
            try:
                self._conn.send(None)
            except (OSError, ValueError):
                pass

    def _finish_close(self, grace_s: float) -> None:
        if not self._exited(grace_s):
            self._kill()
            self._exited(5)
        self._conn.close()
        self._release_host_memory()
        self._fail_pending(f"worker for {self.device} was shut down")

    def _exited(self, wait_s: float = 0.0) -> bool:
        """Whether the worker process has exited (waiting up to wait_s for
        it to), reaping it if so. Every wait, liveness check and kill goes
        through _reap_lock, because asking the process itself races:
        is_alive() and join() call waitpid, and when two threads do at once
        (the reader thread and close() both wait on a dying worker) the
        loser finds no child and reports it still running (CPython's
        Popen.poll) -- so close() could see a killed worker as alive and
        keep its host reservations. The wait is on the process sentinel,
        outside the lock, so a health check never waits behind it. Once the
        sentinel is ready, a blocking reap returns at once (a process's fds
        close just before its exit status is available); a sentinel a forked
        grandchild still holds never becomes ready, so the exit is then read
        without blocking."""
        ready = wait_for_handles([self._proc.sentinel], wait_s)
        with self._reap_lock:
            if ready:
                self._proc.join(5)
            return self._proc.exitcode is not None

    def _kill(self) -> None:
        """SIGKILL the worker unless it was reaped (its pid may be reused)."""
        with self._reap_lock:
            if self._proc.exitcode is None:
                self._proc.kill()

    @property
    def _exit_code(self) -> int | None:
        with self._reap_lock:
            return self._proc.exitcode

    def _release_host_memory(self) -> None:
        """Return host memory the (now exited) worker had reserved."""
        if self._host_guard is not None and self._exited():
            self._host_guard.release_owner(self._index)

    @property
    def alive(self) -> bool:
        return self._dead is None and not self._exited()

    @property
    def death_reason(self) -> str | None:
        if self._dead is None and self._exited():
            return f"worker for {self.device} exited (code {self._exit_code})"
        return self._dead

    # ---- transport ---------------------------------------------------

    def _read_loop(self) -> None:
        while True:
            try:
                req_id, ok, payload, snap, timings = self._conn.recv()
            except (EOFError, OSError):
                break
            except Exception:  # noqa: BLE001 -- an unreadable reply
                # Its request can't be identified, so the stream is no longer
                # trustworthy: retire the worker (its calls fail, health
                # checks fail, the serving framework restarts the replica).
                logger.exception("unreadable reply from worker for %s", self.device)
                self._kill()
                break
            self._apply(snap)
            with self._state_lock:
                fut = self._pending.pop(req_id, None)
            if fut is None:
                continue
            if ok:
                fut.set_result((payload, timings))
            else:
                fut.set_exception(payload)
        exited = self._exited(5)  # reap it, for a meaningful exit code
        if self._dead is None and not exited:
            self._kill()  # its pipe is gone: it can't serve, and must
            self._exited(5)  # exit before its reservations are returned
        self._release_host_memory()
        reason = (
            f"worker for {self.device} exited (code {self._exit_code}); "
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

    def _call(
        self, method: str, *args: Any, timeout_s: float | None = None, **kwargs: Any
    ) -> Any:
        fut: Future = Future()
        with self._state_lock:
            if self._dead is not None:
                raise WorkerDiedError(self._dead)
            req_id = next(self._ids)
            self._pending[req_id] = fut
        try:
            with self._send_lock:
                self._conn.send((req_id, method, args, kwargs))
        except (OSError, ValueError, EOFError) as e:
            self._forget(req_id)
            raise WorkerDiedError(f"worker for {self.device} is gone: {e}") from e
        except Exception:  # unpicklable arguments: the caller's problem
            self._forget(req_id)
            raise
        try:
            # Everything between the router and the worker's own phases:
            # pickling, the pipe, the worker's thread pool queue.
            with timing.phase("worker_call"):
                result, timings = fut.result(timeout_s)
        except FutureTimeoutError:
            self._forget(req_id)
            raise TimeoutError(
                f"worker for {self.device} did not answer {method} in {timeout_s}s"
            ) from None
        collector = timing.current()
        if collector is not None and timings:
            collector.merge(timings)
        return result

    def _forget(self, req_id: int) -> None:
        with self._state_lock:
            self._pending.pop(req_id, None)

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

    def residency_nowait(self) -> tuple[frozenset[str], frozenset[str]]:
        if not self.alive:
            return frozenset(), frozenset()
        snap = self._snap()
        return snap.resident, snap.known

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
        # A dead or unresponsive worker never fits: placement skips it rather
        # than failing a fit other devices can take.
        if not self.alive:
            return _NEVER_FITS
        try:
            return self._call(
                "estimate_fit_bytes",
                n_train,
                n_features,
                backend,
                timeout_s=QUICK_CALL_TIMEOUT_S,
            )
        except (WorkerDiedError, TimeoutError):
            return _NEVER_FITS

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        task: Task | None = None,
        dataset_id: str | None = None,
        backend: str | None = None,
        schema: TableSchema | None = None,
    ) -> str:
        return self._call(
            "fit",
            X,
            y,
            task=task,
            dataset_id=dataset_id,
            backend=backend,
            schema=schema,
        )

    def predict(
        self, dataset_id: str, X_test: ArrayLike, return_proba: bool = False
    ) -> PredictOutcome:
        return self._call("predict", dataset_id, X_test, return_proba=return_proba)

    def schema(self, dataset_id: str) -> TableSchema | None:
        return self._call("schema", dataset_id)

    def evict(self, dataset_id: str) -> None:
        self._call("evict", dataset_id)

    def report(self) -> dict:
        """The worker's spill stats and real GPU memory (see _report)."""
        return self._call("report", timeout_s=QUICK_CALL_TIMEOUT_S)
