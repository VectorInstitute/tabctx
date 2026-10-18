"""Host (CPU) memory admission: the limit GPU admission doesn't see.

Large TabICL fits offload stage outputs to host memory while they run.
Measured on an A100 pod (benchmarks/bench_host_memory.py,
benchmarks/baselines/host-tabicl-a100.jsonl): fits up to ~5M cells grow
host RSS by < 0.5 GiB, but 10M-cell fits that offload (20,000 x 500,
100,000 x 100) peak at ~46 GiB -- transient, returned afterwards (see
TabICLBackend._release_host_memory). TabPFN stays small: at most 6.7 GiB
at 30M cells (host-tabpfn-a100.jsonl). One huge fit fits a 64 GiB pod;
two at once get it OOM-killed with every tenant's context.

HostMemoryGuard admits calls by their estimated host bytes against one
budget for the whole process -- or, built with mp_context, for every
per-GPU worker process at once (host RAM is shared by the node). Each
backend estimates its own calls (host_bytes_estimate: TabICL ~5 KB per
cell when it offloads, TabPFN ~0.6 KB per cell); backends without an
estimate count as one large fit above a cell threshold.

Admission rules:
- Calls under ``min_guarded_bytes`` (1 GiB, inside the reserve that
  detect_host_budget_bytes already holds back) never wait, so small
  predicts and fits stay lock-free however busy the guard is.
- Others are admitted first come, first served: a call waits until it is
  the oldest waiter AND its bytes fit beside those already admitted (a
  call bigger than the whole budget runs alone). No barging, so a large
  fit can't be starved by a stream of modest ones.
- At most ``slots`` calls that offload run at once: the number of large
  fits the budget holds. Each offloading call budgets its own staging
  as a 1/slots share of free host memory (TabICLBackend's
  cpu_safety_factor), so concurrent ones can't all claim the same memory.

Robustness. Reservations are kept per owner (a worker process, or 0
in-process), so when a worker dies holding bytes -- the host OOM kill
this guard exists to prevent -- its parent returns them (release_owner)
instead of leaking them until the replica restarts. The shared lock is
taken with a timeout: a worker killed while holding it (a window of
microseconds) makes calls fail with a retryable HostMemoryUnavailableError
and /readyz report unhealthy, never hang.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tabctx.errors import HostMemoryUnavailableError

GiB = 1024**3
logger = logging.getLogger(__name__)
# Calibrated on an A100-40GB (see module docstring): the fallback for
# backends without a host_bytes_estimate.
LARGE_FIT_CELLS = 3_000_000
LARGE_FIT_HOST_BYTES = 48 * GiB
# Calls below this never wait: detect_host_budget_bytes holds back a larger
# reserve (8 GiB) for the rest of serving.
MIN_GUARDED_BYTES = 1 * GiB
LOCK_TIMEOUT_S = 10.0
# Waiters re-check at least this often (a dead owner's release, a timeout).
_POLL_S = 1.0
# Per owner: reserved bytes, active calls, offloading calls, then queue
# slots for waiting calls (a worker serves at most 16 calls at once).
_QUEUE_SLOTS = 32
_OWNER_BLOCK = 3 + _QUEUE_SLOTS


CGROUP_LIMIT_PATHS = (
    "/sys/fs/cgroup/memory.max",
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",
)


def detect_host_budget_bytes(
    fraction: float = 0.85,
    reserve_bytes: int = 8 * GiB,
    limit_paths: tuple[str, ...] = CGROUP_LIMIT_PATHS,
    meminfo_path: str = "/proc/meminfo",
    status_path: str = "/proc/self/status",
) -> int | None:
    """Host memory available to large calls: this process's memory limit
    (cgroup v2/v1, else physical RAM) x fraction, minus what the process
    already holds (CUDA contexts, model weights) and a reserve for the
    rest of serving (uploads, contexts' host state). None when the limit
    can't be read."""
    limit = _read_limit(limit_paths, meminfo_path)
    if limit is None:
        return None
    return max(0, int(limit * fraction) - _rss_bytes(status_path) - reserve_bytes)


def _rss_bytes(status_path: str) -> int:
    try:
        for line in Path(status_path).read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _read_limit(limit_paths: tuple[str, ...], meminfo_path: str) -> int | None:
    for path in limit_paths:
        try:
            raw = Path(path).read_text().strip()
        except OSError:
            continue
        if raw.isdigit() and int(raw) < 1 << 60:
            return int(raw)
    try:
        for line in Path(meminfo_path).read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


class HostMemoryGuard:
    """Node- or process-wide host memory admission (see module docstring).

    Shared state (a plain list in-process, an mp Array across worker
    processes): ``[next_ticket]`` then, per owner, ``[reserved bytes,
    active calls, offloading calls, queue slots...]``. A waiting call
    parks its ticket in one of its owner's queue slots; the oldest parked
    ticket is the head of the queue.
    """

    def __init__(
        self,
        budget_bytes: int,
        large_fit_cells: int = LARGE_FIT_CELLS,
        large_fit_bytes: int = LARGE_FIT_HOST_BYTES,
        mp_context: Any | None = None,
        owners: int = 1,
        min_guarded_bytes: int = MIN_GUARDED_BYTES,
        lock_timeout_s: float = LOCK_TIMEOUT_S,
    ) -> None:
        self.budget_bytes = budget_bytes
        self.large_fit_bytes = large_fit_bytes
        self.slots = max(1, budget_bytes // large_fit_bytes)
        if budget_bytes < large_fit_bytes:
            logger.warning(
                "host memory budget %.1f GiB is below one large fit's measured "
                "peak (%.0f GiB): large tables may exhaust this pod; give it more "
                "memory or rely on disk offload (TABCTX_OFFLOAD_DIR)",
                budget_bytes / GiB,
                large_fit_bytes / GiB,
            )
        self._large_fit_cells = large_fit_cells
        self._min_guarded = min_guarded_bytes
        self._lock_timeout = lock_timeout_s
        self._owners = max(1, owners)
        self._owner = 0
        size = 1 + self._owners * _OWNER_BLOCK
        if mp_context is None:
            self._cond: Any = threading.Condition()
            self._state: Any = [0] * size
        else:
            self._cond = mp_context.Condition(mp_context.Lock())
            self._state = mp_context.Array("q", size, lock=False)

    def bind(self, owner: int) -> None:
        """Account this process's calls to `owner` (a worker's index), so
        its parent can release them if it dies (release_owner)."""
        if not 0 <= owner < self._owners:
            raise ValueError(f"owner {owner} out of range [0, {self._owners})")
        self._owner = owner

    def is_large(self, n_rows: int, n_features: int) -> bool:
        """The fallback test for backends that can't estimate their host
        use: a cell count below which no offload was ever measured."""
        return n_rows * n_features >= self._large_fit_cells

    @contextlib.contextmanager
    def reserve(self, nbytes: int | None, offloads: bool = False) -> Iterator[None]:
        """Hold `nbytes` of the host budget (and, if the call offloads, one
        of the `slots`) for the duration. Calls under min_guarded_bytes
        that don't offload pass straight through."""
        nbytes = max(0, int(nbytes or 0))
        if nbytes < self._min_guarded and not offloads:
            yield
            return
        state, base = self._state, self._block(self._owner)
        with self._locked():
            ticket = state[0] = state[0] + 1
            slot = None
            try:
                while slot is None:
                    slot = self._park(base, ticket)
                    if slot is None:
                        self._cond.wait(_POLL_S)
                while not (self._head() == ticket and self._admits(nbytes, offloads)):
                    self._cond.wait(_POLL_S)
            finally:
                if slot is not None:
                    state[slot] = 0
                self._cond.notify_all()  # the next waiter is now the head
            state[base] += nbytes
            state[base + 1] += 1
            state[base + 2] += int(offloads)
        try:
            yield
        finally:
            with self._locked():
                state[base] = max(0, state[base] - nbytes)
                state[base + 1] = max(0, state[base + 1] - 1)
                state[base + 2] = max(0, state[base + 2] - int(offloads))
                self._cond.notify_all()

    def release_owner(self, owner: int) -> None:
        """Return everything a dead owner held or queued for."""
        base = self._block(owner)
        try:
            with self._locked():
                for i in range(base, base + _OWNER_BLOCK):
                    self._state[i] = 0
                self._cond.notify_all()
        except HostMemoryUnavailableError:
            logger.error("could not release host reservations of worker %d", owner)

    def stats(self) -> dict[str, int]:
        """Raises HostMemoryUnavailableError if the shared lock is stuck."""
        with self._locked():
            blocks = [self._block(o) for o in range(self._owners)]
            return {
                "budget_bytes": self.budget_bytes,
                "reserved_bytes": sum(self._state[b] for b in blocks),
                "calls_active": sum(self._state[b + 1] for b in blocks),
                "calls_offloading": sum(self._state[b + 2] for b in blocks),
                "calls_waiting": sum(
                    1
                    for b in blocks
                    for i in range(b + 3, b + _OWNER_BLOCK)
                    if self._state[i]
                ),
                "large_fit_slots": self.slots,
            }

    # ---- internals (callers hold the lock) ----------------------------

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        if not self._cond.acquire(True, self._lock_timeout):
            raise HostMemoryUnavailableError(
                f"host memory accounting lock not acquired in {self._lock_timeout}s "
                "(a worker may have died holding it); retry"
            )
        try:
            yield
        finally:
            self._cond.release()

    @staticmethod
    def _block(owner: int) -> int:
        return 1 + owner * _OWNER_BLOCK

    def _park(self, base: int, ticket: int) -> int | None:
        for i in range(base + 3, base + _OWNER_BLOCK):
            if self._state[i] == 0:
                self._state[i] = ticket
                return i
        return None

    def _head(self) -> int:
        return min(
            (
                self._state[i]
                for o in range(self._owners)
                for i in range(self._block(o) + 3, self._block(o) + _OWNER_BLOCK)
                if self._state[i]
            ),
            default=0,
        )

    def _admits(self, nbytes: int, offloads: bool) -> bool:
        blocks = [self._block(o) for o in range(self._owners)]
        reserved = sum(self._state[b] for b in blocks)
        if offloads and sum(self._state[b + 2] for b in blocks) >= self.slots:
            return False
        return reserved == 0 or reserved + nbytes <= self.budget_bytes
