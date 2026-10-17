"""Host (CPU) memory admission: the limit GPU admission doesn't see.

Large TabICL fits offload stage outputs to pinned host memory while they
run. Measured on an A100 pod (benchmarks/bench_host_memory.py,
benchmarks/baselines/host-tabicl-a100.jsonl): fits up to ~5M cells grow
host RSS by < 0.5 GiB, but 10-15M-cell fits (20,000 x 500, 100,000 x 100)
peak at ~46 GiB of host memory -- transient, returned afterwards (see
TabICLBackend._release_host_memory). One such fit fits a 64 GiB pod; two
at once, on two GPUs of one replica, get it OOM-killed with every tenant's
context.

HostMemoryGuard lets at most budget // large_fit_bytes host-offloading
calls run at once, process-wide (shared by every device's engine). Which
calls may offload is the backend's call (TabICLBackend.may_offload mirrors
tabicl's trigger: a stage output above half the device's currently free
GPU memory); backends without that hint use a cell-count threshold. A
large call waits for a slot instead of exhausting the pod; small ones
never wait. Each slot's call should budget at most 1/slots of the host
headroom for itself (see TabICLBackend's cpu_safety_factor), so
concurrent calls can't all claim the same free memory.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Iterator
from pathlib import Path

GiB = 1024**3
logger = logging.getLogger(__name__)
# Calibrated on an A100-40GB (see module docstring).
LARGE_FIT_CELLS = 3_000_000
LARGE_FIT_HOST_BYTES = 48 * GiB


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
    def __init__(
        self,
        budget_bytes: int,
        large_fit_cells: int = LARGE_FIT_CELLS,
        large_fit_bytes: int = LARGE_FIT_HOST_BYTES,
    ) -> None:
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
        self._semaphore = threading.BoundedSemaphore(self.slots)
        self._waiting = 0
        self._active = 0
        self._lock = threading.Lock()

    def is_large(self, n_rows: int, n_features: int) -> bool:
        """The fallback test for backends that can't say whether a call may
        offload: a cell count below which no offload was ever measured."""
        return n_rows * n_features >= self._large_fit_cells

    @contextlib.contextmanager
    def reserve(self, large: bool) -> Iterator[None]:
        """Hold a large-call slot for the duration (no-op when not large)."""
        if not large:
            yield
            return
        with self._lock:
            self._waiting += 1
        self._semaphore.acquire()
        with self._lock:
            self._waiting -= 1
            self._active += 1
        try:
            yield
        finally:
            with self._lock:
                self._active -= 1
            self._semaphore.release()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "large_fit_slots": self.slots,
                "large_fits_active": self._active,
                "large_fits_waiting": self._waiting,
            }
