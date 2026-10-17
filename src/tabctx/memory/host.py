"""Host (CPU) memory admission: the limit GPU admission doesn't see.

Large TabICL fits offload stage outputs to pinned host memory while they
run. Measured on an A100 pod (benchmarks/bench_host_memory.py,
benchmarks/baselines/host-tabicl-a100.jsonl): fits up to ~5M cells grow
host RSS by < 0.5 GiB, but 10-15M-cell fits (20,000 x 500, 100,000 x 100)
peak at ~46 GiB of host memory -- transient, returned afterwards (see
TabICLBackend._release_host_memory). One such fit fits a 64 GiB pod; two
at once, on two GPUs of one replica, get it OOM-killed with every tenant's
context.

HostMemoryGuard lets at most budget // large_fit_bytes large fits run at
once, process-wide (shared by every device's engine). A large fit waits
for a slot instead of exhausting the pod; small fits never wait. "Large"
is a cell count below which no offload was ever measured.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from pathlib import Path

GiB = 1024**3
# Calibrated on an A100-40GB (see module docstring).
LARGE_FIT_CELLS = 3_000_000
LARGE_FIT_HOST_BYTES = 48 * GiB


def detect_host_budget_bytes(fraction: float = 0.85) -> int | None:
    """This process's memory limit (cgroup v2/v1, else physical RAM) times
    fraction, or None when it can't be read."""
    for path in (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ):
        try:
            raw = Path(path).read_text().strip()
        except OSError:
            continue
        if raw.isdigit() and int(raw) < 1 << 60:
            return int(int(raw) * fraction)
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(int(line.split()[1]) * 1024 * fraction)
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
        self._large_fit_cells = large_fit_cells
        self._semaphore = threading.BoundedSemaphore(self.slots)
        self._waiting = 0
        self._active = 0
        self._lock = threading.Lock()

    def is_large(self, n_train: int, n_features: int) -> bool:
        return n_train * n_features >= self._large_fit_cells

    @contextlib.contextmanager
    def reserve(self, n_train: int, n_features: int) -> Iterator[None]:
        """Hold a large-fit slot for the duration (no-op for small fits)."""
        if not self.is_large(n_train, n_features):
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
