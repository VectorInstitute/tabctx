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
cell when it may offload, TabPFN ~0.6 KB per cell); backends without an
estimate count as one large fit above a cell threshold. A call waits
until its bytes fit beside those already admitted (a call bigger than the
whole budget runs alone, as the only one); calls estimated at 0 never
wait. Calls that do offload budget their own staging against ``slots``,
the number of large fits the budget holds (see TabICLBackend's
cpu_safety_factor), so concurrent ones can't all claim the same free
memory.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

GiB = 1024**3
logger = logging.getLogger(__name__)
# Calibrated on an A100-40GB (see module docstring): the fallback for
# backends without a host_bytes_estimate.
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
        mp_context: Any | None = None,
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
        if mp_context is None:
            self._cond: Any = threading.Condition()
            self._counts: Any = [0, 0, 0]  # reserved bytes, active, waiting
        else:
            self._cond = mp_context.Condition(mp_context.Lock())
            self._counts = mp_context.Array("q", 3, lock=False)

    def is_large(self, n_rows: int, n_features: int) -> bool:
        """The fallback test for backends that can't estimate their host
        use: a cell count below which no offload was ever measured."""
        return n_rows * n_features >= self._large_fit_cells

    @contextlib.contextmanager
    def reserve(self, nbytes: int) -> Iterator[None]:
        """Hold `nbytes` of the host budget for the duration (no-op for 0)."""
        if nbytes <= 0:
            yield
            return
        counts = self._counts
        with self._cond:
            counts[2] += 1
            self._cond.wait_for(
                lambda: counts[0] == 0 or counts[0] + nbytes <= self.budget_bytes
            )
            counts[2] -= 1
            counts[0] += nbytes
            counts[1] += 1
        try:
            yield
        finally:
            with self._cond:
                counts[0] -= nbytes
                counts[1] -= 1
                self._cond.notify_all()

    def stats(self) -> dict[str, int]:
        with self._cond:
            return {
                "budget_bytes": self.budget_bytes,
                "reserved_bytes": self._counts[0],
                "calls_active": self._counts[1],
                "calls_waiting": self._counts[2],
                "large_fit_slots": self.slots,
            }
