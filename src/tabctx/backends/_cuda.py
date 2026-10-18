"""Device helpers shared by the torch backends.

A backend bound to e.g. ``cuda:2`` must measure and allocate on THAT
device: torch's memory counters default to the *current* device, and
library internals (tabicl/tabpfn) allocate scratch tensors there too.
Multi-GPU replicas (see tabctx/multidevice.py) run one backend instance
per device, so every GPU call is wrapped in ``device_scope``.
"""

from __future__ import annotations

import contextlib
from typing import Any


def is_cuda(device: str) -> bool:
    return device == "cuda" or device.startswith("cuda:")


def ensure_initialized(device: str) -> None:
    """Initialize CUDA for a backend bound to a GPU. The memory-stat calls
    that take an explicit device (reset_peak_memory_stats("cuda:1"), ...)
    raise "Invalid device argument" before CUDA is initialized -- unlike
    their no-argument forms, which initialize it via current_device(). A
    replica's first fit would otherwise fail."""
    if is_cuda(device):
        import torch

        torch.cuda.init()


def release_cached_device_memory() -> None:
    """Return the CUDA caching allocator's unused blocks (a retry after an
    OOM then sees an unfragmented device). A no-op without torch or CUDA."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def device_available_bytes(device: str) -> int | None:
    """Device memory a new allocation can use now: free on the device plus
    what torch's caching allocator holds reserved but unallocated (reused
    before asking the driver). None off CUDA."""
    if not is_cuda(device):
        return None
    import torch

    d = torch.device(device)
    free, _ = torch.cuda.mem_get_info(d)
    cached = torch.cuda.memory_reserved(d) - torch.cuda.memory_allocated(d)
    return int(free + max(0, cached))


def device_scope(device: str) -> contextlib.AbstractContextManager[Any]:
    """Make ``device`` torch's current CUDA device for the block; a no-op
    on CPU."""
    if not is_cuda(device):
        return contextlib.nullcontext()
    import torch

    return torch.cuda.device(torch.device(device))


class QuiescedMeasurement:
    """Mixin for backends that measure a fit's footprint as a delta of
    torch's allocator counters (memory_allocated before vs after).

    The engine lets light predicts run beside a fit on the same device
    (engine.py); a predict in flight at either reading would skew the
    delta (in flight at the start: the context is understated -- the
    unsafe direction). The engine lends its predict lock via
    set_measurement_quiesce; readings taken under it see no predict's
    temporaries (predicts free them before releasing the lock). Peaks
    still include any predict that overlapped the fit: overstated,
    which is the safe direction."""

    _quiesce: Any = None

    def set_measurement_quiesce(self, lock: Any) -> None:
        self._quiesce = lock

    def quiesced(self) -> contextlib.AbstractContextManager[Any]:
        return self._quiesce if self._quiesce is not None else contextlib.nullcontext()
