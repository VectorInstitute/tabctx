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


def device_scope(device: str) -> contextlib.AbstractContextManager[Any]:
    """Make ``device`` torch's current CUDA device for the block; a no-op
    on CPU."""
    if not is_cuda(device):
        return contextlib.nullcontext()
    import torch

    return torch.cuda.device(torch.device(device))
