"""tabctx error types.

Kept HTTP-agnostic on purpose -- a serving layer (e.g. serve/app.py) maps
these to status codes at its own boundary, so the engine stays usable
outside of any particular transport.
"""

from __future__ import annotations


class TabctxError(Exception):
    """Base class for all tabctx errors."""


class AdmissionRejected(TabctxError):
    """Raised when a request's estimated memory footprint exceeds the
    configured ceiling, before any backend/GPU work is attempted."""


class DatasetNotFoundError(TabctxError):
    """Raised when predict() is called with a dataset_id that isn't
    cached (never fit, evicted, or lost to a replica restart)."""


class CacheCapacityError(TabctxError):
    """Raised when a single context's estimated size alone exceeds the
    cache's total capacity -- no amount of eviction would make room."""


class BackendComputeError(TabctxError):
    """Wraps a backend-level compute failure (e.g. a CUDA OOM that slipped
    past admission control) so callers don't need to import torch to catch
    tabctx errors."""
