"""tabctx error types.

Kept HTTP-agnostic on purpose -- a serving layer (e.g. serve/app.py) maps
these to status codes at its own boundary, so the engine stays usable
outside of any particular transport.
"""

from __future__ import annotations


class TabctxError(Exception):
    """Base class for all tabctx errors."""


class InvalidInputError(TabctxError):
    """Raised for malformed input (mismatched X/y lengths, empty tables,
    ragged rows, a predict() feature count that doesn't match the cached
    context) -- checked before any backend/GPU work is attempted. Found via
    load testing: without this, malformed input reached the backend and
    raised an unhandled, untranslated exception (numpy/sklearn shape
    errors), surfacing as a raw 500 that leaked no useful detail to the
    caller -- exactly the failure mode a multi-tenant service can't afford
    (one careless client's bad request shouldn't look like a server bug)."""


class AdmissionRejected(TabctxError):
    """Raised when a request's estimated memory footprint exceeds the
    configured ceiling, before any backend/GPU work is attempted."""


class DatasetNotFoundError(TabctxError):
    """Raised when predict() is called with a dataset_id that isn't
    cached (never fit, evicted, or lost to a replica restart)."""


class CacheCapacityError(TabctxError):
    """Raised when a single context's estimated size alone exceeds the
    cache's total capacity -- no amount of eviction would make room."""


class UploadNotFoundError(TabctxError):
    """Raised when fit()/predict() references an upload_id that isn't
    present on this replica (never uploaded, expired, already consumed,
    or -- in a multi-replica deployment -- uploaded without the session
    affinity header, so it landed on a different replica)."""


class UploadTooLargeError(TabctxError):
    """Raised when a streamed upload exceeds the configured size cap,
    before the excess is written to disk."""


class BackendComputeError(TabctxError):
    """Wraps a backend-level compute failure (e.g. a CUDA OOM that slipped
    past admission control) so callers don't need to import torch to catch
    tabctx errors."""
