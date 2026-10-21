"""tabctx: multi-tenant context caching and serving for tabular
in-context-learning foundation models (TabICL, TabPFN, and similar).

The core idea: these models already cache their encoded training context
internally to speed up repeated predict() calls against the same training
set (both TabICL and TabPFN call this a "KV cache"), but only as a
single-process, single-estimator-object feature. tabctx makes that a real
multi-tenant, evictable, memory-governed resource -- the analog of what
PagedAttention did for per-request LLM KV caches.
"""

from tabctx.cache.manager import CachedContext, ContextCacheManager
from tabctx.engine import TabctxEngine
from tabctx.errors import (
    AdmissionRejected,
    BackendComputeError,
    CacheCapacityError,
    DatasetNotFoundError,
    HostMemoryUnavailableError,
    InvalidInputError,
    TabctxError,
    UploadNotFoundError,
    UploadTooLargeError,
)
from tabctx.types import EngineStats, PredictOutcome, Task

# The scikit-learn estimators need scikit-learn, which tabctx itself
# doesn't: imported on first use (`from tabctx import TabctxClassifier`).
_ESTIMATORS = ("TabctxClassifier", "TabctxRegressor")


def __getattr__(name: str):
    if name in _ESTIMATORS:
        try:
            from tabctx import estimators
        except ImportError as e:
            raise ImportError(
                f"{name} needs scikit-learn and pyarrow: pip install 'tabctx[client]'"
            ) from e
        return getattr(estimators, name)
    raise AttributeError(f"module 'tabctx' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *_ESTIMATORS])


__all__ = [
    "AdmissionRejected",
    "BackendComputeError",
    "CacheCapacityError",
    "CachedContext",
    "ContextCacheManager",
    "DatasetNotFoundError",
    "EngineStats",
    "HostMemoryUnavailableError",
    "InvalidInputError",
    "PredictOutcome",
    "TabctxEngine",
    "TabctxError",
    "Task",
    "UploadNotFoundError",
    "UploadTooLargeError",
]
