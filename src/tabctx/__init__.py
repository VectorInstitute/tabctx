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
    TabctxError,
)
from tabctx.types import EngineStats, PredictOutcome, Task

__all__ = [
    "AdmissionRejected",
    "BackendComputeError",
    "CacheCapacityError",
    "CachedContext",
    "ContextCacheManager",
    "DatasetNotFoundError",
    "EngineStats",
    "PredictOutcome",
    "TabctxEngine",
    "TabctxError",
    "Task",
]
