"""Environment-driven construction of a TabctxEngine for serving.

Kept separate from the Ray Serve deployment (app.py) so the wiring is
unit-testable without a cluster, and so alternative hosts (a plain
FastAPI process, a notebook, a future gRPC front) can reuse the exact
same construction logic.

Environment variables:

- ``TABCTX_BACKEND``: ``"tabicl"`` (default; requires torch + tabicl),
  ``"tabpfn"`` (requires torch + tabpfn), or ``"fake"`` (deterministic
  stand-in with no GPU/torch dependency -- what makes multi-replica
  routing testable on a laptop or in CI).
- ``TABCTX_GPU_MEMORY_FRACTION``: fraction of the calibrated GPU budget
  this engine may use, in (0, 1]; default 1.0. Set below 1.0 when
  several replicas share one physical GPU (e.g. two replicas at
  ``num_gpus: 0.5`` each on a single A100 should each run with 0.45-ish,
  leaving headroom) -- the estimator's admission ceiling and the cache's
  capacity budget both scale by it.
- ``TABCTX_KV_CACHE``: ``"kv"`` (default), ``"repr"``, or ``"off"`` --
  TabICL's fit-time context cache mode (see backends/tabicl.py). "kv" is
  fastest per predict; "repr" uses far less cache memory per context;
  "off" re-encodes the training set on every predict (tabicl's own
  default, kept only as an escape hatch).
- ``TABCTX_BATCH_WINDOW_MS``: coalescing window for same-context predict
  batching (see batching.py); default 5. 0 disables coalescing.
- ``TABCTX_MAX_UPLOAD_BYTES`` (default 4GiB) and ``TABCTX_UPLOAD_TTL_S``
  (default 3600): size cap and expiry for the large-table upload path
  (see serve/uploads.py).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from tabctx.backends.base import TabularICLBackend
from tabctx.cache.manager import ContextCacheManager
from tabctx.engine import TabctxEngine
from tabctx.memory import (
    A100_40GB_TABICL_CALIBRATION,
    AdaptiveMemoryEstimator,
    PowerLawMemoryEstimator,
)
from tabctx.memory.estimator import (
    DEFAULT_GPU_CAPACITY_BYTES,
    DEFAULT_HARD_CEILING_BYTES,
    MemoryEstimator,
)

BACKEND_ENV_VAR = "TABCTX_BACKEND"
GPU_MEMORY_FRACTION_ENV_VAR = "TABCTX_GPU_MEMORY_FRACTION"
KV_CACHE_ENV_VAR = "TABCTX_KV_CACHE"
BATCH_WINDOW_MS_ENV_VAR = "TABCTX_BATCH_WINDOW_MS"
MAX_UPLOAD_BYTES_ENV_VAR = "TABCTX_MAX_UPLOAD_BYTES"
UPLOAD_TTL_S_ENV_VAR = "TABCTX_UPLOAD_TTL_S"

BackendKind = Literal["tabicl", "tabpfn", "fake"]
KvCacheMode = Literal["kv", "repr", "off"]


@dataclass(frozen=True)
class ServeSettings:
    backend: BackendKind = "tabicl"
    gpu_memory_fraction: float = 1.0
    kv_cache: KvCacheMode = "kv"
    batch_window_ms: float = 5.0
    max_upload_bytes: int = 4 * 1024**3
    upload_ttl_s: float = 3600.0

    @classmethod
    def from_env(cls) -> "ServeSettings":
        backend = os.environ.get(BACKEND_ENV_VAR, "tabicl").strip().lower()
        if backend not in ("tabicl", "tabpfn", "fake"):
            raise ValueError(
                f"{BACKEND_ENV_VAR}={backend!r} is not a known backend "
                "(expected 'tabicl', 'tabpfn', or 'fake')"
            )
        raw_fraction = os.environ.get(GPU_MEMORY_FRACTION_ENV_VAR, "1.0")
        try:
            fraction = float(raw_fraction)
        except ValueError as e:
            raise ValueError(
                f"{GPU_MEMORY_FRACTION_ENV_VAR}={raw_fraction!r} is not a float"
            ) from e
        if not (0.0 < fraction <= 1.0):
            raise ValueError(
                f"{GPU_MEMORY_FRACTION_ENV_VAR} must be in (0, 1], got {fraction}"
            )
        kv_cache = os.environ.get(KV_CACHE_ENV_VAR, "kv").strip().lower()
        if kv_cache not in ("kv", "repr", "off"):
            raise ValueError(
                f"{KV_CACHE_ENV_VAR}={kv_cache!r} is not a known mode "
                "(expected 'kv', 'repr', or 'off')"
            )
        raw_window = os.environ.get(BATCH_WINDOW_MS_ENV_VAR, "5")
        try:
            batch_window_ms = float(raw_window)
        except ValueError as e:
            raise ValueError(
                f"{BATCH_WINDOW_MS_ENV_VAR}={raw_window!r} is not a float"
            ) from e
        if batch_window_ms < 0:
            raise ValueError(
                f"{BATCH_WINDOW_MS_ENV_VAR} must be >= 0, got {batch_window_ms}"
            )
        try:
            max_upload_bytes = int(os.environ.get(MAX_UPLOAD_BYTES_ENV_VAR, str(4 * 1024**3)))
            upload_ttl_s = float(os.environ.get(UPLOAD_TTL_S_ENV_VAR, "3600"))
        except ValueError as e:
            raise ValueError(
                f"{MAX_UPLOAD_BYTES_ENV_VAR}/{UPLOAD_TTL_S_ENV_VAR} must be numeric"
            ) from e
        if max_upload_bytes <= 0 or upload_ttl_s <= 0:
            raise ValueError(
                f"{MAX_UPLOAD_BYTES_ENV_VAR} and {UPLOAD_TTL_S_ENV_VAR} must be positive"
            )
        return cls(
            backend=backend,
            gpu_memory_fraction=fraction,
            kv_cache=kv_cache,
            batch_window_ms=batch_window_ms,
            max_upload_bytes=max_upload_bytes,
            upload_ttl_s=upload_ttl_s,
        )


@dataclass(frozen=True)
class BuiltEngine:
    engine: TabctxEngine
    estimator: MemoryEstimator
    backend: TabularICLBackend
    device: str


def build_estimator(settings: ServeSettings) -> AdaptiveMemoryEstimator:
    """Adaptive estimator over the calibrated static fallback, with both
    ceilings scaled by the configured GPU-memory fraction."""
    fraction = settings.gpu_memory_fraction
    fallback = PowerLawMemoryEstimator(
        A100_40GB_TABICL_CALIBRATION,
        hard_ceiling_bytes=int(DEFAULT_HARD_CEILING_BYTES * fraction),
        gpu_capacity_bytes=int(DEFAULT_GPU_CAPACITY_BYTES * fraction),
    )
    return AdaptiveMemoryEstimator(fallback=fallback)


def _build_backend(settings: ServeSettings) -> tuple[TabularICLBackend, str]:
    """Returns (backend, device). Imports torch/tabicl only on the path
    that needs them, so the fake backend runs with core deps alone."""
    if settings.backend == "fake":
        from tabctx.backends.fake import FakeBackend

        return FakeBackend(), "cpu (fake backend)"

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if settings.backend == "tabpfn":
        from tabctx.backends.tabpfn import TabPFNBackend

        # TABCTX_KV_CACHE maps onto TabPFN's fit_mode (see backends/tabpfn.py).
        return TabPFNBackend(device=device, cache_mode=settings.kv_cache), device

    from tabctx.backends.tabicl import TabICLBackend

    kv_cache: bool | str = False if settings.kv_cache == "off" else settings.kv_cache
    return TabICLBackend(device=device, kv_cache=kv_cache), device


def build_engine(settings: ServeSettings | None = None) -> BuiltEngine:
    settings = settings or ServeSettings.from_env()
    backend, device = _build_backend(settings)
    estimator = build_estimator(settings)
    cache = ContextCacheManager(capacity_bytes=estimator.ceiling_bytes())
    engine = TabctxEngine(backend=backend, cache=cache, estimator=estimator)
    return BuiltEngine(
        engine=engine, estimator=estimator, backend=backend, device=device
    )
