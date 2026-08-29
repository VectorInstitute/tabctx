"""Environment-driven construction of a TabctxEngine for serving.

Kept separate from the Ray Serve deployment (app.py) so the wiring is
unit-testable without a cluster, and so alternative hosts (a plain
FastAPI process, a notebook, a future gRPC front) can reuse the exact
same construction logic.

Environment variables:

- ``TABCTX_BACKEND``: ``"tabicl"`` (default; requires torch + tabicl) or
  ``"fake"`` (deterministic stand-in with no GPU/torch dependency --
  what makes multi-replica routing testable on a laptop or in CI).
- ``TABCTX_GPU_MEMORY_FRACTION``: fraction of the calibrated GPU budget
  this engine may use, in (0, 1]; default 1.0. Set below 1.0 when
  several replicas share one physical GPU (e.g. two replicas at
  ``num_gpus: 0.5`` each on a single A100 should each run with 0.45-ish,
  leaving headroom) -- the estimator's admission ceiling and the cache's
  capacity budget both scale by it.
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

BackendKind = Literal["tabicl", "fake"]


@dataclass(frozen=True)
class ServeSettings:
    backend: BackendKind = "tabicl"
    gpu_memory_fraction: float = 1.0

    @classmethod
    def from_env(cls) -> "ServeSettings":
        backend = os.environ.get(BACKEND_ENV_VAR, "tabicl").strip().lower()
        if backend not in ("tabicl", "fake"):
            raise ValueError(
                f"{BACKEND_ENV_VAR}={backend!r} is not a known backend "
                "(expected 'tabicl' or 'fake')"
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
        return cls(backend=backend, gpu_memory_fraction=fraction)


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

    from tabctx.backends.tabicl import TabICLBackend

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return TabICLBackend(device=device), device


def build_engine(settings: ServeSettings | None = None) -> BuiltEngine:
    settings = settings or ServeSettings.from_env()
    backend, device = _build_backend(settings)
    estimator = build_estimator(settings)
    cache = ContextCacheManager(capacity_bytes=estimator.ceiling_bytes())
    engine = TabctxEngine(backend=backend, cache=cache, estimator=estimator)
    return BuiltEngine(
        engine=engine, estimator=estimator, backend=backend, device=device
    )
