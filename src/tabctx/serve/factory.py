"""Environment-driven construction of a TabctxEngine for serving.

Kept separate from the Ray Serve deployment (app.py) so the wiring is
unit-testable without a cluster, and so alternative hosts (a plain
FastAPI process, a notebook, a future gRPC front) can reuse the exact
same construction logic.

Environment variables:

- ``TABCTX_BACKEND``: comma-separated list of models to serve behind the
  one endpoint (requests pick one via their ``model`` field; the FIRST
  listed is the default). Known kinds: ``"tabicl"`` (default; requires
  torch + tabicl), ``"tabpfn"`` (requires torch + tabpfn), ``"fake"``
  (deterministic stand-in, no GPU/torch -- what makes serving testable
  on a laptop or in CI). E.g. ``TABCTX_BACKEND=tabicl,tabpfn`` serves
  both models over one shared cache and GPU budget.
- ``TABCTX_GPU_MEMORY_FRACTION``: fraction of the GPU budget this engine
  may use, in (0, 1]; default 1.0. Set below 1.0 when several replicas
  share one physical GPU (e.g. two replicas at ``num_gpus: 0.5`` each on
  a single A100 should each run with 0.45-ish, leaving headroom) -- the
  estimator's admission ceiling and the cache's capacity budget both
  scale by it. The budget itself is the visible CUDA device's real
  total memory when torch can see one (an L4's 24GB is not an A100's
  40GB, and admission must not believe otherwise); the A100-40GB
  calibration defaults apply only when no device is visible.
- ``TABCTX_KV_CACHE``: ``"kv"`` (default), ``"repr"``, or ``"off"`` --
  TabICL's fit-time context cache mode (see backends/tabicl.py). "kv" is
  fastest per predict; "repr" uses far less cache memory per context;
  "off" re-encodes the training set on every predict (tabicl's own
  default, kept only as an escape hatch).
- ``TABCTX_KV_PRECISION``: ``"int8"`` (default), ``"fp16"`` or ``"native"``
  -- storage of TabICL's ICL key/value cache, which dominates a TabICL
  context (see backends/tabicl_kv.py). ``"int8"`` about halves resident
  memory per context, so it doubles the longest table and the contexts per
  GPU. Measured on an A100 over 7 datasets
  (benchmarks/baselines/kv-precision-tabicl-a100.jsonl): identical labels
  and metrics, max probability drift 0.009, at up to ~1.4x warm predict
  latency on large contexts. TabPFN already stores its cache in int8.
- ``TABCTX_OFFLOAD_DIR``: where TabICL may spill stage outputs too large
  for GPU and host memory (memory-mapped files), instead of exhausting the
  pod's RAM; default ``<TABCTX_SPILL_DIR>/offload``, else a temporary
  directory. One subdirectory per device.
- ``TABCTX_BATCH_WINDOW_MS``: coalescing window for same-context predict
  batching (see batching.py); default 5. 0 disables coalescing.
- ``TABCTX_MAX_UPLOAD_BYTES`` (default 4GiB) and ``TABCTX_UPLOAD_TTL_S``
  (default 3600): size cap and expiry for the large-table upload path
  (see serve/uploads.py).
- ``TABCTX_SPILL_DIR`` (default unset = spillover off) and
  ``TABCTX_SPILL_CAPACITY_BYTES`` (default 50GiB): disk spillover tier
  for capacity-evicted contexts (see cache/spill.py). With several
  devices, each gets its own subdirectory (``cuda_0/``, ...) and an equal
  share of the capacity (the setting is the total for the replica).
- ``TABCTX_DEVICES``: ``"auto"`` (default: every visible CUDA device, or
  the CPU when there is none) or a comma-separated list such as
  ``cuda:0,cuda:1``. Two or more devices build a MultiDeviceEngine (see
  tabctx/multidevice.py): one engine per device; a fit goes where it fits
  without eviction on the least active device, else to the device with
  the most headroom. ``TABCTX_GPU_MEMORY_FRACTION`` applies to each
  device. The replica only sees the GPUs its process is given: under Ray
  Serve that is ``ray_actor_options.num_gpus`` (or, for a ``num_gpus: 0``
  actor on a GPU pod, ``RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1``).
- ``TABCTX_WORKER_PROCESSES``: ``1`` (default) runs each device's engine
  in its own worker process (serve/workers.py) when there are several
  devices; ``0`` runs them as threads of the serving process. Fits are
  host-bound, so threads contend for the GIL: measured on 4x A100, four
  concurrent fits took 2x as long each from threads, single-GPU time
  from processes.
"""

from __future__ import annotations

import importlib
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tabctx.backends.base import TabularICLBackend
from tabctx.backends.kv_precision import KV_PRECISIONS, KVPrecision
from tabctx.cache.manager import ContextCacheManager
from tabctx.cache.spill import DEFAULT_SPILL_CAPACITY_BYTES, DiskSpillStore
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
from tabctx.memory.host import (
    QUEUE_SLOTS,
    HostMemoryGuard,
    detect_host_budget_bytes,
)
from tabctx.memory.requirement import ResidentRateModel, WorkingFloorModel
from tabctx.multidevice import MultiDeviceEngine
from tabctx.serve.uploads import DEFAULT_MAX_UPLOAD_BYTES, DEFAULT_TTL_S

BACKEND_ENV_VAR = "TABCTX_BACKEND"
GPU_MEMORY_FRACTION_ENV_VAR = "TABCTX_GPU_MEMORY_FRACTION"
KV_CACHE_ENV_VAR = "TABCTX_KV_CACHE"
KV_PRECISION_ENV_VAR = "TABCTX_KV_PRECISION"
OFFLOAD_DIR_ENV_VAR = "TABCTX_OFFLOAD_DIR"
BATCH_WINDOW_MS_ENV_VAR = "TABCTX_BATCH_WINDOW_MS"
MAX_UPLOAD_BYTES_ENV_VAR = "TABCTX_MAX_UPLOAD_BYTES"
SPILL_DIR_ENV_VAR = "TABCTX_SPILL_DIR"
SPILL_CAPACITY_ENV_VAR = "TABCTX_SPILL_CAPACITY_BYTES"
UPLOAD_TTL_S_ENV_VAR = "TABCTX_UPLOAD_TTL_S"
DEVICES_ENV_VAR = "TABCTX_DEVICES"
WORKER_PROCESSES_ENV_VAR = "TABCTX_WORKER_PROCESSES"
COMPILE_ENV_VAR = "TABCTX_COMPILE"
FAST_PREPROCESSING_ENV_VAR = "TABCTX_FAST_PREPROCESSING"

logger = logging.getLogger(__name__)

BackendKind = Literal["tabicl", "tabpfn", "fake"]
KvCacheMode = Literal["kv", "repr", "off"]
BACKEND_KINDS: tuple[BackendKind, ...] = ("tabicl", "tabpfn", "fake")
KV_CACHE_MODES: tuple[KvCacheMode, ...] = ("kv", "repr", "off")


@dataclass(frozen=True)
class ServeSettings:
    backends: tuple[BackendKind, ...] = ("tabicl",)
    gpu_memory_fraction: float = 1.0
    kv_cache: KvCacheMode = "kv"
    kv_precision: KVPrecision = "int8"
    batch_window_ms: float = 5.0
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    upload_ttl_s: float = DEFAULT_TTL_S
    spill_dir: str | None = None
    spill_capacity_bytes: int = DEFAULT_SPILL_CAPACITY_BYTES
    # Empty = "auto" (see resolve_devices).
    devices: tuple[str, ...] = ()
    # None = <spill_dir>/offload, else a temporary directory.
    offload_dir: str | None = None
    # Several devices: one worker process per GPU (default) or threads of
    # this process (see _build_process_engine for why processes).
    worker_processes: bool = True
    # torch.compile the models' hot blocks where measured to help (TabICL's
    # attention blocks: backends/tabicl_compile.py).
    compile_models: bool = True
    # Vectorized (GPU) Yeo-Johnson fits in TabICL's preprocessing
    # (backends/yeo_johnson.py).
    fast_preprocessing: bool = True

    @classmethod
    def from_env(cls) -> ServeSettings:
        raw_backends = os.environ.get(BACKEND_ENV_VAR, "tabicl")
        backends = tuple(
            b.strip().lower() for b in raw_backends.split(",") if b.strip()
        )
        if not backends or any(b not in BACKEND_KINDS for b in backends):
            raise ValueError(
                f"{BACKEND_ENV_VAR}={raw_backends!r} must be a comma-separated "
                f"subset of: {', '.join(BACKEND_KINDS)}"
            )
        if len(set(backends)) != len(backends):
            raise ValueError(f"{BACKEND_ENV_VAR} lists a backend twice")
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
        kv_cache = os.environ.get(KV_CACHE_ENV_VAR, cls.kv_cache).strip().lower()
        if kv_cache not in KV_CACHE_MODES:
            raise ValueError(
                f"{KV_CACHE_ENV_VAR}={kv_cache!r} is not a known mode "
                "(expected 'kv', 'repr', or 'off')"
            )
        kv_precision = (
            os.environ.get(KV_PRECISION_ENV_VAR, cls.kv_precision).strip().lower()
        )
        if kv_precision not in KV_PRECISIONS:
            raise ValueError(
                f"{KV_PRECISION_ENV_VAR}={kv_precision!r} must be one of "
                f"{', '.join(KV_PRECISIONS)}"
            )
        raw_window = os.environ.get(BATCH_WINDOW_MS_ENV_VAR, str(cls.batch_window_ms))
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
            max_upload_bytes = int(
                os.environ.get(MAX_UPLOAD_BYTES_ENV_VAR, str(cls.max_upload_bytes))
            )
            upload_ttl_s = float(
                os.environ.get(UPLOAD_TTL_S_ENV_VAR, str(cls.upload_ttl_s))
            )
        except ValueError as e:
            raise ValueError(
                f"{MAX_UPLOAD_BYTES_ENV_VAR}/{UPLOAD_TTL_S_ENV_VAR} must be numeric"
            ) from e
        if max_upload_bytes <= 0 or upload_ttl_s <= 0:
            raise ValueError(
                f"{MAX_UPLOAD_BYTES_ENV_VAR} and {UPLOAD_TTL_S_ENV_VAR} must be positive"
            )
        spill_dir = os.environ.get(SPILL_DIR_ENV_VAR) or None
        try:
            spill_capacity = int(
                os.environ.get(SPILL_CAPACITY_ENV_VAR, str(cls.spill_capacity_bytes))
            )
        except ValueError as e:
            raise ValueError(f"{SPILL_CAPACITY_ENV_VAR} must be an int") from e
        if spill_capacity <= 0:
            raise ValueError(f"{SPILL_CAPACITY_ENV_VAR} must be positive")
        devices = _parse_devices(os.environ.get(DEVICES_ENV_VAR, "auto"))
        offload_dir = os.environ.get(OFFLOAD_DIR_ENV_VAR) or None
        return cls(
            backends=backends,
            gpu_memory_fraction=fraction,
            kv_cache=kv_cache,
            kv_precision=kv_precision,
            batch_window_ms=batch_window_ms,
            max_upload_bytes=max_upload_bytes,
            upload_ttl_s=upload_ttl_s,
            spill_dir=spill_dir,
            spill_capacity_bytes=spill_capacity,
            devices=devices,
            offload_dir=offload_dir,
            worker_processes=_env_flag(WORKER_PROCESSES_ENV_VAR, cls.worker_processes),
            compile_models=_env_flag(COMPILE_ENV_VAR, cls.compile_models),
            fast_preprocessing=_env_flag(
                FAST_PREPROCESSING_ENV_VAR, cls.fast_preprocessing
            ),
        )


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "1" if default else "0").strip().lower()
    if raw not in ("0", "1", "true", "false"):
        raise ValueError(f"{name}={raw!r} must be 0 or 1")
    return raw in ("1", "true")


def _parse_devices(raw: str) -> tuple[str, ...]:
    raw = raw.strip().lower()
    if raw in ("", "auto"):
        return ()
    devices = tuple(d.strip() for d in raw.split(",") if d.strip())
    for d in devices:
        index = d.removeprefix("cuda:")
        if d not in ("cpu", "cuda") and not (d.startswith("cuda:") and index.isdigit()):
            raise ValueError(
                f"{DEVICES_ENV_VAR}={raw!r}: {d!r} is not 'cpu', 'cuda' or 'cuda:<n>'"
            )
    if len(set(devices)) != len(devices):
        raise ValueError(f"{DEVICES_ENV_VAR} lists a device twice")
    if len(devices) > 1:
        if "cpu" in devices:
            raise ValueError(
                f"{DEVICES_ENV_VAR}={raw!r}: several devices must all be GPUs "
                "(a CPU engine would get a GPU budget and an equal share of users)"
            )
        if "cuda" in devices:
            raise ValueError(
                f"{DEVICES_ENV_VAR}={raw!r}: name GPUs by index (cuda:0,cuda:1); "
                "a bare 'cuda' may be the same GPU as one listed by index"
            )
    return devices


@dataclass(frozen=True)
class BuiltEngine:
    engine: TabctxEngine | MultiDeviceEngine
    # Keyed by model/backend name; `default` names the first-listed model.
    estimators: dict[str, MemoryEstimator]
    backend_names: list[str]
    default: str
    device: str
    spill_store: object | None = None
    # Real device memory the budget was derived from (per device; the
    # smallest when several); None = defaults.
    gpu_capacity_bytes: int | None = None
    devices: tuple[str, ...] = ()
    # One per device when there are several in-process (spill_store is
    # then None); worker processes report their own (worker_shards).
    spill_stores: tuple = ()
    host_guard: HostMemoryGuard | None = None
    # Per-GPU worker processes (serve/workers.py), when used.
    worker_shards: tuple = ()

    # Single-model convenience (the common deployment):
    @property
    def estimator(self) -> MemoryEstimator:
        return self.estimators[self.default]

    def close(self) -> None:
        """Stop worker processes, if any."""
        for shard in self.worker_shards:
            shard.close()


def _preloaded_observations(
    kind: BackendKind, kv_cache: KvCacheMode
) -> tuple[tuple, tuple]:
    """Factory-installed calibration grid matching one model + cache mode
    (see memory/calibration_tabicl_a100.py), so admission rests on real
    measurements from the first request onward. Only TabICL-on-A100
    grids exist so far; other configurations start with no preload and
    learn from their own fits."""
    if kind != "tabicl":
        return (), ()
    try:
        from tabctx.memory import calibration_tabicl_a100 as grids
    except ImportError:  # generated module absent (pre-calibration tree)
        return (), ()
    fit_grid = {
        "kv": getattr(grids, "A100_40GB_TABICL_KV_PEAK_GRID", ()),
        "repr": getattr(grids, "A100_40GB_TABICL_REPR_PEAK_GRID", ()),
        "off": getattr(grids, "A100_40GB_TABICL_OFF_PEAK_GRID", ()),
    }[kv_cache]
    predict_grid = {
        "kv": getattr(grids, "A100_40GB_TABICL_KV_PREDICT_PEAK_GRID", ()),
        "repr": getattr(grids, "A100_40GB_TABICL_REPR_PREDICT_PEAK_GRID", ()),
        "off": getattr(grids, "A100_40GB_TABICL_OFF_PREDICT_PEAK_GRID", ()),
    }[kv_cache]
    return fit_grid, predict_grid


# Generated requirement calibrations (benchmarks/gen_requirement_grid.py),
# per (model, cache mode, compiled): module under tabctx.memory and its name
# prefix. Compiled TabICL needs up to ~2 GiB more than eager at some shapes
# (its minimum, not its peak: tabicl sizes batches from free memory), so it
# has its own grid -- per shape the larger of the two measurements.
_REQUIREMENT_CALIBRATIONS: dict[tuple[str, str, bool], tuple[str, str]] = {
    ("tabicl", "kv", False): (
        "calibration_tabicl_a100_requirement",
        "A100_40GB_TABICL_KV",
    ),
    ("tabicl", "kv", True): (
        "calibration_tabicl_compiled_a100_requirement",
        "A100_40GB_TABICL_KV_COMPILED",
    ),
    ("tabpfn", "kv", False): (
        "calibration_tabpfn_a100_requirement",
        "A100_40GB_TABPFN_KV",
    ),
}
# Backends whose compile_models setting changes their memory profile.
_COMPILED_KINDS = frozenset({"tabicl"})
# Models whose per-row resident cache grows with features (see
# ResidentRateModel): TabPFN caches per (row, feature group); TabICL's ICL
# cache is per row only.
_WIDTH_DEPENDENT_RESIDENT = frozenset({"tabpfn"})


def _requirement_calibration(
    kind: BackendKind, kv_cache: KvCacheMode, compiled: bool = False
) -> tuple[WorkingFloorModel | None, tuple, tuple]:
    """Measured working floors + requirements (memory/requirement.py), so
    admission bounds resident + floor instead of an unconstrained peak.
    Configurations without a measured calibration keep peak semantics."""
    compiled = compiled and kind in _COMPILED_KINDS
    entry = _REQUIREMENT_CALIBRATIONS.get((kind, kv_cache, compiled))
    if entry is None:
        return None, (), ()
    module_name, prefix = entry
    try:
        grid = importlib.import_module(f"tabctx.memory.{module_name}")
    except ImportError:  # generated module absent: peak semantics, no preload
        return None, (), ()
    points = getattr(grid, f"{prefix}_FLOOR_POINTS")
    return (
        WorkingFloorModel(points),
        getattr(grid, f"{prefix}_REQUIREMENT_GRID"),
        points,
    )


def detect_gpu_capacity_bytes(devices: tuple[str, ...] = ("cuda",)) -> int | None:
    """Total memory of the given CUDA devices (the smallest, since one
    estimator budgets them all), or None when torch is absent or sees no
    GPU (the fake backend, CPU hosts)."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    cuda = [torch.device(d) for d in devices if d.startswith("cuda")]
    if not cuda:
        return None
    return min(
        int(torch.cuda.get_device_properties(d.index or 0).total_memory) for d in cuda
    )


def _visible_cuda_count() -> int:
    try:
        import torch
    except ImportError:
        return 0
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def measure_device_baseline(device: str) -> int:
    """Device memory in use before any context (call after the backends'
    warmup): total - free, so it includes the CUDA context and allocator
    overhead as well as the model weights. 0 off CUDA."""
    if not device.startswith("cuda"):
        return 0
    try:
        import torch
    except ImportError:
        return 0
    if not torch.cuda.is_available():
        return 0
    free, total = torch.cuda.mem_get_info(torch.device(device))
    return int(total - free)


def resolve_devices(settings: ServeSettings) -> tuple[str, ...]:
    """The devices to build one engine for each. Explicit settings win
    (GPU indices are checked against what this process can see); "auto"
    means every visible GPU. A single auto-detected GPU stays "cuda" (the
    pre-multi-device layout); the fake backend never touches a GPU, so
    auto means one CPU engine for it."""
    if settings.devices:
        if set(settings.backends) != {"fake"}:
            visible = _visible_cuda_count()
            for d in settings.devices:
                index = d.removeprefix("cuda:")
                if (
                    d == "cuda"
                    and visible == 0
                    or index.isdigit()
                    and int(index) >= visible
                ):
                    raise ValueError(
                        f"{DEVICES_ENV_VAR} names {d}, but this process sees "
                        f"{visible} GPU(s)"
                    )
        return settings.devices
    if set(settings.backends) == {"fake"}:
        return ("cpu",)
    n = _visible_cuda_count()
    if n == 0:
        return ("cpu",)
    if n == 1:
        return ("cuda",)
    return tuple(f"cuda:{i}" for i in range(n))


def build_estimator(
    settings: ServeSettings,
    kind: BackendKind | None = None,
    gpu_capacity_bytes: int | None = None,
) -> AdaptiveMemoryEstimator:
    """Adaptive estimator for one model: the calibrated static fallback
    with ceilings scaled by the GPU-memory fraction, plus that model's
    measured calibration grid preloaded (v0.9.0). Each model gets its
    own estimator (they peak differently for the same shape) even though
    all share one device budget.

    gpu_capacity_bytes: the device's real total memory (see
    detect_gpu_capacity_bytes); None means the A100-40GB defaults. The
    cache's hard ceiling keeps its calibrated ratio to device capacity
    (24GB of 40GB), so a smaller card gets a proportionally smaller
    resident budget rather than one that covers the whole device."""
    kind = kind or settings.backends[0]
    fraction = settings.gpu_memory_fraction
    capacity = gpu_capacity_bytes or DEFAULT_GPU_CAPACITY_BYTES
    ceiling_ratio = DEFAULT_HARD_CEILING_BYTES / DEFAULT_GPU_CAPACITY_BYTES
    fallback = PowerLawMemoryEstimator(
        A100_40GB_TABICL_CALIBRATION,
        hard_ceiling_bytes=int(capacity * ceiling_ratio * fraction),
        gpu_capacity_bytes=int(capacity * fraction),
    )
    fit_grid, predict_grid = _preloaded_observations(kind, settings.kv_cache)
    working_floor, requirements, floor_points = _requirement_calibration(
        kind, settings.kv_cache, settings.compile_models
    )
    return AdaptiveMemoryEstimator(
        fallback=fallback,
        # Both are valid upper bounds; the tightest dominating one wins.
        preloaded=tuple(fit_grid) + tuple(requirements),
        preloaded_predict=predict_grid,
        working_floor=working_floor,
        resident_rate=ResidentRateModel(
            floor_points, width_dependent=kind in _WIDTH_DEPENDENT_RESIDENT
        )
        if working_floor
        else None,
    )


def _build_real_backend(
    kind: BackendKind, settings: ServeSettings, device: str, host_slots: int = 1
) -> TabularICLBackend:  # pragma: no cover
    # Requires torch + tabicl/tabpfn (GPU or licensed weights), which --
    # like backends/tabicl.py and backends/tabpfn.py themselves -- are not
    # part of any unit- or integration-test CI extra; see those modules'
    # docstrings. Isolated in its own function so the pragma doesn't also
    # exclude the (fully unit-tested) fake-backend and dispatch logic in
    # _build_backend below.
    if kind == "tabpfn":
        from tabctx.backends.tabpfn import TabPFNBackend

        # TABCTX_KV_CACHE maps onto TabPFN's fit_mode (see backends/tabpfn.py).
        backend: TabularICLBackend = TabPFNBackend(
            device=device, cache_mode=settings.kv_cache
        )
    else:
        from tabctx.backends.tabicl import TabICLBackend

        kv_cache: bool | str = (
            False if settings.kv_cache == "off" else settings.kv_cache
        )
        backend = TabICLBackend(
            device=device,
            kv_cache=kv_cache,
            kv_precision=settings.kv_precision,
            offload_dir=_offload_dir(settings, device),
            # Concurrent offloading calls each get their share of host
            # headroom (see memory/host.py).
            cpu_safety_factor=0.6 / host_slots,
            compile_blocks=settings.compile_models,
            fast_preprocessing=settings.fast_preprocessing,
        )
    # Weights load at replica start, not inside (and charged to) the first
    # tenant's fit.
    backend.warmup()
    return backend


def _offload_dir(settings: ServeSettings, device: str) -> str:
    """Per-device directory for TabICL's disk offload (created)."""
    if settings.offload_dir:
        root = settings.offload_dir
    elif settings.spill_dir:
        root = os.path.join(settings.spill_dir, "offload")
    else:
        root = os.path.join(tempfile.gettempdir(), "tabctx-offload")
    path = os.path.join(root, device.replace(":", "_"))
    os.makedirs(path, exist_ok=True)
    return path


def _build_backend(
    kind: BackendKind, settings: ServeSettings, device: str = "cpu", host_slots: int = 1
) -> TabularICLBackend:
    """One backend bound to `device`. Imports torch/tabicl/tabpfn only on
    the paths that need them, so the fake backend runs with core deps
    alone."""
    if kind == "fake":
        from tabctx.backends.fake import FakeBackend

        return FakeBackend()
    return _build_real_backend(kind, settings, device, host_slots)


def _warn_on_spill_layout(spill_dir: str, multi: bool) -> None:
    """Spill files from the other layout (one device writes to the root,
    several to per-device subdirectories) are invisible to this process:
    never restored, counted or cleaned. Say so instead of silently
    leaking disk."""
    root = Path(spill_dir)
    if not root.is_dir():
        return
    stray = (
        list(root.glob("*.meta.json"))
        if multi
        else [p for p in root.glob("cuda_*/*.meta.json")]
    )
    if stray:
        logger.warning(
            "%d spilled context(s) under %s were written by a %s-device replica "
            "and will not be restored by this one (layout changed); remove them "
            "to reclaim disk",
            len(stray),
            spill_dir,
            "single" if multi else "multi",
        )


def _build_spill_store(
    settings: ServeSettings,
    backends: dict[str, TabularICLBackend],
    subdir: str | None,
    capacity_bytes: int,
) -> DiskSpillStore | None:
    if not settings.spill_dir:
        return None
    # Backends may provide backbone-aware serialization; pickle is the
    # default for those that don't (see cache/spill.py). Restores use the
    # serializers of the device that spilled, so a reloaded context comes
    # back onto that device.
    serializers = {
        name: (b.dumps_payload, b.loads_payload)
        for name, b in backends.items()
        if hasattr(b, "dumps_payload")
    }
    path = os.path.join(settings.spill_dir, subdir) if subdir else settings.spill_dir
    return DiskSpillStore(path, capacity_bytes=capacity_bytes, serializers=serializers)


@dataclass
class Shard:
    """One device's complete engine and the parts the serving layer reports
    on (see build_shard)."""

    engine: TabctxEngine
    backends: dict[str, TabularICLBackend]
    estimators: dict[str, MemoryEstimator]
    spill_store: DiskSpillStore | None


def build_shard(
    settings: ServeSettings,
    device: str,
    *,
    n_devices: int,
    gpu_capacity: int | None,
    host_guard: HostMemoryGuard | None,
    estimators: dict[str, MemoryEstimator] | None = None,
) -> Shard:
    """Everything one device serves from: a backend per model bound to the
    device, one cache (and spill tier) shared by those models, and the
    engine over them. Used in-process and by per-GPU worker processes
    (serve/workers.py) alike.

    estimators: shared across shards of one process (identical cards
    budgeted alike, so observations learned on one hold on the others);
    filled in on first use. A worker process passes none and learns alone.
    """
    estimators = {} if estimators is None else estimators
    fake_only = set(settings.backends) == {"fake"}
    multi = n_devices > 1
    backends: dict[str, TabularICLBackend] = {}
    for kind in settings.backends:
        backend = _build_backend(
            kind, settings, device, host_guard.slots if host_guard else 1
        )
        backends[backend.name] = backend
        if backend.name not in estimators:
            estimators[backend.name] = build_estimator(settings, kind, gpu_capacity)
    # Backends are keyed by model id (e.g. "tabicl-v2"), not by kind
    # ("tabicl"); insertion order follows settings.backends, so the
    # first-listed kind's model is the default.
    default = next(iter(backends))
    spill_store = _build_spill_store(
        settings,
        backends,
        device.replace(":", "_") if multi else None,
        settings.spill_capacity_bytes // n_devices,
    )
    # ONE cache and budget shared by every model on a device -- the whole
    # point of serving them behind one endpoint (see engine.py).
    cache = ContextCacheManager(
        capacity_bytes=estimators[default].ceiling_bytes(), spill_store=spill_store
    )
    engine = TabctxEngine(
        backends=backends,
        estimators={name: estimators[name] for name in backends},
        cache=cache,
        default_backend=default,
        host_guard=host_guard,
        device_baseline_bytes=0 if fake_only else measure_device_baseline(device),
    )
    return Shard(
        engine=engine,
        backends=backends,
        estimators={name: estimators[name] for name in backends},
        spill_store=spill_store,
    )


def build_engine(settings: ServeSettings | None = None) -> BuiltEngine:
    settings = settings or ServeSettings.from_env()
    devices = resolve_devices(settings)
    fake_only = set(settings.backends) == {"fake"}
    # Budget against the real device, not the calibration card's size;
    # the fake backend never touches a GPU even when one is visible.
    gpu_capacity = None if fake_only else detect_gpu_capacity_bytes(devices)
    multi = len(devices) > 1
    if settings.spill_dir:
        _warn_on_spill_layout(settings.spill_dir, multi)
    host_budget = None if fake_only else detect_host_budget_bytes()
    if multi and settings.worker_processes:
        return _build_process_engine(settings, devices, gpu_capacity, host_budget)
    # One host-memory guard per process (host RAM is shared by every
    # device), with queue slots for each device's engine's waiting calls.
    host_guard = (
        HostMemoryGuard(host_budget, queue_slots=QUEUE_SLOTS * len(devices))
        if host_budget
        else None
    )
    estimators: dict[str, MemoryEstimator] = {}
    shards = [
        build_shard(
            settings,
            device,
            n_devices=len(devices),
            gpu_capacity=gpu_capacity,
            host_guard=host_guard,
            estimators=estimators,
        )
        for device in devices
    ]
    first = shards[0]
    engine: TabctxEngine | MultiDeviceEngine = (
        MultiDeviceEngine([s.engine for s in shards], list(devices))
        if multi
        else first.engine
    )
    return BuiltEngine(
        engine=engine,
        estimators=first.estimators,
        backend_names=list(first.backends),
        default=next(iter(first.backends)),
        device="cpu (fake backend)" if fake_only else ",".join(devices),
        spill_store=None if multi else first.spill_store,
        gpu_capacity_bytes=gpu_capacity,
        devices=devices,
        spill_stores=tuple(s.spill_store for s in shards if s.spill_store is not None)
        if multi
        else (),
        host_guard=host_guard,
    )


def _build_process_engine(
    settings: ServeSettings,
    devices: tuple[str, ...],
    gpu_capacity: int | None,
    host_budget: int | None,
) -> BuiltEngine:
    """One worker process per GPU (serve/workers.py), routed by a
    MultiDeviceEngine in this process. Measured on 4x A100 (benchmarks/
    bench_scaling.py): concurrent fits on four GPUs from threads of one
    process took 2x as long each (the GIL -- fits are host-bound), while
    a process per GPU ran each at single-GPU speed."""
    import multiprocessing as mp

    from tabctx.serve.workers import ProcessShard

    ctx = mp.get_context("spawn")
    host_guard = (
        HostMemoryGuard(host_budget, mp_context=ctx, owners=len(devices))
        if host_budget
        else None
    )
    shards = ProcessShard.start_all(
        settings,
        devices,
        gpu_capacity=gpu_capacity,
        host_guard=host_guard,
        mp_context=ctx,
    )
    if host_guard is not None:
        # Measured before the workers existed; now that each holds its
        # models, CUDA context and compiled kernels, budget what is left.
        rebudget = detect_host_budget_bytes()
        if rebudget is not None:
            host_guard.set_budget(rebudget)
    # This process holds no backends: /limits and /readyz use estimators
    # built from the same settings (calibration, not this replica's
    # learned observations).
    estimators = {
        name: build_estimator(settings, kind, gpu_capacity)
        for name, kind in zip(shards[0].backend_names, settings.backends, strict=True)
    }
    return BuiltEngine(
        engine=MultiDeviceEngine(list(shards), list(devices)),
        estimators=estimators,
        backend_names=list(shards[0].backend_names),
        default=shards[0].default_backend,
        device=",".join(devices),
        gpu_capacity_bytes=gpu_capacity,
        devices=devices,
        host_guard=host_guard,
        worker_shards=tuple(shards),
    )
