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
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tabctx.backends.base import TabularICLBackend
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
from tabctx.multidevice import MultiDeviceEngine
from tabctx.serve.uploads import DEFAULT_MAX_UPLOAD_BYTES, DEFAULT_TTL_S

BACKEND_ENV_VAR = "TABCTX_BACKEND"
GPU_MEMORY_FRACTION_ENV_VAR = "TABCTX_GPU_MEMORY_FRACTION"
KV_CACHE_ENV_VAR = "TABCTX_KV_CACHE"
BATCH_WINDOW_MS_ENV_VAR = "TABCTX_BATCH_WINDOW_MS"
MAX_UPLOAD_BYTES_ENV_VAR = "TABCTX_MAX_UPLOAD_BYTES"
SPILL_DIR_ENV_VAR = "TABCTX_SPILL_DIR"
SPILL_CAPACITY_ENV_VAR = "TABCTX_SPILL_CAPACITY_BYTES"
UPLOAD_TTL_S_ENV_VAR = "TABCTX_UPLOAD_TTL_S"
DEVICES_ENV_VAR = "TABCTX_DEVICES"

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
    batch_window_ms: float = 5.0
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    upload_ttl_s: float = DEFAULT_TTL_S
    spill_dir: str | None = None
    spill_capacity_bytes: int = DEFAULT_SPILL_CAPACITY_BYTES
    # Empty = "auto" (see resolve_devices).
    devices: tuple[str, ...] = ()

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
        return cls(
            backends=backends,
            gpu_memory_fraction=fraction,
            kv_cache=kv_cache,
            batch_window_ms=batch_window_ms,
            max_upload_bytes=max_upload_bytes,
            upload_ttl_s=upload_ttl_s,
            spill_dir=spill_dir,
            spill_capacity_bytes=spill_capacity,
            devices=devices,
        )


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
    backends: dict[str, TabularICLBackend]
    default: str
    device: str
    spill_store: object | None = None
    # Real device memory the budget was derived from (per device; the
    # smallest when several); None = defaults.
    gpu_capacity_bytes: int | None = None
    devices: tuple[str, ...] = ()
    # One per device when there are several (spill_store is then None).
    spill_stores: tuple = ()

    # Single-model conveniences (the common deployment):
    @property
    def estimator(self) -> MemoryEstimator:
        return self.estimators[self.default]

    @property
    def backend(self) -> TabularICLBackend:
        return self.backends[self.default]


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
    return AdaptiveMemoryEstimator(
        fallback=fallback,
        preloaded=fit_grid,
        preloaded_predict=predict_grid,
    )


def _build_real_backend(
    kind: BackendKind, settings: ServeSettings, device: str
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
        return TabPFNBackend(device=device, cache_mode=settings.kv_cache)

    from tabctx.backends.tabicl import TabICLBackend

    kv_cache: bool | str = False if settings.kv_cache == "off" else settings.kv_cache
    return TabICLBackend(device=device, kv_cache=kv_cache)


def _build_backend(
    kind: BackendKind, settings: ServeSettings, device: str = "cpu"
) -> TabularICLBackend:
    """One backend bound to `device`. Imports torch/tabicl/tabpfn only on
    the paths that need them, so the fake backend runs with core deps
    alone."""
    if kind == "fake":
        from tabctx.backends.fake import FakeBackend

        return FakeBackend()
    return _build_real_backend(kind, settings, device)


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


def build_engine(settings: ServeSettings | None = None) -> BuiltEngine:
    settings = settings or ServeSettings.from_env()
    devices = resolve_devices(settings)
    # Budget against the real device, not the calibration card's size;
    # the fake backend never touches a GPU even when one is visible.
    gpu_capacity = (
        None
        if set(settings.backends) == {"fake"}
        else detect_gpu_capacity_bytes(devices)
    )
    multi = len(devices) > 1
    if settings.spill_dir:
        _warn_on_spill_layout(settings.spill_dir, multi)
    estimators: dict[str, MemoryEstimator] = {}
    shards: list[TabctxEngine] = []
    spill_stores: list[DiskSpillStore | None] = []
    first_backends: dict[str, TabularICLBackend] = {}
    for device in devices:
        backends: dict[str, TabularICLBackend] = {}
        for kind in settings.backends:
            backend = _build_backend(kind, settings, device)
            backends[backend.name] = backend
            # One estimator per model, shared by every device: they are
            # identical cards budgeted alike, and observations learned on
            # one device are valid on the others.
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
            settings.spill_capacity_bytes // len(devices),
        )
        # ONE cache and budget shared by every model on a device -- the
        # whole point of serving them behind one endpoint (see engine.py).
        cache = ContextCacheManager(
            capacity_bytes=estimators[default].ceiling_bytes(), spill_store=spill_store
        )
        shards.append(
            TabctxEngine(
                backends=backends,
                estimators={name: estimators[name] for name in backends},
                cache=cache,
                default_backend=default,
            )
        )
        spill_stores.append(spill_store)
        first_backends = first_backends or backends
    engine: TabctxEngine | MultiDeviceEngine = (
        MultiDeviceEngine(shards, list(devices)) if multi else shards[0]
    )
    fake_only = set(settings.backends) == {"fake"}
    return BuiltEngine(
        engine=engine,
        estimators={name: estimators[name] for name in first_backends},
        backends=first_backends,
        default=next(iter(first_backends)),
        device="cpu (fake backend)" if fake_only else ",".join(devices),
        spill_store=None if multi else spill_stores[0],
        gpu_capacity_bytes=gpu_capacity,
        devices=devices,
        spill_stores=tuple(s for s in spill_stores if s is not None) if multi else (),
    )
