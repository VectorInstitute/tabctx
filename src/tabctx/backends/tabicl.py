"""TabICL/TabICLv2 backend (github.com/soda-inria/tabicl).

Fixes two bugs found in the hand-rolled wrapper this library replaces
(inference-platform's tests/gke-tabicl-test/app.py):

1. That wrapper called model.predict() and then model.predict_proba()
   separately when return_proba was requested -- two full inference passes
   for one request. Here, predict_proba() is called once and predictions are
   derived from it via argmax, so return_proba never costs an extra pass.
2. That wrapper built a fresh estimator per request and never reused
   anything across requests, discarding the entire benefit of TabICL's own
   fit-time context caching. Here, fit() still builds a fresh estimator per
   call (required -- see backends/base.py), but the returned estimator is
   retained by tabctx's ContextCacheManager and reused across many predict()
   calls, which is the actual point of this library.

torch/tabicl are imported lazily inside this module (not at package import
time) so the rest of tabctx has no hard dependency on either.

EMPIRICAL CHECK (per the tabctx build plan, run locally on CPU before
finalizing this file): does constructing a fresh TabICLClassifier/Regressor
reload the checkpoint every time, or is it memoized in-process? Confirmed by
reading tabicl/_sklearn/classifier.py's _load_model(): huggingface_hub only
avoids re-downloading (the file is cached on disk after the first fetch),
but torch.load() + load_state_dict() run fresh on every instantiation --
there is no in-process memoization. Measured cost of that reload alone (CPU,
checkpoint already on disk): ~0.1s per fit() call, separate from and in
addition to the actual context-encoding cost. Fine for v1 (a fresh instance
per fit() is required regardless -- see backends/base.py), but a real
optimization opportunity for later: load the backbone weights once per
backend instance and share them across fit() calls, keeping only the
per-fit training-encoding state instance-specific.

REAL GPU MEMORY MEASUREMENT (found via extensive multi-tenant load testing
on a real A100-40GB, 2026-08-28): the formula-based MemoryEstimator, when
used to size a context for cache-capacity accounting, reported ~14x more
bytes than what was actually resident on the device for a realistic shape
(cache accounting said ~21.85GB used; torch.cuda.memory_allocated() said
~1.56GB) -- because that estimator is calibrated on active train+test
forward-pass memory, not resting post-fit context size, and (separately)
already over-estimates small inputs (see memory/estimator.py). This
needlessly throttled effective multi-tenant capacity to a fraction of what
the hardware actually supports. Fixed here by measuring the real
torch.cuda.memory_allocated() delta across fit() and reporting it via
context_bytes_hint(). Predicts may run beside a fit on the same device, so
the readings are taken under the engine's predict lock (see
_cuda.QuiescedMeasurement): no predict's temporaries pollute the delta.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from tabctx.backends import yeo_johnson
from tabctx.backends._cuda import (
    QuiescedMeasurement,
    device_scope,
    ensure_initialized,
    is_cuda,
)
from tabctx.backends.tabicl_kv import KV_PRECISIONS, KVPrecision, compress_icl_cache
from tabctx.errors import BackendComputeError
from tabctx.types import ArrayLike, PredictOutcome, Task

# Set (per thread) while loads_payload unpickles: tabicl's __setstate__
# calls _load_model(), which would otherwise torch.load the full checkpoint
# and move a second backbone onto the GPU on every spill restore.
_restoring = threading.local()
_warned = [False]  # one missing-hook warning per process
# Offloaded stage-output bytes per cell (see TabICLBackend.may_offload).
_OUTPUT_BYTES_PER_CELL = 2560
# Host bytes per cell of an offloading call (see host_bytes_estimate).
_HOST_BYTES_PER_OFFLOADED_CELL = 5_000
logger = logging.getLogger(__name__)
_TASK_TAGS = {"classification": b"C", "regression": b"R"}


def _install_restore_hook(cls: type) -> None:
    """Wrap cls._load_model: while a restore is in progress on this
    thread, attach the backend's shared backbone instead of loading the
    checkpoint. Idempotent; other threads and normal loads are untouched."""
    if getattr(cls, "_tabctx_restore_hook", False):
        return
    original = cls._load_model

    def _load_model(self: Any) -> None:
        shared = getattr(_restoring, "backbone", None)
        if shared is None:
            return original(self)
        for attr, value in shared.items():
            setattr(self, attr, value)
        return None

    cls._load_model = _load_model
    cls._tabctx_restore_hook = True


@dataclass
class _PreparedFit:
    """A fit whose CPU half is done (see TabICLBackend.prepare_fit)."""

    task: Task
    model: Any


def _build_cache(model: Any) -> None:
    """tabicl's fit() from the kv-cache step on (it follows the ensemble's
    preprocessing in both TabICLClassifier and TabICLRegressor): pick the
    cache mode from the estimator's kv_cache setting and build it."""
    mode = model.kv_cache
    if not mode:
        model.model_kv_cache_ = None
        return
    model.cache_mode_ = "repr" if mode == "repr" else "kv"
    model._build_kv_cache()


def install_cached_logn() -> bool:
    """Replace tabicl's ssmax._logn with a cached equivalent.

    _logn builds a 0-dim CUDA tensor from a Python float on every attention
    call -- a host-to-device copy that blocks the CPU until the GPU queue
    drains, 42 times in a 40k x 100 fit, so kernel launches never run ahead
    of the GPU. The values depend only on (n, device, dtype); caching them
    gives bit-identical results with one sync per distinct n. Measured on
    an A100: 10,000 x 50 fits 1.98 s -> 1.66 s, 40,000 x 100 11.1 s -> 11.0 s,
    max probability difference 0. Returns False (and changes nothing) if
    tabicl no longer has the function."""
    import functools

    try:
        from tabicl._model import ssmax
    except ImportError:
        return False
    original = getattr(ssmax, "_logn", None)
    if original is None or getattr(original, "_tabctx_cached", False):
        return original is not None

    # The cached tensors are created under whatever grad/inference mode the
    # first call ran in; fine for serving (always inference), but anything
    # that runs ssmax with autograd (fine-tuning) must not use this cache.
    @functools.lru_cache(maxsize=4096)
    def cached(n: int, device: Any, dtype: Any) -> Any:
        return original(n, device, dtype)

    def logn(n: Any, device: Any, dtype: Any) -> Any:
        import torch

        if torch.compiler.is_compiling():
            # Inside a compiled block (tabicl_compile.py) n is symbolic:
            # math.log would specialize the graph to one row count, so
            # compute log(n) as a tensor op (in fp32, then cast, as the
            # eager path rounds a float64 log to dtype).
            ones = torch.ones((), device=device, dtype=torch.float32)
            return (ones * torch.sym_max(n, 1)).log().to(dtype)
        return cached(n, device, dtype)

    logn._tabctx_cached = True  # type: ignore[attr-defined]
    ssmax._logn = logn
    return True


def make_cache_thread_local(module: Any) -> None:
    """Give a shared TabICL backbone one active-context cache per thread.

    tabicl's forward_with_cache keeps the context it is working with on
    the MODULE (``model_._cache``), and every call clears it when done
    (see _unpin_shared_module). With one backbone shared by every context,
    two calls running at once -- a predict interleaved with a fit on the
    same GPU -- would read each other's cache: a predict answered from
    another tenant's context, or None mid-forward (seen on an A100 as
    AttributeError: 'NoneType' object has no attribute 'icl_cache').
    A per-instance subclass whose ``_cache`` is a thread-local property
    gives each thread its own slot; nothing else about the module changes.
    """
    if getattr(type(module), "_tabctx_thread_local_cache", False):
        return
    local = threading.local()
    base = type(module)

    def _get(_self: Any) -> Any:
        return getattr(local, "cache", None)

    def _set(_self: Any, value: Any) -> None:
        local.cache = value

    module.__dict__.pop("_cache", None)
    module.__class__ = type(
        base.__name__,
        (base,),
        {
            "_cache": property(_get, _set),
            "_tabctx_thread_local_cache": True,
            "__module__": base.__module__,
        },
    )


class TabICLBackend(QuiescedMeasurement):
    # The exact model id callers select via the API's `model` field --
    # named for the checkpoint actually loaded (tabicl-classifier-v2-*).
    name = "tabicl-v2"
    _fast_preprocessing = False

    def __init__(
        self,
        device: str | None = None,
        kv_cache: bool | str = "kv",
        kv_precision: KVPrecision = "native",
        offload_dir: str | None = None,
        cpu_safety_factor: float = 0.6,
        compile_blocks: bool = False,
        fast_preprocessing: bool = False,
    ) -> None:
        """kv_cache: passed through to TabICL ("kv", "repr", True, or
        False). tabicl's own default is False, which makes every
        predict() re-encode the ENTIRE training set through all three
        transformer stages -- i.e. the exact cost tabctx exists to avoid
        paying twice (confirmed by tracing predict_proba: with no cache,
        transform(mode="both") re-concatenates and re-encodes train+test
        on every call). tabctx defaults it ON ("kv": fastest, most
        memory-hungry -- the memory cost lands inside fit()'s measured
        delta, so cache accounting and the adaptive admission gate see it
        automatically; "repr" trades ~24x less ICL-cache memory for
        re-running the ICL stack per predict).

        compile_blocks: torch.compile tabicl's attention blocks (CUDA only;
        see backends/tabicl_compile.py): 1.7-2.2x faster GPU half, ~20%
        lower peak, paid for by a one-time compile in warmup().

        fast_preprocessing: fit the ensemble's Yeo-Johnson power transforms
        for all columns at once, on the GPU (backends/yeo_johnson.py)
        instead of scikit-learn's per-column scipy search -- 80%+ of a
        fit's CPU half. Lambdas agree to ~1e-6; not bit-identical."""
        import torch

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ensure_initialized(self._device)
        self._kv_cache = kv_cache
        if kv_precision not in KV_PRECISIONS:
            raise ValueError(f"kv_precision must be one of {KV_PRECISIONS}")
        # Storage of the ICL key/value cache (see backends/tabicl_kv.py).
        self._kv_precision: KVPrecision = kv_precision
        # Large fits offload stage outputs off the GPU (tabicl's
        # offload="auto"). Without a disk directory, an output too big for
        # host memory falls through to "cpu (swap)" -- an OOM kill in a
        # container with no swap (measured: 65,000 x 500 on a 64 GiB pod).
        # With one, it degrades to disk memmaps instead. tabicl budgets host
        # memory as cgroup headroom x cpu_safety_factor but doesn't count
        # its own pinned staging buffers or other devices' concurrent work,
        # hence 0.6 rather than its 0.85 default.
        self._offload_dir = offload_dir
        self._offload_config = {"cpu_safety_factor": cpu_safety_factor}
        self._last_context_bytes: int | None = None
        self._last_fit_peak_bytes: int | None = None
        self._last_fit_resident_bytes: int | None = None
        # task -> attrs of an already-loaded backbone, shared across
        # fits. Loading is ~hundreds of MB of torch.load + H2D transfer
        # PER FIT without this. Sharing one nn.Module across estimators
        # is tabicl's own sanctioned pattern (see
        # _unsupervised/unsupervised.py and _finetune/base.py, both of
        # which do exactly this with the comment "prevents redundant
        # torch.load() calls"). Calls on the shared module may overlap (a
        # predict beside a fit, see engine.py): its active-context cache is
        # per thread (make_cache_thread_local), and the InferenceManager
        # config every call reapplies is identical for this backend.
        self._shared_backbones: dict[str, dict[str, Any]] = {}
        from tabicl import TabICLClassifier, TabICLRegressor

        _install_restore_hook(TabICLClassifier)
        _install_restore_hook(TabICLRegressor)
        if not install_cached_logn():
            logger.warning("tabicl.ssmax._logn not found: per-call GPU syncs remain")
        self._fast_preprocessing = fast_preprocessing and yeo_johnson.install()
        if fast_preprocessing and not self._fast_preprocessing:
            logger.warning("tabicl preprocessing not patchable: scipy power transforms")
        self._compiled = False
        if compile_blocks and is_cuda(self._device):
            from tabctx.backends.tabicl_compile import install_compiled_blocks

            self._compiled = install_compiled_blocks()

    def _make_estimator(self, task: Task) -> Any:
        options = {
            "device": self._device,
            "kv_cache": self._kv_cache,
            "disk_offload_dir": self._offload_dir,
            "inference_config": {
                stage: dict(self._offload_config)
                for stage in ("COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG")
            },
        }
        if task == "classification":
            from tabicl import TabICLClassifier

            est = TabICLClassifier(**options)
        else:
            from tabicl import TabICLRegressor

            est = TabICLRegressor(**options)

        shared = self._shared_backbones.get(task)
        if shared is not None:
            for attr, value in shared.items():
                setattr(est, attr, value)
            # Instance attribute shadows the method: fit() calls
            # self._load_model() and gets this no-op instead of a fresh
            # torch.load + load_state_dict.
            est._load_model = lambda: None
        return est

    def _stash_backbone(self, task: Task, model: Any) -> None:
        if task in self._shared_backbones:
            return
        make_cache_thread_local(model.model_)
        shared = {"model_": model.model_}
        # Keep save()/pickling of estimators working: __setstate__ only
        # rebuilds from weights when model_config_ exists (see tabicl
        # _sklearn/base.py); copying these mirrors _finetune/base.py.
        for attr in ("model_config_", "model_path_"):
            if hasattr(model, attr):
                shared[attr] = getattr(model, attr)
        self._shared_backbones[task] = shared

    def fit(self, X: ArrayLike, y: ArrayLike, task: Task) -> Any:
        return self.fit_prepared(self.prepare_fit(X, y, task))

    def prepare_fit(self, X: ArrayLike, y: ArrayLike, task: Task) -> _PreparedFit:
        """The CPU half of a fit: validation, label encoding and the
        ensemble's preprocessing (per-feature power transforms, shuffles) --
        tabicl's fit() up to building the kv cache, run with the cache
        disabled. 30-37% of a fit's time (A100, 2k x 20 to 40k x 100) with
        scikit-learn's power transforms, ~10% with fast_preprocessing. It
        needs no device memory beyond the power transforms' bounded,
        measurement-locked temporaries, so the engine runs it before
        taking the device, overlapping another fit's GPU half."""
        # Fresh estimator instance every call (load-bearing -- see
        # backends/base.py), but the underlying pretrained nn.Module is
        # shared across fits after the first (see __init__).
        model = self._make_estimator(task)
        kv_cache = model.kv_cache
        model.kv_cache = False
        # With fast_preprocessing, power transforms fit on this backend's GPU
        # (backends/yeo_johnson.py), under the measurement lock: this runs
        # beside another fit's GPU half, whose readings must not see it.
        fast = (
            yeo_johnson.on_device(
                self._device if is_cuda(self._device) else None, self.quiesced
            )
            if self._fast_preprocessing
            else contextlib.nullcontext()
        )
        try:
            with device_scope(self._device), fast:
                model.fit(X, y)
        finally:
            model.kv_cache = kv_cache
        return _PreparedFit(task, model)

    def fit_prepared(self, prepared: _PreparedFit) -> Any:
        """The GPU half: build the context's kv cache from a prepared fit,
        measuring what it occupies. Re-runnable after a device OOM (tabicl
        rebuilds the cache from scratch)."""
        import torch

        model = prepared.model
        device = self._device
        measuring = is_cuda(device)
        before_bytes = 0
        if measuring:
            with self.quiesced():
                before_bytes = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
        self._last_context_bytes = None
        self._last_fit_peak_bytes = None
        self._last_fit_resident_bytes = None
        try:
            with device_scope(device):
                _build_cache(model)
                # The context as the fit built it, at full precision: what
                # the fit needed resident, and so what admission must
                # learn -- compression only shrinks it afterwards.
                fit_resident = None
                if measuring:
                    with self.quiesced():
                        fit_resident = (
                            torch.cuda.memory_allocated(device) - before_bytes
                        )
                compress_icl_cache(model, self._kv_precision)
        except torch.cuda.OutOfMemoryError as e:
            raise BackendComputeError(f"CUDA OOM during fit(): {e}") from e
        finally:
            # Even on failure: a large fit that dies after offloading leaves
            # tens of GiB pinned, and its host slot is released right after.
            self._unpin_shared_module(model)
            self._release_host_memory(model)
        # Three real measurements, for different consumers:
        # - resident (context_bytes_hint): what the fitted context occupies
        #   afterwards, compressed -> cache capacity/eviction accounting.
        # - fit-time resident (fit_resident_bytes_hint): the context before
        #   compression -> what admission learns the fit needed resident.
        # - peak (fit_peak_bytes_hint): the transient high-water during fit.
        # None on CPU; the engine falls back to the formula estimator.
        if measuring:
            with self.quiesced():
                after_bytes = torch.cuda.memory_allocated(device)
                peak_bytes = torch.cuda.max_memory_allocated(device)
            self._last_context_bytes = max(0, after_bytes - before_bytes)
            self._last_fit_resident_bytes = max(0, fit_resident or 0)
            self._last_fit_peak_bytes = max(0, peak_bytes - before_bytes)
        self._stash_backbone(prepared.task, model)
        return model

    def _unpin_shared_module(self, model: Any) -> None:
        """Drop the shared backbone's pointer to the last-used context.

        tabicl's `forward_with_cache` stores the active context's cache
        on the MODULE (`model_._cache = cache`) and leaves it there.
        With the backbone shared across all contexts (see __init__),
        that pointer pins the most recently used context's multi-GB kv
        cache even after tabctx evicts it -- found on a real A100 as
        ~15GB that eviction "freed" in the accounting but never left the
        device, eventually OOMing a fit the books said was safe. The
        context's own cache lives on its estimator (model_kv_cache_), so
        clearing the module pointer costs nothing and unpins eviction."""
        module = getattr(model, "model_", None)
        if module is not None and hasattr(module, "clear_cache"):
            module.clear_cache()

    def may_offload(self, n_rows: int, n_features: int) -> bool:
        """Whether a call over n_rows x n_features may offload stage outputs
        to host memory. Mirrors tabicl's auto-offload trigger -- an output
        above auto_offload_threshold (0.5) x the device's CURRENTLY free GPU
        memory -- with the output size calibrated on an A100 (no offload at
        5M cells, offload at 10M, ~38 GB free): ~2.5 KB per cell for 8
        estimators. So on a busy card, smaller tables offload too."""
        if not is_cuda(self._device):
            return False
        import torch

        free, _ = torch.cuda.mem_get_info(torch.device(self._device))
        return _OUTPUT_BYTES_PER_CELL * n_rows * n_features > 0.4 * free

    def host_bytes_estimate(
        self, n_rows: int, n_features: int, *, offloads: bool = False
    ) -> int:
        """Host memory a call over n_rows x n_features may use: when it
        offloads, its staged stage outputs -- which scale with cells (the
        offloaded tensors are per-cell activations); measured ~46 GiB for
        10M-cell fits on an A100, ~4.9 KB per cell, budgeted at 5 KB. A
        call that doesn't offload grows host RSS by < 0.5 GiB up to ~5M
        cells: nothing worth reserving."""
        return _HOST_BYTES_PER_OFFLOADED_CELL * n_rows * n_features if offloads else 0

    def _release_host_memory(self, model: Any) -> None:
        """Return the offload staging memory a large call leaves pinned.

        Above ~5M cells tabicl offloads stage outputs to pinned host memory.
        Each stage's InferenceManager keeps its PinnedBufferPool afterwards,
        on the SHARED backbone, and the CUDA pinned-host allocator caches
        what was freed. Measured on an A100 (20,000 x 500): +43 GiB of host
        RSS after one fit, 10 GiB of it pooled buffers and 33 GiB allocator
        cache, none of it the context's -- two large tables would exhaust a
        64 GiB pod and get the replica OOM-killed. Clearing the pools and
        emptying the host cache returns it (only unused cached blocks are
        freed; cudaFreeHost synchronizes, so a concurrent call on another
        device may stall briefly, but its buffers are untouched)."""
        module = getattr(model, "model_", None)
        if module is None:
            return
        pools = [
            getattr(getattr(sub, "inference_mgr", None), "_buffer_pool", None)
            for sub in module.modules()
        ]
        for pool in pools:
            if pool is not None:
                pool.clear()
        import torch

        empty_host_cache = getattr(torch._C, "_host_emptyCache", None)
        if callable(empty_host_cache):
            empty_host_cache()
        if (not any(pools) or not callable(empty_host_cache)) and not _warned[0]:
            # Private hooks (tabicl's pools, torch's host cache): a rename
            # would silently bring back tens of GiB of retained host memory.
            _warned[0] = True
            logger.warning(
                "tabctx could not release TabICL's pinned host memory "
                "(buffer pools found: %s, torch host cache API: %s); large "
                "fits may retain host RAM -- check tabicl/torch versions",
                any(pools),
                callable(empty_host_cache),
            )

    def predict(
        self, payload: Any, X_test: ArrayLike, return_proba: bool = False
    ) -> PredictOutcome:
        import torch

        model = payload
        try:
            with device_scope(self._device):
                return self._predict(model, X_test, return_proba)
        except torch.cuda.OutOfMemoryError as e:
            raise BackendComputeError(f"CUDA OOM during predict(): {e}") from e
        finally:
            self._unpin_shared_module(model)
            self._release_host_memory(model)

    def _predict(
        self, model: Any, X_test: ArrayLike, return_proba: bool
    ) -> PredictOutcome:
        if hasattr(model, "predict_proba"):
            if return_proba:
                proba = model.predict_proba(X_test)
                class_idx = np.argmax(proba, axis=1)
                classes = [str(c) for c in model.classes_]
                predictions = [classes[i] for i in class_idx]
                return PredictOutcome(
                    predictions=predictions,
                    probabilities=proba.tolist(),
                    classes=classes,
                )
            return PredictOutcome(predictions=model.predict(X_test).tolist())
        # Regressor: no predict_proba to derive from.
        return PredictOutcome(predictions=model.predict(X_test).tolist())

    def warmup(self) -> None:
        """Load the model weights for both tasks now, with throwaway fits,
        instead of inside a tenant's first fit: that fit would otherwise
        report the weights as part of its context (resident bytes for a
        tiny table were the whole checkpoint), overcharging the cache and
        leaving the books wrong once it is evicted -- and would pay the
        load latency. With compiled blocks, also compile the common call
        families here (fp32 and fp16 paths, fits and cached predicts,
        1-feature tables and 1-row predicts) rather than in tenants'
        calls."""
        rng = np.random.default_rng(0)
        # (rows, features): with compiled blocks, tabicl's fp16 autocast path
        # (1,024+ rows; the only one compiled), including a 1-feature table
        # (dynamo specializes size-1 dimensions, so they get their own
        # graphs).
        shapes = ((32, 4), (1100, 4), (1100, 1)) if self._compiled else ((32, 4),)
        for rows, features in shapes:
            X = rng.normal(size=(rows, features))
            labels = (X[:, 0] > 0).astype(int)
            for task, y in (("classification", labels), ("regression", X[:, 0])):
                payload = self.fit(X, y, task)
                if self._compiled:
                    proba = task == "classification"
                    for n in (8, 1):  # 1-row predicts: also size-1 graphs
                        self.predict(payload, X[:n], return_proba=proba)
        if self._compiled:
            from tabctx.backends.tabicl_compile import warn_if_not_compiled

            warn_if_not_compiled()
        self._last_context_bytes = None
        self._last_fit_peak_bytes = None
        self._last_fit_resident_bytes = None

    def context_bytes_hint(self, n_train: int, n_features: int) -> int | None:
        del n_train, n_features
        # Real measurement from the fit() call this is queried after (see
        # module docstring); None on CPU or if fit() OOM'd, in which case
        # the engine falls back to the formula-based MemoryEstimator.
        return self._last_context_bytes

    def fit_resident_bytes_hint(self) -> int | None:
        """The most recent fit's context before compression -- what it
        needed resident while it ran (== context_bytes_hint at native
        precision); None on CPU/after OOM."""
        return self._last_fit_resident_bytes

    def fit_peak_bytes_hint(self) -> int | None:
        """Transient high-water memory of the most recent fit() -- the
        admission-relevant quantity (see fit()); None on CPU/after OOM."""
        return self._last_fit_peak_bytes

    # ---- spillover serialization (see cache/spill.py) ----------------

    def dumps_payload(self, payload: Any) -> bytes:
        """Pickle a fitted estimator for the disk spill tier WITHOUT the
        shared pretrained backbone (hundreds of MB, identical across all
        contexts, unpicklable anyway once _load_model is shadowed with a
        lambda): strip the shared attributes, pickle, restore. A one-byte
        task tag leads the blob so loads_payload can attach the right
        backbone during unpickling."""
        import pickle

        task: Task = (
            "classification" if hasattr(payload, "predict_proba") else "regression"
        )
        stripped = {}
        for attr in ("model_", "_load_model"):
            if attr in payload.__dict__:
                stripped[attr] = payload.__dict__.pop(attr)
        try:
            return _TASK_TAGS[task] + pickle.dumps(payload)
        finally:
            payload.__dict__.update(stripped)

    def _ensure_backbone(self, task: Task) -> None:
        if task in self._shared_backbones:
            return
        # Cold restore on a fresh replica: borrow a throwaway estimator's
        # load path to populate the shared backbone.
        probe = self._make_estimator(task)
        with device_scope(self._device):
            probe._load_model()
        self._stash_backbone(task, probe)

    def loads_payload(self, data: bytes) -> Any:
        """Inverse of dumps_payload: unpickle and re-attach this
        process's shared backbone (loading it first if this replica has
        never fit this task type -- e.g. right after a restart)."""
        import pickle

        tagged = {tag: task for task, tag in _TASK_TAGS.items()}
        task_hint = tagged.get(data[:1])
        if task_hint is None:
            # Untagged blob (spilled by tabctx < 0.12): tabicl's
            # __setstate__ reloads the checkpoint; the shared backbone
            # replaces it below.
            with device_scope(self._device):
                model = pickle.loads(data)
        else:
            self._ensure_backbone(task_hint)
            _restoring.backbone = self._shared_backbones[task_hint]
            try:
                with device_scope(self._device):
                    model = pickle.loads(data[1:])
            finally:
                _restoring.backbone = None
        task: Task = (
            "classification" if hasattr(model, "predict_proba") else "regression"
        )
        self._ensure_backbone(task)
        for attr, value in self._shared_backbones[task].items():
            setattr(model, attr, value)
        model._load_model = lambda: None
        return model
