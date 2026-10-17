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
context_bytes_hint() -- safe because the engine holds its cache lock across
the whole fit() call (see engine.py), so no concurrent GPU work can pollute
the before/after delta.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from tabctx.backends._cuda import device_scope, ensure_initialized, is_cuda
from tabctx.backends.tabicl_kv import KV_PRECISIONS, KVPrecision, compress_icl_cache
from tabctx.errors import BackendComputeError
from tabctx.types import ArrayLike, PredictOutcome, Task

# Set (per thread) while loads_payload unpickles: tabicl's __setstate__
# calls _load_model(), which would otherwise torch.load the full checkpoint
# and move a second backbone onto the GPU on every spill restore.
_restoring = threading.local()
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


class TabICLBackend:
    # The exact model id callers select via the API's `model` field --
    # named for the checkpoint actually loaded (tabicl-classifier-v2-*).
    name = "tabicl-v2"

    def __init__(
        self,
        device: str | None = None,
        kv_cache: bool | str = "kv",
        kv_precision: KVPrecision = "native",
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
        re-running the ICL stack per predict)."""
        import torch

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ensure_initialized(self._device)
        self._kv_cache = kv_cache
        if kv_precision not in KV_PRECISIONS:
            raise ValueError(f"kv_precision must be one of {KV_PRECISIONS}")
        # Storage of the ICL key/value cache (see backends/tabicl_kv.py).
        self._kv_precision: KVPrecision = kv_precision
        self._last_context_bytes: int | None = None
        self._last_fit_peak_bytes: int | None = None
        # task -> attrs of an already-loaded backbone, shared across
        # fits. Loading is ~hundreds of MB of torch.load + H2D transfer
        # PER FIT without this. Sharing one nn.Module across estimators
        # is tabicl's own sanctioned pattern (see
        # _unsupervised/unsupervised.py and _finetune/base.py, both of
        # which do exactly this with the comment "prevents redundant
        # torch.load() calls"). Safe here because the engine serializes
        # every backend call behind its cache lock -- the shared module's
        # per-call mutable state (its _cache pointer, InferenceManager
        # config) is never touched concurrently.
        self._shared_backbones: dict[str, dict[str, Any]] = {}
        from tabicl import TabICLClassifier, TabICLRegressor

        _install_restore_hook(TabICLClassifier)
        _install_restore_hook(TabICLRegressor)

    def _make_estimator(self, task: Task) -> Any:
        if task == "classification":
            from tabicl import TabICLClassifier

            est = TabICLClassifier(device=self._device, kv_cache=self._kv_cache)
        else:
            from tabicl import TabICLRegressor

            est = TabICLRegressor(device=self._device, kv_cache=self._kv_cache)

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
        shared = {"model_": model.model_}
        # Keep save()/pickling of estimators working: __setstate__ only
        # rebuilds from weights when model_config_ exists (see tabicl
        # _sklearn/base.py); copying these mirrors _finetune/base.py.
        for attr in ("model_config_", "model_path_"):
            if hasattr(model, attr):
                shared[attr] = getattr(model, attr)
        self._shared_backbones[task] = shared

    def fit(self, X: ArrayLike, y: ArrayLike, task: Task) -> Any:
        import torch

        # Fresh estimator instance every call (load-bearing -- see
        # backends/base.py), but the underlying pretrained nn.Module is
        # shared across fits after the first (see __init__).
        model = self._make_estimator(task)

        device = self._device
        measuring = is_cuda(device)
        before_bytes = torch.cuda.memory_allocated(device) if measuring else 0
        if measuring:
            torch.cuda.reset_peak_memory_stats(device)
        try:
            with device_scope(device):
                model.fit(X, y)
                # Before measuring: the context's resident size is the
                # compressed one.
                compress_icl_cache(model, self._kv_precision)
        except torch.cuda.OutOfMemoryError as e:
            self._last_context_bytes = None
            self._last_fit_peak_bytes = None
            raise BackendComputeError(f"CUDA OOM during fit(): {e}") from e
        # Two DIFFERENT real measurements, for two different consumers --
        # conflating them was the v0.4.0 design flaw fixed in v0.9.0:
        # - resident delta (context_bytes_hint): what the fitted context
        #   occupies afterward -> cache capacity/eviction accounting.
        # - peak delta (fit_peak_bytes_hint): the transient high-water
        #   during fit -> what ADMISSION must bound, since this is what
        #   actually OOMs. Peak can exceed resident by orders of
        #   magnitude (activations, ensemble intermediates).
        # None on CPU; the engine falls back to the formula estimator.
        if measuring:
            after_bytes = torch.cuda.memory_allocated(device)
            self._last_context_bytes = max(0, after_bytes - before_bytes)
            self._last_fit_peak_bytes = max(
                0, torch.cuda.max_memory_allocated(device) - before_bytes
            )
        else:
            self._last_context_bytes = None
            self._last_fit_peak_bytes = None
        self._stash_backbone(task, model)
        # Explicit even though tabicl's fit path clears internally today:
        # a future tabicl change must not silently re-pin (see
        # _unpin_shared_module for the OOM this prevents).
        self._unpin_shared_module(model)
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
        load latency."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(32, 4))
        self.fit(X, (X[:, 0] > 0).astype(int), "classification")
        self.fit(X, X[:, 0], "regression")
        self._last_context_bytes = None
        self._last_fit_peak_bytes = None

    def context_bytes_hint(self, n_train: int, n_features: int) -> int | None:
        del n_train, n_features
        # Real measurement from the fit() call this is queried after (see
        # module docstring); None on CPU or if fit() OOM'd, in which case
        # the engine falls back to the formula-based MemoryEstimator.
        return self._last_context_bytes

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
