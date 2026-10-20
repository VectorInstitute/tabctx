"""TabPFN backend (github.com/PriorLabs/TabPFN).

The second backend behind the `TabularICLBackend` protocol -- proving
the engine/cache/estimator stack really is backend-agnostic (the design
claim in backends/base.py; nothing outside this module imports tabpfn).

The kv-cache lesson from TabICL (see backends/tabicl.py and CHANGELOG
0.7.0) applies verbatim here: TabPFN's default `fit_mode` is
"fit_preprocessors", which caches training-data PREPROCESSING but still
re-runs the transformer's training-context forward pass on every
predict. Its equivalent of the kv cache is `fit_mode="fit_with_cache"`
("the transformer key-value cache is also initialized, allowing for much
faster inference on the same data at a large cost of memory" -- exactly
the cost tabctx's memory accounting exists to price). tabctx maps its
cache modes accordingly:

    tabctx mode  ->  TabPFN fit_mode
    "kv"             "fit_with_cache"   (fastest predicts, memory-heavy)
    "repr"           "fit_preprocessors" (cheap contexts, slower predicts)
    "off"            "low_memory"        (escape hatch; re-preprocesses
                                          per predict)

TabPFN enforces pretraining limits (~10k train rows, ~500 features on
v2-series checkpoints) by raising before any meaningful GPU work; those
surface as InvalidInputError so a multi-tenant caller gets a clean 422,
not a 500. Unlike TabICL, TabPFN memoizes model loading internally per
(model_path, device) via its model-cache mechanism, so there is no
per-fit torch.load to elide here; a fresh estimator per fit() (required
by the protocol -- fit() mutates estimator state) is already cheap.
"""

from __future__ import annotations

import io
import os
import pickle
from typing import Any

import numpy as np

from tabctx.backends._cuda import (
    QuiescedMeasurement,
    device_scope,
    ensure_initialized,
    is_cuda,
)
from tabctx.backends._outcome import classification_outcome
from tabctx.errors import BackendComputeError, InvalidInputError
from tabctx.types import ArrayLike, PredictOutcome, Task

_FIT_MODES = {
    "kv": "fit_with_cache",
    "repr": "fit_preprocessors",
    "off": "low_memory",
}


def served_model_id(model_version: str) -> str:
    """The API model id for a tabpfn `model_version`: "v3" -> "tabpfn-3",
    "v3.5" -> "tabpfn-3.5", "v3.5-fast" -> "tabpfn-3.5-fast"."""
    return f"tabpfn-{model_version.removeprefix('v')}"


# TabPFN builds a fresh model for every estimator unless its built-model
# cache is on, so every cached context held its own ~850 MB copy of the
# weights on the GPU (measured: 4 contexts of a 300x6 table = 3.4 GB). With
# the cache, estimators on one device share one model per task and a
# context costs only its own state (~20 MB for that table), with
# bit-identical predictions. Keyed by device and task, so multi-GPU
# replicas get one shared model per (device, task).
_BUILT_MODEL_CACHE_ENTRIES = "16"
_SPILL_TAG = b"P1"


class _SharedModelPickler(pickle.Pickler):
    def __init__(self, file, shared: dict[int, tuple]) -> None:
        super().__init__(file, protocol=pickle.HIGHEST_PROTOCOL)
        self._shared = shared

    def persistent_id(self, obj: Any) -> tuple | None:
        return self._shared.get(id(obj))


class _SharedModelUnpickler(pickle.Unpickler):
    def __init__(self, file, resolve) -> None:
        super().__init__(file)
        self._resolve = resolve

    def persistent_load(self, pid: Any) -> Any:
        return self._resolve(pid)


class TabPFNBackend(QuiescedMeasurement):
    def __init__(self, device: str | None = None, cache_mode: str = "kv") -> None:
        import torch
        from tabpfn.settings import settings

        if cache_mode not in _FIT_MODES:
            raise ValueError(
                f"cache_mode must be one of {sorted(_FIT_MODES)}, got {cache_mode!r}"
            )
        # The exact model id callers select via the API's `model` field,
        # named for the checkpoint tabpfn will actually load. That is
        # whatever its `model_version` setting (TABPFN_MODEL_VERSION)
        # resolves to; tabpfn >= 9 defaults to v3.5, a different model
        # (and license) from v3, so a fixed id would mislabel it.
        self.model_version = settings.tabpfn.model_version.value
        self.name = served_model_id(self.model_version)
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ensure_initialized(self._device)
        self._fit_mode = _FIT_MODES[cache_mode]
        self._last_context_bytes: int | None = None
        self._last_fit_peak_bytes: int | None = None
        # (task, index) -> the shared model module(s) on this device.
        self._shared_models: dict[tuple[str, int], Any] = {}
        os.environ.setdefault("TABPFN_MODEL_CACHE_SIZE", _BUILT_MODEL_CACHE_ENTRIES)

    def _make_estimator(self, task: Task) -> Any:
        if task == "classification":
            from tabpfn import TabPFNClassifier

            return TabPFNClassifier(device=self._device, fit_mode=self._fit_mode)
        from tabpfn import TabPFNRegressor

        return TabPFNRegressor(device=self._device, fit_mode=self._fit_mode)

    def fit(self, X: ArrayLike, y: ArrayLike, task: Task) -> Any:
        import torch

        model = self._make_estimator(task)
        device = self._device
        measuring = is_cuda(device)
        before_bytes = 0
        if measuring:
            with self.quiesced():
                before_bytes = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
        try:
            with device_scope(device):
                model.fit(np.asarray(X, dtype=float), np.asarray(y))
        except torch.cuda.OutOfMemoryError as e:
            self._last_context_bytes = None
            self._last_fit_peak_bytes = None
            raise BackendComputeError(f"CUDA OOM during fit(): {e}") from e
        except ValueError as e:
            # TabPFN's own input gate (pretraining limits: too many rows,
            # features, or classes) raises before meaningful GPU work; a
            # caller's oversized table is bad input, not a server fault.
            self._last_context_bytes = None
            raise InvalidInputError(f"tabpfn rejected the training table: {e}") from e
        # Resident vs peak: two different consumers -- see
        # backends/tabicl.py fit() for the full rationale.
        if measuring:
            with self.quiesced():
                after_bytes = torch.cuda.memory_allocated(device)
                peak_bytes = torch.cuda.max_memory_allocated(device)
            self._last_context_bytes = max(0, after_bytes - before_bytes)
            self._last_fit_peak_bytes = max(0, peak_bytes - before_bytes)
        else:
            self._last_context_bytes = None
            self._last_fit_peak_bytes = None
        for index, module in enumerate(getattr(model, "models_", None) or []):
            self._shared_models.setdefault((task, index), module)
        return model

    def host_bytes_estimate(
        self, n_rows: int, n_features: int, *, offloads: bool = False
    ) -> int:
        """Host memory a fit or predict may use (preprocessing copies):
        measured on an A100 pod at most ~530 B per cell (100,000 x 100 grew
        RSS 4.9 GiB; 30M-cell tables 6-6.7 GiB), budgeted at 600."""
        return 600 * n_rows * n_features

    def predict(
        self, payload: Any, X_test: ArrayLike, return_proba: bool = False
    ) -> PredictOutcome:
        import torch

        X_arr = np.asarray(X_test, dtype=float)
        try:
            with device_scope(self._device):
                return self._predict(payload, X_arr, return_proba)
        except torch.cuda.OutOfMemoryError as e:
            raise BackendComputeError(f"CUDA OOM during predict(): {e}") from e

    def _predict(
        self, model: Any, X_arr: np.ndarray, return_proba: bool
    ) -> PredictOutcome:
        if hasattr(model, "predict_proba"):
            # Single pass regardless of return_proba (protocol
            # requirement): derive labels from probabilities.
            proba = model.predict_proba(X_arr)
            return classification_outcome(proba, model.classes_, return_proba)
        return PredictOutcome(predictions=model.predict(X_arr).tolist())

    # ---- spillover serialization (see cache/spill.py) ----------------

    def dumps_payload(self, payload: Any) -> bytes:
        """Pickle a fitted estimator for the spill tier WITHOUT the shared
        model (pickled whole, a TabPFN payload was ~900 MB of weights, and a
        restore built a private copy again): every reference to a shared
        module is written as a (task, index) token instead."""
        shared = {
            id(m): ("tabpfn-model", task, i)
            for (task, i), m in self._shared_models.items()
        }
        buf = io.BytesIO()
        _SharedModelPickler(buf, shared).dump(payload)
        return _SPILL_TAG + buf.getvalue()

    def loads_payload(self, data: bytes) -> Any:
        """Inverse of dumps_payload, re-attaching this device's shared model
        (loading it first after a restart). Untagged blobs from older
        versions are plain pickles and still load."""
        if not data.startswith(_SPILL_TAG):
            with device_scope(self._device):
                return pickle.loads(data)

        def resolve(pid: Any) -> Any:
            _, task, index = pid
            if (task, index) not in self._shared_models:
                self.warmup()
            return self._shared_models[(task, index)]

        with device_scope(self._device):
            return _SharedModelUnpickler(
                io.BytesIO(data[len(_SPILL_TAG) :]), resolve
            ).load()

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
        # Real measurement from the fit() this is queried after; None on
        # CPU, where the engine falls back to its MemoryEstimator. NOTE:
        # that estimator's static calibration is TabICL-on-A100 data --
        # treat capacity numbers for TabPFN deployments as provisional
        # until real fits accumulate in the AdaptiveMemoryEstimator.
        return self._last_context_bytes

    def fit_peak_bytes_hint(self) -> int | None:
        """Transient high-water memory of the most recent fit() -- the
        admission-relevant quantity; None on CPU/after OOM."""
        return self._last_fit_peak_bytes
