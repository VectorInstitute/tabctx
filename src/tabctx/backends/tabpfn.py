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

from typing import Any

import numpy as np

from tabctx.errors import BackendComputeError, InvalidInputError
from tabctx.types import ArrayLike, PredictOutcome, Task

_FIT_MODES = {
    "kv": "fit_with_cache",
    "repr": "fit_preprocessors",
    "off": "low_memory",
}


class TabPFNBackend:
    name = "tabpfn"

    def __init__(self, device: str | None = None, cache_mode: str = "kv") -> None:
        import torch

        if cache_mode not in _FIT_MODES:
            raise ValueError(
                f"cache_mode must be one of {sorted(_FIT_MODES)}, got {cache_mode!r}"
            )
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._fit_mode = _FIT_MODES[cache_mode]
        self._last_context_bytes: int | None = None
        self._last_fit_peak_bytes: int | None = None

    def _make_estimator(self, task: Task) -> Any:
        if task == "classification":
            from tabpfn import TabPFNClassifier

            return TabPFNClassifier(device=self._device, fit_mode=self._fit_mode)
        from tabpfn import TabPFNRegressor

        return TabPFNRegressor(device=self._device, fit_mode=self._fit_mode)

    def fit(self, X: ArrayLike, y: ArrayLike, task: Task) -> Any:
        import torch

        model = self._make_estimator(task)
        measuring = self._device == "cuda"
        before_bytes = torch.cuda.memory_allocated() if measuring else 0
        if measuring:
            torch.cuda.reset_peak_memory_stats()
        try:
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
            self._last_context_bytes = max(
                0, torch.cuda.memory_allocated() - before_bytes
            )
            self._last_fit_peak_bytes = max(
                0, torch.cuda.max_memory_allocated() - before_bytes
            )
        else:
            self._last_context_bytes = None
            self._last_fit_peak_bytes = None
        return model

    def predict(
        self, payload: Any, X_test: ArrayLike, return_proba: bool = False
    ) -> PredictOutcome:
        import torch

        model = payload
        X_arr = np.asarray(X_test, dtype=float)
        try:
            if hasattr(model, "predict_proba"):
                # Single pass regardless of return_proba (protocol
                # requirement): derive labels from probabilities.
                proba = model.predict_proba(X_arr)
                class_idx = np.argmax(proba, axis=1)
                classes = [str(c) for c in model.classes_]
                predictions = [classes[i] for i in class_idx]
                if return_proba:
                    return PredictOutcome(
                        predictions=predictions,
                        probabilities=proba.tolist(),
                        classes=classes,
                    )
                return PredictOutcome(predictions=predictions)
            return PredictOutcome(predictions=model.predict(X_arr).tolist())
        except torch.cuda.OutOfMemoryError as e:
            raise BackendComputeError(f"CUDA OOM during predict(): {e}") from e

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
