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
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tabctx.errors import BackendComputeError
from tabctx.types import ArrayLike, PredictOutcome, Task


class TabICLBackend:
    name = "tabicl"

    def __init__(self, device: str | None = None) -> None:
        import torch

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X: ArrayLike, y: ArrayLike, task: Task) -> Any:
        import torch

        # Fresh instance every call -- see backends/base.py docstring for why
        # this is load-bearing, not just a style choice.
        if task == "classification":
            from tabicl import TabICLClassifier

            model = TabICLClassifier(device=self._device)
        else:
            from tabicl import TabICLRegressor

            model = TabICLRegressor(device=self._device)
        try:
            model.fit(X, y)
        except torch.cuda.OutOfMemoryError as e:
            raise BackendComputeError(f"CUDA OOM during fit(): {e}") from e
        return model

    def predict(
        self, payload: Any, X_test: ArrayLike, return_proba: bool = False
    ) -> PredictOutcome:
        import torch

        model = payload
        try:
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
                predictions = model.predict(X_test).tolist()
                return PredictOutcome(predictions=predictions)
            # Regressor: no predict_proba to derive from.
            predictions = model.predict(X_test).tolist()
            return PredictOutcome(predictions=predictions)
        except torch.cuda.OutOfMemoryError as e:
            raise BackendComputeError(f"CUDA OOM during predict(): {e}") from e

    def context_bytes_hint(self, n_train: int, n_features: int) -> int | None:
        del n_train, n_features
        # No backend-specific opinion in v1 -- the engine falls back to its
        # own MemoryEstimator, calibrated empirically (see memory/estimator.py).
        return None
