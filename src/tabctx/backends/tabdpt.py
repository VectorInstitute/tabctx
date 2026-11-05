"""TabDPT backend (github.com/layer6ai-labs/TabDPT-inference, Layer 6).

TabDPT's code and weights are both Apache-2.0, so with TabICLv2 (BSD-3)
it gives tabctx a second model a business can serve without a separate
license.

How it differs from TabICL and TabPFN: TabDPT keeps no encoding of the
training table. Its fit stores the (normalized) table; each predict
draws a context from it (a random subsample by default, or the nearest
rows) and runs the transformer over context and test rows together. So
a cached TabDPT context is mostly its table, and predict costs what it
costs in-process; what tabctx adds for it is serving: one endpoint
beside the other models, admission control, tenancy and list/delete.

One network per task per device. A TabDPT estimator loads the full
checkpoint in its constructor; built per fit, every tenant's context
would hold its own copy of the weights. TabDPT supports sharing (its
`_load_model` docstring): each fit builds an estimator whose
`_load_model` hands back this device's shared network, and the spill tier
pickles a context without it (dumps_payload / loads_payload).

Labels: TabDPT's classifier indexes its outputs by label value, so
labels are sent as codes 0..k-1 and mapped back to the caller's.
"""

from __future__ import annotations

import copy
import io
import logging
import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np

from tabctx.backends._cuda import device_scope, ensure_initialized
from tabctx.backends._outcome import classification_outcome, regression_outcome
from tabctx.errors import BackendComputeError, InvalidInputError
from tabctx.labels import regression_hint
from tabctx.types import ArrayLike, PredictOutcome, Task

logger = logging.getLogger(__name__)

_SPILL_TAG = b"D1"
# TabDPT draws each predict's context and ensemble at random; a fixed seed
# makes the same request give the same answer, as the other models do.
_SEED = 0


def served_model_id(weights_version: str) -> str:
    """The API model id for TabDPT's checkpoint version: "1_3" -> "tabdpt-1.3"."""
    return f"tabdpt-{weights_version.replace('_', '.')}"


@dataclass
class TabDPTContext:
    """A fitted TabDPT estimator, and for classification the caller's
    labels in code order."""

    estimator: Any
    classes: list | None


class TabDPTBackend:
    def __init__(self, device: str | None = None) -> None:
        import torch
        from tabdpt import estimator as tabdpt_estimator

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ensure_initialized(self._device)
        self.name = served_model_id(tabdpt_estimator._VERSION)
        self._shared: dict[str, Any] = {}  # task -> this device's network
        self._last_context_bytes: int | None = None

    # ---- the shared network --------------------------------------------

    def _estimator_class(self, task: Task) -> type:
        from tabdpt import TabDPTClassifier, TabDPTRegressor

        return TabDPTClassifier if task == "classification" else TabDPTRegressor

    def _new_estimator(self, task: Task) -> Any:
        """A fresh estimator for one fit, on this device's shared network
        (loaded on first use)."""
        cls = self._estimator_class(task)
        options = {"device": self._device, "compile": False, "verbose": False}
        shared = self._shared.get(task)
        if shared is None:
            estimator = cls(**options)
            self._shared[task] = estimator.model
            return estimator

        class Sharing(cls):
            def _load_model(self) -> None:
                self.model = shared

        estimator = Sharing(**options)
        estimator.__class__ = cls  # pickles (spill) as TabDPT's own class
        return estimator

    # ---- protocol ------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike, task: Task) -> TabDPTContext:
        import torch

        X_arr = np.asarray(X, dtype=float)
        classes = None
        if task == "classification":
            labels = np.asarray(y)
            classes, codes = np.unique(labels.astype(str), return_inverse=True)
            classes = classes.tolist()
            y_arr = codes.astype(np.int64)
        else:
            y_arr = np.asarray(y, dtype=float)
        estimator = self._new_estimator(task)
        try:
            with device_scope(self._device):
                estimator.fit(X_arr, y_arr)
        except torch.cuda.OutOfMemoryError as e:
            self._last_context_bytes = None
            raise BackendComputeError(f"CUDA OOM during fit(): {e}") from e
        except (AssertionError, ValueError) as e:
            # TabDPT's own input checks (e.g. a single class).
            self._last_context_bytes = None
            raise InvalidInputError(
                f"tabdpt rejected the training table: {e}{regression_hint(y, task)}"
            ) from e
        self._last_context_bytes = _context_bytes(estimator)
        return TabDPTContext(estimator=estimator, classes=classes)

    def predict(
        self,
        payload: TabDPTContext,
        X_test: ArrayLike,
        return_proba: bool = False,
        quantiles: tuple[float, ...] | None = None,
    ) -> PredictOutcome:
        import torch

        X_arr = np.asarray(X_test, dtype=float)
        try:
            with device_scope(self._device):
                return self._predict(payload, X_arr, return_proba, quantiles)
        except torch.cuda.OutOfMemoryError as e:
            raise BackendComputeError(f"CUDA OOM during predict(): {e}") from e

    def _predict(
        self,
        context: TabDPTContext,
        X_arr: np.ndarray,
        return_proba: bool,
        quantiles: tuple[float, ...] | None,
    ) -> PredictOutcome:
        estimator = context.estimator
        if context.classes is not None:
            proba = estimator.predict_proba(X_arr, seed=_SEED)
            return classification_outcome(proba, context.classes, return_proba)
        if quantiles is None:
            return regression_outcome(estimator.predict(X_arr, seed=_SEED))
        # The full predictive distribution, once: its mean and quantiles.
        from tabdpt.bar_distribution import distribution_mean, distribution_quantiles

        full = estimator.predict(X_arr, seed=_SEED, output_type="full")
        levels = [float(q) for q in quantiles]
        return regression_outcome(
            distribution_mean(full),
            np.stack(distribution_quantiles(full, levels), axis=1),
            len(levels),
        )

    def context_bytes_hint(self, n_train: int, n_features: int) -> int | None:
        del n_train, n_features
        return self._last_context_bytes

    def warmup(self) -> None:
        """Load the network for both tasks now, not inside a tenant's fit."""
        for task in ("classification", "regression"):
            if task not in self._shared:
                self._new_estimator(task)
        self._last_context_bytes = None

    # ---- spill tier ----------------------------------------------------

    def dumps_payload(self, payload: TabDPTContext) -> bytes:
        """A context without the shared network (re-attached on load)."""
        estimator = copy.copy(payload.estimator)
        estimator.model = None
        estimator.faiss_knn = None  # rebuilt lazily from the table
        data = pickle.dumps(
            TabDPTContext(estimator=estimator, classes=payload.classes),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        return _SPILL_TAG + data

    def loads_payload(self, data: bytes) -> TabDPTContext:
        if not data.startswith(_SPILL_TAG):
            raise ValueError("not a TabDPT context")
        with device_scope(self._device):
            context: TabDPTContext = pickle.load(io.BytesIO(data[len(_SPILL_TAG) :]))
        task: Task = "classification" if context.classes is not None else "regression"
        if task not in self._shared:
            self._new_estimator(task)
        context.estimator.model = self._shared[task]
        return context


def _context_bytes(estimator: Any) -> int:
    """What a fitted TabDPT context holds beyond the shared network: its
    normalized training table (and a PCA projection for wide tables)."""
    total = 0
    for name in ("X_train", "y_train", "projection"):
        value = getattr(estimator, name, None)
        if value is None:
            continue
        nbytes = getattr(value, "nbytes", None)
        if nbytes is None and hasattr(value, "element_size"):  # a torch tensor
            nbytes = value.element_size() * value.nelement()
        total += int(nbytes or 0)
    return total
