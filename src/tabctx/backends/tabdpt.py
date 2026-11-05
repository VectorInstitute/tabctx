"""TabDPT backend (github.com/layer6ai-labs/TabDPT-inference, Layer 6).

TabDPT's code and weights are both Apache-2.0, so with TabICLv2 (BSD-3)
it gives tabctx a second model a business can serve without a separate
license.

How it differs from TabICL and TabPFN: TabDPT keeps no encoding of the
training table. Its fit stores the (normalized) table; each predict
draws a context of at most `context_size` rows from it (a subsample, the
same for every test row of a call) and runs the transformer over context
and test rows together. Unbounded, the context is the whole table, and
a predict's cost grows with the table's square (8,000 rows: 45 s on a
CPU, against 4 s at 2,048). So a cached TabDPT context is its table, and
what tabctx adds for it is serving: one endpoint beside the other
models, admission control, tenancy and list/delete.

One network per device. A TabDPT estimator loads the full checkpoint in
its constructor; built per fit, every tenant's context would hold its
own copy of the weights. TabDPT supports sharing (its `_load_model`
docstring): each fit builds an estimator whose `_load_model` hands back
this device's network, the same one for both tasks (the task picks the
head). The spill tier pickles a context without it.

Predictions are repeatable: a fixed seed for the context subsample and
the ensemble's permutations. Labels go to TabDPT as codes 0..k-1 (it
indexes outputs by label value) and come back as given.
"""

from __future__ import annotations

import io
import logging
import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np

from tabctx.backends._cuda import device_scope, ensure_initialized, is_cuda
from tabctx.backends._outcome import classification_outcome, regression_outcome
from tabctx.errors import BackendComputeError, InvalidInputError
from tabctx.labels import regression_hint
from tabctx.types import ArrayLike, PredictOutcome, Task

logger = logging.getLogger(__name__)

# Rows of training context per predict (TabDPT's own evaluation setting).
DEFAULT_CONTEXT_SIZE = 2048
# Ensemble members per predict, for both tasks. TabDPT's default is 8;
# measured on a CPU, 8 cost 8x the time of 1 for a 2-3% lower loss.
DEFAULT_ENSEMBLES = 4
# TabDPT samples the context and each member's permutation at random; a
# fixed seed makes the same request give the same answer.
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
    def __init__(
        self,
        device: str | None = None,
        context_size: int = DEFAULT_CONTEXT_SIZE,
        n_ensembles: int = DEFAULT_ENSEMBLES,
    ) -> None:
        import torch
        from tabdpt import estimator as tabdpt_estimator

        if context_size < 1 or n_ensembles < 1:
            raise ValueError("context_size and n_ensembles must be positive")
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ensure_initialized(self._device)
        self._weights_version = tabdpt_estimator._VERSION
        self.name = served_model_id(self._weights_version)
        self._spill_tag = f"D1:{self._weights_version}:".encode()
        self._context_size = context_size
        self._n_ensembles = n_ensembles
        self._network: Any = None  # this device's, shared by every context
        self._last_context_bytes: int | None = None

    # ---- the shared network --------------------------------------------

    def _new_estimator(self, task: Task) -> Any:
        """A fresh estimator for one fit, on this device's network (loaded
        on first use). TabDPT enables flash attention only for the device
        string "cuda" exactly, so it is set for "cuda:N" too."""
        from tabdpt import TabDPTClassifier, TabDPTRegressor

        cls = TabDPTClassifier if task == "classification" else TabDPTRegressor
        options = {"device": self._device, "compile": False, "verbose": False}
        estimator = cls.__new__(cls)
        if self._network is not None:
            network = self._network
            estimator._load_model = lambda: setattr(estimator, "model", network)
        cls.__init__(estimator, **options)
        estimator.__dict__.pop("_load_model", None)
        flash = is_cuda(self._device)
        estimator.use_flash = flash
        estimator.model.use_flash = flash
        self._network = estimator.model
        return estimator

    # ---- protocol ------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike, task: Task) -> TabDPTContext:
        import torch

        X_arr = np.asarray(X, dtype=float)
        classes = None
        if task == "classification":
            classes, y_arr = _label_codes(y)
        else:
            y_arr = np.asarray(y, dtype=float)
            if len(y_arr) < 2:
                raise InvalidInputError(
                    "tabdpt needs at least 2 training rows for regression"
                )
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
        except (AssertionError, ValueError) as e:  # e.g. an infinite value
            raise InvalidInputError(f"tabdpt rejected the test rows: {e}") from e
        finally:
            # TabDPT keeps the last test rows on the estimator; a cached
            # context must not.
            payload.estimator.X_test = None

    def _predict(
        self,
        context: TabDPTContext,
        X_arr: np.ndarray,
        return_proba: bool,
        quantiles: tuple[float, ...] | None,
    ) -> PredictOutcome:
        estimator = context.estimator
        options = {
            "n_ensembles": self._n_ensembles,
            "context_size": self._context_size,
            "seed": _SEED,
        }
        if context.classes is not None:
            proba = estimator.ensemble_predict_proba(X_arr, **options)
            return classification_outcome(proba, context.classes, return_proba)
        # The full predictive distribution, always: its mean (so a predict
        # gives the same mean with or without quantiles) and quantiles.
        from tabdpt.bar_distribution import distribution_mean, distribution_quantiles

        full = estimator.predict(X_arr, output_type="full", **options)
        if quantiles is None:
            return regression_outcome(distribution_mean(full))
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
        """Load the network now, not inside a tenant's first fit."""
        if self._network is None:
            self._new_estimator("regression")
        self._last_context_bytes = None

    # ---- spill tier ----------------------------------------------------

    def dumps_payload(self, payload: TabDPTContext) -> bytes:
        """A context without the network (re-attached on load), tagged with
        the checkpoint version it was fit with."""
        estimator = payload.estimator
        saved = {k: getattr(estimator, k, None) for k in ("model", "faiss_knn")}
        try:
            estimator.model = None
            estimator.faiss_knn = None  # rebuilt lazily from the table
            estimator.X_test = None
            data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        finally:
            for name, value in saved.items():
                setattr(estimator, name, value)
        return self._spill_tag + data

    def loads_payload(self, data: bytes) -> TabDPTContext:
        if not data.startswith(self._spill_tag):
            raise ValueError("not a context from this TabDPT version")
        with device_scope(self._device):
            context: TabDPTContext = pickle.load(
                io.BytesIO(data[len(self._spill_tag) :])
            )
        if self._network is None:
            self.warmup()
        context.estimator.model = self._network
        return context


def _label_codes(y: ArrayLike) -> tuple[list[str], np.ndarray]:
    """(labels as strings in their natural order, each row's code). Numbers
    sort as numbers (1, 2, 10), as the other models order classes."""
    labels = np.asarray(y)
    try:
        uniques, codes = np.unique(labels, return_inverse=True)
    except TypeError:  # mixed types: order them as text
        uniques, codes = np.unique(labels.astype(str), return_inverse=True)
    return [str(u) for u in uniques], codes.astype(np.int64)


def _context_bytes(estimator: Any) -> int:
    """What a context is charged in the cache: its table at float32, plus a
    PCA projection kept on the device for wide tables. The table itself
    stays in host memory (each predict copies a context-size sample of it
    to the device); charging it here makes the cache's budget also bound
    how much TabDPT data a server keeps, and evicting it frees that."""
    total = 0
    X_train = getattr(estimator, "X_train", None)
    if X_train is not None:
        total += int(np.prod(X_train.shape)) * 4 + len(X_train) * 4
    projection = getattr(estimator, "projection", None)
    if projection is not None:
        total += projection.element_size() * projection.nelement()
    return total
