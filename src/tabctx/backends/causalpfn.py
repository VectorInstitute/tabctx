"""CausalPFN backend (github.com/vdblm/CausalPFN; Balazadeh, Kamkari,
Thomas, Ma, Li, Cresswell and Krishnan, NeurIPS 2025): the effect of a
binary treatment on an outcome, estimated from observational data.

The task (tabctx.tasks): each training row has an outcome and a
treatment flag (1 treated, 0 control). A predict gives each row's
conditional average treatment effect (CATE), the expected outcome if
treated minus if not:

- predictions: the CATE (the model's mean outcome under treatment minus
  under control),
- quantiles (levels in (0, 1)): quantiles of a treated sample minus a
  control sample from the model's two outcome distributions, which is how
  CausalPFN makes its own intervals, seeded so a predict is repeatable.
  With CausalPFN's temperature calibration off (as here), they are wide:
  on simulated data they grew with the outcome's noise, and a 5-95% range
  held the true effect for every row. Read them as the spread of an
  individual's effect, not as a confidence interval for the CATE.

The average effect over a population (ATE) is the mean of its rows'
CATEs: predict on those rows and average.

The estimate is only causal under CausalPFN's assumptions, which no data
can check: every confounder of treatment and outcome is among the
features (no unmeasured confounding), and every kind of row could have
been treated or not (overlap).

How it is served: CausalPFN keeps no encoding of the table. Its fit
stores the table and trains a small gradient-boosting model (100 trees,
on the CPU) whose effect estimates pick, for each group of test rows,
the up to 4,096 training rows most like them as the context. That fit is
CPU work, so it runs as the fit's prepare step (prepare_fit), before the
fit takes the device. Missing feature values are filled with the
training table's column medians (the gradient-boosting model can't take
them); a table wider than the model's 99 features is reduced by
truncated SVD, as CausalPFN does.

One network per device (backends/_pfn_common.py): each fit builds a
CATEEstimator whose load_model hands it this device's network.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np

from tabctx.backends._cuda import device_scope, ensure_initialized, is_cuda
from tabctx.backends._outcome import regression_outcome
from tabctx.backends._pfn_common import (
    array_bytes,
    dumps_context,
    float32_rows,
    load_checkpoint,
    loads_context,
)
from tabctx.errors import BackendComputeError, InvalidInputError
from tabctx.tasks import split_targets
from tabctx.types import ArrayLike, PredictOutcome, Task

logger = logging.getLogger(__name__)

REPO_ID = "vdblm/causalpfn"
CHECKPOINT = "causalpfn_v0.pt"
# The Hugging Face commit of causalpfn_v0.pt.
REVISION = "83aad07da1cb077cfda4236878a1b07dc9f72a54"
MODEL_ID = "causalpfn-v0"

# Samples per outcome distribution for quantiles.
QUANTILE_SAMPLES = 2000
_SEED = 0


@dataclass
class CausalPFNContext:
    """A fitted CATEEstimator, and the training columns' medians (filling
    missing values the same way at predict)."""

    estimator: Any
    medians: np.ndarray


class CausalPFNBackend:
    name = MODEL_ID
    tasks = frozenset({"causal"})

    def __init__(self, device: str | None = None) -> None:
        import torch

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ensure_initialized(self._device)
        self._network: Any = None
        self._spill_tag = f"C1:{MODEL_ID}:".encode()
        self._last_context_bytes: int | None = None

    def _shared_network(self) -> Any:
        if self._network is None:
            from causalpfn.models import InContextModel

            ckpt = load_checkpoint(REPO_ID, CHECKPOINT, REVISION)
            network = InContextModel.load(
                model_state=ckpt["model_state_dict"], model_config=ckpt["model_config"]
            )
            self._network = network.to(self._device).eval()
        return self._network

    def warmup(self) -> None:
        """Load the network now, not inside a tenant's first fit."""
        self._shared_network()

    def _new_estimator(self) -> Any:
        """A CATEEstimator for one fit, on this device's network. Its own
        load_model reads the checkpoint (with weights_only=False) on every
        fit; this one sets what that sets from the shared network."""
        from causalpfn import CATEEstimator
        from sklearn.decomposition import TruncatedSVD

        estimator = CATEEstimator(device=self._device)
        network = self._shared_network()

        def load_model() -> None:
            estimator.icl_model = network
            config = network.model_config
            if config["model_type"] == "tabdpt":
                width = config["model"]["max_num_features"] - 1
                estimator.max_feature_size = width
                estimator.x_dim_transformer = TruncatedSVD(
                    n_components=width, algorithm="arpack"
                )

        estimator.load_model = load_model
        return estimator

    # ---- protocol ------------------------------------------------------

    def prepare_fit(self, X: ArrayLike, y: ArrayLike, task: Task) -> CausalPFNContext:
        """The whole fit, on the CPU: fill missing values, reduce a wide
        table, train the gradient-boosting model that picks contexts."""
        if task != "causal":
            raise InvalidInputError(f"{self.name} fits causal, not {task}")
        X_arr = float32_rows(X)
        outcome, treatment = (a.astype(np.float32) for a in split_targets(y))
        medians = _column_medians(X_arr)
        X_arr = _fill(X_arr, medians)
        estimator = self._new_estimator()
        try:
            estimator.fit(X_arr, treatment, outcome)
        except ValueError as e:  # e.g. a wide table too small to reduce
            self._last_context_bytes = None
            raise InvalidInputError(
                f"{self.name} rejected the training table: {e}"
            ) from e
        finally:
            estimator.__dict__.pop("load_model", None)
        # On the device: fit_prepared attaches the network. (The load_model
        # closure is gone too: the spill tier can't pickle one.)
        estimator.icl_model = None
        context = CausalPFNContext(estimator=estimator, medians=medians)
        self._last_context_bytes = array_bytes(
            estimator.X_train, estimator.t_train, estimator.y_train, medians
        ) + len(pickle.dumps(estimator.stratifier))
        return context

    def fit_prepared(self, prepared: CausalPFNContext) -> CausalPFNContext:
        prepared.estimator.icl_model = self._shared_network()
        return prepared

    def fit(self, X: ArrayLike, y: ArrayLike, task: Task) -> CausalPFNContext:
        return self.fit_prepared(self.prepare_fit(X, y, task))

    def predict(
        self,
        payload: CausalPFNContext,
        X_test: ArrayLike,
        return_proba: bool = False,
        quantiles: tuple[float, ...] | None = None,
    ) -> PredictOutcome:
        import torch

        del return_proba  # a causal context has no class probabilities
        X_arr = _fill(float32_rows(X_test), payload.medians)
        try:
            with device_scope(self._device), _seeded(self._device):
                effect, samples = _cate(
                    payload.estimator,
                    X_arr,
                    None if quantiles is None else QUANTILE_SAMPLES,
                )
        except torch.cuda.OutOfMemoryError as e:
            raise BackendComputeError(f"CUDA OOM during predict(): {e}") from e
        except ValueError as e:
            raise InvalidInputError(f"{self.name} rejected the test rows: {e}") from e
        if quantiles is None:
            return regression_outcome(effect)
        assert samples is not None  # asked for with the quantiles
        levels = [float(q) for q in quantiles]
        return regression_outcome(
            effect, np.quantile(samples, levels, axis=1).T, len(levels)
        )

    def context_bytes_hint(self, n_train: int, n_features: int) -> int | None:
        del n_train, n_features
        return self._last_context_bytes

    # ---- spill tier ----------------------------------------------------

    def dumps_payload(self, payload: CausalPFNContext) -> bytes:
        return dumps_context(payload, self._spill_tag)

    def loads_payload(self, data: bytes) -> CausalPFNContext:
        return loads_context(
            data, self._spill_tag, self._shared_network(), self._device
        )


def _cate(
    estimator: Any, X: np.ndarray, n_samples: int | None
) -> tuple[np.ndarray, np.ndarray | None]:
    """(CATE per row, CATE samples per row or None): CATEEstimator's
    estimate_cate, from one pass that also samples when asked to."""
    estimator._check_fitted()
    if (
        estimator.max_feature_size is not None
        and X.shape[1] > estimator.max_feature_size
    ):
        X = estimator.x_dim_transformer.transform(X).astype(np.float32)
    n = X.shape[0]
    result = estimator._predict_cepo(
        X_context=estimator.X_train,
        t_context=estimator.t_train,
        y_context=estimator.y_train,
        X_query=np.concatenate([X, X], axis=0),
        t_query=np.concatenate(
            [np.zeros(n, dtype=X.dtype), np.ones(n, dtype=X.dtype)], axis=0
        ),
        temperature=estimator.prediction_temperature,
        n_samples=n_samples,
    )
    if n_samples is None:
        mu = np.asarray(result)
        return mu[n:] - mu[:n], None
    mu, draws = (np.asarray(r) for r in result)
    return mu[n:] - mu[:n], draws[n:] - draws[:n]


def _column_medians(X: np.ndarray) -> np.ndarray:
    """Each column's median over its present values (0 for a column with
    none)."""
    with np.errstate(all="ignore"):
        present = ~np.isnan(X).all(axis=0)
        medians = np.zeros(X.shape[1], dtype=np.float32)
        if present.any():
            medians[present] = np.nanmedian(X[:, present], axis=0)
    return medians


def _fill(X: np.ndarray, medians: np.ndarray) -> np.ndarray:
    missing = np.isnan(X)
    if not missing.any():
        return X
    return np.where(missing, medians[None, :], X).astype(np.float32)


class _seeded:
    """Seed torch's generators for one predict, restoring them after."""

    def __init__(self, device: str) -> None:
        self._device = device
        self._fork: Any = None

    def __enter__(self) -> None:
        import torch

        devices = [torch.device(self._device)] if is_cuda(self._device) else []
        self._fork = torch.random.fork_rng(devices=devices)
        self._fork.__enter__()
        torch.manual_seed(_SEED)

    def __exit__(self, *exc: object) -> None:
        self._fork.__exit__(*exc)
