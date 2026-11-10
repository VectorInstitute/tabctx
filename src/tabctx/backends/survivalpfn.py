"""SurvivalPFN backend (github.com/rgklab/SurvivalPFN; Qi, Balazadeh,
Cooper, Greiner and Krishnan, 2026): time-to-event prediction under right
censoring. Code and weights are Apache-2.0.

The task (tabctx.tasks): each training row has a time and an event flag
(1: the event happened then; 0: censored, still event-free then). A
predict gives each row's event-time distribution, as

- predictions: its median event time,
- quantiles (levels in (0, 1)): its event-time quantiles,
- survival (times): the probability of no event by each time.

Each is None past the longest time in the training context: the data
says nothing beyond its follow-up, and the model puts next to no
probability there, so a median or survival probability out there would
read as an estimate when it isn't one (with heavy censoring, true
survival of 0.74 came back as 0.001 past the follow-up).

How it is served: SurvivalPFN keeps no encoding of the table. Its fit
stores the table; every predict runs the transformer over the table and
the test rows, which attend to the table only (never to each other), so
test rows are batched freely without changing their answers. Its own
estimator passes the whole table as context; here a table above
`context_size` rows is subsampled at fit, keeping the share of events,
seeded. The model was trained on contexts up to 2,048 rows (16,384 in
an earlier phase); the default keeps 4,096 (on one simulated 10,500-row
table, contexts of 1,024 to 10,000 rows scored the same C-index, while a
predict took 20 times longer at 10,000). A table wider than the model's 99 features is reduced
by truncated SVD, as SurvivalPFN does.

Assumptions the model makes, which a caller should know: censoring
independent of the event given the features, and right censoring only
(no competing risks or time-varying features). Missing feature values
are allowed (the model zero-fills them after normalizing).

One network per device (backends/_pfn_common.py): each fit builds a
SurvivalEstimator around it (its `icl_model` argument).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from tabctx.backends._cuda import device_scope, ensure_initialized
from tabctx.backends._pfn_common import (
    array_bytes,
    dumps_context,
    float32_rows,
    load_checkpoint,
    loads_context,
    subsample,
)
from tabctx.errors import BackendComputeError, InvalidInputError
from tabctx.tasks import split_targets
from tabctx.types import ArrayLike, PredictOutcome, Task

logger = logging.getLogger(__name__)

REPO_ID = "shi-ang/SurvivalPFN"
CHECKPOINT = "survivalpfn_v0.1.pt"
# The Hugging Face commit of survivalpfn_v0.1.pt.
REVISION = "a4ccf9950c73723ca20296f624e649f344e2165a"
MODEL_ID = "survivalpfn-v0.1"

DEFAULT_CONTEXT_SIZE = 4096
# Test rows per forward pass.
DEFAULT_BATCH_ROWS = 2048
# The context subsample of a table above context_size rows.
_SEED = 0


@dataclass
class SurvivalPFNContext:
    """A fitted SurvivalEstimator (its table and feature transform), and
    the longest time in it: nothing is estimated past it."""

    estimator: Any
    longest: float


class SurvivalPFNBackend:
    name = MODEL_ID
    tasks = frozenset({"survival"})

    def __init__(
        self,
        device: str | None = None,
        context_size: int = DEFAULT_CONTEXT_SIZE,
        batch_rows: int = DEFAULT_BATCH_ROWS,
    ) -> None:
        import torch

        if context_size < 2 or batch_rows < 1:
            raise ValueError("context_size must be >= 2 and batch_rows positive")
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ensure_initialized(self._device)
        self._context_size = context_size
        self._batch_rows = batch_rows
        self._network: Any = None
        self._spill_tag = f"S1:{MODEL_ID}:".encode()
        self._last_context_bytes: int | None = None

    def _shared_network(self) -> Any:
        if self._network is None:
            from survivalpfn.models import InContextModel

            ckpt = load_checkpoint(REPO_ID, CHECKPOINT, REVISION)
            network = InContextModel.load(
                model_state=ckpt["model_state_dict"], model_config=ckpt["model_config"]
            )
            self._network = network.to(self._device).eval()
        return self._network

    def warmup(self) -> None:
        """Load the network now, not inside a tenant's first fit."""
        self._shared_network()

    # ---- protocol ------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike, task: Task) -> SurvivalPFNContext:
        from survivalpfn import SurvivalEstimator

        if task != "survival":
            raise InvalidInputError(f"{self.name} fits survival, not {task}")
        X_arr = float32_rows(X)
        times, events = (a.astype(np.float32) for a in split_targets(y))
        keep = subsample(events, self._context_size, _SEED)
        X_arr, times, events = X_arr[keep], times[keep], events[keep]
        estimator = SurvivalEstimator(
            device=self._device, model_path=None, icl_model=self._shared_network()
        )
        try:
            with device_scope(self._device):
                estimator.fit(X_arr, events, times)
        except ValueError as e:  # e.g. a wide table too small to reduce
            self._last_context_bytes = None
            raise InvalidInputError(
                f"{self.name} rejected the training table: {e}"
            ) from e
        self._last_context_bytes = array_bytes(
            estimator.X_train, estimator.delta_train, estimator.T_train
        )
        return SurvivalPFNContext(estimator=estimator, longest=float(times.max()))

    def predict(
        self,
        payload: SurvivalPFNContext,
        X_test: ArrayLike,
        return_proba: bool = False,
        quantiles: tuple[float, ...] | None = None,
        times: tuple[float, ...] | None = None,
    ) -> PredictOutcome:
        import torch

        del return_proba  # a survival context has no class probabilities
        X_arr = float32_rows(X_test)
        parts = []
        try:
            with device_scope(self._device), torch.no_grad():
                for start in range(0, len(X_arr), self._batch_rows):
                    batch = X_arr[start : start + self._batch_rows]
                    parts.append(self._predict(payload, batch, quantiles, times))
        except torch.cuda.OutOfMemoryError as e:
            raise BackendComputeError(f"CUDA OOM during predict(): {e}") from e
        except ValueError as e:  # the estimator's own input checks
            raise InvalidInputError(f"{self.name} rejected the test rows: {e}") from e
        return PredictOutcome.concat(parts)

    def _predict(
        self,
        context: SurvivalPFNContext,
        X: np.ndarray,
        quantiles: tuple[float, ...] | None,
        times: tuple[float, ...] | None,
    ) -> PredictOutcome:
        import torch

        dist = context.estimator.predict_event_distribution(X)
        limit = context.longest
        medians = _within(dist.median().cpu().numpy(), limit)
        q = None
        if quantiles is not None:
            q = _within(event_time_quantiles(dist, quantiles), limit)
        s = None
        if times is not None:
            at = torch.tensor(times, dtype=torch.float32, device=dist.bin_edges.device)
            probs = dist.survival_function(at).cpu().numpy().astype(float)
            past = np.asarray(times) > limit
            s = _within(np.where(past[None, :], np.inf, probs), np.inf)
        return PredictOutcome(predictions=medians, quantiles=q, survival=s)

    def context_bytes_hint(self, n_train: int, n_features: int) -> int | None:
        del n_train, n_features
        return self._last_context_bytes

    # ---- spill tier ----------------------------------------------------

    def dumps_payload(self, payload: SurvivalPFNContext) -> bytes:
        return dumps_context(payload, self._spill_tag)

    def loads_payload(self, data: bytes) -> SurvivalPFNContext:
        return loads_context(
            data, self._spill_tag, self._shared_network(), self._device
        )


def event_time_quantiles(dist: Any, levels: tuple[float, ...]) -> np.ndarray:
    """(rows, levels) event-time quantiles of SurvivalPFN's histogram
    distribution, interpolated within a bin as its median() is; inf
    where the level falls in the residual mass past the last bin edge."""
    import torch

    probs = dist.probs  # (rows, bins): the last is the residual mass
    cdf = torch.cumsum(probs, dim=-1)
    n_bins = cdf.shape[-1]
    at = torch.tensor(levels, dtype=cdf.dtype, device=cdf.device)
    at = at.expand(cdf.shape[0], len(levels)).contiguous()
    index = torch.searchsorted(cdf, at, side="left")
    residual = index >= n_bins - 1
    finite = index.clamp(min=0, max=n_bins - 2)
    cdf_left = torch.cat([torch.zeros_like(cdf[..., :1]), cdf[..., :-1]], dim=-1)
    lo = torch.gather(cdf_left, -1, finite)
    hi = torch.gather(cdf, -1, finite)
    left = torch.gather(dist.bin_edges, -1, finite)
    right = torch.gather(dist.bin_edges, -1, finite + 1)
    alpha = ((at - lo) / torch.clamp(hi - lo, min=1e-12)).clamp(0.0, 1.0)
    values = left + alpha * (right - left)
    values = torch.where(residual, torch.full_like(values, float("inf")), values)
    return values.cpu().numpy()


def _within(values: np.ndarray, limit: float) -> list:
    """Values as JSON-able floats, None where not finite (the open-ended
    last bin) or past `limit` (the training follow-up)."""
    arr = np.asarray(values, dtype=float)
    out = np.where(np.isfinite(arr) & (arr <= limit), arr, np.nan).tolist()
    if arr.ndim == 1:
        return [None if v != v else v for v in out]
    return [[None if v != v else v for v in row] for row in out]
