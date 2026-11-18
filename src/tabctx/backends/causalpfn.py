"""CausalPFN backend (github.com/vdblm/CausalPFN; Balazadeh, Kamkari,
Thomas, Ma, Li, Cresswell and Krishnan, NeurIPS 2025): the effect of a
binary treatment on an outcome, estimated from observational data.

The task (tabctx.tasks): each training row has an outcome and a
treatment flag (1 treated, 0 control). A predict gives each row's
conditional average treatment effect (CATE), the expected outcome if
treated minus if not:

- predictions: the CATE, the mean of the model's outcome distribution
  under treatment minus its mean under control,
- potential_outcomes: [expected outcome if not treated, if treated]
  (the two means the CATE is the difference of),
- quantiles (levels in (0, 1)): quantiles of a treated outcome minus an
  independent control outcome, from the model's two distributions, which
  is how CausalPFN makes its intervals. CausalPFN draws random samples;
  here each row's two distributions are read at a fixed grid of levels,
  paired the same fixed way for every row, so a row gets the same answer
  every time. The distributions are calibrated, as CausalPFN's
  `calibrate=True` does, at the context's first predict that asks for
  quantiles (so a fit never asked for them never pays for it): a temperature chosen by 3-fold cross-validation
  so that the model's outcome intervals cover the held-out outcomes at
  their stated rates (computed exactly here; CausalPFN samples). Like
  CausalPFN's, they describe the spread of an individual's effect, not a
  confidence interval for the CATE.

The average effect over a population (ATE) is the mean of its rows'
CATEs: predict on those rows and average.

The estimate is only causal under CausalPFN's assumptions, which no data
can check: every confounder of treatment and outcome is among the
features (no unmeasured confounding), and every kind of row could have
been treated or not (overlap).

How it is served, and how that differs from CausalPFN's own estimator:
the context is the training table, at most `context_size` rows of it (a
larger table is subsampled at fit, seeded, keeping each arm's share and
the spread of outcomes within it), and
every test row is predicted against that same context. CausalPFN's
estimator instead picks a context per group of test rows (the training
rows nearest them by a gradient-boosting model's effect estimate), so a
row's answer depends on which other rows share its call, and a server
merges concurrent requests and splits large ones. Test rows attend only
to the context, never to each other, so here a row's answer depends on
the row alone. The model was trained on contexts up to 2,048 rows;
CausalPFN's default context is 4,096, the default here. A larger context
costs predict time (roughly with the context's square: 2,000 test rows
took 0.26 s against 4,096 rows and 101 s against 100,000, on an A40) and
bought no accuracy on one simulated dataset (PEHE, the root-mean-square
error of each row's effect: 0.35 at 2,048 rows, 0.34 at 100,000);
TABCTX_CAUSALPFN_CONTEXT_SIZE raises it.

Missing feature values are filled with the training table's column
medians. A table wider than the model's 99 features is reduced by
truncated SVD, as CausalPFN does, to at most one component fewer than
its rows.

One network per device, shared by every context, loaded once as tensors
only (backends/_pfn_common.py). A context holds no network: the spill
tier pickles it as it is.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from tabctx.backends._cuda import device_scope, ensure_initialized
from tabctx.backends._outcome import regression_outcome
from tabctx.backends._pfn_common import (
    float32_rows,
    load_checkpoint,
    strata,
    subsample,
    tagged_dumps,
    tagged_loads,
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

DEFAULT_CONTEXT_SIZE = 4096
# Test rows per forward pass (each is queried as control and as treated).
DEFAULT_BATCH_ROWS = 1024
# Points of each outcome distribution read for quantiles.
QUANTILE_POINTS = 2000
_SEED = 0


@dataclass
class CausalPFNContext:
    """The context table, as the model gets it (missing values filled, a
    wide table reduced), and how to do the same to test rows."""

    X: np.ndarray  # float32 (rows, features)
    treated: np.ndarray  # float32 0/1
    outcome: np.ndarray  # float32
    medians: np.ndarray  # each raw column's training median
    svd: Any  # a fitted TruncatedSVD for a wide table, else None
    nbytes: int
    # The calibrated temperature of the outcome distributions for
    # quantiles: found at the first predict that asks for quantiles (None
    # until then), so a fit never asked for them never pays for it.
    temperature: float | None = None

    def features(self, X: np.ndarray) -> np.ndarray:
        X = _fill(X, self.medians)
        if self.svd is not None:
            X = self.svd.transform(X).astype(np.float32)
        return X


class CausalPFNBackend:
    name = MODEL_ID
    tasks = frozenset({"causal"})

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
        self._network_lock = threading.Lock()
        self._spill_tag = f"C2:{MODEL_ID}:".encode()
        self._last_context_bytes: int | None = None
        # One fixed pairing of the two arms' points, the same for every row.
        self._pairing = np.random.default_rng(_SEED).permutation(QUANTILE_POINTS)

    def _shared_network(self) -> Any:
        with self._network_lock:  # two first fits must not load it twice
            return self._load_network()

    def _load_network(self) -> Any:
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

    # ---- protocol ------------------------------------------------------

    def prepare_fit(self, X: ArrayLike, y: ArrayLike, task: Task) -> CausalPFNContext:
        """The whole fit, on the CPU: subsample, fill missing values, reduce
        a wide table. Its size travels with it: fits prepare concurrently,
        and fit_prepared reports it under the device's lock."""
        from scipy.sparse.linalg import ArpackError
        from sklearn.decomposition import TruncatedSVD

        if task != "causal":
            raise InvalidInputError(f"{self.name} fits causal, not {task}")
        X_arr = float32_rows(X)
        outcome, treated = (a.astype(np.float32) for a in split_targets(y))
        keep = subsample(
            strata(treated, outcome, self._context_size), self._context_size, _SEED
        )
        X_arr, outcome, treated = X_arr[keep], outcome[keep], treated[keep]
        medians = _column_medians(X_arr)
        X_arr = _fill(X_arr, medians)
        svd = None
        width = self._shared_network().model_config["model"]["max_num_features"] - 1
        if X_arr.shape[1] > width:
            components = min(width, min(X_arr.shape) - 1)
            if components < 1:
                raise InvalidInputError(
                    f"{self.name} reduces a table wider than {width} features, "
                    "which needs at least 2 rows"
                )
            svd = TruncatedSVD(
                n_components=components, algorithm="arpack", random_state=_SEED
            )
            try:
                X_arr = svd.fit_transform(X_arr).astype(np.float32)
            except ArpackError as e:  # e.g. every column constant
                raise InvalidInputError(
                    f"{self.name} couldn't reduce this wide table (too few "
                    f"columns that vary): {e}"
                ) from e
        nbytes = sum(a.nbytes for a in (X_arr, treated, outcome, medians))
        if svd is not None:
            nbytes += svd.components_.nbytes
        return CausalPFNContext(X_arr, treated, outcome, medians, svd, nbytes)

    def fit_prepared(self, prepared: CausalPFNContext) -> CausalPFNContext:
        self._last_context_bytes = prepared.nbytes
        return prepared

    def fit(self, X: ArrayLike, y: ArrayLike, task: Task) -> CausalPFNContext:
        return self.fit_prepared(self.prepare_fit(X, y, task))

    def predict(
        self,
        payload: CausalPFNContext,
        X_test: ArrayLike,
        return_proba: bool = False,
        quantiles: tuple[float, ...] | None = None,
        potential_outcomes: bool = False,
    ) -> PredictOutcome:
        import torch

        del return_proba  # a causal context has no class probabilities
        X_arr = payload.features(float32_rows(X_test))
        parts = []
        try:
            with device_scope(self._device), torch.no_grad():
                if quantiles is not None and payload.temperature is None:
                    # Predicts of a context run one at a time per device.
                    payload.temperature = calibrated_temperature(
                        self._shared_network(), payload, self._device
                    )
                for start in range(0, len(X_arr), self._batch_rows):
                    batch = X_arr[start : start + self._batch_rows]
                    parts.append(self._effects(payload, batch, quantiles))
        except torch.cuda.OutOfMemoryError as e:
            raise BackendComputeError(f"CUDA OOM during predict(): {e}") from e
        effect = np.concatenate([p[0] for p in parts])
        if not np.isfinite(effect).all():
            raise InvalidInputError(
                f"{self.name} overflowed on this table's outcomes (too large for "
                "float32): rescale the outcome, e.g. to thousands"
            )
        outcome = (
            regression_outcome(effect)
            if quantiles is None
            else regression_outcome(
                effect, np.concatenate([p[1] for p in parts]), len(quantiles)
            )
        )
        if not potential_outcomes:
            return outcome
        both = np.concatenate([p[2] for p in parts])
        return dataclasses.replace(outcome, potential_outcomes=both.tolist())

    def _effects(
        self,
        context: CausalPFNContext,
        X: np.ndarray,
        quantiles: tuple[float, ...] | None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """(CATE per row, (rows, levels) quantiles or None, (rows, 2)
        expected outcomes if not treated and if treated) for one batch:
        each row queried as control and as treated, in one pass."""
        import torch

        network = self._shared_network()
        n = len(X)
        logits, (y0_shift, y0_scale, y1_shift, y1_scale) = outcome_logits(
            network,
            context,
            np.concatenate([X, X]),
            np.concatenate([np.zeros(n), np.ones(n)]).astype(np.float32),
            self._device,
        )
        centers = network.bin_centers
        # Means at temperature 1, as CausalPFN's point estimates are.
        mean = torch.softmax(logits, dim=-1) @ centers
        mu0 = mean[:n] * y0_scale + y0_shift
        mu1 = mean[n:] * y1_scale + y1_shift
        effect = (mu1 - mu0).cpu().numpy().astype(float)
        both = torch.stack([mu0, mu1], dim=1).cpu().numpy().astype(float)
        if quantiles is None:
            return effect, None, both
        # Each row's two calibrated distributions read at the same fixed
        # levels, and a treated point minus a control point by one fixed
        # pairing.
        probs = torch.softmax(logits / (context.temperature or 1.0), dim=-1)
        levels = (torch.arange(QUANTILE_POINTS, device=probs.device) + 0.5) / (
            QUANTILE_POINTS
        )
        points = centers[_read_at(probs, levels)]
        control = points[:n] * y0_scale + y0_shift
        treated = points[n:] * y1_scale + y1_shift
        pairing = torch.as_tensor(self._pairing, device=probs.device)
        diff = treated - control[:, pairing]
        q = torch.tensor(quantiles, dtype=diff.dtype, device=diff.device)
        spread = torch.quantile(diff, q, dim=1).T.cpu().numpy().astype(float)
        return effect, spread, both

    def context_bytes_hint(self, n_train: int, n_features: int) -> int | None:
        del n_train, n_features
        return self._last_context_bytes

    # ---- spill tier ----------------------------------------------------

    def dumps_payload(self, payload: CausalPFNContext) -> bytes:
        return tagged_dumps(payload, self._spill_tag)

    def loads_payload(self, data: bytes) -> CausalPFNContext:
        return tagged_loads(data, self._spill_tag)


def outcome_logits(
    network: Any,
    context: CausalPFNContext | tuple[np.ndarray, np.ndarray, np.ndarray],
    X_query: np.ndarray,
    t_query: np.ndarray,
    device: str,
) -> tuple[Any, tuple[Any, Any, Any, Any]]:
    """The model's outcome distribution for each query row under its
    treatment value: (rows, bins) logits over network.bin_centers, in the
    standardized scale, and the (control shift, control scale, treated
    shift, treated scale) that map it back. This is CausalPFN's
    InContextModel.predict_cepo up to its logits. context: a context, or
    its (X, treated, outcome) arrays."""
    import torch

    def tensor(a: np.ndarray) -> Any:
        return torch.from_numpy(np.ascontiguousarray(a)).to(device).float().unsqueeze(0)

    if isinstance(context, CausalPFNContext):
        context = (context.X, context.treated, context.outcome)
    X_c, t_c, y_c = (tensor(a) for a in context)
    X_q, t_q = tensor(X_query), tensor(t_query)
    y0_shift, y0_scale, y1_shift, y1_scale = network._get_y0_y1_shift_scale(t_c, y_c)
    y_standard = torch.where(
        t_c == 1, (y_c - y1_shift) / y1_scale, (y_c - y0_shift) / y0_scale
    )
    rows = torch.cat(
        [
            torch.cat([t_c.unsqueeze(-1), X_c], dim=2),
            torch.cat([t_q.unsqueeze(-1), X_q], dim=2),
        ],
        dim=1,
    )
    x_src, y_src = network.prepare_input(rows, y_standard)
    logits = network.model(x_src.transpose(0, 1), y_src.transpose(0, 1)).transpose(
        0, 1
    )[:, :, -network.nbins :]
    scale = tuple(v.squeeze() for v in (y0_shift, y0_scale, y1_shift, y1_scale))
    return logits[0], scale


def _read_at(probs: Any, levels: Any) -> Any:
    """Each row's bin index at each level (rows, levels): the first bin
    whose cumulative probability reaches it."""
    import torch

    cdf = torch.cumsum(probs, dim=-1)
    at = levels.to(cdf.dtype).expand(*cdf.shape[:-1], len(levels)).contiguous()
    return torch.searchsorted(cdf, at).clamp(max=probs.shape[-1] - 1)


# Calibration: CausalPFN's grid of temperatures and interval levels.
TEMPERATURES = np.logspace(np.log10(0.001), np.log10(10.0), 500)
_ALPHAS = np.linspace(0.01, 1.0, 10)
CALIBRATION_FOLDS = 3
# Below this many rows per fold there is too little to calibrate on;
# above this many held-out rows per fold, a sample of them (bounding the
# calibration's time and its memory: batch x rows x bins floats).
_MIN_CALIBRATION_ROWS = 10
_MAX_HELD_ROWS = 1024
_TEMPERATURE_BATCH = 10


def calibrated_temperature(
    network: Any, context: CausalPFNContext, device: str
) -> float:
    """The temperature whose outcome intervals best cover held-out
    outcomes: CausalPFN's calibration (3 folds; for each temperature, the
    mean gap between each interval's coverage and its stated rate, over
    10 interval widths), with intervals read exactly from each held-out
    row's distribution instead of from samples. 1.0 for a table too small
    to calibrate on."""
    import torch

    n = len(context.X)
    if n < CALIBRATION_FOLDS * _MIN_CALIBRATION_ROWS:
        return 1.0
    order = np.random.default_rng(_SEED).permutation(n)
    temperatures = torch.tensor(TEMPERATURES, device=device, dtype=torch.float32)
    alphas = torch.tensor(_ALPHAS, device=device, dtype=torch.float32)
    levels = torch.cat([alphas / 2, 1 - alphas / 2])
    gap = torch.zeros(len(TEMPERATURES), device=device)
    centers = network.bin_centers
    for held in np.array_split(order, CALIBRATION_FOLDS):
        rest = np.setdiff1d(order, held)
        held = held[:_MAX_HELD_ROWS]  # already in random order
        arrays = (context.X[rest], context.treated[rest], context.outcome[rest])
        t_held = context.treated[held]
        logits, (y0_shift, y0_scale, y1_shift, y1_scale) = outcome_logits(
            network, arrays, context.X[held], t_held, device
        )
        treated = torch.as_tensor(t_held, device=device) == 1
        shift = torch.where(treated, y1_shift, y0_shift)[:, None]
        scale = torch.where(treated, y1_scale, y0_scale)[:, None]
        y = torch.as_tensor(context.outcome[held], device=device)[:, None]
        for start in range(0, len(TEMPERATURES), _TEMPERATURE_BATCH):
            batch = temperatures[start : start + _TEMPERATURE_BATCH]
            probs = torch.softmax(logits[None] / batch[:, None, None], dim=-1)
            bounds = centers[_read_at(probs, levels)] * scale + shift
            low, high = bounds[..., : len(alphas)], bounds[..., len(alphas) :]
            covered = ((low <= y) & (y <= high)).float().mean(dim=1)
            gap[start : start + len(batch)] += (covered - (1 - alphas)).abs().mean(-1)
    return float(TEMPERATURES[int(torch.argmin(gap))])


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
