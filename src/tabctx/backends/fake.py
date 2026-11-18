"""Deterministic backend with no GPU/torch/tabicl dependency, for unit tests.

Not a serious model: classification predicts the training majority class for
every test row, regression predicts the training mean, survival the training
times' median (and their empirical survival curve), causal the difference of
the treated and control means. What matters for
tabctx's own tests is the *shape* of the contract (fit returns an opaque
payload, predict does one pass, context_bytes_hint is controllable), not
prediction quality.
"""

from __future__ import annotations

import statistics
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from tabctx.backends._outcome import classification_outcome
from tabctx.tasks import TASKS, split_targets
from tabctx.types import ArrayLike, PredictOutcome, Task


@dataclass
class _FakePayload:
    task: Task
    n_train: int
    n_features: int
    class_fractions: dict[str, float] | None = None
    mean_y: float | None = None
    std_y: float = 0.0
    # Survival: the training times, sorted.
    times: np.ndarray | None = None


class FakeBackend:
    """Test double. `fit_delay_s`/`predict_delay_s` let tests simulate work
    without a real GPU, e.g. to assert cache reuse actually skips fit()."""

    name = "fake"
    tasks = frozenset(TASKS)

    def __init__(
        self,
        bytes_hint: int | None = None,
        fit_delay_s: float = 0.0,
        predict_delay_s: float = 0.0,
        peak_bytes_hint: int | None = None,
        name: str = "fake",
    ) -> None:
        self.name = name  # instance override (tests deploy several fakes)
        self._bytes_hint = bytes_hint
        self._peak_bytes_hint = peak_bytes_hint
        self._fit_delay_s = fit_delay_s
        self._predict_delay_s = predict_delay_s
        self.fit_calls = 0
        self.predict_calls = 0

    def fit(self, X: ArrayLike, y: ArrayLike, task: Task) -> Any:
        self.fit_calls += 1
        if self._fit_delay_s:
            time.sleep(self._fit_delay_s)
        n_train = len(X)
        n_features = len(X[0]) if n_train else 0
        if task == "classification":
            counts = Counter(str(v) for v in y)
            total = sum(counts.values()) or 1
            return _FakePayload(
                task=task,
                n_train=n_train,
                n_features=n_features,
                class_fractions={k: v / total for k, v in counts.items()},
            )
        if task == "survival":
            times, _ = split_targets(y)
            return _FakePayload(
                task=task,
                n_train=n_train,
                n_features=n_features,
                mean_y=float(np.median(times)),
                times=np.sort(times),
            )
        if task == "causal":
            outcome, treated = split_targets(y)
            effect = outcome[treated == 1].mean() - outcome[treated == 0].mean()
            return _FakePayload(
                task=task, n_train=n_train, n_features=n_features, mean_y=float(effect)
            )
        values = [float(v) for v in y]
        mean_y = statistics.fmean(values) if values else 0.0
        std_y = statistics.pstdev(values) if len(values) > 1 else 0.0
        return _FakePayload(
            task=task,
            n_train=n_train,
            n_features=n_features,
            mean_y=mean_y,
            std_y=std_y,
        )

    def predict(
        self,
        payload: _FakePayload,
        X_test: ArrayLike,
        return_proba: bool = False,
        quantiles: tuple[float, ...] | None = None,
        times: tuple[float, ...] | None = None,
        rmst_horizon: float | None = None,
        potential_outcomes: bool = False,
    ) -> PredictOutcome:
        self.predict_calls += 1
        if self._predict_delay_s:
            time.sleep(self._predict_delay_s)
        n_test = len(X_test)
        if payload.task == "classification":
            fractions = payload.class_fractions or {}
            classes = sorted(fractions)
            if not classes:
                return PredictOutcome(predictions=[None] * n_test)
            row = [fractions[c] for c in classes]
            return classification_outcome([row] * n_test, classes, return_proba)
        if payload.task == "survival":
            return _survival_outcome(payload, n_test, quantiles, times, rmst_horizon)
        if payload.task == "causal" and potential_outcomes:
            # Untreated 0, treated the effect.
            both = [[0.0, payload.mean_y or 0.0] for _ in range(n_test)]
            return PredictOutcome(
                predictions=[payload.mean_y] * n_test, potential_outcomes=both
            )
        # A normal distribution around the training mean, with its spread.
        row = None
        if quantiles is not None:
            dist = statistics.NormalDist(payload.mean_y or 0.0, payload.std_y or 1e-9)
            row = [dist.inv_cdf(q) for q in quantiles]
        return PredictOutcome(
            predictions=[payload.mean_y] * n_test,
            quantiles=None if row is None else [list(row) for _ in range(n_test)],
        )

    def context_bytes_hint(self, n_train: int, n_features: int) -> int | None:
        del n_train, n_features  # unused: this backend's hint is fixed at construction
        return self._bytes_hint

    def fit_peak_bytes_hint(self) -> int | None:
        return self._peak_bytes_hint


def _survival_outcome(
    payload: _FakePayload,
    n_test: int,
    quantiles: tuple[float, ...] | None,
    times: tuple[float, ...] | None,
    rmst_horizon: float | None = None,
) -> PredictOutcome:
    """The training times' median, quantiles, empirical survival curve and
    the mean of the times capped at the horizon, the same for every row."""
    observed = payload.times if payload.times is not None else np.zeros(1)
    q_row = None if quantiles is None else np.quantile(observed, quantiles).tolist()
    s_row = None
    if times is not None:
        s_row = [float((observed > t).mean()) for t in times]
    rmst = None
    if rmst_horizon is not None:
        rmst = [float(np.minimum(observed, rmst_horizon).mean())] * n_test
    return PredictOutcome(
        predictions=[payload.mean_y] * n_test,
        quantiles=None if q_row is None else [list(q_row) for _ in range(n_test)],
        survival=None if s_row is None else [list(s_row) for _ in range(n_test)],
        rmst=rmst,
    )
