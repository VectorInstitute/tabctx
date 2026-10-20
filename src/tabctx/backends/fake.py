"""Deterministic backend with no GPU/torch/tabicl dependency, for unit tests.

Not a serious model: classification predicts the training majority class for
every test row, regression predicts the training mean. What matters for
tabctx's own tests is the *shape* of the contract (fit returns an opaque
payload, predict does one pass, context_bytes_hint is controllable), not
prediction quality.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

from tabctx.backends._outcome import classification_outcome
from tabctx.types import ArrayLike, PredictOutcome, Task


@dataclass
class _FakePayload:
    task: Task
    n_train: int
    n_features: int
    class_fractions: dict[str, float] | None = None
    mean_y: float | None = None


class FakeBackend:
    """Test double. `fit_delay_s`/`predict_delay_s` let tests simulate work
    without a real GPU, e.g. to assert cache reuse actually skips fit()."""

    name = "fake"

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
        mean_y = sum(float(v) for v in y) / len(y) if len(y) else 0.0
        return _FakePayload(
            task=task, n_train=n_train, n_features=n_features, mean_y=mean_y
        )

    def predict(
        self, payload: _FakePayload, X_test: ArrayLike, return_proba: bool = False
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
        return PredictOutcome(predictions=[payload.mean_y] * n_test)

    def context_bytes_hint(self, n_train: int, n_features: int) -> int | None:
        del n_train, n_features  # unused: this backend's hint is fixed at construction
        return self._bytes_hint

    def fit_peak_bytes_hint(self) -> int | None:
        return self._peak_bytes_hint
