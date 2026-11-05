"""How a backend reports a classification predict (#45).

Every backend derives a classification's labels from its probabilities,
in one pass (the protocol in backends/base.py), and reports them the same
way: as its classes' strings, whatever return_proba is. A backend that
returned predict()'s labels when probabilities weren't asked for gave
numbers one way and strings the other.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from tabctx.errors import BackendComputeError
from tabctx.types import ArrayLike, PredictOutcome


def classification_outcome(
    proba: ArrayLike, classes: Iterable[Any], return_proba: bool
) -> PredictOutcome:
    """A classification predict's outcome from its probabilities (one row
    per test sample, one column per class, in `classes` order): each label
    is the row's most probable class as a string, so a client gets the same
    labels, drawn from `classes`, either way."""
    names = [str(c) for c in classes]
    predictions = [names[i] for i in np.argmax(proba, axis=1)]
    if not return_proba:
        return PredictOutcome(predictions=predictions)
    return PredictOutcome(
        predictions=predictions, probabilities=np.asarray(proba).tolist(), classes=names
    )


def regression_outcome(
    mean: ArrayLike, quantiles: ArrayLike | None = None, n_levels: int = 0
) -> PredictOutcome:
    """A regression predict's outcome: the mean per test row and, when
    asked for, a (rows, levels) array of quantiles. Its orientation is
    checked, never reshaped into place: a (levels, rows) array from a
    model would otherwise be silently scrambled."""
    means = np.asarray(mean, dtype=float).ravel()
    if quantiles is None:
        return PredictOutcome(predictions=means.tolist())
    values = np.asarray(quantiles, dtype=float)
    if values.shape != (len(means), n_levels):
        raise BackendComputeError(
            f"the model returned quantiles of shape {values.shape}, expected "
            f"{(len(means), n_levels)} (rows, levels)"
        )
    return PredictOutcome(predictions=means.tolist(), quantiles=values.tolist())
