"""What the training labels say about the task.

A caller shouldn't need to name the task for the obvious cases, so when
it's omitted it comes from the labels (resolve_task): non-integer numbers
are a continuous target, anything else (text, integer-coded classes) is
classification. When the caller does say classification but sends
non-integer numbers, that's a mistake worth a clear 422 naming the fix,
not a backend error about "573 classes" or a fit that predicts nonsense.
"""

from __future__ import annotations

import numpy as np

from tabctx.errors import InvalidInputError
from tabctx.types import ArrayLike, Task

_REGRESSION_HINT = "if train_y is a continuous target, pass task='regression'"


def _as_numbers(y: ArrayLike) -> np.ndarray | None:
    """Labels as floats when every one is a number (or numeric text, as
    CSV labels arrive), else None."""
    try:
        return np.asarray(y, dtype=float)
    except (TypeError, ValueError):
        return None


def _fractional(values: np.ndarray) -> np.ndarray:
    return ~np.isfinite(values) | (values != np.round(values))


def infer_task(y: ArrayLike) -> Task:
    """Regression when every label is a number and some aren't whole;
    classification otherwise, which keeps integer-coded classes classes."""
    values = _as_numbers(y)
    if values is not None and values.size and _fractional(values).any():
        return "regression"
    return "classification"


def resolve_task(y: ArrayLike, task: Task | None) -> Task:
    """The task to fit: the caller's, checked against the labels, or the
    one the labels imply when the caller didn't say."""
    if task is None:
        return infer_task(y)
    validate_labels(y, task)
    return task


def validate_labels(y: ArrayLike, task: Task) -> None:
    """Rejects classification labels that can only be a continuous target:
    numbers that aren't whole. Integer-coded classes are left alone --
    they're legitimate, and a backend's class limit catches the rest
    (with regression_hint in its message)."""
    if task != "classification":
        return
    values = _as_numbers(y)
    if values is None:
        return
    fractional = _fractional(values)
    if fractional.any():
        example = values[np.argmax(fractional)]
        raise InvalidInputError(
            f"classification needs discrete class labels, but train_y has "
            f"non-integer numbers (e.g. {example:g}); {_REGRESSION_HINT}"
        )


def regression_hint(y: ArrayLike, task: Task) -> str:
    """A suffix for a backend's rejection of a classification table whose
    labels are all numbers: the likeliest cause is a missing task."""
    if task == "classification" and _as_numbers(y) is not None:
        return f" ({_REGRESSION_HINT})"
    return ""
