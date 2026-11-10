"""What a fit learns, and which models can learn it.

Four tasks:

- classification and regression: one label per training row (train_y).
- survival: time to an event, right-censored. Each training row has a
  time (when the event happened, or when follow-up ended without it)
  and an event flag (1: the event happened at that time; 0: censored,
  only known not to have happened by then). A predict returns each
  row's event-time distribution: its median, its quantiles, and the
  probability of no event by given times.
- causal: the effect of a binary treatment on an outcome, from
  observational data. Each training row has its outcome and its
  treatment (1: treated, 0: control). A predict returns each row's
  conditional average treatment effect (CATE): its expected outcome if
  treated minus if not, and quantiles of that effect.

A survival or causal fit carries two values per row. Every layer below
the API (engine, workers, spill tier) keeps taking one `y`, so they
travel together as a (rows, 2) array: [time, event] for survival,
[outcome, treatment] for causal (pack_targets). The API names the second
column (train_event / event_column, train_treatment / treatment_column).

Which models serve which task: a backend lists its tasks in a `tasks`
attribute; without one it serves classification and regression
(PREDICTIVE). A fit that names no model gets the deployment's default if
it serves the task, else the first listed model that does (choose_model).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, get_args

import numpy as np

from tabctx.errors import InvalidInputError
from tabctx.types import ArrayLike, Task

TASKS: tuple[Task, ...] = get_args(Task)
PREDICTIVE: frozenset[Task] = frozenset({"classification", "regression"})
# Tasks whose fit carries a second column, and its name in the API.
SECOND_COLUMN: dict[Task, str] = {"survival": "event", "causal": "treatment"}
# Tasks whose predicts are a distribution, so they can return quantiles.
WITH_QUANTILES: frozenset[Task] = frozenset({"regression", "survival", "causal"})


def tasks_of(backend: Any) -> frozenset[Task]:
    """The tasks a backend serves."""
    return frozenset(getattr(backend, "tasks", PREDICTIVE))


def has_second_column(task: Task | None) -> bool:
    return task in SECOND_COLUMN


def choose_model(
    task: Task,
    requested: str | None,
    served: Mapping[str, frozenset[Task]],
    default: str,
) -> str:
    """The model to fit `task` with: the requested one, which must serve
    it; else the default if it does; else the first listed that does.
    InvalidInputError naming the models that would, when none is chosen."""
    able = [name for name, tasks in served.items() if task in tasks]
    if requested is not None:
        if requested not in served:
            raise InvalidInputError(
                f"unknown backend {requested!r}; this deployment serves "
                f"{sorted(served)}"
            )
        if task in served[requested]:
            return requested
        raise InvalidInputError(
            f"model {requested!r} serves {_listing(served[requested])}, not "
            f"{task}; " + _who_does(task, able)
        )
    if task in served.get(default, frozenset()):
        return default
    if able:
        return able[0]
    raise InvalidInputError(_who_does(task, able))


def _listing(tasks: frozenset[Task]) -> str:
    return " and ".join(t for t in TASKS if t in tasks)


def _who_does(task: Task, able: list[str]) -> str:
    if able:
        return f"for {task}, use model {' or '.join(map(repr, able))}"
    return f"no model this deployment serves fits {task}"


def pack_targets(task: Task, y: ArrayLike, second: ArrayLike | None) -> ArrayLike:
    """A fit's targets as the engine takes them: y itself for
    classification and regression, else (rows, 2) floats [y, second].
    InvalidInputError when the second column is missing or not wanted."""
    if not has_second_column(task):
        if second is not None:
            raise InvalidInputError(
                f"an event or treatment column is for survival and causal "
                f"fits, not {task}"
            )
        return y
    name = SECOND_COLUMN[task]
    if second is None:
        raise InvalidInputError(f"a {task} fit needs its {name} column ({_HELP[task]})")
    first = _floats(y, "train_y")
    other = _floats(second, name)
    if len(first) != len(other):
        raise InvalidInputError(
            f"train_y has {len(first)} values but {name} has {len(other)}"
        )
    return np.column_stack([first, other])


_HELP: dict[Task, str] = {
    "survival": "1 where the event happened at the row's time, 0 where the row "
    "was censored then",
    "causal": "1 for treated rows, 0 for control rows",
}


def _floats(values: ArrayLike, name: str) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as e:
        raise InvalidInputError(f"{name} must be numbers: {e}") from e
    if arr.ndim != 1:
        raise InvalidInputError(f"{name} must be one value per training row")
    return arr


def split_targets(y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """A survival or causal fit's (rows, 2) targets as its two columns."""
    arr = np.asarray(y, dtype=float)
    return arr[:, 0], arr[:, 1]


def validate_targets(y: ArrayLike, task: Task) -> None:
    """InvalidInputError unless a survival or causal fit's targets are
    usable: (rows, 2) finite numbers, a flag column of 0s and 1s, and the
    rows the model needs (an event; both treated and control rows)."""
    arr = np.asarray(y)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise InvalidInputError(
            f"a {task} fit takes two values per training row "
            f"(train_y and {SECOND_COLUMN[task]})"
        )
    try:
        values = arr.astype(float)
    except (TypeError, ValueError) as e:
        raise InvalidInputError(f"{task} targets must be numbers: {e}") from e
    if not np.isfinite(values).all():
        raise InvalidInputError(
            f"{task} targets have missing or infinite values (NaN/inf); every "
            "training row needs both"
        )
    first, flag = values[:, 0], values[:, 1]
    name = SECOND_COLUMN[task]
    if not np.isin(flag, (0.0, 1.0)).all():
        raise InvalidInputError(f"{name} must be 0 or 1 ({_HELP[task]})")
    if task == "survival":
        if (first < 0).any():
            raise InvalidInputError("survival times can't be negative")
        if not flag.any():
            raise InvalidInputError(
                "every training row is censored: a survival fit needs at least "
                "one event (event = 1)"
            )
    elif flag.all() or not flag.any():
        raise InvalidInputError(
            "a causal fit needs both treated (treatment = 1) and control "
            "(treatment = 0) rows"
        )


def infer_task_with_second(event: bool, treatment: bool) -> Task | None:
    """The task a request implies by the second column it sends."""
    if event and treatment:
        raise InvalidInputError(
            "send an event column (survival) or a treatment column (causal), not both"
        )
    if event:
        return "survival"
    if treatment:
        return "causal"
    return None
