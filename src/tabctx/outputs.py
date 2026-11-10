"""What a predict returns beyond its predictions.

Every predict returns one prediction per test row; a caller can ask for
more, each for the tasks whose models can give it:

| Output | Asked with | Tasks | Each row gets |
|---|---|---|---|
| quantiles | levels in (0, 1) | regression, survival, causal | the predicted distribution at each level |
| survival | times >= 0 | survival | the probability of no event by each time |
| rmst | a horizon > 0 | survival | the expected event-free time up to the horizon |
| potential_outcomes | true | causal | the expected outcome if not treated and if treated |

(Classification's class probabilities are return_proba, as they always
were.)

PredictOutputs carries what was asked through every layer (engine,
devices, worker processes, the coalescer, whose batches it keys) as one
validated, hashable value. A backend's predict takes each output it can
give as a keyword argument of the same name, and is passed only those
asked for (asked()); check() refuses an output the context's task or its
model can't give, before any work. Adding an output is a field here, an
entry in OUTPUT_TASKS, its validation in of(), and the backends that
give it.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Any

from tabctx.errors import InvalidInputError
from tabctx.types import Task
from tabctx.uncertainty import check_quantiles, check_times

# The tasks each output is for, and what to say to a caller who asks for
# it elsewhere.
OUTPUT_TASKS: dict[str, tuple[frozenset[Task], str]] = {
    "quantiles": (
        frozenset({"regression", "survival", "causal"}),
        "quantiles are for regression, survival and causal contexts; a "
        "classification context returns class probabilities (return_proba) "
        "instead",
    ),
    "times": (
        frozenset({"survival"}),
        "times are for survival contexts (the probability of no event by each time)",
    ),
    "rmst_horizon": (
        frozenset({"survival"}),
        "rmst_horizon is for survival contexts (the expected event-free time "
        "up to the horizon)",
    ),
    "potential_outcomes": (
        frozenset({"causal"}),
        "potential_outcomes are for causal contexts (each row's expected "
        "outcome if not treated and if treated)",
    ),
}


@dataclass(frozen=True)
class PredictOutputs:
    """The outputs a predict asked for; None (or False): not asked."""

    quantiles: tuple[float, ...] | None = None
    times: tuple[float, ...] | None = None
    rmst_horizon: float | None = None
    potential_outcomes: bool = False

    @classmethod
    def of(
        cls,
        quantiles: Sequence[float] | None = None,
        times: Sequence[float] | None = None,
        rmst_horizon: float | None = None,
        potential_outcomes: bool = False,
    ) -> PredictOutputs:
        """Validated outputs (InvalidInputError for a bad value)."""
        return cls(
            quantiles=check_quantiles(quantiles),
            times=check_times(times),
            rmst_horizon=check_horizon(rmst_horizon),
            potential_outcomes=bool(potential_outcomes),
        )

    def asked(self) -> dict[str, Any]:
        """The outputs asked for, as a backend's predict keyword arguments."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) not in (None, False)
        }

    def check(self, task: Task, backend: Any) -> None:
        """InvalidInputError (422) unless a context of this task, fit by
        this backend, can give every output asked for."""
        accepted = inspect.signature(backend.predict).parameters
        for name in self.asked():
            tasks, wrong_task = OUTPUT_TASKS[name]
            if task not in tasks:
                raise InvalidInputError(f"{wrong_task}; this context is {task}")
            if name not in accepted:
                raise InvalidInputError(
                    f"model {backend.name!r} can't return {name.replace('_', ' ')}: "
                    "it predicts a single value per row"
                )


NONE = PredictOutputs()


def check_horizon(horizon: float | None) -> float | None:
    """A validated RMST horizon: finite and positive (None: not asked)."""
    if horizon is None:
        return None
    try:
        value = float(horizon)
    except (TypeError, ValueError) as e:
        raise InvalidInputError(f"rmst_horizon must be a number: {e}") from e
    if not (math.isfinite(value) and value > 0):
        raise InvalidInputError(
            f"rmst_horizon must be finite and positive, got {horizon!r}"
        )
    return value
