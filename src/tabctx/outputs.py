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
model can't give, before any work.

OUTPUTS is the one registry of them: each output's tasks, the
PredictOutcome and response field it fills, the request value echoed
back beside it, and the release that added it. The service's response
and the client's check of a server's response are derived from it.
Adding an output: a field on PredictOutputs (validated in of()), its
entry here, its PredictOutcome field, the request and response fields
in serve/service.py and the client's arguments (both typed by hand, so
the API documents itself), and the backends that give it.
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


@dataclass(frozen=True)
class Output:
    """One output a predict can ask for (an entry of OUTPUTS)."""

    tasks: frozenset[Task]
    # Said to a caller who asks for it on another task.
    wrong_task: str
    # The PredictOutcome / response field holding it, and the response
    # field echoing the request's value (None: nothing echoed).
    field: str
    echo: str | None
    # How a message names it, and the release whose servers return it.
    label: str
    since: str


OUTPUTS: dict[str, Output] = {
    "quantiles": Output(
        frozenset({"regression", "survival", "causal"}),
        "quantiles are for regression, survival and causal contexts; a "
        "classification context returns class probabilities (return_proba) "
        "instead",
        field="quantiles",
        echo="quantile_levels",
        label="quantiles",
        since="0.19",
    ),
    "times": Output(
        frozenset({"survival"}),
        "times are for survival contexts (the probability of no event by each time)",
        field="survival",
        echo="survival_times",
        label="survival probabilities",
        since="0.21",
    ),
    "rmst_horizon": Output(
        frozenset({"survival"}),
        "rmst_horizon is for survival contexts (the expected event-free time "
        "up to the horizon)",
        field="rmst",
        echo="rmst_horizon",
        label="restricted mean survival times (rmst)",
        since="0.21",
    ),
    "potential_outcomes": Output(
        frozenset({"causal"}),
        "potential_outcomes are for causal contexts (each row's expected "
        "outcome if not treated and if treated)",
        field="potential_outcomes",
        echo=None,
        label="potential outcomes",
        since="0.21",
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
        if potential_outcomes not in (True, False):  # "false" is not False
            raise InvalidInputError(
                f"potential_outcomes must be true or false, got {potential_outcomes!r}"
            )
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
            if getattr(self, f.name) != f.default
        }

    def check(self, task: Task, backend: Any) -> None:
        """InvalidInputError (422) unless a context of this task, fit by
        this backend, can give every output asked for."""
        accepted = inspect.signature(backend.predict).parameters
        for name in self.asked():
            output = OUTPUTS[name]
            if task not in output.tasks:
                raise InvalidInputError(f"{output.wrong_task}; this context is {task}")
            if name not in accepted:
                raise InvalidInputError(
                    f"model {backend.name!r} can't return {output.label}: "
                    "it predicts a single value per row"
                )

    def response(self, outcome: Any) -> dict[str, Any]:
        """Each asked output's response fields: its values from the
        outcome, and the request's value echoed beside them."""
        out: dict[str, Any] = {}
        for name, value in self.asked().items():
            spec = OUTPUTS[name]
            values = getattr(outcome, spec.field)
            out[spec.field] = values
            if spec.echo is not None and values is not None:
                out[spec.echo] = list(value) if isinstance(value, tuple) else value
        return out


def offered(tasks: frozenset[Task], backend: Any) -> list[str]:
    """The outputs a model can give for any of its tasks (its predict takes
    them), for GET /v1/models."""
    accepted = inspect.signature(backend.predict).parameters
    return [
        name
        for name, output in OUTPUTS.items()
        if name in accepted and output.tasks & tasks
    ]


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
