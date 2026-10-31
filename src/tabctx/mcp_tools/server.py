"""The tools as an MCP server (stdio), for agents that speak MCP."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from tabctx.mcp_tools.tables import json_safe
from tabctx.mcp_tools.tools import MODELS, TabularTools

INSTRUCTIONS = """\
Tabular foundation models (TabPFN, TabICL) on a GPU server. They learn
from a table you give them, in one call: no training loop, no
hyperparameters, and usually as accurate as a tuned gradient-boosting
model, or more. Use them whenever a question needs a prediction from a
table (CSV or Parquet): classification or regression.

Workflow: inspect_table (with the target) -> evaluate (is it accurate
enough? which model?) -> fit_model (on all labelled rows) -> predict
(new rows, writes a CSV) -> what_if (one case, with a column changed) and
explain (which columns matter).

Paths are relative to the agent's working directory. Drop id columns and
columns that leak the answer (known only after the outcome) before
fitting. tabicl-v2 takes at most 10 classes; tabpfn-3.5 up to 160.
"""

Path = Annotated[str, Field(description="A CSV, TSV or Parquet file")]
Target = Annotated[str, Field(description="The column to predict")]
Model = Annotated[str, Field(description=f"One of {list(MODELS)}")]
Drop = Annotated[
    list[str],
    Field(description="Columns to leave out: ids, and anything that leaks the answer"),
]
Task = Annotated[
    str | None,
    Field(
        description="'classification' or 'regression'; default: inferred from the target"
    ),
]


def build_server(tools: TabularTools | None = None) -> MCPServer:
    t = tools or TabularTools()
    server = MCPServer("tabctx", instructions=INSTRUCTIONS)
    tool = _tool(server)

    @tool
    def inspect_table(path: Path, target: Target | None = None) -> dict[str, Any]:
        """Describe a table: rows, and for each column its kind, missing
        values and typical values (flags likely id columns). With target:
        the task it implies and the class balance."""
        return _run(t.inspect_table, path, target)

    @tool
    def evaluate(
        path: Path,
        target: Target,
        models: Annotated[
            list[str] | None, Field(description=f"Default: {list(MODELS)}")
        ] = None,
        baseline: bool = True,
        test_size: float = 0.25,
        drop: Drop | None = None,
        task: Task = None,
    ) -> dict[str, Any]:
        """How accurate is each model on this table? Holds out test_size of
        the labelled rows, fits each model on the rest and scores it
        (classification: accuracy, ROC AUC, log loss; regression: R2,
        RMSE, MAE). baseline adds scikit-learn gradient boosting."""
        return _run(
            t.evaluate,
            path,
            target,
            list(MODELS) if models is None else models,
            baseline,
            test_size,
            drop or [],
            task,
        )

    @tool
    def fit_model(
        path: Path,
        target: Target,
        model: Model = MODELS[0],
        name: Annotated[
            str | None, Field(description="Name to keep the model under")
        ] = None,
        drop: Drop | None = None,
        task: Task = None,
    ) -> dict[str, Any]:
        """Fit a model on every labelled row of the table (seconds, no
        tuning) and keep it under a name, for predict and what_if."""
        return _run(t.fit_model, path, target, model, name, drop or [], task)

    @tool
    def predict(
        name: Annotated[str, Field(description="A fitted model's name")],
        path: Path,
        output: Annotated[
            str | None, Field(description="CSV to write; default next to the input")
        ] = None,
        top: Annotated[int, Field(description="How many top rows to return")] = 10,
        rank_by: Annotated[
            str | None,
            Field(
                description="Class whose probability ranks the rows; default the rarer class"
            ),
        ] = None,
        where: Annotated[
            dict[str, Any] | None,
            Field(description="Rank only rows matching {column: value}"),
        ] = None,
        show: Annotated[
            list[str] | None,
            Field(description="Input columns to include in the top rows"),
        ] = None,
    ) -> dict[str, Any]:
        """Predict every row of a table. Writes the table plus
        predicted_<target> and, for classification, p_<target>=<class>
        probability columns to a CSV. Returns a summary and the top rows
        (highest risk first, with the id column). If the table has the
        target column, also scores the predictions against it."""
        return _run(t.predict, name, path, output, top, rank_by, where, show or [])

    @tool
    def what_if(
        name: Annotated[str, Field(description="A fitted model's name")],
        rows: Annotated[
            list[dict[str, Any]],
            Field(
                description="Rows as {column: value}. With a base row: only the "
                "changes, and {} for the base row as it is"
            ),
        ],
        base_file: Annotated[
            str | None, Field(description="A table holding the base row")
        ] = None,
        base_where: Annotated[
            dict[str, Any] | None,
            Field(description="{column: value} matching one row, e.g. an id"),
        ] = None,
    ) -> dict[str, Any]:
        """Predict a few hand-made rows: one case, or the same case with a
        column changed, to see how the prediction and probabilities move.
        To start from a real row (one customer), give base_file and
        base_where; each row then lists only what to change."""
        return _run(t.what_if, name, rows, base_file, base_where)

    @tool
    def explain(
        path: Path,
        target: Target,
        model: Model = MODELS[0],
        drop: Drop | None = None,
        task: Task = None,
        top: int = 20,
    ) -> dict[str, Any]:
        """Which columns drive the predictions (permutation importance on
        held-out rows): the score drop when each column is shuffled."""
        return _run(t.explain, path, target, model, drop or [], task, top=top)

    @tool
    def list_models() -> dict[str, Any]:
        """The fitted models kept so far, with their targets and features."""
        return _run(t.list_models)

    return server


def _run(
    method: Callable[..., dict[str, Any]], *args: Any, **kwargs: Any
) -> dict[str, Any]:
    """Call a tool, its result as valid JSON. Any failure goes back to
    the agent as a tool error with its message: the SDK would replace an
    exception's message with "Error executing tool <name>", and the agent
    needs it to correct the call (a ValueError's message is written for
    the agent; others get their type, as TabctxError's 422 detail does)."""
    try:
        return json_safe(method(*args, **kwargs))
    except ValueError as e:
        raise ToolError(str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise ToolError(f"{type(e).__name__}: {e}") from e


def _tool(server: MCPServer) -> Callable:
    """server.tool(), with the docstring's indentation removed: the
    description is what the agent reads."""

    def register(fn: Callable) -> Callable:
        return server.tool(description=inspect.cleandoc(fn.__doc__ or ""))(fn)

    return register


def tool_specs(server: MCPServer) -> list[dict[str, Any]]:
    """Each tool's name, description and JSON Schema, as an MCP client
    sees them: for agents without MCP that register the tools themselves
    (the pi extension)."""
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in asyncio.run(server.list_tools())
    ]


def call_tool(
    server: MCPServer, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Call a tool the way an MCP client would (arguments are validated
    against its schema). Raises ToolError, with a message for the agent."""
    if name not in {t["name"] for t in tool_specs(server)}:
        raise ToolError(f"no tool named {name!r}")
    result = asyncio.run(server.call_tool(name, arguments))
    return result[1] if isinstance(result, tuple) else result.structured_content
