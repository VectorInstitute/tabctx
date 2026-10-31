"""`tabctx-mcp`: the MCP server, and the same tools as shell commands that
print JSON (for agents without MCP).

    tabctx-mcp                      # serve MCP over stdio (also: serve)
    tabctx-mcp inspect data.csv --target churn
    tabctx-mcp evaluate data.csv --target churn --drop customer_id
    tabctx-mcp fit data.csv --target churn --name churn --drop customer_id
    tabctx-mcp predict churn new.csv
    tabctx-mcp what-if churn '[{"tenure": 3, "Contract": "One year"}]'
    tabctx-mcp what-if churn '[{}, {"Contract": "One year"}]' \\
        --base-file current.csv --base-where '{"customer_id": "C00042"}'
    tabctx-mcp explain data.csv --target churn
    tabctx-mcp models
    tabctx-mcp skill                # how an agent should use these
    tabctx-mcp tools                # the MCP tools' schemas, as JSON
    tabctx-mcp call predict '{"name": "churn", "path": "new.csv"}'
    tabctx-mcp pi-extension > ~/.pi/agent/extensions/tabctx.ts
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import resources
from typing import Any

from mcp.server.mcpserver.exceptions import ToolError

from tabctx.mcp_tools.tools import MODELS, TabularTools


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tabctx-mcp",
        description="Tabular foundation models (TabPFN, TabICL) as tools for agents.",
    )
    sub = p.add_subparsers(dest="command")
    sub.add_parser("serve", help="serve the tools over MCP (stdio); the default")
    sub.add_parser("skill", help="print the guide for agents (SKILL.md)")
    sub.add_parser("models", help="list fitted models")
    sub.add_parser("tools", help="the MCP tools' names, descriptions and schemas")
    s = sub.add_parser("call", help="call an MCP tool with JSON arguments")
    s.add_argument("tool")
    s.add_argument("arguments", nargs="?", default="{}", help='JSON; "-" reads stdin')
    sub.add_parser(
        "pi-extension", help="print the pi extension that registers these tools"
    )

    def table(cmd, help_, target=True):
        s = sub.add_parser(cmd, help=help_)
        s.add_argument("path", help="CSV, TSV or Parquet file")
        s.add_argument("--target", required=target, help="column to predict")
        return s

    def model_opts(s):
        s.add_argument(
            "--drop", default="", help="comma-separated columns to leave out"
        )
        s.add_argument("--task", choices=["classification", "regression"])

    table("inspect", "describe a table", target=False)

    s = table("evaluate", "hold out rows and score each model")
    s.add_argument("--models", default=",".join(MODELS))
    s.add_argument("--no-baseline", action="store_true")
    s.add_argument("--test-size", type=float, default=0.25)
    model_opts(s)

    s = table("fit", "fit a model and keep it under a name")
    s.add_argument("--model", default=MODELS[0], choices=MODELS)
    s.add_argument("--name")
    model_opts(s)

    s = sub.add_parser("predict", help="predict every row of a table")
    s.add_argument("name")
    s.add_argument("path")
    s.add_argument("--output")
    s.add_argument("--top", type=int, default=10, help="top rows to print")
    s.add_argument("--rank-by", help="class whose probability ranks rows")
    s.add_argument("--where", help="JSON {column: value} to rank only matching rows")
    s.add_argument("--show", default="", help="comma-separated columns to show")

    s = sub.add_parser("what-if", help="predict rows given as a JSON list of dicts")
    s.add_argument("name")
    s.add_argument("rows", help='JSON, e.g. \'[{"tenure": 3}]\'; "-" reads stdin')
    s.add_argument("--base-file", help="table holding the base row")
    s.add_argument(
        "--base-where",
        help='JSON {column: value} matching one row, e.g. \'{"id": "C7"}\'',
    )

    s = table("explain", "rank columns by permutation importance")
    s.add_argument("--model", default=MODELS[0], choices=MODELS)
    s.add_argument("--top", type=int, default=20)
    model_opts(s)
    return p


def _split(text: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()]


def run(
    argv: list[str] | None = None, tools: TabularTools | None = None
) -> dict[str, Any] | str | None:
    """The command's result (a dict to print as JSON, or text), or None
    after serving. tools: what runs the commands (default TabularTools())."""
    args = _parser().parse_args(argv)
    cmd = args.command or "serve"
    if cmd == "serve":
        from tabctx.mcp_tools.server import build_server

        build_server(tools).run("stdio")
        return None
    if cmd == "skill":
        return _resource("SKILL.md")
    if cmd == "pi-extension":
        return _resource("pi_extension.ts")
    if cmd in ("tools", "call"):
        from tabctx.mcp_tools import server

        mcp = server.build_server(tools)
        if cmd == "tools":
            return {"tools": server.tool_specs(mcp)}
        text = sys.stdin.read() if args.arguments == "-" else args.arguments
        return server.call_tool(mcp, args.tool, json.loads(text))

    tools = tools or TabularTools()
    if cmd == "models":
        return tools.list_models()
    if cmd == "inspect":
        return tools.inspect_table(args.path, args.target)
    if cmd == "evaluate":
        return tools.evaluate(
            args.path,
            args.target,
            _split(args.models),
            not args.no_baseline,
            args.test_size,
            _split(args.drop),
            args.task,
        )
    if cmd == "fit":
        return tools.fit_model(
            args.path, args.target, args.model, args.name, _split(args.drop), args.task
        )
    if cmd == "predict":
        return tools.predict(
            args.name,
            args.path,
            args.output,
            args.top,
            args.rank_by,
            json.loads(args.where) if args.where else None,
            _split(args.show),
        )
    if cmd == "what-if":
        text = sys.stdin.read() if args.rows == "-" else args.rows
        rows = json.loads(text)
        where = json.loads(args.base_where) if args.base_where else None
        return tools.what_if(
            args.name, rows if isinstance(rows, list) else [rows], args.base_file, where
        )
    if cmd == "explain":
        return tools.explain(
            args.path,
            args.target,
            args.model,
            _split(args.drop),
            args.task,
            top=args.top,
        )
    raise AssertionError(cmd)


def _resource(name: str) -> str:
    return resources.files("tabctx.mcp_tools").joinpath(name).read_text()


def main(argv: list[str] | None = None, tools: TabularTools | None = None) -> int:
    try:
        result = run(argv, tools)
    except (ValueError, json.JSONDecodeError, ToolError) as e:
        print(json.dumps({"error": str(e)}), file=sys.stdout)
        return 1
    except Exception as e:  # the deployment's errors, for the agent to read
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}), file=sys.stdout)
        return 1
    if isinstance(result, str):
        print(result, end="")
    elif result is not None:
        print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
