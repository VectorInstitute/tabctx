"""tabctx-mcp: tabular foundation models as tools for coding agents.

A coding agent (Claude Code, Codex, Gemini CLI, pi, ...) reads files and
writes code well, but it cannot learn from a table by itself. These tools
give it TabPFN and TabICL on a tabctx deployment: inspect a table, fit it
in one call (no training loop, no tuning), predict new rows, ask what-if
questions about single rows, measure accuracy against a gradient-boosting
baseline, and rank which columns drive the predictions.

The same tools have two front ends:

- an MCP server (`tabctx-mcp`, or `tabctx-mcp serve`), for agents that
  speak MCP;
- a command line that prints JSON (`tabctx-mcp fit ...`), for agents that
  only run shell commands.

tools.TabularTools holds the logic; server and cli only translate.
Fitted models are kept on disk by store.ModelStore, so a model fitted in
one call (or one CLI process) is there for the next.

(The package is mcp_tools, not mcp, so it never shadows the MCP SDK.)

Requires `pip install "tabctx[mcp]"`, plus TABCTX_BASE_URL and
TABCTX_API_KEY for the deployment.
"""
