# Tabular models as tools for coding agents

Coding agents read files and write code well, but they can't learn
from a table on their own. `tabctx-mcp` gives them TabPFN and TabICL as
tools, so an agent can answer "which customers will leave, and why?"
from a CSV. One MCP server works with Claude Code, Codex and Gemini
CLI. The same tools also run as a JSON command line, and as a pi
extension for agents without MCP.

`TABCTX_BASE_URL` is any tabctx deployment: one you run yourself (see
the [Quickstart](../README.md#quickstart)), or Vector's gateway with a
`vp_` key, as below.

```bash
pip install "tabctx[mcp]"
export TABCTX_BASE_URL=https://proxy.vectorinstitute.ai TABCTX_API_KEY=vp_...
claude mcp add tabctx -- tabctx-mcp          # Claude Code (passes its environment on)
codex mcp add tabctx -- tabctx-mcp           # Codex; then in ~/.codex/config.toml, under
                                             #   [mcp_servers.tabctx]:
                                             #   env_vars = ["TABCTX_BASE_URL", "TABCTX_API_KEY"]
gemini mcp add tabctx tabctx-mcp \
  -e 'TABCTX_BASE_URL=$TABCTX_BASE_URL' -e 'TABCTX_API_KEY=$TABCTX_API_KEY'   # Gemini CLI
tabctx-mcp pi-extension > ~/.pi/agent/extensions/tabctx.ts                     # pi (no MCP)
```

The single quotes matter. Gemini CLI expands `$VARS` when it starts the
server, so the key isn't written into its settings file. Agents without
MCP or an extension can run the same tools as shell commands that print
JSON: `tabctx-mcp fit history.csv --target Churn`, `tabctx-mcp predict
churn current.csv`, and so on (`tabctx-mcp --help`).

| Tool | What it does |
|------|--------------|
| `inspect_table` | Columns, their kinds, missing values and likely ids. With a target, the task and class balance. |
| `evaluate` | Held-out accuracy of each model next to a gradient-boosting baseline. |
| `fit_model` | Fits on every labelled row in seconds and saves the model under a name. |
| `predict` | Writes the table with prediction and probability columns, and returns the top-ranked rows. |
| `what_if` | Predicts hand-made rows, or one real row with some columns changed. |
| `explain` | Permutation importance: which columns the model relies on. |

Fitted models live in `.tabctx-mcp/` (or `$TABCTX_MCP_HOME`), so a model
fitted in one call is still there for the next. `tabctx-mcp skill`
prints the guide for agents. The readmission demo in
[`examples/agent_demo/`](../examples/agent_demo/) runs the same task with
each agent and scores their predictions against what the patients
actually did.
