# Churn demo: a coding agent with tabular foundation models

A coding agent gets two CSVs and a business question. The question needs
a prediction from a table, which an agent can't make on its own. With
`tabctx-mcp` it can: it checks how accurate the models are against
gradient boosting, fits TabPFN or TabICL in seconds, ranks today's
customers by churn risk, finds what drives churn, tests a retention
offer on the riskiest customers with what-if predictions, and writes a
report. The agent never sees the real outcomes. `score.py` grades its
predictions against them afterwards.

Data: the public Telco customer churn table (IBM, OpenML #42178). 5,543
past customers with their outcome (27% churned) and 1,500 current
customers.

```bash
uv sync --extra mcp
uv run --extra mcp python examples/agent_demo/prepare_data.py
export TABCTX_API_KEY=vp_...                       # Vector gateway key
examples/agent_demo/run.sh claude                  # or codex, gemini, pi
uv run --extra mcp python examples/agent_demo/score.py examples/agent_demo/runs/claude
```

## Web UI

```bash
export TABCTX_API_KEY=vp_...
uv run --extra mcp --with fastapi --with uvicorn python examples/agent_demo/ui/app.py
# open http://127.0.0.1:8765
```

Pick an agent, edit the task if you like, and click **Run agent**. Each
tabctx call appears as a card as soon as the agent makes it:

- **Inspect:** a map of the table's columns.
- **Evaluate:** both models against gradient boosting on held-out rows.
- **Predict:** the predicted split, and a risk table.
- **Explain:** which columns matter.
- **What-if:** the probability with and without a change, for one customer.

When the agent finishes, the page shows its score against the real
outcomes and its report. **Replay last run** plays a finished run back
without running anything. Hover over any mark for its numbers. The page
has light and dark themes, and listens on 127.0.0.1 only, since it
starts agents on this machine.

The UI reads the same `runs/<agent>/transcript.jsonl` files that
`run.sh` writes. `ui/transcripts.py` turns each agent's format into one
stream of events.

The task is in [`PROMPT.md`](PROMPT.md). Each run gets its own copy of
the data in `runs/<agent>/` and leaves `report.md` and
`transcript.jsonl` there.

Tools by agent:

- **Claude Code** gets the tabctx MCP tools plus file tools. It has no shell.
- **Codex** gets the MCP tools plus its sandboxed shell.
- **Gemini CLI** gets the MCP tools plus file tools. It has no shell.
- **pi** has no MCP. The extension from `tabctx-mcp pi-extension`
  registers the same tools as pi tools. It also gets file tools, and no
  bash.

The agent's LLM:

- Claude Code and Gemini CLI use your own login.
- Codex and pi go through the Vector gateway with the same key as the
  tabular models. Codex uses gpt-5 by default, because for newer models
  Codex adds a `tool_search` tool that the gateway refuses. pi uses
  claude-sonnet-5-5. `CODEX_MODEL` and `PI_MODEL` change them.
- `CLAUDE_VIA_GATEWAY=1` sends Claude Code through the gateway as well.

## Results (2026-10-31, prod gateway)

| Agent (LLM) | tabctx calls | Wall time | Model chosen | ROC AUC on real outcomes | Churned among its top 10 / top 100 |
|---|---|---|---|---|---|
| Claude Code 2.1 (Opus 5.5, own login) | 11 | 76 s | tabpfn-3.5 | 0.867 | 90% / 86% |
| Codex 0.137 (gpt-5, via gateway) | 9 | 100 s | tabicl-v2 | 0.866 | 90% / 87% |
| pi 0.73 (Sonnet 5.5, via gateway) | 11 | 72 s | tabicl-v2 | 0.866 | 90% / 87% |
| Gemini CLI 0.38 | not run | | | | |

Gemini CLI wasn't run because its free-tier Google login is no longer
accepted ("This client is no longer supported for Gemini Code Assist
for individuals"). `run.sh gemini` needs `GEMINI_API_KEY` or Vertex AI
auth.

27% of the current customers churned, so a random top 10 would hold
about 3 churners.

In `evaluate`, on 1,386 held-out past customers, both models beat
scikit-learn's gradient boosting. ROC AUC was 0.841 for tabpfn-3.5,
0.841 for tabicl-v2 and 0.820 for gradient boosting; log loss was 0.420,
0.423 and 0.459. Fitting all 5,543 rows took about 2.5 s. Predicting
1,500 rows took 0.8 s.

All three agents used every tool, and all three dropped `customer_id`,
which `inspect_table` flags as an id. In `explain`, the contract type
mattered most for all three, with an AUC drop of 0.075 to 0.092. The
columns after it came in a different order (internet service, tenure,
charges). All three found the same three riskiest month-to-month
customers: C03380, C04800 and C02577. For them, a one-year contract
lowered predicted churn by 17 to 22 points, for example from 0.94 to
0.76. That checks the offer against the model; it does not prove cause
and effect, and the reports say so.
