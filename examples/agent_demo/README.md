# Readmission demo: a coding agent with tabular foundation models

A coding agent gets two CSVs and a hospital's question: which patients
being discharged now are most likely to be back within 30 days, and what
could lower that? The answer needs a prediction from a table, which an
agent can't make on its own. With `tabctx-mcp` it can: it checks how
accurate the models are against gradient boosting, fits TabPFN or TabICL
in seconds, ranks today's patients by risk, finds what drives
readmission, tests care changes on the riskiest patients with what-if
predictions, and writes a report. The agent never sees the real
outcomes. `score.py` grades its predictions against them afterwards.

The same steps fit any risk question on a table (credit default, fraud,
claims, churn): only the data and the prompt change.

Data: the public Diabetes 130-US Hospitals table (UCI; OpenML #4541),
101,766 hospital stays of diabetic patients, 1999-2008. `prepare_data.py`
prepares it as in most work on it (Strack et al., 2014) and takes 30,000
past stays with their outcome (11.0% readmitted within 30 days), and one
stay each for 2,000 other patients, who are the "current" patients and
appear nowhere in the history (11.0% of them were readmitted). Current
patients are picked with a chance in proportion to their number of
stays, as the history's stays are; picking them uniformly would make
the current set healthier than the history. 33 features: demographics,
admission and discharge, prior visits, diagnoses, lab tests, diabetes
medication.

`task.py` names the files and columns, the outcome's labels and the
task's steps (the page's step icons; keep them in line with
`PROMPT.md`). A different dataset means changing `task.py`,
`prepare_data.py` and `PROMPT.md`; the runner, scoring and web page read
the rest from `task.py`.

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
- **What-if:** the probability with and without a change, for one patient.

When the agent finishes, the page shows its score against the real
outcomes and its report. **Replay last run** plays a finished run back
without running anything. Hover over any mark for its numbers. The page
has light and dark themes, and listens on 127.0.0.1 only, since it
starts agents on this machine.

The UI reads the same `runs/<agent>/transcript.jsonl` files that
`run.sh` writes. `ui/transcripts.py` turns each agent's format into one
stream of events.

## Hosted demo (share a link)

`deploy.sh` builds [`Dockerfile`](Dockerfile) with Cloud Build and deploys it
to Cloud Run (project `coderd`, `us-central1`, service `tabctx-agent-demo`).
The image holds the page, Claude Code, Codex and pi, and `tabctx-mcp`, and
nothing secret.

- **Anyone with the link** can replay the showcase runs in
  [`showcase/`](showcase/). These are three real runs, copied by
  `make_showcase.py` with session details and local paths removed.
- **To run an agent**, a visitor signs in with their own gateway key. The
  app checks the key's project models (`/admin/me/models`) and gives each
  agent the best model that project can use. Agents the project can't run
  stay off.
- **Every agent's model calls go through the gateway with that key**, so
  they are billed to the visitor's project. The tabular calls are free.
- **The task is fixed.** Gemini CLI isn't offered, because it can't use
  the gateway.
- **The key stays in server memory**, linked to an HttpOnly, Secure,
  SameSite=Strict session cookie. It is never written to disk or logs.
  Sessions end on sign-out, after 4 hours idle, or when the instance
  stops.
- **Each run is isolated.** It runs as a Unix user of its own, with its
  own HOME and a folder only that user can read. One visitor's agent can't
  read another's files or process environment, which holds that visitor's
  key. Nobody's agent can read the answers.
- **Limits:** 3 runs at once, one per visitor, each stopped after 15
  minutes. Sign-in attempts are rate-limited.

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

| Agent (LLM) | tabctx calls | Wall time | Model chosen | ROC AUC on real outcomes | Readmitted among its top 10 / top 100 |
|---|---|---|---|---|---|
| Claude Code 2.1 (Opus 5.5, own login) | 10 | 160 s | tabpfn-3.5 | 0.674 | 80% / 36% |
| Codex 0.137 (gpt-5, via gateway) | 9 | 188 s | tabpfn-3.5 | 0.674 | 80% / 36% |
| pi 0.73 (Sonnet 5.5, via gateway) | 10 | 134 s | tabpfn-3.5 | 0.674 | 80% / 36% |
| Gemini CLI 0.38 | not run | | | | |

Gemini CLI wasn't run because its free-tier Google login is no longer
accepted ("This client is no longer supported for Gemini Code Assist
for individuals"). `run.sh gemini` needs `GEMINI_API_KEY` or Vertex AI
auth.

11.0% of the current patients were readmitted within 30 days, so a
random top 10 would hold about 1, and the agents' top 100 held 3.3 times
the base rate. Readmission is hard to predict from administrative data:
published risk scores (LACE, HOSPITAL) reach an AUC of about 0.65-0.72.

In `evaluate`, on 7,500 held-out past stays, ROC AUC was 0.697 for
tabpfn-3.5, 0.690 for tabicl-v2 and 0.683 for gradient boosting; log
loss 0.321, 0.325 and 0.325. Fitting 30,000 rows took 20 to 23 s.
Predicting 2,000 took about 1.5 s, and `explain` (50,000 rows in one
predict) about 30 s.

All three agents used every tool, dropped `patient_id` (which
`inspect_table` flags as an id), picked tabpfn-3.5 and got the same
ranking. In `explain`, the number of inpatient stays in the past year
mattered most (AUC drop 0.056), then where the patient is discharged to
(0.039), emergency visits (0.018) and the primary diagnosis (0.014).
For the three riskiest patients going home, discharge with home health
service lowered predicted risk by 2 to 10 points; insulin, metformin and
A1C changes moved it by about 1 point or less. All three reports say
these are associations the model learned from past records, not proven
effects, and recommend a transitional-care program targeted by risk
score, measured against a comparison group.

In an earlier run of this task, Codex stopped after predicting, with a
progress message and no report (gpt-5 in `codex exec` sometimes ends its
turn early). Running it again finished.
