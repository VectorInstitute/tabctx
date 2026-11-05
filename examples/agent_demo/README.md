# Agent demo: a coding agent with tabular foundation models

Two scenarios, each a question about a table, in [`scenarios/`](scenarios/):

- **readmission** (the default, and the hosted demo): which hospital
  patients will be back within 30 days? Public data, below.
- **insurance**: which policyholders will cancel at renewal, and what
  might keep them? The agent first cleans two messy raw exports with a
  script it writes, then does the same steps. Synthetic data, see
  [Insurance scenario](#insurance-scenario).

`DEMO_SCENARIO=insurance` picks the second for every command below.

## Readmission


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

Each scenario is a folder in `scenarios/` with three files: `task.py`
names the files and columns, the outcome's labels and the task's steps
(the page's step icons; keep them in line with `PROMPT.md`),
`prepare_data.py` builds the data, and `PROMPT.md` is the agent's task.
A new scenario is a new folder; the runner, scoring and web page find it
through [`scenario.py`](scenario.py). Its data goes in
`workspace/<scenario>/` (what the agent gets) and `answers/<scenario>/`
(what it never sees), its runs in `runs/<scenario>/<agent>/`.

```bash
uv sync --extra mcp
uv run --extra mcp python examples/agent_demo/prepare_data.py
export TABCTX_API_KEY=vp_...                       # Vector gateway key
examples/agent_demo/run.sh claude                  # or codex, gemini, pi
uv run --extra mcp python examples/agent_demo/score.py examples/agent_demo/runs/readmission/claude
```

## Insurance scenario

Maple Mutual, a made-up Canadian insurer, wants to know which
policyholders whose renewal is due in the next three months will cancel,
and what might keep them. `scenarios/insurance/prepare_data.py` makes a
synthetic book: 6,000 of last year's renewals with the outcome (about
16% cancelled) and 1,200 renewals due. The patterns follow what insurers
report: a premium increase, missed payments and paying by cheque raise
the chance of cancelling; years as a customer, more products and
contact with an advisor lower it, and an advisor's check-in halves the
effect of an increase.

The agent gets both tables as a policy system exports them: headers
with spaces, four date formats, amounts as `$1,234.50`, `1234.50` and
`CAD 1,235`, a province spelled five ways, yes/no as Y, 1 and TRUE,
blanks as N/A or `-`, impossible ages, rows exported twice, six internal
test policies and a totals row. Its first step is to write `clean.py`,
run it, and list every fix in its report, so the task gives each agent
its shell (`SHELL` in `task.py`): Claude Code's Bash runs in its
sandbox, Codex in its own, Gemini CLI and pi without one.

```bash
export DEMO_SCENARIO=insurance
uv run --extra mcp python examples/agent_demo/prepare_data.py
examples/agent_demo/run.sh claude
uv run --extra mcp python examples/agent_demo/score.py examples/agent_demo/runs/insurance/claude
```

A laptop is enough: run a tabctx server with TabICL
(`TABCTX_BACKEND=tabicl python -m tabctx.serve`) and point the tools at
it (`TABCTX_BASE_URL=http://127.0.0.1:8000`, any `TABCTX_API_KEY`). A
Claude Code run there takes about five minutes.

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
  [`showcase/readmission/`](showcase/readmission/). These are three real runs, copied by
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
  Qwen3_8-27B, an open model served on Vector's own GPUs (hosted
  visitors whose project lacks it get claude-sonnet-5-5). `CODEX_MODEL`
  and `PI_MODEL` change them.
- `CLAUDE_VIA_GATEWAY=1` sends Claude Code through the gateway as well.

## Results (2026-10-31; pi on 2026-11-01, prod gateway)

| Agent (LLM) | tabctx calls | Wall time | Model chosen | ROC AUC on real outcomes | Readmitted among its top 10 / top 100 |
|---|---|---|---|---|---|
| Claude Code 2.1 (Opus 5.5, own login) | 10 | 160 s | tabpfn-3.5 | 0.674 | 80% / 36% |
| Codex 0.137 (gpt-5, via gateway) | 9 | 188 s | tabpfn-3.5 | 0.674 | 80% / 36% |
| pi 0.73 (Qwen3.8 27B, on-prem via gateway) | 9 | 404 s | tabpfn-3.5 | 0.674 | 80% / 36% |
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

pi on the open Qwen3.8 27B model, served on Vector's own GPUs, made
the same choices and got the same score as the two frontier models. It
took longer: 404 s, against 134 s for pi on Sonnet 5.5.

In an earlier run of this task, Codex stopped after predicting, with a
progress message and no report (gpt-5 in `codex exec` sometimes ends its
turn early). Running it again finished.
