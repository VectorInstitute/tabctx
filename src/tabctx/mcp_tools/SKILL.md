---
name: tabctx
description: Predict from tables (CSV/Parquet) with tabular foundation models (TabPFN, TabICL) on a GPU server: classification or regression, fitted in seconds with no training or tuning. Use for churn, risk, pricing, demand, or any "which rows will..." question.
---

# Tabular predictions with tabctx

`tabctx-mcp` gives you two tabular foundation models, `tabpfn-3.5` (the
default) and `tabicl-v2`. They learn from a labelled table in one call,
on a GPU server: there is no training loop and no hyperparameters, and
they are usually as accurate as a tuned gradient-boosting model, or more.
Prefer them to writing your own model code.

With MCP, the tools are `inspect_table`, `evaluate`, `fit_model`,
`predict`, `what_if`, `explain` and `list_models`. Without MCP, run the
same tools as shell commands; each prints JSON:

```bash
tabctx-mcp inspect history.csv --target Churn
tabctx-mcp evaluate history.csv --target Churn --drop customer_id
tabctx-mcp fit history.csv --target Churn --drop customer_id --name churn
tabctx-mcp predict churn current.csv            # writes current.churn.predictions.csv
tabctx-mcp what-if churn '[{"tenure": 2, "Contract": "Month-to-month"}, {"tenure": 2, "Contract": "One year"}]'
tabctx-mcp explain history.csv --target Churn --drop customer_id
tabctx-mcp models
```

## Workflow

1. **Inspect** the labelled table with its target. Check the task and
   class balance. Note id columns (flagged) and any column that is known
   only after the outcome (it leaks the answer). Pass both as `drop`.
2. **Evaluate** before you trust it. This holds out 25% of the rows and
   compares both models with a gradient-boosting baseline. Report the
   held-out metric, not training accuracy.
3. **Fit** the better model on all labelled rows, under a short name.
4. **Predict** the new rows. The output CSV is the input table plus
   `predicted_<target>` and `p_<target>=<class>` columns. Read it with
   pandas to rank, filter or aggregate.
5. **Explain** which columns drive the predictions, and use **what_if**
   to test an intervention on one case. Change a single column and
   compare the probabilities. That checks a recommendation; it does not
   prove cause and effect.

## Limits

- Requests are at most 30 MiB. That is about 50,000 rows of 100 numeric
  columns. Sample a larger table with pandas first.
- `tabicl-v2` takes at most 10 classes. `tabpfn-3.5` takes up to 160.
- Text columns and missing values are fine as they are. Numbers stored as
  text are converted when the table is read.
- `predict` writes its CSV only inside the working directory.
- A model fitted under an existing name replaces it.
- The server needs `TABCTX_BASE_URL` and `TABCTX_API_KEY` in the
  environment.
