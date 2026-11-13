# HTTP API

The routes, with curl for the ones you'll use most. The
[reference](../reference/http-api.md) has the full request and response
schemas, generated from the service itself.

| Method | Route | What it does |
|---|---|---|
| `POST` | `/v1/tabctx/fit` | Encode a training table into a context. |
| `POST` | `/v1/tabctx/predict` | Predict against a context. |
| `GET` | `/v1/tabctx/datasets` | List your contexts. |
| `DELETE` | `/v1/tabctx/datasets/{dataset_id}` | Delete a context everywhere it is held. |
| `POST` | `/v1/tabctx/upload` | Stream a large table ahead of a fit or predict. |
| `GET` | `/v1/models` | The models served; the default is marked. |
| `GET` | `/v1/tabctx/limits` | What table shapes the deployment will admit right now. |
| `GET` | `/v1/tabctx/residency` | Where your contexts are, per GPU (multi-GPU replicas). |
| `GET` | `/livez`, `/readyz` | Liveness and readiness probes. |

## Fit and predict

A table goes as rows of JSON, or as a file in the same request (CSV,
gzip or zstd CSV, or Parquet).

=== "Table file"

    ```bash
    curl localhost:8000/v1/tabctx/fit \
      -F dataset_id=churn -F target=churned -F file=@train.parquet
    curl localhost:8000/v1/tabctx/predict \
      -F dataset_id=churn -F return_proba=true -F file=@today.csv
    ```

=== "JSON rows"

    ```bash
    curl localhost:8000/v1/tabctx/fit -H 'Content-Type: application/json' -d '{
      "dataset_id": "churn", "model": "tabicl-v2",
      "train_X": [[1.5, "gold", null], [0.2, "basic", 3.0]],
      "train_y": ["yes", "no"]}'
    curl localhost:8000/v1/tabctx/predict -H 'Content-Type: application/json' -d '{
      "dataset_id": "churn", "test_X": [[1.1, "gold", 2.0]], "return_proba": true}'
    ```

Text cells are categories, `null` is a missing value, and the task is
inferred from the labels unless you send `task`. A regression predict
can also ask for `quantiles` of the predicted distribution, e.g.
`"quantiles": [0.05, 0.95]` (`-F quantiles=0.05,0.95` in a form), and
gets `quantiles` (one row per test row) and `quantile_levels` back. When the training table
was a file, a test file's columns are matched by name.

A survival fit sends each row's time as `train_y` and its event flag as
`train_event` (1: the event happened then; 0: censored), or names the
columns of a file (`target`, `event_column`); its predicts return each
row's median event time and, with `times`, the probability of no event
by each time (`survival`), and with `rmst_horizon` the expected
event-free time up to it (`rmst`). Past the longest training time (the
fit response's `summary.max_time`) nothing is estimated, so to rank rows
by risk ask for `rmst` with `rmst_horizon` at that time: it is always
defined (a higher `rmst` is a lower risk). A causal fit sends each row's outcome
and its treatment flag (`train_treatment`, or `treatment_column`); its
predicts return each row's treatment effect, and with
`potential_outcomes: true` the expected outcome if not treated and if
treated. Predicting a survival or causal context with `return_proba` is a
422: there are no class probabilities. `GET /v1/models` lists each model's
`tasks` and `outputs`. A survival or causal fit's response has a
`summary`: `max_time` (outputs past it are `null`), `events` and
`censored`, or `treated` and `control`.

```bash
curl localhost:8000/v1/tabctx/fit -F dataset_id=retention -F target=days \
  -F event_column=cancelled -F file=@policies.csv
curl localhost:8000/v1/tabctx/predict -F dataset_id=retention \
  -F times=90,365 -F file=@current.csv
```

## Large tables

A table file sent with the fit or predict is fine up to
`TABCTX_MAX_UPLOAD_BYTES` (4 GiB by default). To send it ahead, stream
the CSV as the request body to `/v1/tabctx/upload`, then use the
returned id once:

```bash
curl localhost:8000/v1/tabctx/upload -H 'x-session-id: churn' --data-binary @train.csv
# {"upload_id": "3f2a...", ...}
curl localhost:8000/v1/tabctx/fit -H 'Content-Type: application/json' \
  -d '{"dataset_id": "churn", "train_upload_id": "3f2a...", "target": "churned"}'
```

## Retention

A fit can carry `ttl_s`: the server deletes the context that many seconds
later, from every tier (capped by the server's `TABCTX_MAX_CONTEXT_AGE_S`,
which also applies to fits that don't ask). The fit response and the
listing show `expires_at`.

```bash
curl localhost:8000/v1/tabctx/fit -F dataset_id=churn -F target=churned -F ttl_s=86400 -F file=@train.csv
```

## List and delete

```bash
curl localhost:8000/v1/tabctx/datasets
curl -X DELETE localhost:8000/v1/tabctx/datasets/churn
```

## Headers

| Header | When |
|---|---|
| `x-tabctx-tenant-id` | Scopes every dataset id to a tenant. Required when the server runs with `TABCTX_REQUIRE_TENANT=true`. The server trusts it as sent: put it behind a gateway that sets it (see [tenant isolation](../serving.md#tenant-isolation)). |
| `x-session-id` | Equal to the dataset id. Routes the request to the replica holding the context when there are several. |
| `Authorization: Bearer ...` | Through a gateway that needs an API key. |

## Status codes

`422` malformed request · `404` no such context or upload · `413` too
large to admit, or over the upload limit · `503` busy or temporarily
unable, retry · `507` the model failed on the device · `401` tenant
required.
