# HTTP API

Every route, with curl. The [reference](../reference/http-api.md) has the
full request and response schemas, generated from the service itself.

| Method | Route | What it does |
|---|---|---|
| `POST` | `/v1/tabctx/fit` | Encode a training table into a context. |
| `POST` | `/v1/tabctx/predict` | Predict against a context. |
| `GET` | `/v1/tabctx/datasets` | List your contexts. |
| `DELETE` | `/v1/tabctx/datasets/{dataset_id}` | Delete a context everywhere it is held. |
| `POST` | `/v1/tabctx/upload` | Stream a large table ahead of a fit or predict. |
| `GET` | `/v1/models` | The models served; the default is marked. |
| `GET` | `/v1/tabctx/limits` | What table shapes the deployment will admit right now. |
| `GET` | `/v1/tabctx/residency` | Where your contexts are, per GPU. |
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
inferred from the labels unless you send `task`. When the training table
was a file, a test file's columns are matched by name.

## List and delete

```bash
curl localhost:8000/v1/tabctx/datasets
curl -X DELETE localhost:8000/v1/tabctx/datasets/churn
```

## Headers

| Header | When |
|---|---|
| `x-tabctx-tenant-id` | Scopes every dataset id to a tenant. Required when the server runs with `TABCTX_REQUIRE_TENANT=true`. |
| `x-session-id` | Equal to the dataset id. Routes the request to the replica holding the context when there are several. |
| `Authorization: Bearer ...` | Through a gateway that needs an API key. |

## Status codes

`422` malformed request · `404` no such context · `413` too large to
admit · `503` busy, retry · `507` the model failed on the device · `401`
tenant required.
