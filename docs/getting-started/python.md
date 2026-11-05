# Python and scikit-learn

## Estimators

`TabctxClassifier` and `TabctxRegressor` are scikit-learn estimators
whose fitting and predicting happen on a tabctx server.

```python
from tabctx import TabctxClassifier, TabctxRegressor, TabctxClient

client = TabctxClient("http://127.0.0.1:8000")
clf = TabctxClassifier(client=client)                       # the server's default model
reg = TabctxRegressor(model="tabpfn-3.5", client=client)    # or name one
```

- **Input:** DataFrames or arrays. Text columns, categories, dates and
  missing values are sent as they are, as Parquet.
- **Labels** may be strings, ints or bools; `predict` returns them as
  given to `fit`, and `predict_proba`'s columns follow `classes_`.
- **scikit-learn tools:** they clone and pickle, and work with
  `cross_val_score` and `GridSearchCV`.
- **One context per table:** the context id is a hash of the table,
  model and task, so estimators fit on the same data share one cached
  context on the server.
- **Evicted or restarted servers are handled:** an estimator keeps its
  training table and fits again if the server has lost the context.

```python
from sklearn.model_selection import cross_val_score

cross_val_score(TabctxClassifier(client=client), X, y, cv=5)
```

## The client

`TabctxClient` is the HTTP client underneath, pure standard library. It
sets the routing and tenant headers for you.

```python
client = TabctxClient("http://localhost:8000", tenant_id="acme")

client.fit(X_train, y_train, dataset_id="churn-v1")            # DataFrame, array or rows
client.fit_file("train.parquet", target="churned", dataset_id="churn-v2")
result = client.predict("churn-v1", X_test, return_proba=True)
result.predictions, result.probabilities, result.classes

client.datasets()             # what the server holds for this tenant
client.delete("churn-v1")     # gone from GPU, host memory and disk
client.models(), client.limits()
```

`base_url` and `api_key` default to `TABCTX_BASE_URL` and
`TABCTX_API_KEY`. The key is sent as a bearer token, never appears in
the client's `repr`, and is left out when the client is pickled.

## Errors

Server errors come back as tabctx exceptions, importable from `tabctx`.
The in-process engine raises the same types for the same mistakes
(invalid input, an unknown context, a table too large); conditions that
are only temporary come back from a server as one retryable type,
`TabctxBackpressureError`.

| HTTP | Exception | Meaning |
|---|---|---|
| 422 | `InvalidInputError` | A malformed request: retry won't help. |
| 404 | `DatasetNotFoundError` | No such context: never fit, deleted, evicted or lost to a restart. Fit again. |
| 404 | `UploadNotFoundError` | No such upload: expired, already used, or sent to another replica. |
| 413 | `AdmissionRejected` | The table is too large for this deployment's memory, or the request for its upload limit. |
| 503 | `TabctxBackpressureError` | Busy or temporarily unable; the client already retried with backoff. |
| 507 | `BackendComputeError` | The model failed on the device. |
| 401/403 | `PermissionError` | Missing tenant or API key. |

## The engine in your own process

Without a server, `TabctxEngine` gives you the same cache and admission
control in one Python process. This example uses the fake model, so it
runs anywhere; swap in `tabctx.backends.tabicl.TabICLBackend(device="cuda")`
(with the `tabicl` extra) for real predictions:

```python
--8<-- "examples/local_fit_predict.py:8:39"
```
