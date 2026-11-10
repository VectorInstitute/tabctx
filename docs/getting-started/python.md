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

## Prediction intervals

TabICL and TabPFN predict a distribution over the target, not just a
number. `predict_quantiles` returns that distribution's quantiles, from
the same model call as the mean:

```python
reg = TabctxRegressor(client=client).fit(X_train, y_train)
q = reg.predict_quantiles(X_test, quantiles=[0.05, 0.5, 0.95])
low, median, high = q[:, 0], q[:, 1], q[:, 2]      # a 90% interval per row
```

Levels are in (0, 1), in any order; the columns follow it. Check the
intervals on held-out data before relying on them:
[`examples/prediction_intervals.py`](https://github.com/VectorInstitute/tabctx/blob/main/examples/prediction_intervals.py)
does it for California housing, where the 90% interval held 86.5% of the
prices. Classifiers already return a distribution: `predict_proba`.

## The client

`TabctxClient` is the HTTP client underneath, pure standard library. It
sets the routing and tenant headers for you.

```python
client = TabctxClient("http://localhost:8000", tenant_id="acme")

client.fit(X_train, y_train, dataset_id="churn-v1")            # DataFrame, array or rows
client.fit_file("train.parquet", target="churned", dataset_id="churn-v2")
result = client.predict("churn-v1", X_test, return_proba=True)
result.predictions, result.probabilities, result.classes
client.predict("prices", X_test, quantiles=[0.05, 0.95]).quantiles  # regression

client.datasets()             # what the server holds for this tenant
client.delete("churn-v1")     # gone from GPU, host memory and disk
client.models(), client.limits()
```

`base_url` and `api_key` default to `TABCTX_BASE_URL` and
`TABCTX_API_KEY`. The key is sent as a bearer token, never appears in
the client's `repr`, and is left out when the client is pickled.

## Survival and treatment effects

Two more tasks, each served by its own model (`GET /v1/models` lists
every model's `tasks`; a fit that names no model goes to one that fits
its task).

**Time to an event** (`survivalpfn-v0.1`): each training row has a time
and an event flag, 1 if the event happened at that time and 0 if the row
was censored then (still event-free when follow-up ended).

```python
client.fit(X_train, days, event=cancelled, dataset_id="retention")
r = client.predict("retention", X_now, times=[90, 365], quantiles=[0.25, 0.75])
r.predictions   # each row's median time to the event
r.survival      # per row, the probability of no event by day 90 and day 365
```

**Treatment effects** (`causalpfn-v0`): each training row has an outcome
and a treatment flag (1 treated, 0 control), from observational data.

```python
client.fit(X_train, spend, treatment=got_offer, dataset_id="offer")
effect = client.predict("offer", X_now).predictions  # per row: if treated minus if not
sum(effect) / len(effect)                            # the average effect over these rows
```

Nothing is estimated past the longest time in the training data: a
median, quantile or survival probability out there is `None`. Each model
uses at most 4,096 training rows as its context (a larger table is
subsampled, keeping the share of events or of each arm;
`TABCTX_SURVIVALPFN_CONTEXT_SIZE`, `TABCTX_CAUSALPFN_CONTEXT_SIZE`), and
every test row is predicted against that same context, so a row's answer
never depends on the other rows sent with it.

From a file, name the columns: `fit_file(path, target="days",
event_column="cancelled")` or `treatment_column=`; a flag column may be
0/1, true/false or yes/no. What the models
assume: the survival model, that censoring is independent of the event
given the features; the causal model, that every confounder of treatment
and outcome is a feature, and that every kind of row could have been
treated or not. No data can confirm the causal one. The causal model's
`quantiles` are wide (CausalPFN's own interval method, uncalibrated): read
them as the spread of an individual's effect, not as a confidence
interval for the average.

## Errors

Server errors come back as tabctx exceptions, importable from `tabctx`.
The in-process engine raises the same types for the same mistakes
(invalid input, an unknown context, a table too large); conditions that
are only temporary come back from a server as one retryable type,
`TabctxBackpressureError`.

| HTTP | Exception | Meaning |
|---|---|---|
| 422 | `InvalidInputError` | A malformed request: retry won't help. |
| 404 | `DatasetNotFoundError` | No such context: never fit, deleted, evicted, past its retention limit, or lost to a restart. Fit again. |
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
