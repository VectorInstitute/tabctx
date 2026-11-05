# Serving tabctx

How to install, run and talk to a tabctx deployment, and how it scales.

## Installation

Requires Python ≥ 3.12.

```bash
pip install tabctx               # core library (FakeBackend, no GPU deps)
pip install "tabctx[tabicl]"     # + real TabICL backend (torch, tabicl)
pip install "tabctx[tabpfn]"     # + TabPFN backend (weights license-gated:
                                 #   ux.priorlabs.ai + export TABPFN_TOKEN)
pip install "tabctx[serve]"      # + Ray Serve deployment (ray[serve], fastapi,
                                 #   pyarrow and python-multipart for table files)
pip install "tabctx[standalone]" # + the Ray-free host (python -m tabctx.serve):
                                 #   [serve] minus ray, single replica only
pip install "tabctx[client]"     # + scikit-learn estimators (scikit-learn, pyarrow, pandas)
pip install "tabctx[dev]"        # + test dependencies
```

## Ways to run it

**See it work with zero setup** (no GPU, no model download: a fake
backend stands in for TabICL):

```bash
pip install tabctx
curl -sO https://raw.githubusercontent.com/VectorInstitute/tabctx/main/examples/local_fit_predict.py
python local_fit_predict.py
```

**Use it for real**, with TabICL doing the actual predicting:

```bash
pip install "tabctx[tabicl]"
```

```python
from tabctx import TabctxEngine, ContextCacheManager
from tabctx.backends.tabicl import TabICLBackend
from tabctx.memory import AdaptiveMemoryEstimator, PowerLawMemoryEstimator, A100_40GB_TABICL_CALIBRATION

estimator = AdaptiveMemoryEstimator(fallback=PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION))
cache = ContextCacheManager(capacity_bytes=estimator.ceiling_bytes())
engine = TabctxEngine(backend=TabICLBackend(), cache=cache, estimator=estimator)

# task is inferred from y_train when omitted (non-integer numbers, or many
# distinct whole numbers such as prices, mean regression); pass
# task="classification" or "regression" to be explicit. Missing values
# (NaN, or null in JSON) are fine: both models handle them.
dataset_id = engine.fit(X_train, y_train)
result = engine.predict(dataset_id, X_test, return_proba=True)

# The whole point: reuse the same cached context for a new test batch,
# with no re-fit cost.
result2 = engine.predict(dataset_id, other_X_test)
```

**Serve it over HTTP** with Ray Serve: `src/tabctx/serve/app.py` is a
ready-to-run deployment (`fit`/`predict` endpoints, health checks, live
memory-usage reporting). See `benchmarks/README.md` for how to measure it
once it's running.

**Or without Ray**: `python -m tabctx.serve` (the `standalone` extra)
serves the same endpoints from one plain uvicorn process -- for platforms
that don't run Ray, e.g. a Kubernetes Deployment behind a gateway. One
replica only (the multi-replica session-affinity router is Ray's); point
the platform's liveness probe at `GET /livez` and its readiness probe at
`GET /readyz`. `serve/standalone.py` documents what else the Ray host
provided and what replaces it.

**Talk to a deployment** as a scikit-learn model (see the [Quickstart](getting-started/quickstart.md)), or with the bundled client (pure stdlib, sets the
routing/tenancy headers for you, raises the same tabctx exceptions the
in-process engine does):

```python
from tabctx import TabctxClient

client = TabctxClient("http://localhost:8000", tenant_id="acme")
client.fit(X_train, y_train, dataset_id="churn-v1")
result = client.predict("churn-v1", X_test, return_proba=True)
client.fit_file("train.parquet", target="churned", dataset_id="churn-v2")
```

## Table files and text columns

A fit or predict can send its table as rows of JSON, or as a file in
the same request (multipart/form-data, a `file` part beside the other
fields as form fields):

```bash
curl $URL/v1/tabctx/fit -F model=tabicl-v2 -F dataset_id=churn -F target=churned -F file=@train.parquet
curl $URL/v1/tabctx/predict -F dataset_id=churn -F file=@test.csv
```

- **Formats**, detected from the content: CSV with a header row (plain,
  gzip or zstd) and Parquet. Parquet keeps types and nulls and is several
  times smaller than JSON (a 5,000 x 500 float table: 51 MB as JSON,
  13 MB as Parquet), so it is the way to send a large table.
- **Text columns** (`"gold"`, `"CA"`) work in files and JSON rows alike:
  the server codes each text column's values at fit (`tabctx.schema`),
  keeps that encoding with the context, and applies it at predict; a
  value the training table never had becomes missing. **Missing values**:
  JSON `null`, a Parquet null, or a CSV cell that is empty or `NA`,
  `N/A`, `null`, `None`, `NaN`.
- **Columns** of a test file are matched to the training file's by
  name: any order, and extra columns (such as the target) are ignored. A
  missing one is a 422.
- The target is `target` (or `target_column`), default the last column.

Tables too big for GPU and host memory together can still fit: TabICL
offloads stage outputs to memory-mapped files under `TABCTX_OFFLOAD_DIR`
(default `<TABCTX_SPILL_DIR>/offload`), slower but without exhausting the
pod's RAM (one A100 fit 130k x 500 this way).

A file can also be uploaded first and referenced by id, for one that
should travel apart from the request that uses it (uploads stream
straight to the serving replica's disk):

```python
client = TabctxClient("http://localhost:8000")
upload_id = client.upload_csv_file("train.csv", dataset_id="churn-v1")
client.fit_uploaded(upload_id, "churn-v1", target_column="label")

test_upload = client.upload_csv_file("scoring_batch.csv", "churn-v1")
result = client.predict("churn-v1", test_upload_id=test_upload)
```

- Uploads are **single-use** (consumed by the fit/predict that
  references them) and expire after `TABCTX_UPLOAD_TTL_S` (default 1h);
  size-capped by `TABCTX_MAX_UPLOAD_BYTES` (default 4GiB, enforced
  mid-stream).
- Uploads are replica-local, so in multi-replica deployments the upload
  request must carry the same `x-session-id: <dataset_id>` header as
  everything else -- affinity routes the upload, fit, and predicts to
  one replica. (The bundled client does this for you.)
- Transport size and admission are separate layers: a huge file uploads
  fine, and the memory-admission gate then decides whether the *table*
  fits the GPU budget (413 if not).

## Scaling out: multi-replica serving

The context cache is in-process and per-replica, so at `num_replicas >= 2`
requests for one dataset must consistently reach the replica that holds
its context. The deployment ships with Ray Serve's consistent-hash
request router (requires ray >= 2.58) configured for strict session
affinity, and the contract is simply: **send the dataset_id as the
`x-session-id` header on every `/v1/tabctx/fit` and `/v1/tabctx/predict`
call.**

```bash
curl -X POST $URL/v1/tabctx/fit \
  -H "content-type: application/json" -H "x-session-id: my-dataset" \
  -d '{"train_X": [[1,2],[3,4]], "train_y": [0,1], "dataset_id": "my-dataset"}'

curl -X POST $URL/v1/tabctx/predict \
  -H "content-type: application/json" -H "x-session-id: my-dataset" \
  -d '{"dataset_id": "my-dataset", "test_X": [[5,6]]}'
```

Details worth knowing:

- fit() adopts the header value as the dataset_id if the body omits one; a
  header/body mismatch is rejected 422 (a silent mismatch would mis-route).
- Omitting the header still works, but only routes correctly on
  single-replica deployments.
- Affinity is strict (no fallback replica): if the owning replica is
  backpressured, callers see retry-with-backoff/503 rather than a
  spurious 404 from a replica that doesn't hold the context.
- Responses report `served_by` (the replica tag) so affinity is
  observable and testable end to end.
- When replicas share one physical GPU (e.g. 2 replicas at
  `num_gpus: 0.5`), set `TABCTX_GPU_MEMORY_FRACTION` (e.g. `0.45`) so
  each replica budgets its share of GPU memory. The budget is the
  visible device's real memory (an L4's 24GB, an A100's 40GB), detected
  at startup and reported by `/readyz` as `gpu_capacity_bytes`.
- If the deployment renames the affinity header
  (`RAY_SERVE_SESSION_ID_HEADER_KEY`), construct the client with
  `session_header=` to match; the header name is the routing key.
- `TABCTX_BACKEND=fake` runs the whole serve stack without a GPU or
  torch -- `tests/integration/test_multi_replica_affinity.py` uses it to
  prove the multi-replica contract on a laptop.

## Several GPUs in one replica

A replica that can see several GPUs serves from all of them
(`TABCTX_DEVICES=auto`, the default, or an explicit `cuda:0,cuda:1,...`):
one engine per GPU, with its own backends, cache and spill directory,
behind the same API. No client changes and no routing header needed.

- **Placement.** Each fit goes to a GPU where the table fits *without
  evicting* anyone, preferring the least active GPU (calls in flight,
  then calls in the last 30s). If no GPU can take it without eviction,
  it goes to the GPU with the most headroom. Concurrent users spread
  across GPUs, and a large table gets a whole GPU's headroom.
- **Predicts** run on the GPU that holds the context, including after a
  spill and restore, or a replica restart.
- **Limits.** Neither TabICL nor TabPFN can split one context across
  GPUs, so a table's ceiling is one GPU's memory. What more GPUs buy is
  N x the resident cache and N fits/predicts in flight at once.
- **Concurrency.** Routing reads lock-free snapshots, so it never waits
  on a GPU's work: fits and predicts on different GPUs run at the same
  time. Fits and evicts of the *same* dataset_id are serialized, so a
  retried fit can't leave two copies. A re-fit lands first and only then
  replaces the old copy: a failed re-fit (413, OOM) leaves the previous
  context serving.
- **A process per GPU.** Each GPU's engine runs in its own worker process
  (`TABCTX_WORKER_PROCESSES=1`, the default with several GPUs; `0` runs
  them as threads). Fits are host-bound, so threads would contend for
  Python's GIL: on 4x A100, four concurrent fits ran at ~4x aggregate
  from processes and 2-3x from threads. A worker that dies fails the
  replica's health check, so the serving framework restarts it.
- **Small predicts don't wait for fits.** A predict of up to 2,048 rows
  against a resident context runs beside a fit on the same GPU (~0.1 s
  instead of the fit's full duration); larger predicts and restores of
  spilled contexts wait their turn.
- **Placement is pluggable.** The rule above is the default
  `PlacementPolicy` (`tabctx.placement`); pass your own to
  `MultiDeviceEngine(placement=...)` to steer tabular work, e.g. away
  from a GPU a platform scheduler wants for an LLM. Each placement and
  removal is a residency event (`tabctx.events`) for any subscriber;
  `GET /v1/tabctx/residency` returns the latest.
- **The replica must be given the GPUs.** Under Ray Serve that means
  `ray_actor_options.num_gpus: N`, or, for a `num_gpus: 0` actor on a
  GPU pod (as in inference-platform's prod config),
  `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`. A standalone host
  (`python -m tabctx.serve`) simply sees the process's
  `CUDA_VISIBLE_DEVICES`, nothing extra.
- **Spill.** Each GPU spills to its own subdirectory (`cuda_0/`, ...);
  `TABCTX_SPILL_CAPACITY_BYTES` is the replica's total, split evenly.
  Changing between one and several GPUs changes the layout, and the
  replica logs a warning about spilled files it won't restore.
- `/readyz` reports per-device contexts, bytes, activity and real GPU
  memory (`real_gpu_memory` keeps its flat fields, summed, plus
  `per_device`); `/v1/tabctx/limits` reports the best single GPU's
  headroom and each device's.

## Listing and deleting contexts

`GET /v1/tabctx/datasets` lists the contexts the replica holds for the
caller's tenant, resident or spilled to disk: id, model, task, shape,
size, location, GPU and fit time. It shows metadata only, never a table.
`DELETE /v1/tabctx/datasets/{dataset_id}` removes one from device
memory, host memory and the disk tier on every GPU of the replica. It
answers 404 if there is none, and 503 if it can't be done right now
(retry; the client does).

```python
client = TabctxClient("http://localhost:8000", tenant_id="acme")
client.datasets()            # [{"dataset_id": "churn-v1", "model": "tabicl-v2", ...}]
client.delete("churn-v1")    # DatasetNotFoundError if it isn't there
```

```bash
curl localhost:8000/v1/tabctx/datasets
curl -X DELETE localhost:8000/v1/tabctx/datasets/churn-v1
```

What a delete covers:

- **The context and its spilled copy.** A predict already running on it
  finishes, then its memory is freed. A spill or restore of it in
  progress lands first, so no copy reappears afterwards; if one doesn't
  land in time the delete answers 503 and deleted nothing.
- **Every GPU, or none.** While a GPU worker is down, list and delete
  answer 503: that worker's spilled copies stay on disk until the
  replica restarts it, and would come back. If one GPU fails mid-delete,
  the others are already cleared and the answer is 503; the retry
  finishes the job.
- **Uploaded tables** are already gone: an upload is deleted as soon as
  the fit or predict that used it has read it, and unused uploads
  expire after `TABCTX_UPLOAD_TTL_S`.
- **One tenant's ids only.** Another tenant's context of the same name
  is untouched. Callers without a tenant id never see scoped contexts,
  and can't name one: ids starting with `t:` are reserved for them
  (422).

With several Ray replicas, a listing comes from whichever replica
answers (`served_by`) and shows only that replica's contexts. The client
sends a delete with `x-session-id: <dataset_id>`, so it reaches the
replica that holds the context.

## Tenant isolation

`dataset_id` alone is a flat, guessable namespace -- without isolation,
any caller who knows another tenant's dataset_id could predict against
their cached model. The serving layer therefore scopes every dataset_id
by the `x-tabctx-tenant-id` header before it touches the cache: two
tenants can both call their dataset `mnist` and get fully separate
contexts, and a caller without the right tenant id sees a clean 404,
never another tenant's model.

- Set `TABCTX_REQUIRE_TENANT=true` in production: every `/v1/tabctx`
  request must then carry a tenant id (else 401), so there is no
  unscoped namespace at all.
- By default the header is optional (dev-friendly, backward compatible),
  but then unscoped ids coexist with scoped ones and the boundary is
  advisory, not enforced.
- Tenant identity is caller-supplied and unverified by design; put an
  authenticating proxy (API keys -> tenant id) in front for real
  security. See `src/tabctx/serve/tenancy.py` for the full trust-model
  notes.
