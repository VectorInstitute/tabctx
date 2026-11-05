# Configuration

A server reads its settings from environment variables once, at start.
Flags take `1`/`true` or `0`/`false`.

## Models and devices

| Variable | Default | |
|---|---|---|
| `TABCTX_BACKEND` | `tabicl` | Comma-separated models to serve: `tabicl`, `tabpfn`, `fake` (no GPU or weights; for tests). The first is the default model. `tabicl,tabpfn` serves both from one cache. |
| `TABCTX_DEVICES` | `auto` | `auto`: every visible GPU, else the CPU. Or a list such as `cuda:0,cuda:1`. Two or more devices place each table on one GPU. |
| `TABCTX_WORKER_PROCESSES` | `1` | With several GPUs, one worker process per GPU (`0`: threads of one process). |
| `TABCTX_GPU_MEMORY_FRACTION` | `1.0` | The share of each GPU this server may use, in (0, 1]. Lower it when replicas share a GPU. |
| `TABPFN_TOKEN` | | Needed to download TabPFN weights, after accepting Prior Labs' terms. |

## Cache and memory

| Variable | Default | |
|---|---|---|
| `TABCTX_SPILL_DIR` | off | A directory for the disk tier: evicted contexts are written here and restored on their next predict. Measure on your disk first; a re-fit can be faster than a restore from a slow one. |
| `TABCTX_SPILL_CAPACITY_BYTES` | 50 GiB | The disk tier's size, shared across GPUs. |
| `TABCTX_KV_CACHE` | `kv` | TabICL's context: `kv` (fastest predicts), `repr` (much less memory per context), `off` (re-encode on every predict). |
| `TABCTX_KV_PRECISION` | `int8` | TabICL's cached keys and values: `int8` (about half the memory; identical labels and metrics on 7 datasets, probabilities within 0.01), `fp16`, `native`. |
| `TABCTX_OFFLOAD_DIR` | `<spill dir>/offload` | Where TabICL may stage a very large fit's intermediate results on disk. |

## Requests

| Variable | Default | |
|---|---|---|
| `TABCTX_BATCH_WINDOW_MS` | `5` | Concurrent predicts on one context within this window share one model call. `0` turns it off. |
| `TABCTX_MAX_UPLOAD_BYTES` | 4 GiB | The largest table file a request may carry. |
| `TABCTX_UPLOAD_TTL_S` | `3600` | How long an unused `/v1/tabctx/upload` is kept. |

## Performance

| Variable | Default | |
|---|---|---|
| `TABCTX_COMPILE` | `1` | `torch.compile` TabICL's attention blocks at start (slower start, faster calls). |
| `TABCTX_COMPILE_AFTER_WARMUP` | `0` | Also compile shapes first seen in requests, inside those requests. |
| `TABCTX_FAST_PREPROCESSING` | `1` | Fit TabICL's Yeo-Johnson preprocessing on the GPU. |

## Tenancy and routing

| Variable | Default | |
|---|---|---|
| `TABCTX_REQUIRE_TENANT` | `false` | `true`, `1` or `yes`: every request that reads or changes contexts must carry `x-tabctx-tenant-id` (else 401). `/v1/tabctx/limits`, `/v1/models` and the probes don't need it. |
| `RAY_SERVE_SESSION_ID_HEADER_KEY` | `x-session-id` | The replica-affinity header, shared with Ray Serve's router. |

!!! warning "The tenant header is trusted as sent"
    tabctx does not authenticate callers: any caller can send any tenant
    id. Tenancy is a security boundary only behind a gateway that
    authenticates each caller, sets `x-tabctx-tenant-id` itself and
    strips any value the client sent. See
    [tenant isolation](../serving.md#tenant-isolation).

## Standalone host

`python -m tabctx.serve [--host 0.0.0.0] [--port 8000]`

| Variable | Default | |
|---|---|---|
| `TABCTX_MAX_ONGOING_REQUESTS` | `8` | Requests served at once. |
| `TABCTX_MAX_QUEUED_REQUESTS` | `16` | Requests waiting for a slot; beyond that, 503. Health probes are never shed. |

## Clients

| Variable | |
|---|---|
| `TABCTX_BASE_URL` | Where `TabctxClient()` and the estimators send requests. |
| `TABCTX_API_KEY` | Sent as `Authorization: Bearer` (a gateway's key). |
| `TABCTX_MCP_HOME` | Where `tabctx-mcp` keeps fitted models (default `.tabctx-mcp/`). |
| `TABCTX_MCP_KEY_FILE` | The key `tabctx-mcp` signs saved models with (default `~/.config/tabctx-mcp/signing.key`). |
