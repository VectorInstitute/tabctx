# Configuration

A server is configured by environment variables, read once at start.

## Engine

::: tabctx.serve.factory
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: false

## Tenancy and routing

| Variable | Default | |
|---|---|---|
| `TABCTX_REQUIRE_TENANT` | `false` | `true`: every `/v1/tabctx` request must carry `x-tabctx-tenant-id` (else 401). Set it in production. |
| `RAY_SERVE_SESSION_ID_HEADER_KEY` | `x-session-id` | The replica-affinity header, shared with Ray Serve's router. |

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
