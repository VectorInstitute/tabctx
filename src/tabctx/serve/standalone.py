"""Standalone host: the tabctx service as a plain uvicorn process.

`python -m tabctx.serve` serves the same routes as the Ray Serve
deployment (serve/app.py) from one ordinary process, for platforms that
don't run Ray -- e.g. a Kubernetes Deployment behind a gateway. What the
Ray host provides and what replaces it here:

- Replica restart on a dead per-GPU worker (Serve's check_health): point
  the platform's liveness probe at GET /livez, which fails exactly then.
  /readyz stays the readiness probe (it also 503s on transient pressure,
  which must not restart the process).
- Request shedding (the deployment's max_ongoing_requests /
  max_queued_requests): the limiter below, configured by
  TABCTX_MAX_ONGOING_REQUESTS and TABCTX_MAX_QUEUED_REQUESTS (defaults
  8 / 16, matching serve/app.py). Health routes bypass it -- a saturated
  service must still answer its probes.
- Multi-replica session affinity: nothing replaces it. Run ONE replica,
  or front several with a router that pins the session-affinity header
  (serve/affinity.py) to a replica.
- A per-request time ceiling: uvicorn has none; enforce timeouts
  client-side (gateways already read with their own budget).
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from tabctx.serve.service import ROUTE_SPECS, TabctxService, fastapi_app

MAX_ONGOING_ENV_VAR = "TABCTX_MAX_ONGOING_REQUESTS"
MAX_QUEUED_ENV_VAR = "TABCTX_MAX_QUEUED_REQUESTS"
# The Ray deployment's defaults (serve/app.py); see its comment for why 8
# concurrent requests are safe (each device's engine schedules its own
# GPU work; extra in-flight requests queue or coalesce there).
DEFAULT_MAX_ONGOING = 8
DEFAULT_MAX_QUEUED = 16
# Never shed probes: a platform restarts a replica whose liveness probe
# times out, and a loaded replica is exactly when that must not happen.
_UNSHED_PATHS = frozenset({"/livez", "/readyz", "/healthz"})


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError(f"{name}={raw!r} is not an int") from e
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


class _RequestLimiter:
    """Serve-shaped backpressure as pure ASGI middleware: at most
    `max_ongoing` requests execute, at most `max_queued` wait for a slot,
    and the rest get 503 -- honest and retryable, the same contract as
    Ray Serve's max_queued_requests."""

    def __init__(self, app, max_ongoing: int, max_queued: int) -> None:
        self._app = app
        self._slots = asyncio.Semaphore(max_ongoing)
        self._max_queued = max_queued
        self._queued = 0

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope["path"] in _UNSHED_PATHS:
            await self._app(scope, receive, send)
            return
        if self._slots.locked() and self._queued >= self._max_queued:
            response = JSONResponse(
                {
                    "detail": f"the service is at capacity ({self._max_queued} "
                    "requests already queued); retry with backoff"
                },
                status_code=503,
            )
            await response(scope, receive, send)
            return
        self._queued += 1
        try:
            await self._slots.acquire()
        finally:
            self._queued -= 1
        try:
            await self._app(scope, receive, send)
        finally:
            self._slots.release()


def make_app(service: TabctxService | None = None) -> FastAPI:
    """The service's routes rebound to a plain FastAPI app.

    Builds the engine up front (so construction cost and failures happen
    at startup, not on the first request), registers every route captured
    in ROUTE_SPECS with its handler bound to this instance, and closes
    the engine's worker processes on shutdown."""
    service = service or TabctxService()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        service.close()

    app = FastAPI(title=fastapi_app.title, lifespan=lifespan)
    for spec in ROUTE_SPECS:
        app.add_api_route(
            spec.path,
            getattr(service, spec.endpoint_name),
            methods=list(spec.methods),
            response_model=spec.response_model,
            openapi_extra=spec.openapi_extra,
        )
    app.add_middleware(
        _RequestLimiter,
        max_ongoing=_positive_int_env(MAX_ONGOING_ENV_VAR, DEFAULT_MAX_ONGOING),
        max_queued=_positive_int_env(MAX_QUEUED_ENV_VAR, DEFAULT_MAX_QUEUED),
    )
    return app


def main(argv: list[str] | None = None) -> None:
    """`python -m tabctx.serve`: parse arguments and serve until stopped."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m tabctx.serve",
        description="Serve tabctx without Ray, from one uvicorn process. "
        "Engine configuration comes from the TABCTX_* environment "
        "variables (serve/factory.py documents them).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="interface to bind (default %(default)s; a container that "
        "should accept cluster traffic passes 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="port to bind (default %(default)s)"
    )
    args = parser.parse_args(argv)

    import uvicorn

    uvicorn.run(make_app(), host=args.host, port=args.port)
