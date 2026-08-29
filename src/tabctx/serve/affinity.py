"""Session-affinity contract for multi-replica deployments.

tabctx's cache is in-process, per-replica. With 2+ replicas and Ray
Serve's default (power-of-two-choices) routing, a predict() for an
existing dataset_id has no guarantee of landing on the replica that
fit() it, so callers would see intermittent, spurious 404s as traffic
bounces between replicas that don't share state.

The fix (see serve/app.py for the router wiring): route on Ray Serve's
session-stickiness header via its consistent-hash request router, with
the CONTRACT that the session id IS the dataset_id. Clients send
`x-session-id: <dataset_id>` on every /v1/tabctx/fit and
/v1/tabctx/predict call; fit adopts the header value as the dataset_id
when the body doesn't name one. Single-replica deployments keep working
without the header.

This module is deliberately transport-agnostic (plain header mappings in,
tabctx errors out; no ray/fastapi imports) so the contract itself is
unit-testable without a running cluster.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from tabctx.errors import InvalidInputError

# The header name mirrors Ray Serve's own configuration: its proxy
# populates the routing metadata's session id from the header named by
# this env var (default "x-session-id"), so tabctx reads the same env var
# rather than inventing a parallel setting. Must match what the router
# sees, or affinity silently degrades to uniform routing.
SESSION_ID_HEADER_ENV_VAR = "RAY_SERVE_SESSION_ID_HEADER_KEY"
DEFAULT_SESSION_ID_HEADER = "x-session-id"


class SessionAffinityMismatchError(InvalidInputError):
    """The affinity routing key contradicts the request's dataset_id.

    A mismatched pair would *silently* break routing (the router pins the
    request by session id, the cache is keyed by dataset_id), surfacing
    later as confusing intermittent 404s in multi-replica deployments --
    exactly the failure mode sticky routing exists to prevent. A loud
    reject at the source is strictly better.
    """


def session_id_header_name() -> str:
    """The configured session-id header name (read fresh from the env)."""
    return os.environ.get(SESSION_ID_HEADER_ENV_VAR, DEFAULT_SESSION_ID_HEADER)


def _normalize(header_key: str) -> str:
    # Case-insensitive, with `-` and `_` equivalent -- the same matching
    # rule Ray Serve's proxy uses, so intermediate proxies that rewrite
    # the separator (nginx, AWS API Gateway, ...) don't drop affinity.
    return header_key.lower().replace("-", "_")


def session_id_from_headers(headers: Mapping[str, str]) -> str | None:
    """Return the session-id header value, or None if absent."""
    want = _normalize(session_id_header_name())
    for key, value in headers.items():
        if _normalize(key) == want:
            return value
    return None


def resolve_dataset_id(
    session_id: str | None, body_dataset_id: str | None
) -> str | None:
    """Reconcile the affinity header with the request body's dataset_id.

    Returns the dataset_id to use (which may still be None, meaning the
    caller wants a server-generated id -- fine on a single replica, where
    there is no routing to break).

    Raises SessionAffinityMismatchError when both are present and differ.
    """
    if session_id is not None and body_dataset_id is not None:
        if session_id != body_dataset_id:
            raise SessionAffinityMismatchError(
                f"{session_id_header_name()!r} header ({session_id!r}) does "
                f"not match dataset_id ({body_dataset_id!r}). The session "
                "header is the replica-affinity routing key and must equal "
                "the dataset_id (or be omitted, which only routes correctly "
                "on single-replica deployments)."
            )
        return body_dataset_id
    return body_dataset_id if session_id is None else session_id
