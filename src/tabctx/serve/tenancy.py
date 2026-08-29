"""Tenant namespacing for the serving layer.

The problem (tabctx ROADMAP.md Priority 2): dataset_id alone is a flat,
unauthenticated, guessable namespace -- any caller who knows or guesses
another tenant's dataset_id could predict() against their cached model.
That's a data-leakage risk, not a missing nicety.

The minimal fix implemented here is namespacing, not a full auth system:
callers send a tenant id in the `x-tabctx-tenant-id` header, and the
serving layer scopes every dataset_id as `t:{tenant_id}:{dataset_id}`
before it ever touches the cache. Responses show the caller's own
(unscoped) dataset_id; the scoping prefix never leaves the server.

Deployment modes (env var TABCTX_REQUIRE_TENANT):

- "false" (default, backward compatible): the header is honored when
  present, and absent-header requests use the raw dataset_id unscoped.
  NOTE this mode is not a security boundary: nothing stops an unscoped
  caller from crafting a dataset_id that textually collides with a
  scoped one. It exists so single-tenant/dev deployments keep working
  with zero client changes.
- "true": every /v1/tabctx request MUST carry a valid tenant header;
  requests without one are rejected (HTTP 401 at the app layer). With no
  unscoped namespace, tenants are fully isolated from each other --
  the only way to reach a context is to present the same tenant id that
  fit it.

Trust model, stated plainly: tenant identity is CALLER-SUPPLIED and NOT
verified against any identity source. That means this protects against
accidental cross-tenant access and guessed dataset_ids, but not against
a malicious caller who knows another tenant's id and presents it.
Verifying tenant identity (API keys mapped to tenant ids at an
authenticating proxy, mTLS, etc.) is deliberately out of scope here and
belongs in front of the service; this module gives that proxy something
meaningful to enforce.

Interplay with session affinity (serve/affinity.py): the routing key
stays the *unscoped* dataset_id, which is fine -- routing only needs to
be deterministic per dataset_id, and two tenants sharing a dataset_id
string simply land on the same replica while their contexts stay
separate cache entries. Transport-agnostic and unit-tested like
affinity.py.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping

from tabctx.errors import InvalidInputError, TabctxError

TENANT_HEADER = "x-tabctx-tenant-id"
REQUIRE_TENANT_ENV_VAR = "TABCTX_REQUIRE_TENANT"

# Conservative charset: forbids the scoping separator (":") by
# construction, plus anything that would make ids annoying in logs/URLs.
_TENANT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

# Scoped ids are namespaced under this prefix. Kept distinct from bare
# UUIDs/user ids so a scoped id is recognizable in logs and cache stats.
_SCOPE_PREFIX = "t"


class TenantRequiredError(TabctxError):
    """The deployment requires a tenant id and the request carried none.

    Mapped to HTTP 401 at the app layer (it's an authentication-shaped
    failure, not a malformed-input one)."""


class InvalidTenantIdError(InvalidInputError):
    """The tenant id doesn't match the allowed pattern."""


def tenant_required() -> bool:
    return os.environ.get(REQUIRE_TENANT_ENV_VAR, "false").strip().lower() in (
        "true",
        "1",
        "yes",
    )


def _normalize(header_key: str) -> str:
    return header_key.lower().replace("-", "_")


def tenant_id_from_headers(headers: Mapping[str, str]) -> str | None:
    """Extract and validate the tenant header value, or None if absent.

    Same case/-/_ tolerance as the session header (see affinity.py).
    Raises InvalidTenantIdError for a present-but-malformed value --
    silently ignoring a malformed tenant id would drop the caller into
    the unscoped namespace, which is the opposite of what they asked for.
    """
    want = _normalize(TENANT_HEADER)
    for key, value in headers.items():
        if _normalize(key) == want:
            if not _TENANT_ID_RE.match(value):
                raise InvalidTenantIdError(
                    f"{TENANT_HEADER!r} header value {value!r} is invalid: "
                    "expected 1-128 chars of [A-Za-z0-9._-], starting "
                    "alphanumeric"
                )
            return value
    return None


def resolve_tenant_id(headers: Mapping[str, str]) -> str | None:
    """tenant_id_from_headers plus enforcement of the deployment mode."""
    tenant_id = tenant_id_from_headers(headers)
    if tenant_id is None and tenant_required():
        raise TenantRequiredError(
            f"this deployment requires a {TENANT_HEADER!r} header on every "
            "/v1/tabctx request (TABCTX_REQUIRE_TENANT=true)"
        )
    return tenant_id


def scope_dataset_id(tenant_id: str | None, dataset_id: str) -> str:
    """The cache-facing dataset id for (tenant, dataset)."""
    if tenant_id is None:
        return dataset_id
    return f"{_SCOPE_PREFIX}:{tenant_id}:{dataset_id}"


def unscope_dataset_id(tenant_id: str | None, scoped_id: str) -> str:
    """Inverse of scope_dataset_id, for responses -- the scoping prefix
    must never leak to callers (it embeds the tenant id, and callers
    should see exactly the ids they chose or were given)."""
    if tenant_id is None:
        return scoped_id
    prefix = f"{_SCOPE_PREFIX}:{tenant_id}:"
    if not scoped_id.startswith(prefix):
        raise ValueError(
            f"scoped id {scoped_id!r} does not belong to tenant {tenant_id!r}"
        )
    return scoped_id[len(prefix):]
