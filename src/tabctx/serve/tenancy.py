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

Trust model: by default tenant identity is CALLER-SUPPLIED and NOT
verified -- that protects against accidental cross-tenant access and
guessed dataset_ids, but not against a malicious caller presenting
another tenant's id. For VERIFIED identity, set ``TABCTX_API_KEYS`` to a
comma-separated ``key:tenant`` map (inject via a k8s Secret in real
deployments): every /v1/tabctx request must then carry
``Authorization: Bearer <key>``, and the tenant is derived from the key
(401 for a missing/unknown key). A tenant header may still be sent but
must agree with the key's tenant. One tenant may own several keys;
rotating a key is adding the new one and dropping the old. Stronger
schemes (OIDC, mTLS) still belong in a fronting proxy -- this covers the
common shared-secret case without one.

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
API_KEYS_ENV_VAR = "TABCTX_API_KEYS"

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


class InvalidApiKeyError(TabctxError):
    """API-key mode is on and the request's bearer key is missing,
    unknown, or contradicts a tenant header. Mapped to HTTP 401. The
    message never confirms whether a presented key exists."""


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


def api_key_map() -> dict[str, str] | None:
    """Parse TABCTX_API_KEYS ("key:tenant,key2:tenant2"); None when the
    deployment doesn't use API-key mode. Malformed entries fail loudly at
    call time rather than silently dropping a key."""
    raw = os.environ.get(API_KEYS_ENV_VAR, "").strip()
    if not raw:
        return None
    mapping: dict[str, str] = {}
    for entry in raw.split(","):
        key, sep, tenant = entry.strip().partition(":")
        if not sep or not key or not _TENANT_ID_RE.match(tenant):
            raise ValueError(
                f"{API_KEYS_ENV_VAR} entries must be 'key:tenant' with a "
                f"valid tenant id; got {entry.strip()!r}"
            )
        mapping[key] = tenant
    return mapping


def _bearer_key(headers: Mapping[str, str]) -> str | None:
    for k, v in headers.items():
        if k.lower() == "authorization":
            scheme, _, key = v.partition(" ")
            if scheme.lower() == "bearer" and key.strip():
                return key.strip()
            return None
    return None


def resolve_tenant_id(headers: Mapping[str, str]) -> str | None:
    """The request's tenant, per the deployment's mode:

    - API-key mode (TABCTX_API_KEYS set): tenant is DERIVED from the
      verified bearer key -- the only mode where identity is trustworthy.
      A tenant header may accompany it but must agree.
    - Required-header mode (TABCTX_REQUIRE_TENANT=true): caller-supplied
      header, mandatory.
    - Default: caller-supplied header, optional (unscoped when absent).
    """
    keys = api_key_map()
    header_tenant = tenant_id_from_headers(headers)
    if keys is not None:
        key = _bearer_key(headers)
        if key is None or key not in keys:
            raise InvalidApiKeyError(
                "this deployment requires 'Authorization: Bearer <api key>' "
                "on every /v1/tabctx request, and the key was missing or "
                "not recognized"
            )
        tenant_id = keys[key]
        if header_tenant is not None and header_tenant != tenant_id:
            raise InvalidApiKeyError(
                f"{TENANT_HEADER!r} header does not match the tenant this "
                "API key belongs to"
            )
        return tenant_id
    if header_tenant is None and tenant_required():
        raise TenantRequiredError(
            f"this deployment requires a {TENANT_HEADER!r} header on every "
            "/v1/tabctx request (TABCTX_REQUIRE_TENANT=true)"
        )
    return header_tenant


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
