"""Unit tests for tenant namespacing (serve/tenancy.py)."""

import pytest

from tabctx.errors import InvalidInputError
from tabctx.serve.tenancy import (
    REQUIRE_TENANT_ENV_VAR,
    TENANT_HEADER,
    InvalidTenantIdError,
    TenantRequiredError,
    resolve_tenant_id,
    scope_dataset_id,
    tenant_id_from_headers,
    tenant_required,
    unscope_dataset_id,
)


class TestTenantIdFromHeaders:
    def test_absent(self):
        assert tenant_id_from_headers({}) is None

    def test_present(self):
        assert tenant_id_from_headers({TENANT_HEADER: "acme"}) == "acme"

    def test_case_and_separator_tolerant(self):
        assert tenant_id_from_headers({"X-Tabctx-Tenant-Id": "acme"}) == "acme"
        assert tenant_id_from_headers({"x_tabctx_tenant_id": "acme"}) == "acme"

    @pytest.mark.parametrize(
        "bad",
        [
            "",  # empty
            "a:b",  # scoping separator -- would break namespacing
            ".starts-with-dot",
            "-starts-with-dash",
            "has space",
            "x" * 129,  # too long
        ],
    )
    def test_malformed_rejected_not_ignored(self, bad):
        # Silently ignoring a malformed tenant id would drop the caller
        # into the unscoped namespace -- the opposite of what they asked.
        with pytest.raises(InvalidTenantIdError):
            tenant_id_from_headers({TENANT_HEADER: bad})

    def test_invalid_tenant_is_invalid_input(self):
        # Must map to 422 through the app layer's existing error table.
        assert issubclass(InvalidTenantIdError, InvalidInputError)


class TestRequireTenantMode:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv(REQUIRE_TENANT_ENV_VAR, raising=False)
        assert tenant_required() is False
        assert resolve_tenant_id({}) is None

    @pytest.mark.parametrize("truthy", ["true", "True", " 1 ", "yes"])
    def test_enabled(self, monkeypatch, truthy):
        monkeypatch.setenv(REQUIRE_TENANT_ENV_VAR, truthy)
        assert tenant_required() is True
        with pytest.raises(TenantRequiredError):
            resolve_tenant_id({})
        # With a tenant present, required mode passes it through.
        assert resolve_tenant_id({TENANT_HEADER: "acme"}) == "acme"


class TestScoping:
    def test_none_tenant_is_identity(self):
        assert scope_dataset_id(None, "ds") == "ds"
        assert unscope_dataset_id(None, "ds") == "ds"

    def test_roundtrip(self):
        scoped = scope_dataset_id("acme", "ds-1")
        assert scoped != "ds-1"
        assert "acme" in scoped
        assert unscope_dataset_id("acme", scoped) == "ds-1"

    def test_distinct_tenants_never_collide(self):
        assert scope_dataset_id("acme", "ds") != scope_dataset_id("bcme", "ds")

    def test_tenant_dataset_boundary_is_unambiguous(self):
        # The classic namespacing bug: ("a", "b:c") colliding with
        # ("a:b", "c"). The tenant charset forbids ":" so the first
        # separator always ends the tenant id.
        assert scope_dataset_id("a", "b:c") != scope_dataset_id("a.b", "c")

    def test_unscope_foreign_id_refused(self):
        scoped = scope_dataset_id("acme", "ds-1")
        with pytest.raises(ValueError):
            unscope_dataset_id("other", scoped)


def test_callers_see_only_their_own_dataset_ids():
    from tabctx.serve.tenancy import scope_dataset_id, visible_dataset_id

    acme = scope_dataset_id("acme", "churn")
    other = scope_dataset_id("globex", "secret")
    assert visible_dataset_id("acme", acme) == "churn"
    assert visible_dataset_id("acme", other) is None
    assert visible_dataset_id("acme", "unscoped") is None
    assert visible_dataset_id(None, "unscoped") == "unscoped"
    assert visible_dataset_id(None, acme) is None  # never another tenant's
