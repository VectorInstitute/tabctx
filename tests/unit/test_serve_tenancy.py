"""Unit tests for tenant namespacing (serve/tenancy.py)."""

import pytest

from tabctx.errors import InvalidInputError
from tabctx.serve.tenancy import (
    API_KEYS_ENV_VAR,
    REQUIRE_TENANT_ENV_VAR,
    TENANT_HEADER,
    InvalidApiKeyError,
    InvalidTenantIdError,
    TenantRequiredError,
    api_key_map,
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


class TestApiKeyMode:
    def test_off_by_default(self, monkeypatch):
        monkeypatch.delenv(API_KEYS_ENV_VAR, raising=False)
        assert api_key_map() is None

    def test_parse(self, monkeypatch):
        monkeypatch.setenv(API_KEYS_ENV_VAR, "sk-aaa:acme, sk-bbb:globex")
        assert api_key_map() == {"sk-aaa": "acme", "sk-bbb": "globex"}

    @pytest.mark.parametrize("bad", ["justakey", "key:", ":tenant", "k:bad tenant"])
    def test_malformed_map_fails_loudly(self, monkeypatch, bad):
        monkeypatch.setenv(API_KEYS_ENV_VAR, bad)
        with pytest.raises(ValueError):
            api_key_map()

    def test_tenant_derived_from_key(self, monkeypatch):
        monkeypatch.setenv(API_KEYS_ENV_VAR, "sk-aaa:acme")
        headers = {"Authorization": "Bearer sk-aaa"}
        assert resolve_tenant_id(headers) == "acme"

    def test_missing_or_unknown_key_rejected(self, monkeypatch):
        monkeypatch.setenv(API_KEYS_ENV_VAR, "sk-aaa:acme")
        with pytest.raises(InvalidApiKeyError):
            resolve_tenant_id({})
        with pytest.raises(InvalidApiKeyError) as exc_info:
            resolve_tenant_id({"Authorization": "Bearer sk-wrong"})
        # Never confirm whether a presented key exists.
        assert "sk-wrong" not in str(exc_info.value)

    def test_header_must_agree_with_key(self, monkeypatch):
        monkeypatch.setenv(API_KEYS_ENV_VAR, "sk-aaa:acme")
        headers = {"Authorization": "Bearer sk-aaa", TENANT_HEADER: "globex"}
        with pytest.raises(InvalidApiKeyError):
            resolve_tenant_id(headers)
        headers[TENANT_HEADER] = "acme"  # agreeing header is fine
        assert resolve_tenant_id(headers) == "acme"

    def test_key_mode_overrides_header_supplied_identity(self, monkeypatch):
        # The whole point: a caller cannot pick a tenant, only prove one.
        monkeypatch.setenv(API_KEYS_ENV_VAR, "sk-aaa:acme")
        with pytest.raises(InvalidApiKeyError):
            resolve_tenant_id({TENANT_HEADER: "acme"})  # header alone, no key

    def test_non_bearer_scheme_rejected(self, monkeypatch):
        monkeypatch.setenv(API_KEYS_ENV_VAR, "sk-aaa:acme")
        with pytest.raises(InvalidApiKeyError):
            resolve_tenant_id({"Authorization": "Basic c2stYWFh"})


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
