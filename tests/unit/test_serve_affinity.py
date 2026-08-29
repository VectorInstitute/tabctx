"""Unit tests for the session-affinity contract (serve/affinity.py).

Transport-agnostic on purpose: these run with core deps only (no ray, no
fastapi), because the contract itself -- header name resolution, the
session-id == dataset_id rule -- is what multi-replica correctness rests
on, and it must be testable everywhere.
"""

import pytest

from tabctx.serve.affinity import (
    DEFAULT_SESSION_ID_HEADER,
    SESSION_ID_HEADER_ENV_VAR,
    SessionAffinityMismatchError,
    resolve_dataset_id,
    session_id_from_headers,
    session_id_header_name,
)


class TestHeaderName:
    def test_default(self, monkeypatch):
        monkeypatch.delenv(SESSION_ID_HEADER_ENV_VAR, raising=False)
        assert session_id_header_name() == DEFAULT_SESSION_ID_HEADER == "x-session-id"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv(SESSION_ID_HEADER_ENV_VAR, "x-correlation-id")
        assert session_id_header_name() == "x-correlation-id"


class TestSessionIdFromHeaders:
    def test_absent(self):
        assert session_id_from_headers({}) is None
        assert session_id_from_headers({"content-type": "application/json"}) is None

    def test_exact_match(self):
        assert session_id_from_headers({"x-session-id": "ds-1"}) == "ds-1"

    def test_case_insensitive(self):
        assert session_id_from_headers({"X-Session-Id": "ds-1"}) == "ds-1"

    def test_underscore_dash_equivalence(self):
        # Intermediate proxies (nginx, AWS API Gateway) rewrite the
        # separator; affinity must survive that.
        assert session_id_from_headers({"x_session_id": "ds-1"}) == "ds-1"

    def test_env_override_changes_match(self, monkeypatch):
        monkeypatch.setenv(SESSION_ID_HEADER_ENV_VAR, "x-correlation-id")
        headers = {"x-session-id": "wrong", "X-Correlation-ID": "right"}
        assert session_id_from_headers(headers) == "right"


class TestResolveDatasetId:
    def test_neither(self):
        assert resolve_dataset_id(None, None) is None

    def test_body_only(self):
        assert resolve_dataset_id(None, "ds-1") == "ds-1"

    def test_header_only_adopted_as_dataset_id(self):
        assert resolve_dataset_id("ds-1", None) == "ds-1"

    def test_matching_pair(self):
        assert resolve_dataset_id("ds-1", "ds-1") == "ds-1"

    def test_mismatch_rejected_loudly(self):
        with pytest.raises(SessionAffinityMismatchError) as exc_info:
            resolve_dataset_id("ds-1", "ds-2")
        # The error must name both values -- it exists to make a silent
        # routing bug loud and debuggable.
        assert "ds-1" in str(exc_info.value)
        assert "ds-2" in str(exc_info.value)
