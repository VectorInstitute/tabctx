"""Shared fixtures for the unit tests."""

import pytest


@pytest.fixture(autouse=True)
def _mcp_signing_key(tmp_path_factory, monkeypatch):
    """tabctx-mcp signs saved models with a key in the user's config
    directory; tests use a throwaway one."""
    key = tmp_path_factory.mktemp("mcp-key") / "signing.key"
    monkeypatch.setenv("TABCTX_MCP_KEY_FILE", str(key))
