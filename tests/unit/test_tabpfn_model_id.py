"""Unit tests for the TabPFN backend's served model id (no tabpfn needed)."""

import pytest

from tabctx.backends.tabpfn import served_model_id


@pytest.mark.parametrize(
    ("version", "model_id"),
    [
        ("v3", "tabpfn-3"),
        ("v3.5", "tabpfn-3.5"),
        ("v3.5-fast", "tabpfn-3.5-fast"),
        ("v2.5", "tabpfn-2.5"),
    ],
)
def test_served_model_id_follows_model_version(version, model_id):
    assert served_model_id(version) == model_id
