"""Predictive quantiles from real TabICL (CPU): one pass returns the mean
and the requested quantiles, row-aligned and ordered by level.

Skipped without tabicl or its checkpoint in the local HuggingFace cache."""

import numpy as np
import pytest

pytest.importorskip("tabicl")
pytest.importorskip("torch")

from tabctx.backends.tabicl import TabICLBackend  # noqa: E402


@pytest.fixture(scope="module")
def backend():
    try:
        b = TabICLBackend(device="cpu")
        b.warmup()
    except Exception as e:  # noqa: BLE001 -- no checkpoint offline
        pytest.skip(f"TabICL unavailable: {e}")
    return b


def test_quantiles_bracket_the_mean_and_follow_the_levels(backend):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = 3 * X[:, 0] + rng.normal(scale=0.5, size=300)
    payload = backend.fit(X[:250], y[:250], "regression")
    out = backend.predict(payload, X[250:], quantiles=(0.05, 0.5, 0.95))
    q = np.asarray(out.quantiles)
    mean = np.asarray(out.predictions)
    assert q.shape == (50, 3)
    assert (q[:, 0] <= q[:, 1]).all() and (q[:, 1] <= q[:, 2]).all()
    assert ((q[:, 0] <= mean) & (mean <= q[:, 2])).mean() > 0.9
    # A 90% interval should cover most held-out targets.
    covered = ((q[:, 0] <= y[250:]) & (y[250:] <= q[:, 2])).mean()
    assert covered > 0.7
    # Same mean as a plain predict: the quantiles cost no second model.
    np.testing.assert_allclose(backend.predict(payload, X[250:]).predictions, mean)
