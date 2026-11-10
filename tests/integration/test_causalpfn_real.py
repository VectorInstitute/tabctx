"""CausalPFNBackend against real CausalPFN (CPU): one network for every fit,
the fit's CPU half before the device, effects that track the truth,
repeatable quantiles from the same pass as the effect, missing values
filled, and a context through the spill tier without its network.

Skipped without causalpfn or its checkpoint (from Hugging Face)."""

import numpy as np
import pytest

pytest.importorskip("causalpfn")
pytest.importorskip("torch")

from tabctx.backends.causalpfn import CausalPFNBackend  # noqa: E402
from tabctx.errors import InvalidInputError  # noqa: E402


@pytest.fixture(scope="module")
def backend():
    b = CausalPFNBackend(device="cpu")
    try:
        b.warmup()
    except OSError as e:  # the checkpoint can't be downloaded (offline)
        pytest.skip(f"CausalPFN weights unavailable: {e}")
    return b


@pytest.fixture(scope="module")
def data():
    """Confounded treatment (more likely with feature 0, which also raises
    the outcome) whose effect is 1 + feature 1."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(800, 5)).astype(np.float32)
    treated = rng.binomial(1, 1 / (1 + np.exp(-X[:, 0])))
    effect = 1.0 + X[:, 1]
    outcome = X[:, 0] + treated * effect + rng.normal(scale=0.5, size=800)
    return X, np.column_stack([outcome, treated]), effect


def test_fits_share_the_network_and_prepare_off_the_device(backend, data):
    X, y, _ = data
    prepared = backend.prepare_fit(X[:300], y[:300], "causal")
    assert prepared.estimator.icl_model is None  # no network until the device
    context = backend.fit_prepared(prepared)
    other = backend.fit(X, y, "causal")
    assert context.estimator.icl_model is other.estimator.icl_model
    assert "load_model" not in vars(context.estimator)  # picklable
    assert backend.context_bytes_hint(0, 0) > 800 * 5 * 4


def test_effects_track_the_truth(backend, data):
    X, y, effect = data
    context = backend.fit(X[:600], y[:600], "causal")
    out = backend.predict(context, X[600:])
    assert np.corrcoef(out.predictions, effect[600:])[0, 1] > 0.9


def test_quantiles_come_with_the_same_effect_and_repeat(backend, data):
    X, y, _ = data
    context = backend.fit(X[:600], y[:600], "causal")
    plain = backend.predict(context, X[600:650])
    a = backend.predict(context, X[600:650], quantiles=(0.05, 0.5, 0.95))
    b = backend.predict(context, X[600:650], quantiles=(0.05, 0.5, 0.95))
    np.testing.assert_allclose(a.predictions, plain.predictions, rtol=1e-5)
    assert a.quantiles == b.quantiles
    q = np.array(a.quantiles)
    assert (q[:, 0] <= q[:, 1]).all() and (q[:, 1] <= q[:, 2]).all()


def test_missing_values_are_filled_with_training_medians(backend, data):
    X, y, _ = data
    holes = X.copy()
    holes[::7, 2] = np.nan
    context = backend.fit(holes[:600], y[:600], "causal")
    assert not np.isnan(context.estimator.X_train).any()
    out = backend.predict(context, holes[600:620])
    assert np.isfinite(out.predictions).all()


def test_wide_tables_are_reduced(backend, data):
    _, y, _ = data
    wide = np.random.default_rng(1).normal(size=(400, 120)).astype(np.float32)
    context = backend.fit(wide, y[:400], "causal")
    assert len(backend.predict(context, wide[:10]).predictions) == 10


def test_only_causal(backend, data):
    X, y, _ = data
    with pytest.raises(InvalidInputError, match="fits causal, not survival"):
        backend.fit(X, y, "survival")


def test_a_spilled_context_comes_back_on_the_shared_network(backend, data):
    X, y, _ = data
    context = backend.fit(X[:300], y[:300], "causal")
    data_bytes = backend.dumps_payload(context)
    assert len(data_bytes) < 5_000_000  # the table and stratifier, not the network
    back = backend.loads_payload(data_bytes)
    assert back.estimator.icl_model is context.estimator.icl_model
    np.testing.assert_allclose(
        backend.predict(back, X[300:310]).predictions,
        backend.predict(context, X[300:310]).predictions,
        rtol=1e-5,
    )
