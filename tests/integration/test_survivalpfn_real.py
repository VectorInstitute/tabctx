"""SurvivalPFNBackend against real SurvivalPFN (CPU): one network for every
fit, medians and quantiles from one distribution, test rows batched
without changing their answers, large tables subsampled, and a context
through the spill tier without its network.

Skipped without survivalpfn or its checkpoint (from Hugging Face)."""

import numpy as np
import pytest

pytest.importorskip("survivalpfn")
pytest.importorskip("torch")

from tabctx.backends.survivalpfn import SurvivalPFNBackend  # noqa: E402
from tabctx.errors import InvalidInputError  # noqa: E402


@pytest.fixture(scope="module")
def backend():
    b = SurvivalPFNBackend(device="cpu", context_size=600, batch_rows=64)
    try:
        b.warmup()
    except OSError as e:  # the checkpoint can't be downloaded (offline)
        pytest.skip(f"SurvivalPFN weights unavailable: {e}")
    return b


@pytest.fixture(scope="module")
def data():
    """Right-censored times whose event risk rises with feature 0."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(700, 5)).astype(np.float32)
    event = np.exp(1.0 - X[:, 0] + rng.normal(scale=0.35, size=700))
    censor = rng.exponential(3.0, 700) + 0.05
    y = np.column_stack([np.minimum(event, censor), event <= censor])
    return X, y


def test_fits_share_the_network(backend, data):
    X, y = data
    a, b = backend.fit(X[:200], y[:200], "survival"), backend.fit(X, y, "survival")
    assert a.estimator.icl_model is b.estimator.icl_model
    assert len(b.estimator.X_train) == 600  # subsampled to context_size
    assert backend.context_bytes_hint(0, 0) == 600 * 5 * 4 + 600 * 4 * 2


def test_predictions_track_risk_and_agree_with_their_quantiles(backend, data):
    X, y = data
    context = backend.fit(X[:500], y[:500], "survival")
    out = backend.predict(context, X[500:], quantiles=(0.25, 0.5, 0.75), times=(1, 3))
    medians = np.array(out.predictions, dtype=float)
    q = np.array(out.quantiles, dtype=float)
    np.testing.assert_allclose(medians, q[:, 1], rtol=1e-5)
    assert (np.diff(q, axis=1) >= 0).all()
    survival = np.array(out.survival)
    assert ((survival >= 0) & (survival <= 1)).all()
    assert (survival[:, 1] <= survival[:, 0]).all()
    # Higher risk (feature 0), earlier events.
    assert np.corrcoef(X[500:, 0], medians)[0, 1] < -0.5


def test_batching_leaves_answers_unchanged(backend, data):
    X, y = data
    context = backend.fit(X[:300], y[:300], "survival")
    whole = backend.predict(context, X[300:400]).predictions  # two batches of 64
    alone = [backend.predict(context, X[i : i + 1]).predictions[0] for i in (300, 399)]
    np.testing.assert_allclose([whole[0], whole[-1]], alone, rtol=1e-5)


def test_wide_tables_are_reduced(backend, data):
    X, y = data
    wide = np.random.default_rng(1).normal(size=(300, 120)).astype(np.float32)
    context = backend.fit(wide, y[:300], "survival")
    assert len(backend.predict(context, wide[:10]).predictions) == 10


def test_only_survival(backend, data):
    X, y = data
    with pytest.raises(InvalidInputError, match="fits survival, not regression"):
        backend.fit(X, y[:, 0], "regression")


def test_a_spilled_context_comes_back_on_the_shared_network(backend, data):
    X, y = data
    context = backend.fit(X[:300], y[:300], "survival")
    data_bytes = backend.dumps_payload(context)
    assert context.estimator.icl_model is not None  # restored after pickling
    assert len(data_bytes) < 1_000_000  # the table, not the 75 MB network
    back = backend.loads_payload(data_bytes)
    assert back.estimator.icl_model is context.estimator.icl_model
    np.testing.assert_allclose(
        backend.predict(back, X[300:310]).predictions,
        backend.predict(context, X[300:310]).predictions,
    )
    with pytest.raises(ValueError, match="model version"):
        backend.loads_payload(b"X1:other:" + data_bytes)
