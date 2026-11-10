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


def test_nothing_is_estimated_past_the_follow_up(backend):
    """Every row censored by t=1 (true survival exp(-t/10)): survival and
    quantiles past the longest training time are None, not near-zero."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 3)).astype(np.float32)
    event = rng.exponential(10.0, 400)
    y = np.column_stack([np.minimum(event, 1.0), event <= 1.0])
    context = backend.fit(X, y, "survival")
    longest = float(y[:, 0].max())
    out = backend.predict(context, X[:5], quantiles=(0.05, 0.5), times=(0.5, 3.0))
    assert all(row[1] is None for row in out.survival)
    assert all(row[0] is not None and 0.0 < row[0] <= 1.0 for row in out.survival)
    medians = [m for m in out.predictions if m is not None]
    assert all(m <= longest for m in medians)
    assert all(row[1] is None or row[1] <= longest for row in out.quantiles)


def test_rmst_is_the_area_under_the_survival_curve(backend, data):
    X, y = data
    context = backend.fit(X[:500], y[:500], "survival")
    horizon = 2.0
    grid = tuple(np.linspace(0.0, horizon, 2001))
    curve = np.array(backend.predict(context, X[500:510], times=grid).survival)
    out = backend.predict(context, X[500:510], rmst_horizon=horizon)
    np.testing.assert_allclose(out.rmst, np.trapezoid(curve, grid, axis=1), rtol=1e-3)
    assert all(0 < r <= horizon for r in out.rmst)
    with pytest.raises(InvalidInputError, match="past the longest time"):
        backend.predict(context, X[500:510], rmst_horizon=1e6)


def test_quantiles_in_the_open_ended_last_bin_are_inf():
    import torch
    from survivalpfn.models.utils import HistogramDistribution

    from tabctx.backends.survivalpfn import event_time_quantiles

    edges = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    # 0.3 in each finite bin, 0.1 past the last edge.
    logits = torch.log(torch.tensor([[0.3, 0.3, 0.3, 0.1]]))
    dist = HistogramDistribution(edges, (edges[:, 1:] + edges[:, :-1]) / 2, logits)
    q = event_time_quantiles(dist, (0.15, 0.5, 0.95))
    np.testing.assert_allclose(q[0, :2], [0.5, 1 + 0.2 / 0.3], rtol=1e-5)
    assert np.isinf(q[0, 2])


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
