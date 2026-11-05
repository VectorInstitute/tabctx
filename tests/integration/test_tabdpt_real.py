"""TabDPTBackend against real TabDPT (CPU): fits share one network per
task, labels round-trip, quantiles come from one distribution, and a
context survives the spill tier without carrying the network.

Skipped without tabdpt or its checkpoint (downloaded from Hugging Face)."""

import numpy as np
import pytest

pytest.importorskip("tabdpt")
pytest.importorskip("torch")

from tabctx.backends.tabdpt import TabDPTBackend  # noqa: E402


@pytest.fixture(scope="module")
def backend():
    b = TabDPTBackend(device="cpu")
    try:
        b.warmup()
    except OSError as e:  # the checkpoint can't be downloaded (offline)
        pytest.skip(f"TabDPT weights unavailable: {e}")
    return b


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5))
    return X, X[:, 0] + 0.1 * rng.normal(size=300)


def test_fits_share_the_network(backend, data):
    X, y = data
    a = backend.fit(X, y, "regression")
    b = backend.fit(X[:100], y[:100], "regression")
    c = backend.fit(X[:100], (y[:100] > 0).astype(int), "classification")
    # One network for every fit and both tasks.
    assert a.estimator.model is b.estimator.model is c.estimator.model
    assert a.estimator is not b.estimator
    # The context's size is its table, not the network.
    assert 0 < backend.context_bytes_hint(300, 5) < 10_000_000


def test_labels_come_back_as_given(backend, data):
    X, _ = data
    labels = np.where(X[:, 0] > 0, "yes", "no")
    context = backend.fit(X[:250], labels[:250], "classification")
    out = backend.predict(context, X[250:], return_proba=True)
    assert set(out.predictions) <= {"yes", "no"} and out.classes == ["no", "yes"]
    assert np.mean(np.array(out.predictions) == labels[250:]) > 0.8
    np.testing.assert_allclose(np.sum(out.probabilities, axis=1), 1.0, atol=1e-4)


def test_quantiles_follow_the_levels(backend, data):
    X, y = data
    context = backend.fit(X[:250], y[:250], "regression")
    out = backend.predict(context, X[250:], quantiles=(0.95, 0.05, 0.5))
    q = np.asarray(out.quantiles)
    assert q.shape == (50, 3)
    assert (q[:, 1] <= q[:, 2]).all() and (q[:, 2] <= q[:, 0]).all()
    covered = ((q[:, 1] <= y[250:]) & (y[250:] <= q[:, 0])).mean()
    assert covered > 0.7


def test_spill_round_trip_without_the_network(backend, data):
    X, y = data
    context = backend.fit(X[:250], y[:250], "regression")
    blob = backend.dumps_payload(context)
    assert len(blob) < 1_000_000  # the table, not the weights
    restored = backend.loads_payload(blob)
    assert restored.estimator.model is context.estimator.model
    np.testing.assert_allclose(
        backend.predict(restored, X[250:]).predictions,
        backend.predict(context, X[250:]).predictions,
        rtol=1e-5,
    )


def test_the_same_request_gives_the_same_answer(backend, data):
    X, y = data
    context = backend.fit(X[:250], y[:250], "regression")
    first = backend.predict(context, X[250:]).predictions
    assert backend.predict(context, X[250:]).predictions == first
