"""CausalPFNBackend against real CausalPFN (CPU): effects that match
CausalPFN's own estimator on a table that fits its context and track the
truth, a row's answer independent of the rest of its request (the server
merges and splits requests), repeatable quantiles from the same pass as
the effect, missing values, wide tables, accounting under concurrent
fits, and the spill tier.

Skipped without causalpfn or its checkpoint (from Hugging Face)."""

import numpy as np
import pytest

pytest.importorskip("causalpfn")
pytest.importorskip("torch")

from tabctx.backends.causalpfn import CausalPFNBackend  # noqa: E402
from tabctx.errors import InvalidInputError  # noqa: E402


@pytest.fixture(scope="module")
def backend():
    b = CausalPFNBackend(device="cpu", context_size=1024, batch_rows=64)
    try:
        b.warmup()
    except OSError as e:  # the checkpoint can't be downloaded (offline)
        pytest.skip(f"CausalPFN weights unavailable: {e}")
    return b


def make(n, seed=0, features=5):
    """Confounded treatment (more likely with feature 0, which also raises
    the outcome) whose effect is 1 + feature 1."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, features)).astype(np.float32)
    treated = rng.binomial(1, 1 / (1 + np.exp(-X[:, 0])))
    effect = 1.0 + X[:, 1]
    outcome = X[:, 0] + treated * effect + rng.normal(scale=0.5, size=n)
    return X, np.column_stack([outcome, treated]), effect


@pytest.fixture(scope="module")
def data():
    return make(800)


def test_matches_causalpfns_estimator_when_the_table_is_the_context(backend, data):
    from causalpfn import CATEEstimator

    X, y, _ = data
    ours = backend.predict(backend.fit(X[:600], y[:600], "causal"), X[600:650])
    theirs = CATEEstimator(device="cpu", num_neighbours=10**6, max_context_length=10**6)
    theirs.load_model = lambda: setattr(theirs, "icl_model", backend._shared_network())
    theirs.fit(X[:600], y[:600, 1].astype(np.float32), y[:600, 0].astype(np.float32))
    np.testing.assert_allclose(
        ours.predictions, theirs.estimate_cate(X[600:650]), rtol=1e-4, atol=1e-4
    )


def test_effects_track_the_truth(backend, data):
    X, y, effect = data
    out = backend.predict(backend.fit(X[:600], y[:600], "causal"), X[600:])
    assert np.corrcoef(out.predictions, effect[600:])[0, 1] > 0.9


def test_a_rows_answer_does_not_depend_on_its_request(backend):
    # Above the context size, so the table is subsampled (seeded).
    X, y, _ = make(3000, seed=1)
    context = backend.fit(X[:2800], y[:2800], "causal")
    levels = (0.05, 0.5, 0.95)
    together = backend.predict(context, X[2800:], quantiles=levels)  # 4 batches
    for i in (0, 77, 199):
        alone = backend.predict(context, X[2800 + i : 2801 + i], quantiles=levels)
        np.testing.assert_allclose(
            alone.predictions, together.predictions[i : i + 1], rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            alone.quantiles, together.quantiles[i : i + 1], rtol=1e-5, atol=1e-6
        )


def test_quantiles_come_with_the_same_effect_and_repeat(backend, data):
    X, y, _ = data
    context = backend.fit(X[:600], y[:600], "causal")
    plain = backend.predict(context, X[600:650])
    a = backend.predict(context, X[600:650], quantiles=(0.05, 0.5, 0.95))
    b = backend.predict(context, X[600:650], quantiles=(0.05, 0.5, 0.95))
    np.testing.assert_allclose(a.predictions, plain.predictions, rtol=1e-6)
    assert a.quantiles == b.quantiles
    q = np.array(a.quantiles)
    assert (q[:, 0] <= q[:, 1]).all() and (q[:, 1] <= q[:, 2]).all()


def test_concurrent_fits_are_each_charged_their_own_size(backend):
    big = backend.prepare_fit(*make(900)[:2], "causal")
    small = backend.prepare_fit(*make(40, seed=2)[:2], "causal")
    backend.fit_prepared(big)  # prepared before `small`, placed after
    assert backend.context_bytes_hint(0, 0) == big.nbytes > small.nbytes


def test_missing_values_are_filled_with_training_medians(backend, data):
    X, y, _ = data
    holes = X.copy()
    holes[::7, 2] = np.nan
    context = backend.fit(holes[:600], y[:600], "causal")
    assert not np.isnan(context.X).any()
    assert np.isfinite(backend.predict(context, holes[600:620]).predictions).all()


@pytest.mark.parametrize("rows", [400, 50])
def test_wide_tables_are_reduced(backend, rows):
    X, y, _ = make(rows, features=120)
    context = backend.fit(X, y, "causal")
    assert context.X.shape[1] == min(99, rows - 1)
    assert len(backend.predict(context, X[:10]).predictions) == 10


def test_only_causal(backend, data):
    X, y, _ = data
    with pytest.raises(InvalidInputError, match="fits causal, not survival"):
        backend.fit(X, y, "survival")


def test_a_spilled_context_comes_back(backend, data):
    X, y, _ = data
    context = backend.fit(X[:300], y[:300], "causal")
    data_bytes = backend.dumps_payload(context)
    assert len(data_bytes) < 1_000_000  # the table, never the network
    back = backend.loads_payload(data_bytes)
    np.testing.assert_allclose(
        backend.predict(back, X[300:310]).predictions,
        backend.predict(context, X[300:310]).predictions,
    )
