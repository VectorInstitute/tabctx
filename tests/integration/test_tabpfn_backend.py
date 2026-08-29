"""TabPFN backend integration tests (real model, CPU).

Skipped automatically when tabpfn isn't installed (it is not part of any
CI extra -- these run locally/on GPU rigs). First run downloads the
TabPFN checkpoint from Hugging Face.

What this proves: the engine/cache/estimator stack works unchanged with
a second real backend (the backend-agnosticism claim in
backends/base.py), and TabPFN's kv-cache-equivalent fit mode
("fit_with_cache") produces the same predictions as its cheaper modes.
"""

import numpy as np
import pytest

pytest.importorskip("tabpfn")

from sklearn.datasets import make_classification  # noqa: E402

from tabctx.backends.tabpfn import TabPFNBackend  # noqa: E402
from tabctx.errors import InvalidInputError  # noqa: E402
from tabctx.serve.factory import build_engine  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def tabpfn_weights_available():
    """TabPFN v8+ gates its checkpoint behind PriorLabs' own license
    portal: register at https://ux.priorlabs.ai, accept the license on
    the Licenses tab, and export TABPFN_TOKEN=<API key from /account>.
    Skip (with instructions) rather than fail when the machine isn't
    authorized -- accepting a model license is a human decision, not
    something a test run should attempt."""
    try:
        tiny_X = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.2, 0.8]]
        TabPFNBackend(device="cpu").fit(tiny_X, ["a", "b", "a", "b"], "classification")
    except Exception as e:  # noqa: BLE001 -- any auth/download gate
        if "license" in str(e).lower() or "token" in str(e).lower():
            pytest.skip(
                "TabPFN checkpoint not accessible: register/log in at "
                "https://ux.priorlabs.ai, accept the license, and export "
                f"TABPFN_TOKEN=<your API key>. ({type(e).__name__})"
            )
        raise


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(
        n_samples=80, n_features=5, n_informative=3, random_state=0
    )
    return (
        X[:60].tolist(),
        [str(v) for v in y[:60]],
        X[60:].tolist(),
    )


def test_engine_stack_with_tabpfn(data, monkeypatch):
    monkeypatch.setenv("TABCTX_BACKEND", "tabpfn")
    train_X, train_y, test_X = data
    built = build_engine()
    assert built.backend.name == "tabpfn"

    dataset_id = built.engine.fit(train_X, train_y, task="classification")
    outcome = built.engine.predict(dataset_id, test_X, return_proba=True)
    assert len(outcome.predictions) == len(test_X)
    assert outcome.classes == ["0", "1"]
    assert all(abs(sum(row) - 1.0) < 1e-5 for row in outcome.probabilities)
    # Cached context reused across calls -- the tabctx core promise.
    outcome2 = built.engine.predict(dataset_id, test_X[:3])
    assert len(outcome2.predictions) == 3


def test_cache_modes_agree(data):
    """fit_with_cache (tabctx "kv") must predict the same labels as the
    cheaper fit_preprocessors mode -- the same invariant we verified for
    TabICL's kv cache before defaulting it on."""
    train_X, train_y, test_X = data
    fast = TabPFNBackend(device="cpu", cache_mode="kv")
    cheap = TabPFNBackend(device="cpu", cache_mode="repr")
    p_fast = fast.predict(fast.fit(train_X, train_y, "classification"), test_X, True)
    p_cheap = cheap.predict(cheap.fit(train_X, train_y, "classification"), test_X, True)
    assert p_fast.predictions == p_cheap.predictions
    assert np.abs(
        np.array(p_fast.probabilities) - np.array(p_cheap.probabilities)
    ).max() < 1e-3


def test_regression(data):
    train_X, _, test_X = data
    train_y = [float(sum(row)) for row in train_X]
    backend = TabPFNBackend(device="cpu")
    payload = backend.fit(train_X, train_y, "regression")
    outcome = backend.predict(payload, test_X)
    assert len(outcome.predictions) == len(test_X)
    assert all(isinstance(v, float) for v in outcome.predictions)


def test_pretraining_limits_surface_as_invalid_input():
    """TabPFN rejects tables beyond its pretraining limits; a caller's
    oversized table must 422, not 500 (multi-tenant requirement)."""
    rng = np.random.default_rng(0)
    # 600 features exceeds the v2 checkpoint's 500-feature limit.
    X = rng.normal(size=(50, 600)).tolist()
    y = ["a", "b"] * 25
    backend = TabPFNBackend(device="cpu")
    with pytest.raises(InvalidInputError):
        backend.fit(X, y, "classification")
