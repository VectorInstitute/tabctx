"""CausalPFNBackend against CausalPFN's own published benchmark on IHDP.

The CausalPFN authors' reference: Table 1 of their paper (arXiv:2506.07918v2,
"Mean PEHE ± Standard Error"), CausalPFN's out-of-sample CATE error on the
100 IHDP realizations (Hill 2011; Shalit et al. 2017's 672/75 train/test
files), computed as in their code (benchmarks/ihdp.py and
notebooks/causal_effect.ipynb at vdblm/CausalPFN@b7ed2f1): fit on each
realization's training rows, estimate the effect of its 75 test rows, and
score the root mean squared error against the true effect (PEHE).

IHDP's training table (672 rows) fits the model's context whole, so:
- tabctx's effects are CausalPFN's estimator's, row for row;
- their mean PEHE reproduces the published value;
- the API's average effect (the mean effect over a table's own rows, 747)
  is as close to the true average as CausalPFN's ATE estimator's, and
  reproduces the published ATE error.

The data (17 MB, from the authors' repository at that commit, checked by
sha256) is downloaded once into pytest's cache. Needs the causalpfn extra
and its weights; runs on CPU in about a minute.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("causalpfn")
torch = pytest.importorskip("torch")

from tabctx.backends.causalpfn import CausalPFNBackend  # noqa: E402

COMMIT = "b7ed2f1e1d58a83c608d08ced3dee955f0a7c221"
FILES = {
    "ihdp_npci_1-100.train.npz": "750697c71b4f8d7a3aafff771b56a4ac4cd83ec649bf69afb04f8a5aee41a240",
    "ihdp_npci_1-100.test.npz": "a70a8acbcc4e8deb677cc9bf9e9dabeb17caaa37cdbb1d7ba06be7ffb929c41c",
}
URL = f"https://raw.githubusercontent.com/vdblm/CausalPFN/{COMMIT}/benchmarks/IHDP/{{}}"
REALIZATIONS = 100
# Table 1, CausalPFN, IHDP: mean PEHE 0.58 ± 0.07 (standard error).
PUBLISHED_PEHE = 0.58
# Measured here: 0.579 (CPU), with every effect equal to CausalPFN's own.
PEHE_MARGIN = 0.01
# Table 1, CausalPFN, IHDP: mean ATE relative error 0.20 ± 0.04 (measured 0.203).
PUBLISHED_ATE_ERROR = 0.20
ATE_MARGIN = 0.02
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def ihdp(request) -> tuple[dict, dict]:
    cache = Path(request.config.cache.mkdir("causalpfn-ihdp"))
    for name, digest in FILES.items():
        path = cache / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            try:
                with urllib.request.urlopen(URL.format(name), timeout=60) as r:
                    body = r.read()
            except OSError as e:  # offline
                pytest.skip(f"IHDP data unavailable: {e}")
            assert hashlib.sha256(body).hexdigest() == digest, (
                f"{name} changed upstream"
            )
            path.write_bytes(body)
    train, test = (dict(np.load(cache / name)) for name in FILES)
    return train, test


@pytest.fixture(scope="module")
def backend():
    b = CausalPFNBackend(device=DEVICE)
    try:
        b.warmup()
    except OSError as e:  # the checkpoint can't be downloaded (offline)
        pytest.skip(f"CausalPFN weights unavailable: {e}")
    return b


def realization(train: dict, test: dict, i: int):
    """The authors' IHDPDataset[i]: training rows, test rows and their
    true effects, and every row (train then test) for the average effect."""
    f = lambda d, k: d[k][..., i].astype(np.float32)  # noqa: E731
    return (
        f(train, "x"),
        np.column_stack([f(train, "yf"), f(train, "t")]),
        f(test, "x"),
        f(test, "mu1") - f(test, "mu0"),
        np.concatenate([f(train, "x"), f(test, "x")]),
        np.column_stack(
            [
                np.concatenate([f(train, "yf"), f(test, "yf")]),
                np.concatenate([f(train, "t"), f(test, "t")]),
            ]
        ),
        float(train["ate"].item()),
    )


def test_matches_the_published_benchmark(backend, ihdp):
    from causalpfn import ATEEstimator, CATEEstimator

    train, test = ihdp
    pehe, ate_ours, ate_theirs = [], [], []
    for i in range(REALIZATIONS):
        X, y, X_test, effect, X_all, y_all, ate = realization(train, test, i)
        ours = backend.predict(backend.fit(X, y, "causal"), X_test).predictions

        theirs = CATEEstimator(device=DEVICE)
        theirs.load_model = lambda theirs=theirs: setattr(
            theirs, "icl_model", backend._shared_network()
        )
        theirs.fit(X, y[:, 1], y[:, 0])
        np.testing.assert_allclose(
            ours, theirs.estimate_cate(X_test), rtol=1e-4, atol=1e-4
        )
        pehe.append(np.sqrt(np.mean((ours - effect) ** 2)))

        whole = backend.fit(X_all, y_all, "causal")
        ate_ours.append(
            abs(np.mean(backend.predict(whole, X_all).predictions) - ate) / abs(ate)
        )
        est = ATEEstimator(device=DEVICE)
        est.load_model = lambda est=est: setattr(
            est, "icl_model", backend._shared_network()
        )
        est.fit(X_all, y_all[:, 1], y_all[:, 0])
        ate_theirs.append(abs(est.estimate_ate() - ate) / abs(ate))

    assert np.mean(pehe) == pytest.approx(PUBLISHED_PEHE, abs=PEHE_MARGIN)
    assert np.mean(ate_ours) == pytest.approx(np.mean(ate_theirs), abs=1e-3)
    assert np.mean(ate_ours) == pytest.approx(PUBLISHED_ATE_ERROR, abs=ATE_MARGIN)
