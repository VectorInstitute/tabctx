"""CausalPFNBackend against CausalPFN's own benchmarks: IHDP and Lalonde CPS.

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

Lalonde CPS (RealCause's effects, the authors' first 10 realizations) is
larger than the context and only 1% treated, so it checks how a context is
sampled: as accurate as CausalPFN's estimator (see the test's docstring).

The data (17 MB for IHDP, 13 MB for CPS, from the authors' repository at
that commit, checked by sha256) is downloaded once into pytest's cache.
Needs the causalpfn extra and its weights (`uv sync --extra causalpfn`;
CI's integration job doesn't install it, so there it skips); runs on CPU
in about five minutes.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("causalpfn")
torch = pytest.importorskip("torch")

from tabctx.backends.causalpfn import (  # noqa: E402
    DEFAULT_CONTEXT_SIZE,
    CausalPFNBackend,
)

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
# Lalonde CPS with RealCause's effects, the authors' first 10 realizations
# (benchmarks/realcause.py): 16,177 rows, 173 treated (1%), so a training
# table (90%) is larger than the 4,096-row context.
CPS_URL = (
    f"https://raw.githubusercontent.com/vdblm/CausalPFN/{COMMIT}/"
    "benchmarks/realcause_datasets/{}"
)
CPS_FILES = {
    "lalonde_cps_sample0.csv": "20ce33e76b4dd68c3a81cacbf802c768a45ce0133084785bd7d2f98d80c6f865",
    "lalonde_cps_sample1.csv": "84ecb89bbc1b158e9cd00e4d6806fb05f84db7c7bbd93ddc0c7ff49657c47305",
    "lalonde_cps_sample2.csv": "8946bf8c14885049e9bab3e9c95affa1e4cae9b36f6fbd7def414efeddfaf920",
    "lalonde_cps_sample3.csv": "45d2011ce890349f2288a973b41bf8c2f1bc8c85e54406be7173a251e16bcd8a",
    "lalonde_cps_sample4.csv": "d2b8edf7988d5f13b1e695212e427db4fe198eaed14f10bdc34e6263e821cc56",
    "lalonde_cps_sample5.csv": "936e9087a6afc97fe888007289babdef7519a44041bcc94aa53d9a1b1b0e372e",
    "lalonde_cps_sample6.csv": "ffc6aaa830819657a7c897e85e08459dbf62004654c1b29b8849a59b3ffe479b",
    "lalonde_cps_sample7.csv": "9db19839d8bb8dce5188d32cd0305afe0cb63abae343157d85ae2b27095b88bd",
    "lalonde_cps_sample8.csv": "614b5e2aabe60a8c372c97bbc1f3d64cfd6b1a9478af16bf4f04dda67b64029a",
    "lalonde_cps_sample9.csv": "bc4ad40ad305e12ab3c4ce2a5bb120331c747256b214dad3ac75c10fdc6e9b2f",
}
# CPS: tabctx's mean PEHE over these 10, at most this times CausalPFN's
# estimator's (measured 1.003; keeping each arm's share, 1.070), and the
# mean ATE error at most this (measured 0.128; keeping shares, 0.346).
CPS_PEHE_RATIO = 1.02
CPS_ATE_ERROR = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def fetch(request, folder: str, url: str, files: dict[str, str]) -> Path:
    """files (name: sha256) from url, kept in pytest's cache under folder."""
    cache = Path(request.config.cache.mkdir(folder))
    for name, digest in files.items():
        path = cache / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            try:
                with urllib.request.urlopen(url.format(name), timeout=60) as r:
                    body = r.read()
            except OSError as e:  # offline
                pytest.skip(f"benchmark data unavailable: {e}")
            assert hashlib.sha256(body).hexdigest() == digest, (
                f"{name} changed upstream"
            )
            path.write_bytes(body)
    return cache


@pytest.fixture(scope="module")
def ihdp(request) -> tuple[dict, dict]:
    cache = fetch(request, "causalpfn-ihdp", URL, FILES)
    train, test = (dict(np.load(cache / name)) for name in FILES)
    return train, test


def upstream(cls, backend):
    """CausalPFN's own estimator (CATEEstimator, ATEEstimator) on the
    backend's network, so both sides run the same weights."""
    est = cls(device=DEVICE)
    est.load_model = lambda: setattr(est, "icl_model", backend._shared_network())
    return est


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

        theirs = upstream(CATEEstimator, backend)
        theirs.fit(X, y[:, 1], y[:, 0])
        np.testing.assert_allclose(
            ours, theirs.estimate_cate(X_test), rtol=1e-4, atol=1e-4
        )
        pehe.append(np.sqrt(np.mean((ours - effect) ** 2)))

        whole = backend.fit(X_all, y_all, "causal")
        ate_ours.append(
            abs(np.mean(backend.predict(whole, X_all).predictions) - ate) / abs(ate)
        )
        est = upstream(ATEEstimator, backend)
        est.fit(X_all, y_all[:, 1], y_all[:, 0])
        ate_theirs.append(abs(est.estimate_ate() - ate) / abs(ate))

    assert np.mean(pehe) == pytest.approx(PUBLISHED_PEHE, abs=PEHE_MARGIN)
    assert np.mean(ate_ours) == pytest.approx(np.mean(ate_theirs), abs=1e-3)
    assert np.mean(ate_ours) == pytest.approx(PUBLISHED_ATE_ERROR, abs=ATE_MARGIN)


def lalonde_cps(path: Path, i: int):
    """The authors' RealCauseLalondeCPSDataset[i] (benchmarks/realcause.py at
    COMMIT): covariates, then t, y, y0, y1 and ite (the row's true effect);
    a 90/10 train/test split by default_rng(42 + i)."""
    with path.open() as fh:
        names = fh.readline().strip().split(",")
    rows = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    col = {name: rows[:, j] for j, name in enumerate(names)}
    X = rows[:, : len(names) - 5].astype(np.float32)
    t = (col["t"] != 0).astype(np.float32)
    y, ite = col["y"].astype(np.float32), col["ite"].astype(np.float32)
    order = np.random.default_rng(42 + i).permutation(len(X))
    cut = int(len(X) * 0.9)
    return X, t, y, ite, order[:cut], order[cut:]


def test_a_rare_treatment_in_a_table_larger_than_the_context(backend, request):
    """Lalonde CPS: the context keeps every treated row (context_rows), so
    the effects are as accurate as CausalPFN's estimator's. Measured over
    these 10: PEHE 8,851 against the estimator's 8,828, ATE error 0.128; a
    context keeping each arm's share (44 treated rows) scored 9,444 and
    0.346. The ATE is checked against a bound, not the ATE estimator."""
    from causalpfn import CATEEstimator

    cache = fetch(request, "causalpfn-lalonde-cps", CPS_URL, CPS_FILES)
    ours, theirs, ate_error = [], [], []
    for i, name in enumerate(CPS_FILES):
        X, t, y, ite, train, test = lalonde_cps(cache / name, i)
        assert len(train) > DEFAULT_CONTEXT_SIZE
        assert t[train].sum() < DEFAULT_CONTEXT_SIZE // 2

        fit = backend.fit(X[train], np.column_stack([y[train], t[train]]), "causal")
        assert len(fit.treated) == DEFAULT_CONTEXT_SIZE
        assert fit.treated.sum() == t[train].sum()  # every treated row
        ours.append(_pehe(backend.predict(fit, X[test]).predictions, ite[test]))
        est = upstream(CATEEstimator, backend)
        est.fit(X[train], t[train], y[train])
        theirs.append(_pehe(est.estimate_cate(X[test]), ite[test]))

        whole = backend.fit(X, np.column_stack([y, t]), "causal")
        ate = np.mean(backend.predict(whole, X).predictions)
        ate_error.append(abs(ate - ite.mean()) / abs(ite.mean()))

    assert np.mean(ours) <= CPS_PEHE_RATIO * np.mean(theirs)
    assert np.mean(ate_error) <= CPS_ATE_ERROR


def _pehe(estimate, effect) -> float:
    return float(np.sqrt(np.mean((np.asarray(estimate) - effect) ** 2)))
