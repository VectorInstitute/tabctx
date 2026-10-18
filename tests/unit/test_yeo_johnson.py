"""The vectorized Yeo-Johnson PowerTransformer (backends/yeo_johnson.py)
against scikit-learn's, on CPU torch."""

import numpy as np
import pytest

pytest.importorskip("torch")
from sklearn.preprocessing import PowerTransformer  # noqa: E402

from tabctx.backends.yeo_johnson import FastPowerTransformer, fit_lambdas  # noqa: E402


def columns(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    cols = [
        rng.normal(size=n),
        rng.exponential(size=n),
        -3 * rng.exponential(size=n),  # all negative
        rng.lognormal(sigma=1.5, size=n),
        rng.standard_t(3, size=n),
        rng.uniform(-1, 1, n) ** 3,
        np.zeros(n),  # constant: lambda 1, untouched
    ]
    X = np.stack(cols, 1)
    X[:, :-1] = (X[:, :-1] - X[:, :-1].mean(0)) / X[:, :-1].std(0)
    return X.astype(np.float32)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_scikit_learn(seed):
    X = columns(seed=seed)
    ref = PowerTransformer(method="yeo-johnson").fit(X)
    fast = FastPowerTransformer(method="yeo-johnson").fit(X)
    assert fast.lambdas_.dtype == ref.lambdas_.dtype
    np.testing.assert_allclose(fast.lambdas_, ref.lambdas_, atol=1e-5)
    np.testing.assert_allclose(fast.transform(X), ref.transform(X), atol=1e-4)
    assert fast.lambdas_[-1] == 1.0


def test_fit_transform_matches_too():
    X = columns()
    ref = PowerTransformer(method="yeo-johnson").fit_transform(X.copy())
    fast = FastPowerTransformer(method="yeo-johnson").fit_transform(X.copy())
    np.testing.assert_allclose(fast, ref, atol=1e-4)


def test_nans_and_box_cox_use_scikit_learn():
    X = columns()
    X[3, 1] = np.nan
    ref = PowerTransformer(method="yeo-johnson").fit(X)
    fast = FastPowerTransformer(method="yeo-johnson").fit(X)
    np.testing.assert_array_equal(fast.lambdas_, ref.lambdas_)  # same code path
    positive = np.abs(columns()[:, :3]) + 0.1
    np.testing.assert_array_equal(
        FastPowerTransformer(method="box-cox").fit(positive).lambdas_,
        PowerTransformer(method="box-cox").fit(positive).lambdas_,
    )


def test_lambdas_stay_within_scipys_overflow_bounds():
    rng = np.random.default_rng(3)
    X = np.stack([rng.lognormal(sigma=4, size=2000), rng.normal(size=2000)], 1)
    lam = fit_lambdas(X.astype(np.float32))
    out = FastPowerTransformer(method="yeo-johnson").fit_transform(X.astype(np.float32))
    assert np.all(np.isfinite(lam)) and np.all(np.isfinite(out))


def test_the_fitted_transformer_pickles_as_a_power_transformer():
    import pickle

    fast = FastPowerTransformer(method="yeo-johnson").fit(columns())
    again = pickle.loads(pickle.dumps(fast))
    assert isinstance(again, PowerTransformer)
    np.testing.assert_array_equal(again.lambdas_, fast.lambdas_)
