"""The vectorized Yeo-Johnson PowerTransformer (backends/yeo_johnson.py)
against scikit-learn's, on CPU torch."""

import numpy as np
import pytest

pytest.importorskip("torch")
from sklearn.preprocessing import PowerTransformer  # noqa: E402

from tabctx.backends.yeo_johnson import (  # noqa: E402
    FastPowerTransformer,
    fit_lambdas,
    on_device,
)


@pytest.fixture(autouse=True)
def fast_path():
    """The fast fit only runs inside on_device (CPU torch here)."""
    with on_device(None):
        yield


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


def test_the_fitted_transformer_pickles_as_a_plain_power_transformer():
    """Spilled contexts must not depend on this module's class path."""
    import pickle

    X = columns()
    fast = FastPowerTransformer(method="yeo-johnson").fit(X)
    blob = pickle.dumps(fast)
    assert b"tabctx" not in blob
    again = pickle.loads(blob)
    assert type(again) is PowerTransformer
    np.testing.assert_array_equal(again.transform(X), fast.transform(X))


def test_outside_on_device_it_is_scikit_learns(fast_path):
    """A backend without fast_preprocessing shares the patched class."""
    import threading

    X = columns()
    ref = PowerTransformer(method="yeo-johnson").fit(X)
    box = {}
    t = threading.Thread(  # a thread outside any on_device block
        target=lambda: box.setdefault("l", FastPowerTransformer().fit(X).lambdas_)
    )
    t.start()
    t.join()
    np.testing.assert_array_equal(box["l"], ref.lambdas_)  # bit-identical: same code


def test_the_device_section_runs_under_the_guard():
    entered = []

    class Guard:
        def __enter__(self):
            entered.append(True)

        def __exit__(self, *exc):
            return False

    with on_device(None, Guard):
        FastPowerTransformer(method="yeo-johnson").fit(columns())
    assert entered == [True]


def test_device_out_of_memory_falls_back_to_scikit_learn(monkeypatch):
    import torch

    import tabctx.backends.yeo_johnson as yj

    def oom(*a, **k):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(yj, "fit_lambdas", oom)
    X = columns()
    np.testing.assert_array_equal(
        FastPowerTransformer(method="yeo-johnson").fit(X).lambdas_,
        PowerTransformer(method="yeo-johnson").fit(X).lambdas_,
    )


def test_columns_are_fitted_in_bounded_chunks(monkeypatch):
    import tabctx.backends.yeo_johnson as yj

    X = columns()
    whole = fit_lambdas(X[:, :-1])
    monkeypatch.setattr(yj, "_CHUNK_BYTES", yj._BYTES_PER_CELL * X.shape[0] * 2)
    calls = []
    real = yj._fit_lambdas
    monkeypatch.setattr(
        yj, "_fit_lambdas", lambda x, d: calls.append(x.shape[1]) or real(x, d)
    )
    np.testing.assert_allclose(fit_lambdas(X[:, :-1]), whole, atol=1e-12)
    assert calls == [2, 2, 2]  # 6 columns, 2 per chunk


def test_values_beyond_any_safe_lambda_are_left_untransformed():
    """Mixed-sign float32 values near 1e15: scipy raises; lambda 1 here."""
    X = np.array([[1e15, 0.5], [-1e15, -0.5], [3e14, 0.1]] * 10, dtype=np.float32)
    assert fit_lambdas(X)[0] == 1.0


def test_float64_input_matches_scikit_learn():
    X = columns().astype(np.float64)
    np.testing.assert_allclose(
        FastPowerTransformer(method="yeo-johnson").fit(X).lambdas_,
        PowerTransformer(method="yeo-johnson").fit(X).lambdas_,
        atol=1e-6,
    )


def test_install_replaces_tabicls_normalizer():
    preprocessing = pytest.importorskip("tabicl._sklearn.preprocessing")
    from tabctx.backends.yeo_johnson import install

    original = preprocessing.PowerTransformer
    try:
        assert install()
        pipeline = preprocessing.PreprocessingPipeline(normalization_method="power")
        pipeline.fit(columns()[:, :-1])
        assert isinstance(pipeline.normalizer_, FastPowerTransformer)
    finally:
        preprocessing.PowerTransformer = original
