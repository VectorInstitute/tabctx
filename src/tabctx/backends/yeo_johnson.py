"""A vectorized Yeo-Johnson PowerTransformer for TabICL's preprocessing (#31).

TabICL normalizes every ensemble member's features with scikit-learn's
PowerTransformer(method="yeo-johnson"), which fits one lambda per column
with scipy's bounded Brent search -- a Python loop over columns, dozens of
passes over each. On an A100 host that was 80%+ of a fit's CPU half
(40,000 x 100: 3.2 of 3.9 s; 20,000 x 200: 4.0 of 4.6 s), and with the
GPU half compiled (tabicl_compile.py) the CPU half became most of an
isolated fit's latency.

``FastPowerTransformer`` fits all columns at once with torch, on the
backend's GPU when one is set (``on_device``): the same objective
(scipy's yeojohnson_llf), the same per-column bounds (scipy's
yeojohnson_normmax, which keep the transform from overflowing the data's
dtype), and the same tolerance -- a coarse grid over the bounds, then
golden-section refinement of the best bracket. The transform and the
standardization are scikit-learn's (StandardScaler), so the fitted object
is an ordinary PowerTransformer to everything downstream (transform,
pickling for spill).

It is not bit-identical: lambdas agree with scipy's to ~1e-6 (they are
stored in the data's dtype, float32, anyway), and a column whose
likelihood has two local optima may get the better one where Brent stops
at the first. Anything outside the fast path -- Box-Cox, non-finite
values, NaNs -- runs scikit-learn's implementation unchanged.
"""

from __future__ import annotations

import contextlib
import math
import threading
from collections.abc import Iterator
from typing import Any

import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler

# scipy's optimize.brent / fminbound tolerance (yeojohnson_normmax).
_XTOL = 1.48e-08
_GRID_POINTS = 64
_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0

_local = threading.local()


@contextlib.contextmanager
def on_device(device: str | None) -> Iterator[None]:
    """Fit lambdas on `device` (e.g. "cuda:1") within this block, on this
    thread; CPU otherwise."""
    previous = getattr(_local, "device", None)
    _local.device = device
    try:
        yield
    finally:
        _local.device = previous


def _bounds(x_abs_max: np.ndarray, any_neg: np.ndarray, all_neg: np.ndarray, dtype):
    """scipy's yeojohnson_normmax bounds, per column: lambdas for which
    values up to 20x the largest observed transform without over- or
    underflowing half of `dtype`'s exponent range."""
    finfo = np.finfo(dtype)
    log1p_max_x = np.log1p(20 * x_abs_max)
    log_eps = np.log(finfo.eps)
    log_tiny = (np.log(finfo.tiny) - log_eps) / 2
    log_max = (np.log(finfo.max) + log_eps) / 2
    with np.errstate(divide="ignore"):
        lb = log_tiny / log1p_max_x
        ub = log_max / log1p_max_x
    neg_lb, neg_ub = 2 - ub, 2 - lb
    lb_out = np.where(all_neg, neg_lb, np.where(any_neg, np.maximum(neg_lb, lb), lb))
    ub_out = np.where(all_neg, neg_ub, np.where(any_neg, np.minimum(neg_ub, ub), ub))
    return lb_out, ub_out


def _transform(x: Any, lam: Any) -> Any:
    """Yeo-Johnson of each column of x by its lambda (torch, broadcast)."""
    import torch

    pos = x >= 0
    lp = torch.log1p(x.abs())
    eps = torch.finfo(x.dtype).eps
    near0 = lam.abs() < eps
    near2 = (lam - 2).abs() < eps
    safe = torch.where(near0, torch.ones_like(lam), lam)
    safe2 = torch.where(near2, torch.ones_like(lam), 2 - lam)
    pos_out = torch.where(near0, lp, torch.expm1(lam * lp) / safe)
    neg_out = torch.where(near2, -lp, -torch.expm1((2 - lam) * lp) / safe2)
    return torch.where(pos, pos_out, neg_out)


def _neg_llf(x: Any, lam: Any, sum_signed_log: Any) -> Any:
    """scipy's -yeojohnson_llf for every column at once (inf variance and
    zero variance count as rejected, as scipy's _neg_llf does)."""
    import torch

    n = x.shape[0]
    var = _transform(x, lam).var(dim=0, unbiased=False)
    llf = -n / 2 * torch.log(var) + (lam - 1) * sum_signed_log
    return torch.where(torch.isfinite(llf), -llf, torch.full_like(llf, math.inf))


def fit_lambdas(X: np.ndarray, device: str | None = None) -> np.ndarray:
    """The maximum-likelihood Yeo-Johnson lambda of each column of finite X."""
    import torch

    x = torch.as_tensor(np.asarray(X, dtype=np.float64), device=device or "cpu")
    x_abs_max = x.abs().max(dim=0).values.cpu().numpy()
    any_neg = (x < 0).any(dim=0).cpu().numpy()
    all_neg = (x < 0).all(dim=0).cpu().numpy()
    lb_np, ub_np = _bounds(x_abs_max, any_neg, all_neg, X.dtype)
    lb = torch.as_tensor(lb_np, device=x.device, dtype=torch.float64)
    ub = torch.as_tensor(ub_np, device=x.device, dtype=torch.float64)
    sum_signed_log = (torch.sign(x) * torch.log1p(x.abs())).sum(dim=0)

    # Coarse grid over each column's bounds: the best point and its
    # neighbours bracket the optimum.
    steps = torch.linspace(0, 1, _GRID_POINTS, device=x.device, dtype=torch.float64)
    width = ub - lb
    values = torch.stack(
        [_neg_llf(x, lb + t * width, sum_signed_log) for t in steps]
    )  # [grid, columns]
    best = values.argmin(dim=0)
    step = width / (_GRID_POINTS - 1)
    a = torch.maximum(lb, lb + (best - 1).clamp(min=0) * step)
    b = torch.minimum(ub, lb + (best + 1).clamp(max=_GRID_POINTS - 1) * step)

    # Golden-section search within each bracket, to scipy's tolerance.
    c = b - _GOLDEN * (b - a)
    d = a + _GOLDEN * (b - a)
    fc = _neg_llf(x, c, sum_signed_log)
    fd = _neg_llf(x, d, sum_signed_log)
    iterations = math.ceil(
        math.log(_XTOL / max(float((b - a).max()), _XTOL)) / math.log(_GOLDEN)
    )
    for _ in range(max(iterations, 0)):
        left = fc < fd  # the minimum is in [a, d]
        b = torch.where(left, d, b)
        a = torch.where(left, a, c)
        d_new = torch.where(left, c, a + _GOLDEN * (b - a))
        c_new = torch.where(left, b - _GOLDEN * (b - a), d)
        f_new = _neg_llf(x, torch.where(left, c_new, d_new), sum_signed_log)
        fc, fd = torch.where(left, f_new, fd), torch.where(left, fc, f_new)
        c, d = c_new, d_new
    return ((a + b) / 2).cpu().numpy()


class FastPowerTransformer(PowerTransformer):
    """PowerTransformer whose Yeo-Johnson fit is vectorized (module
    docstring); everything else is scikit-learn's."""

    def _fit(self, X, y=None, force_transform=False):
        if self.method != "yeo-johnson":
            return super()._fit(X, y, force_transform)
        X = self._check_input(X, in_fit=True, check_positive=True)
        if not np.all(np.isfinite(X)):  # NaNs: scikit-learn drops them per column
            return super()._fit(X, y, force_transform)
        if not self.copy and not force_transform:
            X = X.copy()
        n_samples = X.shape[0]
        mean = np.mean(X, axis=0, dtype=np.float64)
        var = np.var(X, axis=0, dtype=np.float64)
        constant = _constant_features(var, mean, n_samples) | np.all(X == 0, axis=0)
        lambdas = np.ones(X.shape[1], dtype=X.dtype)
        varying = np.flatnonzero(~constant)
        if varying.size:
            device = getattr(_local, "device", None)
            lambdas[varying] = fit_lambdas(X[:, varying], device)
        self.lambdas_ = lambdas
        if self.standardize or force_transform:
            import torch

            device = getattr(_local, "device", None) or "cpu"
            for start in range(0, varying.size, 256):  # bound device memory
                cols = varying[start : start + 256]
                x = torch.as_tensor(X[:, cols].astype(np.float64), device=device)
                lam = torch.as_tensor(lambdas[cols].astype(np.float64), device=device)
                X[:, cols] = _transform(x, lam).cpu().numpy().astype(X.dtype)
        if self.standardize:
            self._scaler = StandardScaler(copy=False).set_output(transform="default")
            if force_transform:
                X = self._scaler.fit_transform(X)
            else:
                self._scaler.fit(X)
        return X


def _constant_features(var: np.ndarray, mean: np.ndarray, n_samples: int) -> np.ndarray:
    """scikit-learn's test for a constant column (variance within rounding
    of zero), which PowerTransformer leaves untransformed (lambda 1)."""
    eps = np.finfo(np.float64).eps
    upper_bound = n_samples * eps * var + (n_samples * mean * eps) ** 2
    return var <= upper_bound


def install() -> bool:
    """Use FastPowerTransformer in tabicl's preprocessing. Idempotent;
    False if tabicl's preprocessing no longer imports PowerTransformer."""
    try:
        from tabicl._sklearn import preprocessing
    except ImportError:
        return False
    if getattr(preprocessing, "PowerTransformer", None) is None:
        return False
    preprocessing.PowerTransformer = FastPowerTransformer
    return True
