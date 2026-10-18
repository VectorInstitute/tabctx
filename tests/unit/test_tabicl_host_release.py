"""TabICLBackend releases offload staging memory -- also when a fit fails
(the review's finding: a failed large fit kept its pinned buffers while its
host slot went to the next fit)."""

import pytest

torch = pytest.importorskip("torch")

from tabctx.backends.tabicl import TabICLBackend  # noqa: E402


class _Pool:
    def __init__(self):
        self.cleared = 0

    def clear(self):
        self.cleared += 1


class _Stage:
    def __init__(self):
        self.inference_mgr = type("Mgr", (), {"_buffer_pool": _Pool()})()


class _Module:
    def __init__(self):
        self.stages = [_Stage(), _Stage()]

    def modules(self):
        return iter([self, *self.stages])

    def clear_cache(self):
        pass


class _Estimator:
    """tabicl's two halves: fit() preprocesses (kv_cache off, as
    prepare_fit runs it), _build_kv_cache() is the device work."""

    def __init__(self, fail: bool):
        self.model_ = _Module()
        self.fail = fail
        self.kv_cache = "kv"

    def fit(self, X, y):
        assert self.kv_cache is False  # prepare_fit disables the cache

    def _build_kv_cache(self):
        if self.fail:
            raise RuntimeError("Failed to execute even with minimum batch size 1")


def _backend(estimator):
    b = object.__new__(TabICLBackend)  # no tabicl needed
    b._device = "cpu"
    b._kv_precision = "native"
    b._shared_backbones = {}
    b._make_estimator = lambda task: estimator
    b._stash_backbone = lambda task, model: None
    return b


def pools(est):
    return [s.inference_mgr._buffer_pool.cleared for s in est.model_.stages]


def test_pools_are_cleared_after_a_fit():
    est = _Estimator(fail=False)
    _backend(est).fit([[1.0]], [0], "classification")
    assert pools(est) == [1, 1]


def test_pools_are_cleared_even_when_the_fit_fails():
    est = _Estimator(fail=True)
    with pytest.raises(RuntimeError):
        _backend(est).fit([[1.0]], [0], "classification")
    assert pools(est) == [1, 1]
