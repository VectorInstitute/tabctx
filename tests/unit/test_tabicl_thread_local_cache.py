"""The shared TabICL backbone keeps one active-context cache per thread
(backends/tabicl.py make_cache_thread_local)."""

import threading

import pytest

torch = pytest.importorskip("torch")

from tabctx.backends.tabicl import make_cache_thread_local  # noqa: E402


class Backbone(torch.nn.Module):
    """Mimics tabicl's TabICL: _cache is a plain instance attribute set by
    forward_with_cache and cleared by clear_cache."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self._cache = None

    def forward(self, x):
        return self.linear(x)

    def clear_cache(self) -> None:
        self._cache = None


def test_each_thread_sees_its_own_cache():
    m = Backbone()
    make_cache_thread_local(m)
    m._cache = "main"
    seen: dict = {}
    ready, go = threading.Event(), threading.Event()

    def other() -> None:
        assert m._cache is None  # never the main thread's
        m._cache = "other"
        ready.set()
        go.wait(5)
        seen["other"] = m._cache

    t = threading.Thread(target=other)
    t.start()
    ready.wait(5)
    m.clear_cache()  # the main thread's call ends: only its slot clears
    go.set()
    t.join(5)
    assert seen["other"] == "other"
    assert m._cache is None


def test_module_still_works_and_is_idempotent():
    m = Backbone()
    make_cache_thread_local(m)
    make_cache_thread_local(m)
    assert isinstance(m, Backbone)
    assert m(torch.zeros(1, 2)).shape == (1, 2)
    assert "linear" in dict(m.named_modules())
    assert len(list(m.parameters())) == 2


def test_cached_logn_matches_and_is_installed_once():
    ssmax = pytest.importorskip("tabicl._model.ssmax")
    from tabctx.backends.tabicl import install_cached_logn

    original = ssmax._logn
    try:
        assert install_cached_logn()
        patched = ssmax._logn
        assert install_cached_logn() and ssmax._logn is patched  # idempotent
        for n in (1, 7, 40_000):
            expected = original(n, torch.device("cpu"), torch.float16)
            assert torch.equal(patched(n, torch.device("cpu"), torch.float16), expected)
    finally:
        ssmax._logn = original


def test_logn_stays_symbolic_inside_compiled_code():
    """math.log on the row count specialized every compiled graph to one
    table size (each new table recompiled for ~11 s on an A100)."""
    ssmax = pytest.importorskip("tabicl._model.ssmax")
    from torch._dynamo.testing import CompileCounter

    from tabctx.backends.tabicl import install_cached_logn

    original = ssmax._logn
    try:
        install_cached_logn()
        counter = CompileCounter()

        @torch.compile(backend=counter, dynamic=True)
        def scale(q):
            return q * ssmax._logn(q.shape[0], q.device, q.dtype)

        for n in (3000, 5000, 7000, 12000):
            q = torch.ones(n, 4)
            expected = original(n, torch.device("cpu"), torch.float32)
            assert torch.allclose(scale(q)[0, 0], expected, rtol=1e-6)
        assert counter.frame_count == 1  # one graph for every row count
    finally:
        ssmax._logn = original
        torch._dynamo.reset()


def test_compiled_blocks_install_once():
    layers = pytest.importorskip("tabicl._model.layers")
    from tabctx.backends.tabicl_compile import install_compiled_blocks

    original = layers.MultiheadAttentionBlock.forward
    try:
        assert install_compiled_blocks()
        compiled = layers.MultiheadAttentionBlock.forward
        assert compiled is not original
        assert install_compiled_blocks()
        assert layers.MultiheadAttentionBlock.forward is compiled  # idempotent
    finally:
        layers.MultiheadAttentionBlock.forward = original
        torch._dynamo.reset()
