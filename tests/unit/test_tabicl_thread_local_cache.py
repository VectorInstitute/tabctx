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
