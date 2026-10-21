"""Dynamo settings for tabicl's compiled blocks (backends/tabicl_compile.py)."""

import asyncio
import contextvars
import threading

import pytest

from tabctx.backends.tabicl_compile import compile_settings, with_compile_settings

torch = pytest.importorskip("torch")
dynamo = pytest.importorskip("torch._dynamo")
config = dynamo.config


def _settings() -> tuple[int, int, bool]:
    return (
        config.recompile_limit,
        config.accumulated_recompile_limit,
        config.suppress_errors,
    )


def _fresh(fn):
    """Run fn in an empty context, as a new thread would: the module-level
    settings with_compile_settings applies for its caller don't leak into
    what each test checks."""
    return contextvars.Context().run(fn)


def test_settings_raise_both_limits_for_every_device():
    settings = _fresh(lambda: compile_settings(devices=2))
    assert settings == {
        "recompile_limit": 512,
        "accumulated_recompile_limit": 512,
        "suppress_errors": True,
    }


def test_settings_apply_on_any_thread_or_context():
    """Dynamo's config overrides are per thread (a ContextVar): set once at
    install, serving threads saw a recompile limit of 8 and compile errors
    that raised. The settings now travel with each call."""
    wrapped = _fresh(lambda: with_compile_settings(_settings, devices=2))
    seen = {}
    thread = threading.Thread(target=lambda: seen.update(thread=wrapped()))
    thread.start()
    thread.join()
    seen["to_thread"] = asyncio.run(asyncio.to_thread(wrapped))
    assert seen == {"thread": (512, 512, True), "to_thread": (512, 512, True)}


def test_settings_are_restored_after_a_call_even_when_it_raises():
    def default_then_call(wrapped) -> tuple:
        before = _settings()
        with pytest.raises(RuntimeError):
            wrapped()
        return before, _settings()

    def boom():
        raise RuntimeError

    def run():
        wrapped = with_compile_settings(boom, devices=1)
        return _settings(), contextvars.Context().run(default_then_call, wrapped)

    installer, (before, after) = _fresh(run)
    assert installer == (256, 256, True)  # set for the installing thread
    assert before == after  # another context: untouched by the failed call


def test_nested_and_concurrent_calls_restore_their_own_settings():
    """One wrapper serves every block call: nested calls and calls on other
    threads each see the settings and leave their context as they found it."""
    inner = _fresh(lambda: with_compile_settings(_settings, devices=1))
    outer = _fresh(
        lambda: with_compile_settings(lambda: (inner(), _settings()), devices=1)
    )
    results: list = []

    def call() -> None:
        before = _settings()
        results.append((outer(), _settings() == before))

    threads = [threading.Thread(target=call) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [(((256, 256, True), (256, 256, True)), True)] * 8


def test_new_graphs_compile_on_a_serving_thread_past_dynamo_default_limit():
    """The bug end to end, on CPU: more graphs than dynamo's default limit
    of 8 compiled on one thread, then a new shape on another thread. It
    compiled nothing there before (limit 8, hit for the life of the
    process); with the settings carried by the call it compiles."""
    from torch._dynamo.utils import counters

    def f(x):
        return x * 2 + 1

    dynamo.reset()
    compiled = _fresh(
        lambda: with_compile_settings(
            torch.compile(f, backend="eager", dynamic=False), devices=1
        )
    )
    for n in range(1, 11):  # 10 graphs: past the default limit of 8
        compiled(torch.ones(n))
    graphs = counters["stats"]["unique_graphs"]

    thread = threading.Thread(target=lambda: compiled(torch.ones(50)))
    thread.start()
    thread.join()
    assert counters["stats"]["unique_graphs"] == graphs + 1
    dynamo.reset()


def test_after_warmup_uncovered_shapes_run_eagerly_instead_of_compiling():
    from torch._dynamo import eval_frame
    from torch._dynamo.utils import counters

    from tabctx.backends.tabicl_compile import stop_compiling_new_graphs

    @torch.compile(backend="eager", dynamic=False)
    def f(x):
        return x * 2 + 1

    f(torch.ones(3))  # one graph, as warmup would leave
    graphs = counters["stats"]["unique_graphs"]
    before = eval_frame._stance.stance
    try:
        assert stop_compiling_new_graphs() is True
        assert torch.equal(f(torch.ones(3)), torch.full((3,), 3.0))  # cached graph
        assert torch.equal(f(torch.ones(5)), torch.full((5,), 3.0))  # new shape: eager
        results = []
        thread = threading.Thread(target=lambda: results.append(f(torch.ones(7))))
        thread.start()
        thread.join()
        assert torch.equal(results[0], torch.full((7,), 3.0))  # another thread too
        assert counters["stats"]["unique_graphs"] == graphs  # nothing compiled
    finally:
        torch.compiler.set_stance(before)
