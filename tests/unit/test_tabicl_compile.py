"""Dynamo settings for tabicl's compiled blocks (backends/tabicl_compile.py)."""

import asyncio
import threading

import pytest

from tabctx.backends.tabicl_compile import with_compile_settings

config = pytest.importorskip("torch._dynamo").config


def _settings() -> tuple[int, bool]:
    return config.recompile_limit, config.suppress_errors


def test_settings_apply_on_any_thread_or_context():
    """Dynamo's config overrides are per thread (a ContextVar): set once at
    install, serving threads saw a recompile limit of 8 and compile errors
    that raised. The settings now travel with each call."""
    wrapped = with_compile_settings(_settings, devices=2)
    seen = {}
    thread = threading.Thread(target=lambda: seen.update(thread=wrapped()))
    thread.start()
    thread.join()
    seen["to_thread"] = asyncio.run(asyncio.to_thread(wrapped))
    assert seen == {"thread": (512, True), "to_thread": (512, True)}


def test_settings_stay_scoped_to_the_call():
    default = _settings()
    with_compile_settings(_settings, devices=1)()
    assert _settings() == default


def test_nested_and_concurrent_calls_restore_their_own_settings():
    """One wrapper serves every block call: nested calls and calls on other
    threads each restore what they found."""
    default = _settings()
    inner = with_compile_settings(_settings, devices=1)
    outer = with_compile_settings(lambda: (inner(), _settings()), devices=1)
    results: list = []
    threads = [
        threading.Thread(target=lambda: results.append(outer())) for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [((256, True), (256, True))] * 8
    assert outer() == ((256, True), (256, True))
    assert _settings() == default
