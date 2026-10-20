"""CompressedKVEntry against real tabicl structures (CPU): fit, compress,
predict, pickle round trip -- the paths a hand-written stand-in can't
prove (KVCache batching, TabICLCache.to, __getstate__/__setstate__).

Skipped without tabicl or its checkpoint in the local HuggingFace cache."""

import pickle

import numpy as np
import pytest

pytest.importorskip("tabicl")
torch = pytest.importorskip("torch")

from tabctx.backends.tabicl import TabICLBackend  # noqa: E402
from tabctx.backends.tabicl_kv import CompressedKVEntry  # noqa: E402


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1200, 6)).astype(np.float32)  # >1024 rows: AMP path
    return X, (X[:, 0] + X[:, 1] > 0).astype(int)


def _backend(precision):
    try:
        b = TabICLBackend(device="cpu", kv_precision=precision)
        b.warmup()
    except Exception as e:  # noqa: BLE001 -- no checkpoint offline
        pytest.skip(f"TabICL checkpoint unavailable: {e}")
    return b


@pytest.mark.parametrize("precision", ["int8", "fp16"])
def test_compressed_context_predicts_like_native(data, precision):
    X, y = data
    native, comp = _backend("native"), _backend(precision)
    want = native.predict(
        native.fit(X[:1000], y[:1000], "classification"), X[1000:], True
    )
    payload = comp.fit(X[:1000], y[:1000], "classification")
    entries = [
        e for c in payload.model_kv_cache_.values() for e in c.icl_cache.kv.values()
    ]
    assert entries and all(isinstance(e, CompressedKVEntry) for e in entries)
    got = comp.predict(payload, X[1000:], True)
    assert got.predictions == want.predictions
    # Without probabilities, the same labels, as strings from classes (#45).
    plain = comp.predict(payload, X[1000:], False)
    assert plain.predictions == got.predictions
    assert set(plain.predictions) <= set(got.classes)
    assert (
        np.abs(np.array(got.probabilities) - np.array(want.probabilities)).max() < 0.02
    )
    # Spill round trip through tabicl's own pickling (cache.to("cpu") etc.).
    restored = comp.loads_payload(comp.dumps_payload(payload))
    assert comp.predict(restored, X[1000:], True).predictions == got.predictions
    assert pickle.loads(pickle.dumps(entries[0])).is_valid()
