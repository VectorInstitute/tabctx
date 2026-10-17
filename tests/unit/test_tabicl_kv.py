"""Unit tests for compressed TabICL ICL K/V storage (no tabicl needed)."""

import pickle
from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from tabctx.backends.tabicl_kv import CompressedKVEntry, compress_icl_cache  # noqa: E402


@dataclass
class Entry:  # the shape of tabicl's KVCacheEntry
    key: object = None
    value: object = None

    def is_valid(self):
        return self.key is not None and self.value is not None


def entry(dtype=torch.float16, shape=(4, 8, 300, 64), seed=0):
    g = torch.Generator().manual_seed(seed)
    k = (torch.randn(shape, generator=g) * 3).to(dtype)
    v = torch.randn(shape, generator=g).to(dtype)
    return Entry(k, v)


def nbytes(t):
    return t.numel() * t.element_size()


class TestCompressedKVEntry:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_int8_roundtrip_is_close_and_keeps_dtype(self, dtype):
        e = entry(dtype)
        c = CompressedKVEntry.compress(e, "int8")
        for orig, back in ((e.key, c.key), (e.value, c.value)):
            assert back.dtype == dtype and back.shape == orig.shape
            # per-(head,row) absmax/127 quantization: error is half a step,
            # plus fp16 rounding of the scale and the dequantize -- under one
            # step (measured worst case: 0.54 steps, 0.4% of absmax)
            err = (back.float() - orig.float()).abs()
            bound = orig.float().abs().amax(-1, keepdim=True) / 127
            assert bool((err <= bound).all())

    def test_int8_is_about_half_of_fp16(self):
        e = entry(torch.float16)
        c = CompressedKVEntry.compress(e, "int8")
        ratio = c.nbytes / (nbytes(e.key) + nbytes(e.value))
        assert 0.5 < ratio < 0.53  # 1 byte + 2/64 bytes vs 2 bytes

    def test_fp16_halves_fp32_exactly_representable(self):
        e = entry(torch.float32)
        c = CompressedKVEntry.compress(e, "fp16")
        assert c.nbytes * 2 == nbytes(e.key) + nbytes(e.value)
        assert c.key.dtype == torch.float32
        assert torch.equal(c.key, e.key.half().float())

    def test_batch_slicing_matches_full_unpack(self):
        c = CompressedKVEntry.compress(entry(), "int8")
        assert torch.equal(c[1:3].key, c.key[1:3])
        assert torch.equal(c[(slice(0, 2), slice(None))].value, c.value[0:2])

    def test_setitem_writes_a_batch_slice(self):
        c = CompressedKVEntry.compress(entry(seed=0), "int8")
        other = CompressedKVEntry.compress(entry(shape=(2, 8, 300, 64), seed=1), "int8")
        c[0:2] = other
        assert torch.equal(c.key[0:2], other.key)
        plain = entry(shape=(1, 8, 300, 64), seed=2)
        c[3:4] = plain  # a plain entry is compressed into the layout
        assert torch.allclose(c.key[3:4].float(), plain.key.float(), atol=0.1)

    def test_to_and_pickle_roundtrip(self):
        c = CompressedKVEntry.compress(entry(), "int8")
        moved = c.to("cpu", dtype=torch.float32)  # dtype is ignored: storage is fixed
        assert torch.equal(moved.key, c.key)
        restored = pickle.loads(pickle.dumps(c))
        assert torch.equal(restored.value, c.value) and restored.is_valid()


class _Cache:
    def __init__(self, entries):
        self.icl_cache = type("KV", (), {"kv": dict(enumerate(entries))})()


class TestCompressIclCache:
    def test_compresses_every_icl_entry_once(self):
        est = type("Est", (), {})()
        est.model_kv_cache_ = {
            "power": _Cache([entry(), entry()]),
            "none": _Cache([entry()]),
        }
        assert compress_icl_cache(est, "int8") == 3
        assert compress_icl_cache(est, "int8") == 0  # idempotent
        assert all(
            isinstance(e, CompressedKVEntry)
            for c in est.model_kv_cache_.values()
            for e in c.icl_cache.kv.values()
        )

    def test_native_and_fp16_on_fp16_are_noops(self):
        est = type("Est", (), {})()
        est.model_kv_cache_ = {"m": _Cache([entry(torch.float16)])}
        assert compress_icl_cache(est, "native") == 0
        assert compress_icl_cache(est, "fp16") == 0
        assert compress_icl_cache(type("NoCache", (), {})(), "int8") == 0
