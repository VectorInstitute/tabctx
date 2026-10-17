"""Compressed storage for TabICL's ICL key/value cache.

A TabICL context in kv mode is dominated by the ICL transformer's cached
keys and values: ~0.19 MiB per training row for the default 8 estimators
(measured on an A100: 100k x 10 = 17.6 GiB resident), stored in fp16 once
AMP is on (>= 1024 rows) and fp32 below that. That is what bounds the
longest table a GPU can hold, and how many contexts fit.

This module swaps the ICL cache's entries for ``CompressedKVEntry``, which
stores them compactly and decompresses on access:

- ``"int8"``: symmetric int8 per (head, row) over the head dimension, with
  an fp16 scale -- about 0.52x of fp16 (1 byte + 2/head_dim bytes per
  element), 0.26x of fp32.
- ``"fp16"``: half-precision storage of fp32 caches (small tables), 0.5x;
  a no-op for caches already in fp16.

Attention reads ``entry.key`` / ``entry.value`` once per layer per predict
(tabicl/_model/attention.py), so only one layer's K/V is ever materialized
at full precision, transiently. Decompression returns the original dtype,
so the model's compute path is unchanged.

``CompressedKVEntry`` duck-types tabicl's ``KVCacheEntry`` (is_valid,
batch slicing via __getitem__/__setitem__, ``to``), which is everything
tabicl's InferenceManager batching and its pickling (``cache.to("cpu")``
on save, ``.to(device)`` on load) use.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from tabctx.backends.kv_precision import KV_PRECISIONS, KVPrecision

__all__ = ["KV_PRECISIONS", "CompressedKVEntry", "KVPrecision", "compress_icl_cache"]


class _Packed:
    """One compressed tensor: payload (+ scale for int8) and the dtype to
    decompress to."""

    __slots__ = ("data", "scale", "dtype")

    def __init__(self, data: Tensor, scale: Tensor | None, dtype: torch.dtype) -> None:
        self.data = data
        self.scale = scale
        self.dtype = dtype

    @classmethod
    def pack(cls, t: Tensor, precision: KVPrecision) -> _Packed:
        if precision == "int8":
            absmax = t.abs().amax(dim=-1, keepdim=True).float().clamp_min(1e-8)
            scale = (absmax / 127.0).to(torch.float16)
            q = torch.round(t.float() / scale.float()).clamp_(-127, 127).to(torch.int8)
            return cls(q, scale, t.dtype)
        return cls(t.to(torch.float16), None, t.dtype)

    def unpack(self) -> Tensor:
        if self.scale is None:
            return self.data.to(self.dtype)
        return (self.data.to(torch.float16) * self.scale).to(self.dtype)

    def slice(self, indices: Any) -> _Packed:
        # Scale has the same leading dims as data (last dim 1).
        scale = None if self.scale is None else self.scale[indices]
        return _Packed(self.data[indices], scale, self.dtype)

    def write(self, indices: Any, other: _Packed) -> None:
        self.data[indices] = other.data
        if self.scale is not None and other.scale is not None:
            self.scale[indices] = other.scale

    def to(self, device: Any) -> _Packed:
        scale = None if self.scale is None else self.scale.to(device)
        return _Packed(self.data.to(device), scale, self.dtype)

    @property
    def nbytes(self) -> int:
        n = self.data.numel() * self.data.element_size()
        if self.scale is not None:
            n += self.scale.numel() * self.scale.element_size()
        return n


class CompressedKVEntry:
    """A KVCacheEntry stored compressed; ``key``/``value`` decompress."""

    def __init__(self, key: _Packed, value: _Packed) -> None:
        self._key = key
        self._value = value

    @classmethod
    def compress(cls, entry: Any, precision: KVPrecision) -> CompressedKVEntry:
        return cls(
            _Packed.pack(entry.key, precision), _Packed.pack(entry.value, precision)
        )

    @property
    def key(self) -> Tensor:
        return self._key.unpack()

    @property
    def value(self) -> Tensor:
        return self._value.unpack()

    @property
    def nbytes(self) -> int:
        return self._key.nbytes + self._value.nbytes

    def is_valid(self) -> bool:
        return True

    def __getitem__(self, indices: Any) -> CompressedKVEntry:
        return CompressedKVEntry(self._key.slice(indices), self._value.slice(indices))

    def __setitem__(self, indices: Any, other: Any) -> None:
        if isinstance(other, CompressedKVEntry):
            self._key.write(indices, other._key)
            self._value.write(indices, other._value)
        else:  # a plain KVCacheEntry: compress into our layout
            precision: KVPrecision = "int8" if self._key.scale is not None else "fp16"
            self[indices] = CompressedKVEntry.compress(other, precision)

    def to(self, device: Any, dtype: Any = None) -> CompressedKVEntry:
        del dtype  # storage dtype is fixed by the compression mode
        return CompressedKVEntry(self._key.to(device), self._value.to(device))


def compress_icl_cache(estimator: Any, precision: KVPrecision) -> int:
    """Compress the ICL K/V of every cache on a fitted TabICL estimator, in
    place. Returns the number of entries compressed (0 for "native", for
    estimators without a kv cache, or when already compressed)."""
    if precision == "native":
        return 0
    caches = getattr(estimator, "model_kv_cache_", None) or {}
    n = 0
    for cache in caches.values():
        icl = getattr(cache, "icl_cache", None)
        kv = getattr(icl, "kv", None)
        if not kv:
            continue
        for idx, entry in list(kv.items()):
            if isinstance(entry, CompressedKVEntry) or not entry.is_valid():
                continue
            if precision == "fp16" and entry.key.dtype == torch.float16:
                continue  # already half precision
            kv[idx] = CompressedKVEntry.compress(entry, precision)
            n += 1
    return n
