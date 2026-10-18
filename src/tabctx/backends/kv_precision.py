"""KV-cache storage precisions (torch-free, so settings can validate them
without importing a GPU backend). See backends/tabicl_kv.py."""

from __future__ import annotations

from typing import Literal

KVPrecision = Literal["native", "fp16", "int8"]
KV_PRECISIONS: tuple[KVPrecision, ...] = ("native", "fp16", "int8")
