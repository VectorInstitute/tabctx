#!/usr/bin/env python3
"""Context copy bandwidth on a GPU node (#39): what moving a context costs.

Fit/predict disaggregation and rebalancing contexts between GPUs both pay
for a copy. This measures, for buffer sizes spanning real contexts (MBs to
tens of GB): GPU->GPU (peer, when available), GPU->pinned host,
pinned host->GPU, and GPU->pageable host (what pickling a context to the
spill tier starts with).

Usage: bench_copy.py --sizes-gib 0.25 1 4 16 --out copy-a100x4.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch

GiB = 1024**3


def bandwidth(fn, nbytes: int, devices: list[torch.device], repeats: int) -> float:
    fn()  # warm
    for d in devices:
        torch.cuda.synchronize(d)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    for d in devices:
        torch.cuda.synchronize(d)
    return nbytes * repeats / (time.perf_counter() - t0) / GiB


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes-gib", type=float, nargs="+", default=[0.25, 1, 4, 16])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    n = torch.cuda.device_count()
    gpus = [torch.device(f"cuda:{i}") for i in range(n)]
    result = {
        "gpus": [torch.cuda.get_device_name(g) for g in gpus],
        "peer_access": {
            f"{i}->{j}": torch.cuda.can_device_access_peer(i, j)
            for i in range(n)
            for j in range(n)
            if i != j
        },
        "gib_per_s": [],
    }
    for size in args.sizes_gib:
        nbytes = int(size * GiB)
        src = torch.empty(nbytes, dtype=torch.uint8, device=gpus[0])
        pinned = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        row = {"size_gib": size}
        row["gpu_to_pinned_host"] = bandwidth(
            lambda s=src, h=pinned: h.copy_(s, non_blocking=True),
            nbytes,
            gpus[:1],
            args.repeats,
        )
        row["pinned_host_to_gpu"] = bandwidth(
            lambda s=src, h=pinned: s.copy_(h, non_blocking=True),
            nbytes,
            gpus[:1],
            args.repeats,
        )
        row["gpu_to_pageable_host"] = bandwidth(
            lambda s=src: s.cpu(), nbytes, gpus[:1], max(1, args.repeats // 2)
        )
        if n > 1:
            dst = torch.empty(nbytes, dtype=torch.uint8, device=gpus[1])
            row["gpu0_to_gpu1"] = bandwidth(
                lambda s=src, d=dst: d.copy_(s, non_blocking=True),
                nbytes,
                gpus[:2],
                args.repeats,
            )
            dst = None
        row = {k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()}
        result["gib_per_s"].append(row)
        print(json.dumps(row), flush=True)
        src = pinned = None
        torch.cuda.empty_cache()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "gib_per_s"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
