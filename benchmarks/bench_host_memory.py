#!/usr/bin/env python3
"""Host (CPU) memory per fit: the limit GPU admission doesn't see.

With GPU admission bounded by what fits actually need (memory/
requirement.py), the next ceiling is host RAM: the models' CPU-side
preprocessing scales with cells, and a fit that exhausts the pod's memory
gets the whole replica OOM-killed -- every tenant's context with it.

For each shape this runs one fit in a fresh subprocess (an OOM kill takes
only that child), sampling RSS every 20 ms, and records the peak RSS
growth during fit and the RSS still held afterwards (host memory per
cached context), plus the cgroup memory limit.

Usage: bench_host_memory.py --backend tabicl --shapes 20000x100 10000x500 --out host.jsonl
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import threading
import time
from pathlib import Path

GiB = 1024**3


def rss_bytes() -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


def cgroup_limit_bytes() -> int | None:
    for path in (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ):
        try:
            with open(path) as f:
                raw = f.read().strip()
        except OSError:
            continue
        if raw.isdigit() and int(raw) < 1 << 60:
            return int(raw)
    return None


def _child(backend_kind: str, device: str, rows: int, features: int, queue) -> None:
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from profile_fit_predict import build_backend, make_table

    backend = build_backend(backend_kind, device, "kv")
    backend.warmup()
    X, y = make_table(rows, features, "classification", seed=rows + features)
    base = rss_bytes()
    peak = [base]
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            peak[0] = max(peak[0], rss_bytes())
            time.sleep(0.02)

    t = threading.Thread(target=sample, daemon=True)
    t.start()
    t0 = time.perf_counter()
    payload = backend.fit(X, y, "classification")
    fit_s = time.perf_counter() - t0
    stop.set()
    t.join()
    del X, y
    import gc

    gc.collect()
    queue.put(
        {
            "status": "ok",
            "fit_s": round(fit_s, 2),
            "peak_growth_gib": round((peak[0] - base) / GiB, 2),
            "retained_gib": round((rss_bytes() - base) / GiB, 2),
            "table_gib": round(rows * features * 4 / GiB, 3),
        }
    )
    del payload


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["tabicl", "tabpfn"], required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--shapes", nargs="+", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    ctx = mp.get_context("spawn")
    limit = cgroup_limit_bytes()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out = open(args.out, "a") if args.out else None  # noqa: SIM115
    for shape in args.shapes:
        rows, features = (int(v) for v in shape.lower().split("x"))
        q = ctx.Queue()
        child = ctx.Process(
            target=_child, args=(args.backend, args.device, rows, features, q)
        )
        child.start()
        child.join()
        rec = {
            "backend": args.backend,
            "rows": rows,
            "features": features,
            "cells": rows * features,
        }
        rec["cgroup_limit_gib"] = round(limit / GiB, 1) if limit else None
        rec.update(q.get() if not q.empty() else {"status": f"exit {child.exitcode}"})
        print(json.dumps(rec), flush=True)
        if out:
            out.write(json.dumps(rec) + "\n")
            out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
