#!/usr/bin/env python3
"""Does one process scale across a node's GPUs? (#39)

The 4xL4 rehearsal ran the same 6-user workload on one replica using four
GPUs and pinned to one: large fits took 142 s median on four GPUs vs 70 s
on one. Concurrent fits on *different* GPUs slowed each other down. This
isolates why, without Ray or HTTP: N workers each fit the same shape on
their own GPU, all starting together, and report per-fit wall and CPU time.

Modes:
  threads         N threads in one process (how MultiDeviceEngine runs)
  threads-capped  same, with torch's intra-op threads capped at cores // N
  processes       one process per GPU, each capped at cores // N

N=1 in any mode is the single-GPU baseline. If threads scale badly but
processes scale well, the limit is in-process (GIL, a shared CPU thread
pool); if both scale badly, it's the node (CPU cores, host memory or PCIe
bandwidth).

Usage: bench_scaling.py --backend tabicl --shape 40000x100 --gpus 1 2 4 \\
           --modes threads threads-capped processes --out scaling.jsonl
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import resource
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _worker(kind, device, rows, features, fits, threads, start, results) -> None:
    """Build a backend on `device`, wait for every worker, fit `fits` times."""
    import torch
    from profile_fit_predict import build_backend, make_table

    if threads:
        torch.set_num_threads(threads)
    backend = build_backend(kind, device, "kv")
    backend.warmup()
    X, y = make_table(rows, features, "classification", seed=rows + features)
    start.wait()
    for _ in range(fits):
        cpu0 = time.thread_time()
        t0 = time.perf_counter()
        payload = backend.fit(X, y, "classification")
        torch.cuda.synchronize(device)
        results.append(
            {
                "device": device,
                "wall_s": time.perf_counter() - t0,
                "thread_cpu_s": time.thread_time() - cpu0,
            }
        )
        del payload


def _process_entry(kind, device, rows, features, fits, threads, start, queue):
    results: list = []
    _worker(kind, device, rows, features, fits, threads, start, results)
    for r in results:
        queue.put(r)


def run(kind: str, mode: str, n: int, rows: int, features: int, fits: int) -> dict:
    devices = [f"cuda:{i}" for i in range(n)]
    capped = max(1, _cores() // n) if mode != "threads" else 0
    cpu0 = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    if mode == "processes":
        ctx = mp.get_context("spawn")
        start = ctx.Barrier(n)
        queue = ctx.Queue()
        procs = [
            ctx.Process(
                target=_process_entry,
                args=(kind, d, rows, features, fits, capped, start, queue),
            )
            for d in devices
        ]
        for p in procs:
            p.start()
        results = [queue.get() for _ in range(n * fits)]
        for p in procs:
            p.join()
        child = resource.getrusage(resource.RUSAGE_CHILDREN)
        process_cpu = child.ru_utime + child.ru_stime
    else:
        start = threading.Barrier(n)
        results = []
        workers = [
            threading.Thread(
                target=_worker,
                args=(kind, d, rows, features, fits, capped, start, results),
            )
            for d in devices
        ]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        now = resource.getrusage(resource.RUSAGE_SELF)
        process_cpu = (now.ru_utime - cpu0.ru_utime) + (now.ru_stime - cpu0.ru_stime)
    elapsed = time.perf_counter() - t0
    walls = [r["wall_s"] for r in results]
    return {
        "backend": kind,
        "mode": mode,
        "gpus": n,
        "rows": rows,
        "features": features,
        "fits": len(results),
        "torch_threads": capped or None,
        "cores": _cores(),
        "fit_wall_p50_s": round(statistics.median(walls), 2),
        "fit_wall_max_s": round(max(walls), 2),
        "thread_cpu_p50_s": round(
            statistics.median(r["thread_cpu_s"] for r in results), 2
        ),
        # Includes model loading/warmup; compare across modes, not to wall.
        "process_cpu_s": round(process_cpu, 1),
        "elapsed_s": round(elapsed, 1),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["tabicl", "tabpfn"], default="tabicl")
    p.add_argument("--shape", default="40000x100")
    p.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument(
        "--modes",
        nargs="+",
        default=["threads", "threads-capped", "processes"],
        choices=["threads", "threads-capped", "processes"],
    )
    p.add_argument("--fits", type=int, default=2, help="per worker")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    rows, features = (int(v) for v in args.shape.lower().split("x"))
    out = None
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out = open(args.out, "a")  # noqa: SIM115
    for mode in args.modes:
        for n in args.gpus:
            if mode != "threads" and n == 1 and "threads" in args.modes:
                continue  # N=1 baseline is measured once
            rec = run(args.backend, mode, n, rows, features, args.fits)
            print(json.dumps(rec), flush=True)
            if out:
                out.write(json.dumps(rec) + "\n")
                out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
