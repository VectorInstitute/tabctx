#!/usr/bin/env python3
"""How much GPU memory does a fit actually NEED? (#31, #37)

TabICL's InferenceManager sizes each stage's batches from the GPU memory
that is currently free (x a safety factor), so an unconstrained fit's peak
expands to fill the card: it measures "what was free", not "what is
needed". tabctx's admission learns from measured peaks, so it can reject
tables that would fit.

This runs the same fit with only F GiB left free (the rest taken by a
ballast tensor) for decreasing F, and records success, fit time, transient
peak and resident context size. The smallest F that still succeeds at an
acceptable time is the real requirement.

Usage (on a GPU host):
  bench_memory_pressure.py --backend tabicl --shapes 20000x100 100000x10 \
      --free-gib 0 24 16 8 4 2 --out pressure-tabicl.jsonl
(free 0 = no ballast, the whole card)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from profile_fit_predict import build_backend, make_table, parse_shape  # noqa: E402

GiB = 1024**3


def run(args, backend, rows: int, features: int, free_gib: float) -> dict:
    device = torch.device(args.device)
    record = {"rows": rows, "features": features, "free_gib": free_gib or "all"}
    ballast = None
    try:
        torch.cuda.empty_cache()
        if free_gib:
            free_now, _ = torch.cuda.mem_get_info(device)
            take = int(free_now - free_gib * GiB)
            if take <= 0:
                record["status"] = "skip (less free than requested)"
                return record
            ballast = torch.empty(take, dtype=torch.uint8, device=device)
        X, y = make_table(rows, features, args.task, seed=rows + features)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        payload = backend.fit(X, y, args.task)
        torch.cuda.synchronize(device)
        record.update(
            status="ok",
            fit_s=round(time.perf_counter() - t0, 3),
            peak_gib=round((backend.fit_peak_bytes_hint() or 0) / GiB, 2),
            resident_gib=round(
                (backend.context_bytes_hint(rows, features) or 0) / GiB, 2
            ),
        )
        del payload
    except Exception as e:  # noqa: BLE001 -- OOM is the result being measured
        record.update(status=type(e).__name__, error=str(e)[:200])
    finally:
        del ballast
        torch.cuda.empty_cache()
    return record


def search(args, backend, rows: int, features: int) -> dict:
    """Binary-search the least free memory at which the fit still succeeds
    (to --precision-gib), recording the fit time there and unconstrained."""
    free_all = run(args, backend, rows, features, 0)
    if free_all["status"] != "ok":
        return {"rows": rows, "features": features, "status": "fails unconstrained"}
    lo = free_all["resident_gib"]  # can't succeed with less than the context
    hi = torch.cuda.mem_get_info(torch.device(args.device))[0] / GiB
    best = None
    # A model that degrades instead of failing (TabPFN falls back to slower
    # paths under memory pressure) "succeeds" at almost any free memory;
    # its requirement is where it still runs at an acceptable speed.
    max_s = free_all["fit_s"] * args.max_slowdown + 1.0 if args.max_slowdown else None
    while hi - lo > args.precision_gib:
        mid = (lo + hi) / 2
        r = run(args, backend, rows, features, round(mid, 2))
        if r["status"] == "ok" and (max_s is None or r["fit_s"] <= max_s):
            hi, best = mid, r
        else:
            lo = mid
    return {
        "rows": rows,
        "features": features,
        "status": "ok",
        "resident_gib": free_all["resident_gib"],
        "unconstrained_peak_gib": free_all["peak_gib"],
        "unconstrained_fit_s": free_all["fit_s"],
        "min_free_gib": round(hi, 2),
        "min_working_gib": round(hi - free_all["resident_gib"], 2),
        "fit_s_at_min": best["fit_s"] if best else free_all["fit_s"],
        "max_slowdown": args.max_slowdown,
    }


def _measure_shape(args, rows: int, features: int, queue) -> None:
    """One shape in a fresh process: no allocator or library state left
    over from earlier shapes' OOMs can skew its measurements."""
    backend = build_backend(args.backend, args.device, args.cache_mode)
    backend.warmup()
    if args.search:
        queue.put([search(args, backend, rows, features)])
    else:
        queue.put([run(args, backend, rows, features, f) for f in args.free_gib])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["tabicl", "tabpfn"], required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--task", default="classification")
    p.add_argument("--cache-mode", default="kv")
    p.add_argument("--shapes", nargs="+", type=parse_shape, required=True)
    p.add_argument("--free-gib", nargs="+", type=float, default=[0, 24, 16, 8, 4, 2])
    p.add_argument(
        "--search", action="store_true", help="binary-search min free memory"
    )
    p.add_argument("--precision-gib", type=float, default=0.5)
    p.add_argument(
        "--max-slowdown",
        type=float,
        default=None,
        help="with --search: a fit slower than this x unconstrained counts as "
        "failed (for models that degrade instead of failing, e.g. 2 for tabpfn)",
    )
    p.add_argument("--out", default=None)
    args = p.parse_args()

    ctx = mp.get_context("spawn")
    for rows, features in args.shapes:
        queue = ctx.Queue()
        child = ctx.Process(target=_measure_shape, args=(args, rows, features, queue))
        child.start()
        # Read before join: a child blocks exiting until its queued output
        # is consumed.
        records = None
        while records is None and (child.is_alive() or not queue.empty()):
            try:
                records = queue.get(timeout=5)
            except Exception:  # noqa: BLE001 -- queue.Empty; keep waiting
                continue
        child.join()
        if records is None:
            records = [
                {"rows": rows, "features": features, "status": f"exit {child.exitcode}"}
            ]
        for record in records:
            line = json.dumps(record)
            print(line, flush=True)
            if args.out:
                with open(args.out, "a") as out:
                    out.write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
