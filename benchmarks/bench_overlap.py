#!/usr/bin/env python3
"""Can predicts share a GPU with a running fit? (#39, head-of-line)

Each device engine holds its cache lock across whole GPU calls, so a
predict for a small context waits for any fit on that device to finish:
on the 4xL4 rehearsal, predicts waited up to 266 s behind 40k x 100 fits.
Interleaving them is only worth building if the GPU actually overlaps the
work (a predict is a few small kernels; a fit is many large ones) and
neither runs out of memory.

On one device, with a small context already fitted, this measures:
  1. predict latency on an idle device;
  2. a large fit alone;
  3. the same fit with a thread predicting the small context back to
     back throughout (no lock between them): predict latency during the
     fit, the fit's slowdown, and any OOM on either side.

Usage: bench_overlap.py --device cuda:0 --fit 40000x100 --context 2000x20 \\
           --out overlap.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_fit_predict import build_backend, make_table, parse_shape  # noqa: E402


def _pcts(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "p50_ms": round(statistics.median(ordered) * 1000, 1),
        "p95_ms": round(ordered[int(0.95 * (len(ordered) - 1))] * 1000, 1),
        "max_ms": round(ordered[-1] * 1000, 1),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["tabicl", "tabpfn"], default="tabicl")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fit", type=parse_shape, default=(40_000, 100))
    p.add_argument("--context", type=parse_shape, default=(2_000, 20))
    p.add_argument("--test-rows", type=int, default=100)
    p.add_argument("--out", default=None)
    p.add_argument("--compile", action="store_true", help="compiled TabICL blocks")
    args = p.parse_args()
    device = torch.device(args.device)

    backend = build_backend(
        args.backend, args.device, "kv", compile_blocks=args.compile
    )
    backend.warmup()
    cr, cf = args.context
    Xc, yc = make_table(cr, cf, "classification", seed=1)
    small = backend.fit(Xc, yc, "classification")
    X_test, _ = make_table(args.test_rows, cf, "classification", seed=2)

    reference = backend.predict(small, X_test, False).predictions
    mismatches = [0]

    def predict_once() -> float:
        t0 = time.perf_counter()
        out = backend.predict(small, X_test, False)
        torch.cuda.synchronize(device)
        # A predict sharing the backbone with a fit must still answer from
        # its own context.
        if out.predictions != reference:
            mismatches[0] += 1
        return time.perf_counter() - t0

    for _ in range(3):
        predict_once()  # warm
    idle = [predict_once() for _ in range(20)]

    fr, ff = args.fit
    Xf, yf = make_table(fr, ff, "classification", seed=3)

    def timed_fit() -> tuple[float, str]:
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        try:
            payload = backend.fit(Xf, yf, "classification")
            torch.cuda.synchronize(device)
            del payload
            status = "ok"
        except Exception as e:  # noqa: BLE001 -- an OOM is a result
            status = f"{type(e).__name__}: {str(e)[:120]}"
        torch.cuda.empty_cache()
        return time.perf_counter() - t0, status

    alone_s, alone_status = timed_fit()

    during: list[float] = []
    predict_errors: list[str] = []
    stop = threading.Event()

    def predictor() -> None:
        while not stop.is_set():
            try:
                during.append(predict_once())
            except Exception as e:  # noqa: BLE001
                predict_errors.append(f"{type(e).__name__}: {str(e)[:120]}")
                time.sleep(0.1)

    t = threading.Thread(target=predictor, daemon=True)
    t.start()
    shared_s, shared_status = timed_fit()
    stop.set()
    t.join(30)

    rec = {
        "backend": args.backend,
        "device": torch.cuda.get_device_name(device),
        "fit": f"{fr}x{ff}",
        "context": f"{cr}x{cf}",
        "test_rows": args.test_rows,
        "predict_idle": _pcts(idle),
        "fit_alone_s": round(alone_s, 2),
        "fit_alone_status": alone_status,
        "fit_shared_s": round(shared_s, 2),
        "fit_shared_status": shared_status,
        "predict_during_fit": _pcts(during),
        "predict_errors": predict_errors[:5],
        "n_predict_errors": len(predict_errors),
        "predict_mismatches": mismatches[0],
    }
    print(json.dumps(rec), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
