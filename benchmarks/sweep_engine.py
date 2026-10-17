#!/usr/bin/env python3
"""Admission sweep through the engine on one GPU, no serving layer: for
each feature count, fit tables of increasing rows (re-fitting one id) until
the admission gate rejects, recording fit time, peak, resident and any
OOM. Every admitted fit must succeed -- an OOM means admission is too
optimistic. Compare admission policies with --peak-only (the pre-floor
behavior: no working-floor model, peak grids only).

Usage: sweep_engine.py --backend tabicl --features 10 100 500 1000 --out sweep.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

import tabctx.serve.factory as factory
from tabctx.errors import AdmissionRejected, BackendComputeError
from tabctx.serve.factory import ServeSettings, build_engine

ROWS = [1_000, 2_000, 5_000, 10_000, 20_000, 30_000, 45_000, 65_000, 80_000,
        100_000, 130_000, 160_000, 200_000, 250_000]  # fmt: skip


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="tabicl")
    p.add_argument("--features", type=int, nargs="+", default=[10, 100, 500, 1000])
    p.add_argument("--peak-only", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.peak_only:
        factory._requirement_calibration = lambda kind, kv: (None, (), ())
    built = build_engine(ServeSettings(backends=(args.backend,)))
    engine = built.engine
    rng = np.random.default_rng(0)
    out = open(args.out, "a") if args.out else None  # noqa: SIM115
    policy = "peak-only" if args.peak_only else "requirement"
    for d in args.features:
        for n in ROWS:
            X = rng.normal(size=(n, d)).astype(np.float32)
            y = (X[:, 0] > 0).astype(int)
            rec = {"policy": policy, "rows": n, "features": d}
            t0 = time.perf_counter()
            try:
                engine.fit(X, y, dataset_id=f"sweep-{d}")
                rec["status"] = "ok"
            except AdmissionRejected as e:
                rec["status"] = "413"
                rec["detail"] = str(e)[:160]
            except BackendComputeError as e:
                rec["status"] = "OOM"
                rec["detail"] = str(e)[:160]
            rec["fit_s"] = round(time.perf_counter() - t0, 2)
            backend = built.backend
            if rec["status"] == "ok":
                rec["peak_gib"] = round((backend.fit_peak_bytes_hint() or 0) / 2**30, 2)
                rec["resident_gib"] = round(
                    (backend.context_bytes_hint(n, d) or 0) / 2**30, 2
                )
            print(json.dumps(rec), flush=True)
            if out:
                out.write(json.dumps(rec) + "\n")
                out.flush()
            if rec["status"] != "ok":
                break
        engine.evict(f"sweep-{d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
