#!/usr/bin/env python3
"""Fixed-cost stalls across table sizes (#47).

Builds the engine exactly as serving does (serve/factory.build_engine, so
warmup included), then walks a sequence of table shapes per model: one fit,
then three predicts of the same rows against it. For every call it records
wall time and how many graphs torch.compile built during it. A stall that
recompiles once shows as a slow first call that the repeats don't share; one
that recompiles on every call shows on all three; a stall without compiles
points elsewhere.

Usage (on a GPU host, with the serving environment, e.g. TABCTX_BACKEND):
  TORCH_LOGS=recompiles bench_stalls.py --models tabicl tabpfn \
      --features 10 100 500 --rows 100 500 2000 3500 10000 --out stalls.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np


def compiled_graphs() -> int:
    """Graphs torch.compile has built in this process so far (0 without it)."""
    try:
        from torch._dynamo.utils import counters
    except ImportError:
        return 0
    return int(counters["stats"]["unique_graphs"])


def timed(fn, *args, **kwargs):
    """(seconds, graphs compiled) of one call."""
    graphs = compiled_graphs()
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    return round(time.perf_counter() - t0, 3), compiled_graphs() - graphs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["tabicl", "tabpfn"])
    p.add_argument("--features", type=int, nargs="+", default=[10, 100, 500])
    p.add_argument("--rows", type=int, nargs="+", default=[100, 500, 2000, 3500, 10000])
    p.add_argument("--test-rows", type=int, default=100)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out")
    args = p.parse_args()

    from tabctx.serve.factory import ServeSettings, build_engine

    t0 = time.perf_counter()
    built = build_engine(ServeSettings.from_env())
    engine = built.engine
    print(
        json.dumps(
            {
                "event": "built",
                "models": built.backend_names,
                "device": built.device,
                "build_s": round(time.perf_counter() - t0, 1),
                "graphs_after_warmup": compiled_graphs(),
            }
        ),
        flush=True,
    )
    rng = np.random.default_rng(0)
    records = []
    for model in args.models:
        backend = next(b for b in built.backend_names if b.startswith(model))
        for features in args.features:
            for rows in args.rows:
                X = rng.normal(size=(rows, features)).astype(np.float32)
                y = (X[:, 0] + 0.5 * X[:, 1 % features] > 0).astype(int)
                X_test = rng.normal(size=(args.test_rows, features)).astype(np.float32)
                dataset_id = f"stall-{model}-{features}x{rows}"
                fit_s, fit_graphs = timed(
                    engine.fit, X, y, dataset_id=dataset_id, backend=backend
                )
                predicts = []
                for _ in range(args.repeats):
                    s, graphs = timed(engine.predict, dataset_id, X_test)
                    predicts.append({"s": s, "graphs": graphs})
                engine.evict(dataset_id)
                record = {
                    "model": backend,
                    "shape": f"{rows}x{features}",
                    "fit_s": fit_s,
                    "fit_graphs": fit_graphs,
                    "predicts": predicts,
                }
                records.append(record)
                print(json.dumps(record), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)
    print(json.dumps({"event": "done", "graphs_total": compiled_graphs()}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
