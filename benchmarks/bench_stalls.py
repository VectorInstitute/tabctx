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


class _OnThread:
    """The engine, with fit/predict/evict made on a pool thread."""

    def __init__(self, engine, pool) -> None:
        self._engine, self._pool = engine, pool

    def __getattr__(self, name):
        method = getattr(self._engine, name)
        return lambda *a, **kw: self._pool.submit(method, *a, **kw).result()


def stalls(engine, backend, task, rows, features, args, rng) -> dict:
    """One fit of a rows x features table, then repeated predicts against it."""
    X = rng.normal(size=(rows, features)).astype(np.float32)
    signal = X[:, 0] + 0.5 * X[:, 1 % features]
    y = (signal > 0).astype(int) if task == "classification" else signal
    X_test = rng.normal(size=(args.test_rows, features)).astype(np.float32)
    dataset_id = f"stall-{backend}-{task}-{features}x{rows}"
    fit_s, fit_graphs = timed(
        engine.fit, X, y, task=task, dataset_id=dataset_id, backend=backend
    )
    predicts = []
    for _ in range(args.repeats):
        s, graphs = timed(engine.predict, dataset_id, X_test)
        predicts.append({"s": s, "graphs": graphs})
    engine.evict(dataset_id)
    return {"fit_s": fit_s, "fit_graphs": fit_graphs, "predicts": predicts}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["tabicl", "tabpfn"])
    p.add_argument("--features", type=int, nargs="+", default=[10, 100, 500])
    p.add_argument("--rows", type=int, nargs="+", default=[100, 500, 2000, 3500, 10000])
    p.add_argument("--test-rows", type=int, default=100)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--tasks",
        nargs="+",
        default=["classification"],
        choices=["classification", "regression"],
    )
    p.add_argument(
        "--call-thread",
        action="store_true",
        help="make every call on another thread than the one that built the "
        "engine, as serving does (thread-local settings show up only then)",
    )
    p.add_argument("--out")
    args = p.parse_args()

    from tabctx.errors import AdmissionRejected
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
    if args.call_thread:
        from concurrent.futures import ThreadPoolExecutor

        pool = ThreadPoolExecutor(1, thread_name_prefix="call")
        engine = _OnThread(engine, pool)
    rng = np.random.default_rng(0)
    records = []
    for model in args.models:
        backend = next(b for b in built.backend_names if b.startswith(model))
        for task in args.tasks:
            for features in args.features:
                for rows in args.rows:
                    shape = {
                        "model": backend,
                        "task": task,
                        "shape": f"{rows}x{features}",
                    }
                    try:
                        record = {
                            **shape,
                            **stalls(engine, backend, task, rows, features, args, rng),
                        }
                    except AdmissionRejected as e:
                        record = {**shape, "skipped": str(e)[:160]}
                    records.append(record)
                    print(json.dumps(record), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)
    print(json.dumps({"event": "done", "graphs_total": compiled_graphs()}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
