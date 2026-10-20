#!/usr/bin/env python3
"""Where one predict's time goes (#47): cProfile over a cached-context predict
built exactly as serving builds it (serve/factory.build_engine).

Usage (on a GPU host):
  TABCTX_BACKEND=tabicl profile_predict.py --model tabicl --shape 3500x500
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
import time

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="tabicl")
    p.add_argument("--shape", default="3500x500")
    p.add_argument("--test-rows", type=int, default=100)
    p.add_argument("--top", type=int, default=40)
    args = p.parse_args()
    rows, features = (int(v) for v in args.shape.split("x"))

    from tabctx.serve.factory import ServeSettings, build_engine

    built = build_engine(ServeSettings.from_env())
    engine = built.engine
    backend = next(b for b in built.backend_names if b.startswith(args.model))
    rng = np.random.default_rng(0)
    X = rng.normal(size=(rows, features)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    X_test = rng.normal(size=(args.test_rows, features)).astype(np.float32)
    engine.fit(X, y, dataset_id="p", backend=backend)
    engine.predict("p", X_test)  # first call: anything one-off happens here
    t0 = time.perf_counter()
    profiler = cProfile.Profile()
    profiler.runcall(engine.predict, "p", X_test)
    print(f"predict {time.perf_counter() - t0:.2f}s", flush=True)
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
