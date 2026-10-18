#!/usr/bin/env python3
"""Predict throughput against many cached contexts on one GPU (#33).

Before building cross-context batched predict (one kernel launch serving
several tenants' predicts, the continuous-batching analogue), measure
whether there is anything to win: fit N contexts, then T threads each
predict R rows against distinct contexts, through one TabctxEngine (whose
predict lock runs one GPU predict at a time). Reports predicts/s and
latency per T, and the GPU's utilization while it runs (nvidia-smi).
If throughput stays flat with T while utilization is low, predicts are
launch/host bound and batching across contexts would pay; if the GPU is
already busy, it wouldn't.

Usage (on a GPU host):
  bench_predict_throughput.py --contexts 32 --shape 2000x20 --rows 100 \
      --threads 1 2 4 8 --seconds 20 --compile --fast-preprocessing
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from profile_fit_predict import build_backend, make_table, parse_shape  # noqa: E402

from tabctx.cache.manager import ContextCacheManager  # noqa: E402
from tabctx.engine import TabctxEngine  # noqa: E402
from tabctx.memory import (  # noqa: E402
    A100_40GB_TABICL_CALIBRATION,
    AdaptiveMemoryEstimator,
    PowerLawMemoryEstimator,
)


class Utilization:
    """Samples GPU utilization with nvidia-smi while running."""

    def __init__(self, index: int) -> None:
        self._index = index
        self._samples: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(0.2):
            out = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self._index}",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            if out.returncode == 0 and out.stdout.strip().isdigit():
                self._samples.append(int(out.stdout.strip()))

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(5)

    @property
    def mean(self) -> float:
        return round(float(np.mean(self._samples)), 1) if self._samples else 0.0


def run_level(engine, ids, X_test, threads: int, seconds: float, index: int):
    """T threads predicting against distinct contexts for `seconds`:
    (latencies, elapsed, mean GPU utilization)."""
    latencies: list[float] = []
    lock = threading.Lock()
    deadline = time.perf_counter() + seconds

    def user(k: int) -> None:
        n = 0
        while time.perf_counter() < deadline:
            dataset_id = ids[(k + n * threads) % len(ids)]
            t0 = time.perf_counter()
            engine.predict(dataset_id, X_test)
            with lock:
                latencies.append(time.perf_counter() - t0)
            n += 1

    with Utilization(index) as util:
        start = time.perf_counter()
        workers = [threading.Thread(target=user, args=(k,)) for k in range(threads)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        elapsed = time.perf_counter() - start
    return np.array(latencies), elapsed, util.mean


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["tabicl", "tabpfn"], default="tabicl")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--contexts", type=int, default=32)
    p.add_argument("--shape", type=parse_shape, default=(2000, 20))
    p.add_argument("--rows", type=int, default=100)
    p.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--fast-preprocessing", action="store_true")
    p.add_argument("--out")
    args = p.parse_args()

    backend = build_backend(
        args.backend,
        args.device,
        "kv",
        compile_blocks=args.compile,
        fast_preprocessing=args.fast_preprocessing,
    )
    backend.warmup()
    estimator = AdaptiveMemoryEstimator(
        fallback=PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION)
    )
    engine = TabctxEngine(
        backend=backend,
        cache=ContextCacheManager(estimator.ceiling_bytes()),
        estimator=estimator,
    )
    rows, features = args.shape
    ids = []
    for i in range(args.contexts):
        X, y = make_table(rows, features, "classification", seed=i)
        ids.append(engine.fit(X, y, "classification", dataset_id=f"ctx{i}"))
    X_test, _ = make_table(args.rows, features, "classification", seed=999)
    for dataset_id in ids:  # warm every context's predict path
        engine.predict(dataset_id, X_test)

    index = int(args.device.split(":")[1]) if ":" in args.device else 0
    for threads in args.threads:
        lat, elapsed, util = run_level(
            engine, ids, X_test, threads, args.seconds, index
        )
        record = {
            "backend": args.backend,
            "compiled": args.compile,
            "shape": f"{rows}x{features}",
            "rows": args.rows,
            "threads": threads,
            "predicts_per_s": round(len(lat) / elapsed, 2),
            "p50_ms": round(float(np.percentile(lat, 50)) * 1000, 1),
            "p95_ms": round(float(np.percentile(lat, 95)) * 1000, 1),
            "gpu_util_pct": util,
        }
        print(json.dumps(record), flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps(record) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
