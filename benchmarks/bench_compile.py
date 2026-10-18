#!/usr/bin/env python3
"""Does torch.compile speed up a backend's fit and predict, and is it exact
enough? (#32/#36)

Compares one backend eager vs compiled, on one device (compilation patches
the model process-wide, so run eager and compiled as separate processes):

- TabICL: its attention blocks compiled (TabICLBackend(compile_blocks=True),
  see backends/tabicl_compile.py) -- layer norms, residual adds and the
  GELU MLP fused around the attention kernel;
- TabPFN-3.5: the library's own compile path (enable_torch_compile).

Reports the fit time (TabICL: its GPU half, CPU preprocessing excluded),
the first call's compile cost, predict latency, accuracy and log loss on
held-out rows, and the largest probability difference against a saved
eager run.

Usage (on a GPU host with the models' weights):
  bench_compile.py --backend tabicl --shapes 10000x50 40000x100 --save-proba eager.npz
  bench_compile.py --backend tabicl --shapes 10000x50 40000x100 --compile \
      --baseline eager.npz --out compile-a100.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch


def parse_shape(text: str) -> tuple[int, int]:
    rows, features = text.lower().split("x")
    return int(rows), int(features)


def make_table(rows: int, features: int, seed: int):
    """A noisy nonlinear task (so accuracy is informative, not saturated)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(rows, features)).astype(np.float32)
    j = min(2, features - 1)
    signal = X[:, 0] + 0.5 * X[:, min(1, features - 1)] * X[:, j] + np.sin(2 * X[:, j])
    y = (signal + rng.normal(scale=0.8, size=rows) > 0).astype(int)
    return X, y


def enable_tabpfn_compile() -> None:
    """TabPFN-3.5 ships its own torch.compile path (PerformanceOptions.
    enable_torch_compile: selected sub-graphs, dynamic row/feature dims,
    symbolic log n), off by default and not exposed by the estimators;
    turn it on in the model's default options."""
    import dataclasses

    from tabpfn.architectures.tabpfn_v3_5 import TabPFNV3p5

    defaults = TabPFNV3p5.get_default_performance_options

    def with_compile(self):
        return dataclasses.replace(defaults(self), enable_torch_compile=True)

    TabPFNV3p5.get_default_performance_options = with_compile
    # Its SDPA wrapper loops over ceil(batch*heads / 65536) chunks: a Python
    # loop count that depends on the table, so every new size recompiled
    # (60-266 s each). Run it eagerly (a graph break around attention).
    from tabpfn.architectures.shared import scaled_dot_product_attention as sdpa

    sdpa._torch_sdpa = torch.compiler.disable(sdpa._torch_sdpa)


def build(kind: str, device: str, compiled: bool):
    if kind == "tabpfn":
        from tabctx.backends.tabpfn import TabPFNBackend

        if compiled:
            enable_tabpfn_compile()
        return TabPFNBackend(device=device)
    from tabctx.backends.tabicl import TabICLBackend

    return TabICLBackend(device=device, compile_blocks=compiled)


def fit_timed(backend, X, y, device: str):
    """The fit's GPU half where the backend splits it (TabICL), else all."""
    prepare = getattr(backend, "prepare_fit", None)
    if callable(prepare):
        prepared = prepare(X, y, "classification")
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        payload = backend.fit_prepared(prepared)
    else:
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        payload = backend.fit(X, y, "classification")
    torch.cuda.synchronize(device)
    return payload, time.perf_counter() - t0


def run(
    kind: str,
    device: str,
    compiled: bool,
    rows: int,
    features: int,
    repeats: int,
    amp: str,
) -> dict:
    backend = build(kind, device, compiled)
    if (
        amp != "auto" and kind == "tabicl"
    ):  # tabicl's default: fp16 autocast above 1024 rows
        make = backend._make_estimator

        def make_with_amp(task):
            est = make(task)
            est.use_amp = amp == "on"
            return est

        backend._make_estimator = make_with_amp
    backend.warmup()
    torch.cuda.reset_peak_memory_stats(device)
    X, y = make_table(rows, features, seed=0)
    X_test, y_test = make_table(2048, features, seed=1)
    times, payload = [], None
    first = 0.0
    for i in range(repeats + 1):
        payload, elapsed = fit_timed(backend, X, y, device)
        if i == 0:
            first = elapsed
        else:
            times.append(elapsed)
    backend.predict(payload, X_test, return_proba=True)  # warm the predict path
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    outcome = backend.predict(payload, X_test, return_proba=True)
    torch.cuda.synchronize(device)
    proba = np.asarray(outcome.probabilities, dtype=np.float64)
    p_true = np.clip(proba[np.arange(len(y_test)), y_test], 1e-12, 1.0)
    return {
        "accuracy": round(float((proba.argmax(1) == y_test).mean()), 4),
        "log_loss": round(float(-np.log(p_true).mean()), 5),
        "first_s": round(first, 3),
        "fit_s": round(float(np.median(times)), 3),
        "predict_s": round(time.perf_counter() - t0, 3),
        "proba": proba,
        "peak_gib": round(torch.cuda.max_memory_allocated(device) / 1024**3, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shapes", nargs="+", default=["10000x50", "40000x100"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--backend", choices=("tabicl", "tabpfn"), default="tabicl")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--amp", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--baseline", help="npz of eager probabilities to compare against")
    ap.add_argument("--save-proba", help="write this run's probabilities (npz)")
    ap.add_argument("--out")
    args = ap.parse_args()

    baseline = dict(np.load(args.baseline)) if args.baseline else {}
    saved = {}
    for text in args.shapes:
        rows, features = parse_shape(text)
        result = run(
            args.backend,
            args.device,
            args.compile,
            rows,
            features,
            args.repeats,
            args.amp,
        )
        proba = result.pop("proba")
        saved[text] = proba
        if text in baseline:
            ref = baseline[text]
            result["max_abs_proba_diff"] = float(np.abs(proba - ref).max())
            result["label_agreement"] = float((proba.argmax(1) == ref.argmax(1)).mean())
        record = {
            "backend": args.backend,
            "compiled": args.compile,
            "amp": args.amp,
            "shape": text,
            **result,
        }
        print(json.dumps(record), flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps(record) + "\n")
    if args.save_proba:
        np.savez(args.save_proba, **saved)
    return 0


if __name__ == "__main__":
    sys.exit(main())
