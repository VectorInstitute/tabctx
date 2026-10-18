#!/usr/bin/env python3
"""Does torch.compile speed up TabICL's GPU half, and is it exact enough? (#32/#36)

The fit profile (#31) put layer norm at ~21% of the GPU half -- memory-bound
passes over [ensembles, features, rows, 128] activations, each next to a
residual add or a GELU MLP that also streams the tensor. Inductor can fuse
those elementwise passes. This compares, on one device:

- eager (baseline);
- ``block``: each attention block's whole forward compiled (norms,
  residual adds and the GELU MLP fused around the attention kernel).

For each it reports the GPU half's time (TabICLBackend.fit_prepared, CPU
preprocessing excluded), the first call's compile cost, predict latency, and
the largest probability difference against eager on held-out rows.

Usage (on a GPU host with tabicl and the weights):
  bench_compile.py --device cuda:0 --shapes 10000x50 40000x100 --repeats 3 \
      --variant block --baseline eager.npz --out compile-a100.jsonl
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


def install(variant: str, mode: str) -> None:
    """Patch tabicl's attention block class (all instances share one graph:
    nn-module parameters are graph inputs, so instances don't recompile)."""
    from tabicl._model.layers import MultiheadAttentionBlock as Block

    if variant == "eager":
        return
    if variant == "block":
        # Blocks see several static shapes (column vs row vs ICL stages,
        # induced points vs features): allow a graph per shape family.
        torch._dynamo.config.recompile_limit = 64
        Block.forward = torch.compile(Block.forward, dynamic=True, mode=mode)
        return
    raise SystemExit(f"unknown variant {variant}")


def run(device: str, rows: int, features: int, repeats: int, amp: str) -> dict:
    from tabctx.backends.tabicl import TabICLBackend

    backend = TabICLBackend(device=device)
    if amp != "auto":  # tabicl's default: fp16 autocast above 1024 rows
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
        prepared = backend.prepare_fit(X, y, "classification")
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        payload = backend.fit_prepared(prepared)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
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
        "gpu_half_s": round(float(np.median(times)), 3),
        "predict_s": round(time.perf_counter() - t0, 3),
        "proba": proba,
        "peak_gib": round(torch.cuda.max_memory_allocated(device) / 1024**3, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shapes", nargs="+", default=["10000x50", "40000x100"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--variant", default="eager")
    ap.add_argument("--mode", default="default")
    ap.add_argument("--amp", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--baseline", help="npz of eager probabilities to compare against")
    ap.add_argument("--save-proba", help="write this run's probabilities (npz)")
    ap.add_argument("--out")
    args = ap.parse_args()

    install(args.variant, args.mode)
    baseline = dict(np.load(args.baseline)) if args.baseline else {}
    saved = {}
    for text in args.shapes:
        rows, features = parse_shape(text)
        result = run(args.device, rows, features, args.repeats, args.amp)
        proba = result.pop("proba")
        saved[text] = proba
        if text in baseline:
            ref = baseline[text]
            result["max_abs_proba_diff"] = float(np.abs(proba - ref).max())
            result["label_agreement"] = float((proba.argmax(1) == ref.argmax(1)).mean())
        record = {
            "variant": args.variant,
            "mode": args.mode,
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
