#!/usr/bin/env python3
"""GPU memory calibration for the admission gate (ROADMAP Priority 1).

Sweeps a grid of (n_train, n_features) through TabICLBackend with the
kv-cache ON (the shipped default) on a real GPU, measuring the two
quantities the estimator needs and must never conflate:

- **peak fit bytes** (high-water torch.cuda memory during fit, minus the
  pre-fit baseline): the transient cost admission control must bound --
  if this exceeds free GPU memory, the fit OOMs regardless of how small
  the resulting context is;
- **resident context bytes** (post-fit allocated delta): what the fitted
  context actually occupies afterward -- the cache-accounting quantity.

It also measures predict-time peak for a fixed test batch (chunking's
quantity), escalates n_train until the first real OOM per feature count
(the honest capacity boundary -- caught and recorded, the process keeps
going), and writes everything as JSON for `memory/calibration_data.py`.

Run on a CUDA box:

    python benchmarks/calibrate_memory.py --out calib.json

`--quick` runs a tiny grid (works on CPU, records no memory numbers) to
smoke-test the harness itself before spending GPU time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone

import numpy as np

N_TEST_ROWS = 1_000

# (n_features, [n_train grid ... escalation]). Escalation points run in
# order until the first OOM for that feature count, then stop.
DEFAULT_GRID: list[tuple[int, list[int]]] = [
    (10, [1_000, 5_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]),
    (50, [1_000, 5_000, 20_000, 50_000, 100_000, 200_000, 500_000]),
    (100, [1_000, 5_000, 20_000, 50_000, 100_000, 200_000, 400_000]),
    (200, [1_000, 5_000, 20_000, 50_000, 100_000, 200_000, 300_000]),
]
QUICK_GRID: list[tuple[int, list[int]]] = [(5, [200, 500])]


def _make_data(n_train: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_train, n_features)).astype(np.float32)
    y = np.where(X[:, 0] > 0, "a", "b").tolist()
    X_test = rng.normal(size=(N_TEST_ROWS, n_features)).astype(np.float32)
    return X, y, X_test


def calibrate(grid, out_path: str, kv_cache: str = "kv") -> int:
    import torch

    from tabctx.backends.tabicl import TabICLBackend

    cuda = torch.cuda.is_available()
    backend = TabICLBackend(kv_cache=False if kv_cache == "off" else kv_cache)
    records = []

    def gpu(f):
        return f() if cuda else 0

    # Warm the shared backbone once so its footprint never pollutes a
    # per-shape measurement, and record it separately.
    Xw, yw, _ = _make_data(256, 4, seed=0)
    backend.fit(Xw, yw, "classification")
    gpu(torch.cuda.empty_cache)
    backbone_bytes = gpu(torch.cuda.memory_allocated)
    print(f"[info] backbone resident: {backbone_bytes/1e6:.0f}MB "
          f"(measured once, excluded from per-shape numbers)")

    for n_features, n_train_grid in grid:
        for n_train in n_train_grid:
            X, y, X_test = _make_data(n_train, n_features, seed=n_train + n_features)
            gpu(torch.cuda.empty_cache)
            before = gpu(torch.cuda.memory_allocated)
            gpu(torch.cuda.reset_peak_memory_stats)

            rec = {"n_train": n_train, "n_features": n_features}
            start = time.monotonic()
            try:
                payload = backend.fit(X, y, "classification")
            except Exception as e:  # noqa: BLE001 -- OOM is a *result* here
                is_oom = "out of memory" in str(e).lower() or "OOM" in str(e)
                rec.update({
                    "outcome": "oom" if is_oom else "error",
                    "error": f"{type(e).__name__}: {str(e)[:200]}",
                    "peak_fit_bytes": gpu(torch.cuda.max_memory_allocated) - before,
                    "peak_fit_reserved_bytes": gpu(torch.cuda.max_memory_reserved),
                })
                records.append(rec)
                print(f"[{'oom' if is_oom else 'ERR'}] {n_train}x{n_features}: "
                      f"{rec['error'][:100]}")
                gpu(torch.cuda.empty_cache)
                if is_oom:
                    break  # larger n_train at this width will OOM too
                continue

            fit_s = time.monotonic() - start
            peak = gpu(torch.cuda.max_memory_allocated)
            peak_reserved = gpu(torch.cuda.max_memory_reserved)
            resident = gpu(torch.cuda.memory_allocated)

            gpu(torch.cuda.reset_peak_memory_stats)
            start = time.monotonic()
            backend.predict(payload, X_test)
            predict_s = time.monotonic() - start
            predict_peak = gpu(torch.cuda.max_memory_allocated)

            rec.update({
                "outcome": "ok",
                "fit_s": round(fit_s, 2),
                "peak_fit_bytes": peak - before,
                "peak_fit_reserved_bytes": peak_reserved,
                "resident_context_bytes": resident - before,
                "n_test": N_TEST_ROWS,
                "predict_s": round(predict_s, 2),
                "peak_predict_bytes": predict_peak - resident,
            })
            records.append(rec)
            print(f"[ok]  {n_train}x{n_features}: fit={fit_s:6.1f}s "
                  f"peak={rec['peak_fit_bytes']/1e9:6.2f}GB "
                  f"resident={rec['resident_context_bytes']/1e9:6.3f}GB "
                  f"predict_peak={rec['peak_predict_bytes']/1e9:6.3f}GB")

            del payload
            gpu(torch.cuda.empty_cache)

            # Persist incrementally: a crash late in the sweep must not
            # lose the measurements already paid for.
            _write(out_path, records, backbone_bytes, cuda, kv_cache)

    _write(out_path, records, backbone_bytes, cuda, kv_cache)
    print(f"\n[info] wrote {len(records)} records to {out_path}")
    return 0


def _write(out_path, records, backbone_bytes, cuda, kv_cache="kv"):
    import torch

    env = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(0) if cuda else "cpu",
        "total_gpu_bytes": (
            torch.cuda.get_device_properties(0).total_memory if cuda else 0
        ),
        "torch": torch.__version__,
        "kv_cache": kv_cache,
        "backbone_resident_bytes": backbone_bytes,
        "note": (
            "peak_fit_bytes is the admission-relevant transient high-water "
            "delta; resident_context_bytes is the cache-accounting quantity. "
            "Measured via TabICLBackend (tabctx) with kv_cache='kv'."
        ),
    }
    with open(out_path, "w") as f:
        json.dump({"env": env, "records": records}, f, indent=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="calibration.json")
    p.add_argument("--quick", action="store_true",
                   help="tiny smoke grid (CPU ok; memory numbers meaningless)")
    p.add_argument("--kv-cache", default="kv", choices=["kv", "repr", "off"],
                   help="TabICL context-cache mode to calibrate (see backends/tabicl.py)")
    args = p.parse_args()
    return calibrate(
        QUICK_GRID if args.quick else DEFAULT_GRID, args.out, kv_cache=args.kv_cache
    )


if __name__ == "__main__":
    sys.exit(main())
