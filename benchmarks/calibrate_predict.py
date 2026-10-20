#!/usr/bin/env python3
"""Predict-time GPU memory under the serving configuration (#47).

For each model, fits a grid of (n_train, n_features) tables through the
engine exactly as serving builds it (serve/factory.build_engine: compiled
blocks, kv precision and all), then measures one predict's peak: the
high-water mark of torch's allocator during the call, above what was
allocated before it. The predict runs on the backend directly, so the
engine's own chunking (what this calibrates) can't shrink it. Output is one
JSON record per shape, the input to memory/predict.py's cost model.

Usage (on a CUDA host):
  TABCTX_BACKEND=tabicl,tabpfn calibrate_predict.py --out predict.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["tabicl", "tabpfn"])
    p.add_argument("--n-train", type=int, nargs="+", default=[1000, 5000, 20000])
    p.add_argument("--features", type=int, nargs="+", default=[10, 50, 100, 200, 500])
    p.add_argument("--n-test", type=int, nargs="+", default=[100, 1000])
    p.add_argument("--out")
    args = p.parse_args()

    import torch

    from tabctx.errors import AdmissionRejected
    from tabctx.serve.factory import ServeSettings, build_engine

    built = build_engine(ServeSettings.from_env())
    engine = built.engine
    device = torch.device(built.device if ":" in built.device else "cuda")
    rng = np.random.default_rng(0)
    records = []
    for model in args.models:
        name = next(b for b in built.backend_names if b.startswith(model))
        backend = engine.backend_for(name)
        for n_features in args.features:
            for n_train in args.n_train:
                X = rng.normal(size=(n_train, n_features)).astype(np.float32)
                y = (X[:, 0] > 0).astype(int)
                dataset_id = f"cal-{model}-{n_train}x{n_features}"
                try:
                    engine.fit(X, y, dataset_id=dataset_id, backend=name)
                except AdmissionRejected as e:
                    skip = {"model": name, "n_train": n_train, "n_features": n_features}
                    print(json.dumps({**skip, "skipped": str(e)[:200]}), flush=True)
                    continue
                # The cached payload, as the engine would hand it to the
                # backend (calibration needs the unchunked call).
                payload = engine._cache.peek(dataset_id).payload  # noqa: SLF001
                for n_test in args.n_test:
                    X_test = rng.normal(size=(n_test, n_features)).astype(np.float32)
                    torch.cuda.synchronize(device)
                    base = torch.cuda.memory_allocated(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    backend.predict(payload, X_test, return_proba=True)
                    torch.cuda.synchronize(device)
                    peak = torch.cuda.max_memory_allocated(device) - base
                    record = {
                        "model": name,
                        "n_train": n_train,
                        "n_features": n_features,
                        "n_test": n_test,
                        "peak_predict_bytes": int(peak),
                    }
                    records.append(record)
                    print(json.dumps(record), flush=True)
                payload = None
                engine.evict(dataset_id)
    meta = {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "settings": {
            k: str(v) for k, v in vars(ServeSettings.from_env()).items() if k != "token"
        },
    }
    print(json.dumps({"event": "done", **meta}), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            f.write(json.dumps({"meta": meta}) + "\n")
            f.writelines(json.dumps(r) + "\n" for r in records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
