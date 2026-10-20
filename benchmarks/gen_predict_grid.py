#!/usr/bin/env python3
"""Generate the predict calibration module from calibrate_predict.py output
(JSON lines): one tuple of PredictPoint per model, for PredictCostModel (see
memory/predict.py).

Usage:
  gen_predict_grid.py predict_a100.jsonl predict_a100_wide.jsonl \
      --prefix A100_40GB --out src/tabctx/memory/calibration_predict_a100.py
"""

from __future__ import annotations

import argparse
import json
import sys

# Served model id -> the name the factory keys calibrations by.
_KINDS = {"tabicl": "TABICL", "tabpfn": "TABPFN"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+")
    p.add_argument("--prefix", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    metas, rows = [], []
    for path in args.inputs:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if "meta" in record:
                    metas.append(record["meta"])
                else:
                    rows.append(record)
    if not metas:
        sys.exit("no meta record in the inputs: run calibrate_predict.py with --out")
    if any(m != metas[0] for m in metas[1:]):
        sys.exit("inputs were measured on different devices or settings")
    source = [f"{metas[0]['device']}, torch {metas[0]['torch']}"]
    settings = metas[0]["settings"]
    config = ", ".join(
        f"{k}={settings[k]}"
        for k in ("kv_cache", "kv_precision", "compile_models")
        if k in settings
    )
    lines = [
        '"""GENERATED predict calibration -- do not hand-edit.',
        "",
        "Produced by benchmarks/gen_predict_grid.py from",
        "benchmarks/calibrate_predict.py on real hardware: each PredictPoint is",
        "one predict's peak allocation (above what was allocated before it)",
        "against a kv-cached context, under the serving configuration. Read by",
        "serve/factory.py into PredictCostModel (memory/predict.py).",
        "",
        f"Measured on {'; '.join(source)} ({config}).",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from tabctx.memory.predict import PredictPoint",
    ]
    for kind, name in _KINDS.items():
        points = sorted(
            (r for r in rows if r["model"].startswith(kind)),
            key=lambda r: (r["n_features"], r["n_train"], r["n_test"]),
        )
        if not points:
            continue
        lines += [
            "",
            f"{args.prefix}_{name}_PREDICT_POINTS: tuple[PredictPoint, ...] = (",
        ]
        lines += [
            f"    PredictPoint(n_train={r['n_train']}, n_test={r['n_test']}, "
            f"n_features={r['n_features']}, peak_bytes={r['peak_predict_bytes']}),"
            for r in points
        ]
        lines.append(")")
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(rows)} points to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
