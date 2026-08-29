#!/usr/bin/env python3
"""Generate src/tabctx/memory/calibration_tabicl_a100.py from one or
more calibrate_memory.py JSON outputs.

Usage:
  python benchmarks/gen_calibration_data.py kv=calib.json repr=calib-repr.json \
      --out src/tabctx/memory/calibration_tabicl_a100.py
"""

from __future__ import annotations

import argparse
import json
import sys

GRID_NAMES = {
    "kv": "A100_40GB_TABICL_KV_PEAK_GRID",
    "repr": "A100_40GB_TABICL_REPR_PEAK_GRID",
    "off": "A100_40GB_TABICL_OFF_PEAK_GRID",
}
PREDICT_GRID_NAMES = {
    "kv": "A100_40GB_TABICL_KV_PREDICT_PEAK_GRID",
    "repr": "A100_40GB_TABICL_REPR_PREDICT_PEAK_GRID",
    "off": "A100_40GB_TABICL_OFF_PREDICT_PEAK_GRID",
}


def _emit_grid(mode: str, data: dict) -> str:
    env = data["env"]
    ok = [r for r in data["records"] if r["outcome"] == "ok"]
    oom = [r for r in data["records"] if r["outcome"] == "oom"]
    lines = [
        f"# mode={mode!r}: measured {env['generated_at']} on {env['device']} "
        f"({env['total_gpu_bytes']} bytes), torch {env['torch']}.",
    ]
    if oom:
        boundary = ", ".join(f"{r['n_train']}x{r['n_features']}" for r in oom)
        lines.append(f"# First OOM per feature count: {boundary}.")
    lines.append(f"{GRID_NAMES[mode]}: tuple[Observation, ...] = (")
    for r in sorted(ok, key=lambda r: (r["n_features"], r["n_train"])):
        lines.append(
            f"    Observation(n_train={r['n_train']}, "
            f"n_features={r['n_features']}, "
            f"real_bytes={int(r['peak_fit_bytes'])}),"
            f"  # fit {r['fit_s']}s, resident {int(r['resident_context_bytes'])}"
        )
    lines.append(")")
    lines.append("")
    lines.append(
        f"# Measured PREDICT peaks for the same shapes, at n_test="
        f"{ok[0].get('n_test', 1000)} test rows (chunking's quantity;"
        " see memory/adaptive.py)."
    )
    lines.append(f"{PREDICT_GRID_NAMES[mode]}: tuple[Observation, ...] = (")
    for r in sorted(ok, key=lambda r: (r["n_features"], r["n_train"])):
        lines.append(
            f"    Observation(n_train={r['n_train']}, "
            f"n_features={r['n_features']}, "
            f"real_bytes={int(r['peak_predict_bytes'])}),"
        )
    lines.append(")")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", help="mode=path.json pairs")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    blocks = []
    for spec in args.inputs:
        mode, _, path = spec.partition("=")
        if mode not in GRID_NAMES or not path:
            print(f"bad input spec {spec!r} (want kv=file.json)", file=sys.stderr)
            return 1
        with open(path) as f:
            blocks.append(_emit_grid(mode, json.load(f)))

    header = '''"""GENERATED calibration grids -- do not hand-edit.

Produced by benchmarks/gen_calibration_data.py from
benchmarks/calibrate_memory.py sweeps on real hardware. Each Observation
records the measured PEAK fit bytes (transient high-water allocation
delta during fit -- the admission-relevant quantity; see
backends/base.py) for one (n_train, n_features) shape; resident context
size and fit time ride along as comments. OOM boundary shapes are listed
per grid: nothing beyond them has ever succeeded on this hardware.

These grids are preloaded into AdaptiveMemoryEstimator by
serve/factory.py so admission decisions rest on measurements from the
first request onward. They are A100-40GB + TabICLv2 measurements:
different GPUs/models need their own sweep (same script).
"""

from __future__ import annotations

from tabctx.memory.adaptive import Observation

'''
    with open(args.out, "w") as f:
        f.write(header + "\n\n".join(blocks) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
