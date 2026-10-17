#!/usr/bin/env python3
"""Table-size sweep: how large a table does a deployment admit, and how
long do fit and predict take on the way up?

For each feature count, fits tables of increasing row count until the
admission gate answers 413 (or --max-rows), timing fit and a fixed-size
predict at every step -- the method behind the single-L4 "fit latency
vs table size" chart. Each feature count re-fits ONE dataset_id, so
earlier steps never hold memory that would shrink later steps' headroom.
Tables at or above UPLOAD_CELLS go through the CSV upload path.

Usage:
  sweep_admission.py --base-url http://127.0.0.1:8000/tabctx \
      --session-header x-tabctx-session-id --model tabicl-v2 \
      --save benchmarks/baselines/sweep-l4x4.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime

import numpy as np
from bench_workload import UPLOAD_CELLS, fit_table, make_table

from tabctx.client import TabctxClient
from tabctx.errors import AdmissionRejected, TabctxError

ROWS = [
    100, 250, 500, 1_000, 2_000, 3_500, 5_000, 7_500, 10_000, 15_000, 20_000,
    30_000, 45_000, 65_000, 100_000, 150_000, 200_000, 300_000, 500_000,
]  # fmt: skip


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--features", type=int, nargs="+", default=[10, 100, 500, 1000])
    p.add_argument("--max-rows", type=int, default=500_000)
    p.add_argument("--test-rows", type=int, default=100)
    p.add_argument("--model", default=None)
    p.add_argument("--tenant", default="sweep")
    p.add_argument("--session-header", default="x-session-id")
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--save", default=None)
    args = p.parse_args()

    client = TabctxClient(
        args.base_url,
        tenant_id=args.tenant,
        session_header=args.session_header,
        timeout_s=args.timeout,
    )
    rng = np.random.default_rng(0)
    points = []
    for features in args.features:
        dataset_id = f"sweep-{args.model or 'default'}-f{features}"
        for rows in (r for r in ROWS if r <= args.max_rows):
            X, y = make_table(rows, features, rng)
            point = {
                "n_features": features,
                "n_rows": rows,
                "upload": X.size >= UPLOAD_CELLS,
            }
            t0 = time.monotonic()
            try:
                fit_table(client, X, y, dataset_id, args.model)
            except AdmissionRejected as e:
                point.update(
                    status=413,
                    fit_s=round(time.monotonic() - t0, 3),
                    detail=str(e)[:300],
                )
                points.append(point)
                print(json.dumps(point), flush=True)
                break
            except (TabctxError, OSError) as e:
                point.update(status="error", detail=f"{type(e).__name__}: {e}"[:300])
                points.append(point)
                print(json.dumps(point), flush=True)
                break
            point["fit_s"] = round(time.monotonic() - t0, 3)
            X_test, _ = make_table(args.test_rows, features, rng)
            t0 = time.monotonic()
            client.predict(dataset_id, X_test.tolist())
            point.update(status=200, predict_s=round(time.monotonic() - t0, 3))
            points.append(point)
            print(json.dumps(point), flush=True)

    limits = client.limits()
    result = {
        "timestamp": datetime.now(UTC).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "points": points,
        "max_admitted_rows": {
            str(f): max(
                (
                    q["n_rows"]
                    for q in points
                    if q["n_features"] == f and q["status"] == 200
                ),
                default=0,
            )
            for f in args.features
        },
        "limits": limits,
    }
    print(json.dumps(result["max_admitted_rows"]))
    if args.save:
        with open(args.save, "w") as f:
            json.dump(result, f, indent=2)
    return 1 if any(q["status"] == "error" for q in points) else 0


if __name__ == "__main__":
    sys.exit(main())
