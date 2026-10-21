#!/usr/bin/env python3
"""Table-size sweep through the Vector inference gateway.

The same steps as sweep_admission.py, but as a gateway user sees them:
bearer auth, the gateway's own tenant and session headers, and its request
body cap. A 413 can come from that cap or from tabctx's admission gate, so
each rejection records its detail. Comparing the output with a direct
sweep shows what the gateway adds to fit and predict latency, and where
its body cap stops a table before admission would.

Usage:
  VP_API_KEY=... sweep_gateway.py --model tabicl-v2 \
      --save benchmarks/baselines/sweep-gateway-a40x4-tabicl-v2.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_workload import encode_table, make_table  # noqa: E402
from sweep_admission import ROWS  # noqa: E402


class Gateway:
    """The three tabctx endpoints, called the way a gateway client calls them."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: float):
        self._base = base_url.rstrip("/")
        self._auth = {"Authorization": f"Bearer {api_key}"}
        self._model = model
        self._timeout = timeout_s

    def _send(self, path: str, body: bytes, headers: dict) -> tuple[int, dict]:
        req = urllib.request.Request(
            self._base + path, data=body, headers=self._auth | headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                return e.code, json.loads(raw)
            except ValueError:
                return e.code, {"detail": raw[:300].decode(errors="replace")}

    def _post_json(self, path: str, payload: dict) -> tuple[int, dict]:
        body = json.dumps(payload | {"model": self._model}).encode()
        return self._send(path, body, {"Content-Type": "application/json"})

    def fit(
        self, encoded, dataset_id: str, task: str = "classification"
    ) -> tuple[int, dict]:
        kind, data = encoded
        if kind == "csv":
            status, out = self._send(
                "/tabctx/upload",
                data,
                {
                    "Content-Type": "text/csv",
                    "x-tabctx-model": self._model,
                    "x-tabctx-dataset-id": dataset_id,
                },
            )
            if status != 200:
                return status, out
            payload = {"dataset_id": dataset_id, "train_upload_id": out["upload_id"]}
        else:
            X, y = data
            payload = {"dataset_id": dataset_id, "train_X": X, "train_y": y}
        return self._post_json("/tabctx/fit", payload | {"task": task})

    def predict(
        self, dataset_id: str, X_test: list, return_proba: bool = False
    ) -> tuple[int, dict]:
        return self._post_json(
            "/tabctx/predict",
            {"dataset_id": dataset_id, "test_X": X_test, "return_proba": return_proba},
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="https://proxy.vectorinstitute.ai/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--features", type=int, nargs="+", default=[10, 100, 500, 1000])
    p.add_argument("--max-rows", type=int, default=500_000)
    p.add_argument("--test-rows", type=int, default=100)
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--save", default=None)
    args = p.parse_args()

    gateway = Gateway(args.base_url, os.environ["VP_API_KEY"], args.model, args.timeout)
    rng = np.random.default_rng(0)
    points = []
    for features in args.features:
        dataset_id = f"sweep-gw-{args.model}-f{features}"
        for rows in (r for r in ROWS if r <= args.max_rows):
            X, y = make_table(rows, features, rng)
            encoded = encode_table(X, y)
            body_bytes = len(encoded[1]) if encoded[0] == "csv" else None
            point = {
                "n_features": features,
                "n_rows": rows,
                "upload": encoded[0] == "csv",
                "upload_bytes": body_bytes,
            }
            t0 = time.monotonic()
            status, out = gateway.fit(encoded, dataset_id)
            point["fit_s"] = round(time.monotonic() - t0, 3)
            if status != 200:
                point.update(status=status, detail=str(out.get("detail"))[:300])
                points.append(point)
                print(json.dumps(point), flush=True)
                break
            X_test, _ = make_table(args.test_rows, features, rng)
            t0 = time.monotonic()
            status, out = gateway.predict(dataset_id, X_test.tolist())
            point["predict_s"] = round(time.monotonic() - t0, 3)
            point["status"] = 200 if status == 200 else f"predict_{status}"
            if status != 200:
                point["detail"] = str(out.get("detail"))[:300]
            points.append(point)
            print(json.dumps(point), flush=True)

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
    }
    print(json.dumps(result["max_admitted_rows"]))
    if args.save:
        with open(args.save, "w") as f:
            json.dump(result, f, indent=2)
    bad = [q for q in points if q["status"] not in (200, 413)]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
