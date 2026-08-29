#!/usr/bin/env python3
"""tabctx local serving vs. PriorLabs' hosted TabPFN API, on the workload
tabctx exists for: fit ONCE, then predict repeatedly against the same
training context.

What this measures per side, for each (n_train, n_features) shape:

- cold_fit_s: time until the context is usable at all
  (local: TabPFN fit() with kv cache; API: prepare upload + 2 PUTs + /fit)
- warm_predict_s p50/p95: repeated predicts against the fitted context
  (local: in-process cached-context predict; API: prepare_test_upload +
  PUT + /predict -- the provider caches the fitted context server-side,
  but every predict still pays upload + network round trips)

READ THIS BEFORE QUOTING NUMBERS: the two sides run on different
hardware by construction -- the API runs on PriorLabs' cloud GPUs, the
local side runs on whatever this machine has (a laptop CPU, in the run
that produced the first saved baseline). So this is NOT a model-speed
comparison; it is a SERVING-ARCHITECTURE comparison: what does a repeat
query cost when the context sits in-process (tabctx) vs. behind a
per-request upload+HTTP flow (hosted API)? On a GPU box the local side
only gets faster; the API's floor is the network + upload path.

Auth: export TABPFN_TOKEN (from https://ux.priorlabs.ai/account). The
token is read from the environment only -- never hardcode it.

Usage:
  export TABPFN_TOKEN=...
  python benchmarks/bench_vs_hosted_api.py --save-baseline benchmarks/baselines/vs-api.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone

import httpx
from sklearn.datasets import make_classification

API_BASE = "https://api.priorlabs.ai"
SHAPES = [(500, 15), (2000, 50)]  # (n_train, n_features)
N_TEST = 20
N_WARM_PREDICTS = 8


def _pct(values, p):
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * p))]


def _summarize(values):
    return {
        "p50_s": round(_pct(values, 0.50), 3),
        "p95_s": round(_pct(values, 0.95), 3),
        "mean_s": round(statistics.mean(values), 3),
        "n": len(values),
    }


def _to_csv(rows, header):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue().encode()


def make_data(n_train, n_features, seed):
    X, y = make_classification(
        n_samples=n_train + N_TEST,
        n_features=n_features,
        n_informative=max(2, n_features // 2),
        random_state=seed,
    )
    return X[:n_train], y[:n_train], X[n_train:]


# ---- local side: tabctx engine with the TabPFN backend --------------------


def bench_local(X_train, y_train, X_test):
    from tabctx.serve.factory import ServeSettings, build_engine

    built = build_engine(ServeSettings(backend="tabpfn"))
    start = time.monotonic()
    dataset_id = built.engine.fit(
        X_train.tolist(), [str(v) for v in y_train], task="classification"
    )
    cold_fit_s = time.monotonic() - start

    latencies = []
    for _ in range(N_WARM_PREDICTS):
        start = time.monotonic()
        built.engine.predict(dataset_id, X_test.tolist(), return_proba=True)
        latencies.append(time.monotonic() - start)
    return cold_fit_s, latencies


# ---- API side: PriorLabs' documented client recipe ------------------------


class HostedApi:
    def __init__(self, token):
        self._client = httpx.Client(
            base_url=API_BASE,
            headers={"Authorization": f"Bearer {token}"},
            timeout=300.0,
        )

    def _post(self, path, payload):
        resp = self._client.post(path, json=payload)
        resp.raise_for_status()
        return resp.json()

    def _upload(self, info, content):
        httpx.put(
            info["signed_urls"][0],
            content=content,
            headers=info.get("required_headers") or {},
            timeout=300.0,
        ).raise_for_status()

    def fit(self, X_train, y_train, feature_names):
        prep = self._post(
            "/tabpfn/prepare_train_set_upload",
            {"x_train_info": {"format": "csv"}, "y_train_info": {"format": "csv"}},
        )
        self._upload(prep["x_train_info"], _to_csv(X_train.tolist(), feature_names))
        self._upload(prep["y_train_info"], _to_csv([[v] for v in y_train], ["target"]))
        fit_resp = self._post(
            "/tabpfn/fit",
            {
                "train_set_upload_id": prep["train_set_upload_id"],
                "task": "classification",
            },
        )
        return fit_resp["fitted_train_set_id"]

    def predict(self, fitted_id, X_test, feature_names):
        prep = self._post(
            "/tabpfn/prepare_test_set_upload",
            {"fitted_train_set_id": fitted_id, "x_test_info": {"format": "csv"}},
        )
        self._upload(prep["x_test_info"], _to_csv(X_test.tolist(), feature_names))
        return self._post(
            "/tabpfn/predict",
            {
                "test_set_upload_id": prep["test_set_upload_id"],
                "fitted_train_set_id": fitted_id,
                "task_config": {
                    "task": "classification",
                    "predict_params": {"output_type": "probas"},
                },
            },
        )


def bench_api(token, X_train, y_train, X_test, n_features):
    api = HostedApi(token)
    feature_names = [f"f{i}" for i in range(n_features)]

    start = time.monotonic()
    fitted_id = api.fit(X_train, y_train, feature_names)
    cold_fit_s = time.monotonic() - start

    latencies = []
    for _ in range(N_WARM_PREDICTS):
        start = time.monotonic()
        api.predict(fitted_id, X_test, feature_names)
        latencies.append(time.monotonic() - start)
    return cold_fit_s, latencies


# ---- main -----------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save-baseline", default=None)
    args = p.parse_args()

    token = os.environ.get("TABPFN_TOKEN")
    if not token:
        print("export TABPFN_TOKEN first (https://ux.priorlabs.ai/account)", file=sys.stderr)
        return 1

    results = []
    for seed, (n_train, n_features) in enumerate(SHAPES):
        X_train, y_train, X_test = make_data(n_train, n_features, seed)
        print(f"\n=== shape: {n_train} rows x {n_features} features, "
              f"{N_TEST} test rows, {N_WARM_PREDICTS} warm predicts ===")

        local_fit_s, local_lat = bench_local(X_train, y_train, X_test)
        print(f"  tabctx local  cold_fit={local_fit_s:6.2f}s  "
              f"warm p50={_pct(local_lat, .5):6.3f}s  p95={_pct(local_lat, .95):6.3f}s")

        api_fit_s, api_lat = bench_api(token, X_train, y_train, X_test, n_features)
        print(f"  hosted API    cold_fit={api_fit_s:6.2f}s  "
              f"warm p50={_pct(api_lat, .5):6.3f}s  p95={_pct(api_lat, .95):6.3f}s")

        ratio = _pct(api_lat, 0.5) / max(_pct(local_lat, 0.5), 1e-9)
        print(f"  warm-predict p50 ratio (API/local): {ratio:.1f}x")

        results.append({
            "n_train": n_train, "n_features": n_features,
            "n_test": N_TEST, "n_warm_predicts": N_WARM_PREDICTS,
            "tabctx_local": {"cold_fit_s": round(local_fit_s, 3),
                             "warm_predict": _summarize(local_lat)},
            "hosted_api": {"cold_fit_s": round(api_fit_s, 3),
                           "warm_predict": _summarize(api_lat)},
        })

    if args.save_baseline:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "note": (
                "Serving-architecture comparison, not model-speed: local side "
                "ran on this machine's hardware (see 'local_hardware'), the "
                "API on PriorLabs' cloud. Warm predict is the tabctx use case "
                "(fit once, query repeatedly)."
            ),
            "local_hardware": _local_hardware(),
            "results": results,
        }
        with open(args.save_baseline, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[info] saved {args.save_baseline}")
    return 0


def _local_hardware():
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda: {torch.cuda.get_device_name(0)}"
    except ImportError:
        pass
    import platform

    return f"cpu: {platform.machine()} ({platform.processor() or platform.system()})"


if __name__ == "__main__":
    sys.exit(main())
