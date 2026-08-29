#!/usr/bin/env python3
"""Concurrency sweep benchmark for a running tabctx deployment, using the
same vocabulary this org already uses for LLM benchmarking
(inference-platform/benchmarks/bench.py: concurrency sweep, throughput,
latency percentiles, JSON baselines) so results are directly comparable
across model/engine types, adapted for what tabctx's workload actually is:

  LLM concept                    tabctx analog
  ----------------------------   --------------------------------------------
  TTFT (time to first token)     cold_fit_latency_ms -- one-time cost before
                                  a NEW tenant's context is usable at all.
                                  There's no streaming equivalent of "first
                                  token" here; fit() is the all-or-nothing
                                  gate a tenant must clear before anything
                                  is servable, so it plays TTFT's role.
  decode / inter-token latency   warm_predict_latency_ms -- recurring cost
                                  of a request against an ALREADY-cached
                                  context. Not streamed (no per-token
                                  granularity), so this is a whole-response
                                  latency, not a per-token one.
  output tok/s (throughput)      warm_predict_ops_per_sec -- steady-state
                                  throughput of predict() calls against
                                  pre-warmed, distinct tenants' contexts.
  concurrency sweep              same concept: run at c=1,2,4,8,16... and
                                  see how throughput/latency respond.

Why sweep with DISTINCT tenants per concurrency level (not N threads
hammering one dataset_id): the question this answers is "how does the
service behave as MORE DIFFERENT USERS show up," which is what actually
matters for capacity planning -- not lock contention on a single shared
resource (that's covered separately by probe_scale.py's race tests).

v1 is known to serialize all GPU work per replica via one coarse lock (see
tabctx's README) -- unlike an LLM engine's continuous batching, throughput
here is NOT expected to scale with concurrency past c=1; if anything it
should plateau or degrade (queueing) as c grows. That is the headline
number this benchmark exists to track over time: re-run it after any
future concurrency-model change and compare against a saved baseline to
prove whether it actually improved, the same way inference-platform's LLM
benchmarks compare before/after a config change.

Usage:
  bench_concurrency.py --base-url http://127.0.0.1:8000 \
      --concurrency 1 2 4 8 16 --duration 20 \
      --save-baseline benchmarks/baselines/v0.5.0.json
"""

import argparse
import concurrent.futures
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from sklearn.datasets import make_classification

N_TRAIN, N_TEST, N_FEATURES = 500, 20, 15  # default shape for the concurrency sweep


def _post_status(url, payload, timeout, session_id=None):
    # session_id carries the dataset_id in the x-session-id header so the
    # consistent-hash router (v0.6.0) pins all of one tenant's requests to
    # the replica holding its cached context. Harmless at num_replicas=1;
    # required for the benchmark to be valid at num_replicas >= 2.
    headers = {"Content-Type": "application/json"}
    if session_id is not None:
        headers["x-session-id"] = session_id
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode())
        except json.JSONDecodeError:
            body = {}
        return e.code, body


def _get(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _make_tenant_data(seed, n_train=N_TRAIN, n_test=N_TEST, n_features=N_FEATURES):
    X, y = make_classification(
        n_samples=n_train + n_test, n_features=n_features,
        n_informative=max(2, n_features // 2), n_classes=2, random_state=seed,
    )
    return X[:n_train].tolist(), y[:n_train].tolist(), X[n_train:].tolist()


def _percentiles(values):
    if not values:
        return {"p50": None, "p95": None, "p99": None, "mean": None, "max": None}
    s = sorted(values)

    def pct(p):
        idx = min(len(s) - 1, int(len(s) * p))
        return s[idx]

    return {
        "p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99),
        "mean": statistics.mean(s), "max": max(s),
    }


def measure_cold_fit_latency(base_url, timeout, tenant_seed_offset, n_samples=5):
    """TTFT-equivalent: one-shot fit() latency for a brand-new tenant,
    measured serially (this is the "onboarding a new user" cost, not a
    concurrency-swept quantity -- fit() always contends on the same lock
    regardless of how many other predict() calls are in flight, so its
    latency mostly reflects checkpoint-reload + encode cost, not queueing)."""
    latencies = []
    for i in range(n_samples):
        train_X, train_y, _ = _make_tenant_data(tenant_seed_offset + i)
        start = time.monotonic()
        dataset_id = f"bench-cold-{tenant_seed_offset}-{i}"
        status, _ = _post_status(
            f"{base_url}/v1/tabctx/fit",
            {"train_X": train_X, "train_y": train_y, "dataset_id": dataset_id},
            timeout, session_id=dataset_id,
        )
        latencies.append((time.monotonic() - start) * 1000)
        if status != 200:
            print(f"[warn] cold fit sample {i} returned status {status}", file=sys.stderr)
    return _percentiles(latencies)


def run_concurrency_level(base_url, timeout, concurrency, duration_s, tenant_seed_offset):
    """Pre-warms `concurrency` distinct tenant contexts, then hammers
    predict() against them (one dedicated tenant per worker thread) for
    duration_s, measuring steady-state warm-predict throughput/latency."""
    dataset_ids = []
    test_sets = []
    for i in range(concurrency):
        train_X, train_y, test_X = _make_tenant_data(tenant_seed_offset + i)
        dataset_id = f"bench-warm-{tenant_seed_offset}-{i}"
        status, _ = _post_status(
            f"{base_url}/v1/tabctx/fit",
            {"train_X": train_X, "train_y": train_y, "dataset_id": dataset_id},
            timeout, session_id=dataset_id,
        )
        if status != 200:
            raise RuntimeError(f"warmup fit failed for tenant {i}: status {status}")
        dataset_ids.append(dataset_id)
        test_sets.append(test_X)

    def worker(idx):
        # Only successful (200) requests count as completed work for
        # throughput/latency purposes -- a 503 backpressure rejection
        # returns almost instantly (Ray Serve rejects it before any GPU
        # work starts), so counting it as a "fast op" would inflate
        # ops/sec and deflate latency percentiles, making an overloaded,
        # mostly-rejecting concurrency level look BETTER than a healthy
        # one. Found by inspection: an early version of this script did
        # exactly that at high concurrency and produced a misleading
        # "peak throughput" number driven almost entirely by rejections.
        success_latencies = []
        backpressure = 0
        errors = 0
        end_at = time.monotonic() + duration_s
        while time.monotonic() < end_at:
            start = time.monotonic()
            try:
                status, _ = _post_status(
                    f"{base_url}/v1/tabctx/predict",
                    {"dataset_id": dataset_ids[idx], "test_X": test_sets[idx]},
                    timeout, session_id=dataset_ids[idx],
                )
                elapsed_ms = (time.monotonic() - start) * 1000
                if status == 200:
                    success_latencies.append(elapsed_ms)
                elif status == 503:
                    backpressure += 1
                else:
                    errors += 1
            except Exception:  # noqa: BLE001
                errors += 1
        return success_latencies, backpressure, errors

    wall_start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(worker, range(concurrency)))
    wall_s = time.monotonic() - wall_start

    success_latencies = [lat for lats, _, _ in results for lat in lats]
    total_backpressure = sum(bp for _, bp, _ in results)
    total_errors = sum(err for _, _, err in results)
    total_attempts = len(success_latencies) + total_backpressure + total_errors

    return {
        "concurrency": concurrency,
        "duration_s": round(wall_s, 2),
        "n_successful_requests": len(success_latencies),
        "n_attempted_requests": total_attempts,
        "warm_predict_ops_per_sec": round(len(success_latencies) / wall_s, 3) if wall_s else 0,
        "warm_predict_latency_ms": _percentiles(success_latencies),
        "backpressure_503": total_backpressure,
        "backpressure_rate": round(total_backpressure / total_attempts, 3) if total_attempts else 0,
        "errors": total_errors,
    }


def run_shape_point(base_url, timeout, n_train, n_test, n_features, seed, n_samples=8):
    """Single-tenant, single-concurrency (c=1) measurement at one (n_train,
    n_features) shape: fit once, then predict n_samples times against the
    same warm context, holding n_test fixed. Isolates the effect of table
    shape on latency/memory from the effect of concurrency (see
    run_concurrency_level for the concurrency-holding-shape-fixed sweep) --
    the two are deliberately swept separately rather than as a full 2D grid,
    to keep real A100 time bounded."""
    train_X, train_y, test_X = _make_tenant_data(seed, n_train, n_test, n_features)
    dataset_id = f"bench-shape-{n_train}-{n_features}-{seed}"

    fit_start = time.monotonic()
    status, body = _post_status(
        f"{base_url}/v1/tabctx/fit",
        {"train_X": train_X, "train_y": train_y, "dataset_id": dataset_id},
        timeout, session_id=dataset_id,
    )
    fit_ms = (time.monotonic() - fit_start) * 1000
    if status != 200:
        return {"n_train": n_train, "n_test": n_test, "n_features": n_features,
                "fit_status": status, "fit_error": body, "fit_latency_ms": fit_ms}

    predict_latencies = []
    for _ in range(n_samples):
        start = time.monotonic()
        status, body = _post_status(
            f"{base_url}/v1/tabctx/predict",
            {"dataset_id": dataset_id, "test_X": test_X},
            timeout, session_id=dataset_id,
        )
        predict_latencies.append((time.monotonic() - start) * 1000)
        if status != 200:
            print(f"[warn] shape ({n_train},{n_features}) predict returned {status}: {body}",
                  file=sys.stderr)

    readyz = _get(f"{base_url}/readyz", timeout)
    real_gpu = readyz.get("real_gpu_memory") or {}

    return {
        "n_train": n_train, "n_test": n_test, "n_features": n_features,
        "fit_status": 200,
        "fit_latency_ms": round(fit_ms, 1),
        "warm_predict_latency_ms": _percentiles(predict_latencies),
        "real_gpu_allocated_mb_after": real_gpu.get("allocated_mb"),
    }


def run_shape_sweep(base_url, timeout, n_train, n_test, feature_counts):
    print(f"\n=== Shape sweep: n_train={n_train}, n_test={n_test}, "
          f"n_features in {feature_counts} (concurrency=1, isolates column-count effect) ===")
    print(f"{'n_features':>10} {'fit_ms':>9} {'pred_p50_ms':>12} {'pred_p95_ms':>12} "
          f"{'pred_max_ms':>12} {'real_gpu_mb':>12}")
    results = []
    for i, n_features in enumerate(feature_counts):
        r = run_shape_point(base_url, timeout, n_train, n_test, n_features, seed=800_000 + i)
        results.append(r)
        if r["fit_status"] != 200:
            print(f"{n_features:>10}   FIT FAILED: {r.get('fit_error')}")
            continue
        lat = r["warm_predict_latency_ms"]
        print(f"{n_features:>10} {r['fit_latency_ms']:>9.1f} {lat['p50']:>12.1f} "
              f"{lat['p95']:>12.1f} {lat['max']:>12.1f} "
              f"{r['real_gpu_allocated_mb_after']:>12.1f}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--duration", type=int, default=20, help="seconds per concurrency level")
    p.add_argument("--skip-concurrency-sweep", action="store_true",
                   help="skip the concurrency sweep (useful when only --sweep-features is wanted)")
    p.add_argument("--sweep-features", type=int, nargs="+", default=None,
                   help="if given, also run a column-count sweep at concurrency=1 "
                        "(e.g. --sweep-features 15 100 300 500)")
    p.add_argument("--sweep-train-rows", type=int, default=500,
                   help="n_train held fixed during --sweep-features")
    p.add_argument("--sweep-test-rows", type=int, default=20,
                   help="n_test held fixed during --sweep-features")
    p.add_argument("--save-baseline", default=None, help="path to write JSON results")
    args = p.parse_args()

    readyz = _get(f"{args.base_url}/readyz", args.timeout)
    print(f"[info] target: device={readyz.get('device')} "
          f"cache_stats={readyz.get('cache_stats')}")

    cold_fit = None
    levels = []
    if not args.skip_concurrency_sweep:
        print("\n=== cold_fit_latency_ms (TTFT-equivalent: onboarding a new tenant) ===")
        cold_fit = measure_cold_fit_latency(args.base_url, args.timeout, tenant_seed_offset=900_000)
        print(f"  p50={cold_fit['p50']:.1f}  p95={cold_fit['p95']:.1f}  "
              f"p99={cold_fit['p99']:.1f}  mean={cold_fit['mean']:.1f}  max={cold_fit['max']:.1f}")

        print(f"\n{'concurrency':>11} {'ops/sec':>9} {'p50_ms':>9} {'p95_ms':>9} "
              f"{'p99_ms':>9} {'max_ms':>9} {'success':>8} {'503_rate':>9} {'errors':>7}")
        for i, c in enumerate(args.concurrency):
            result = run_concurrency_level(
                args.base_url, args.timeout, c, args.duration, tenant_seed_offset=i * 100_000
            )
            lat = result["warm_predict_latency_ms"]
            p50 = f"{lat['p50']:.1f}" if lat["p50"] is not None else "n/a"
            p95 = f"{lat['p95']:.1f}" if lat["p95"] is not None else "n/a"
            p99 = f"{lat['p99']:.1f}" if lat["p99"] is not None else "n/a"
            mx = f"{lat['max']:.1f}" if lat["max"] is not None else "n/a"
            print(f"{c:>11} {result['warm_predict_ops_per_sec']:>9.2f} "
                  f"{p50:>9} {p95:>9} {p99:>9} {mx:>9} "
                  f"{result['n_successful_requests']:>8} {result['backpressure_rate']:>9.1%} "
                  f"{result['errors']:>7}")
            levels.append(result)

        # ops/sec ONLY counts successful requests (see run_concurrency_level's
        # worker() docstring) -- a level with a high backpressure_rate is
        # overloaded, not fast, even if its ops/sec number looks fine in
        # isolation. Only consider levels with a low rejection rate as
        # candidates for "peak" throughput.
        healthy_levels = [lv for lv in levels if lv["backpressure_rate"] < 0.05] or levels
        peak = max(healthy_levels, key=lambda r: r["warm_predict_ops_per_sec"])
        print(f"\n[info] peak warm_predict_ops_per_sec={peak['warm_predict_ops_per_sec']} "
              f"at concurrency={peak['concurrency']}")
        if levels[-1]["warm_predict_ops_per_sec"] < levels[0]["warm_predict_ops_per_sec"] * 1.2:
            print("[info] throughput did NOT scale meaningfully with concurrency -- consistent "
                  "with v1's documented single coarse lock serializing GPU work per replica. "
                  "This is the number to watch for improvement after any future concurrency work.")

    shape_sweep_results = None
    if args.sweep_features:
        shape_sweep_results = run_shape_sweep(
            args.base_url, args.timeout,
            args.sweep_train_rows, args.sweep_test_rows, args.sweep_features,
        )

    payload = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "shape": {"n_train": N_TRAIN, "n_test": N_TEST, "n_features": N_FEATURES},
        "cold_fit_latency_ms": cold_fit,
        "concurrency_levels": levels,
        "feature_count_sweep": shape_sweep_results,
    }
    if args.save_baseline:
        with open(args.save_baseline, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[info] saved baseline to {args.save_baseline}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
