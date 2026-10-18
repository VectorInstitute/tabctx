#!/usr/bin/env python3
"""Mixed-workload benchmark: several concurrent users, small AND large
tables, the way a shared deployment is actually used.

Each simulated user loops for --duration seconds: fit a table drawn from a
size mix (mostly small, some medium, occasionally large), then run a few
predicts against it, then move to a new table. Users run concurrently,
each with its own tenant and session, through TabctxClient (so the
gateway's header contract is exercised exactly).

Reports, per size class: fit and predict latency percentiles, 413s
(admission rejections), and errors; overall predict throughput; and, on a
multi-GPU replica, how contexts and activity were spread across devices
(from /readyz). Compare runs before/after a change with --save.

Usage:
  bench_workload.py --base-url http://127.0.0.1:8000/tabctx \
      --users 6 --duration 120 --session-header x-tabctx-session-id \
      --save benchmarks/baselines/workload-a40x4.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from tabctx.client import TabctxClient
from tabctx.errors import AdmissionRejected, TabctxError

# Tables at least this many cells go through the CSV upload path, the way
# a real large-table user sends them; smaller ones are inline JSON.
UPLOAD_CELLS = 200_000

# (name, weight, rows, features). Defaults follow the single-L4 sweep:
# "large" is near what one GPU admits, "small" is a typical user table.
DEFAULT_MIX = [
    ("small", 0.6, 2_000, 20),
    ("medium", 0.3, 10_000, 50),
    ("large", 0.1, 40_000, 100),
]


@dataclass
class ClassStats:
    fit_s: list[float] = field(default_factory=list)
    predict_s: list[float] = field(default_factory=list)
    rejected: int = 0
    errors: list[str] = field(default_factory=list)


def make_table(rows: int, features: int, rng: np.random.Generator):
    X = rng.normal(size=(rows, features)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int).astype(str).tolist()
    return X, y


def to_csv(X: np.ndarray, y: list[str]) -> bytes:
    header = ",".join([f"f{i}" for i in range(X.shape[1])] + ["target"])
    lines = [header]
    lines += [
        ",".join(f"{v:.5g}" for v in row) + f",{label}"
        for row, label in zip(X, y, strict=True)
    ]
    return ("\n".join(lines) + "\n").encode()


def encode_table(X: np.ndarray, y: list[str]):
    """Client-side encoding (CSV or JSON lists), done BEFORE timing a fit:
    building a large CSV is seconds of GIL-bound work that is not the
    server's."""
    if X.size >= UPLOAD_CELLS:
        return ("csv", to_csv(X, y))
    return ("json", (X.tolist(), y))


def fit_table(client: TabctxClient, encoded, dataset_id: str, model):
    kind, data = encoded
    if kind == "csv":
        upload_id = client.upload_csv(data, dataset_id)
        return client.fit_uploaded(upload_id, dataset_id, model=model)
    X, y = data
    return client.fit(X, y, dataset_id=dataset_id, model=model)


def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(p / 100 * len(ordered)))], 3)


def user_loop(user: int, args, mix, stats: dict[str, ClassStats], lock, stop_at):
    rng = np.random.default_rng(args.seed + user)
    pick = random.Random(args.seed + user)
    client = TabctxClient(
        args.base_url,
        tenant_id=f"{args.tenant}-{user}",
        session_header=args.session_header,
        timeout_s=args.timeout,
        max_retries=10,
    )
    n = 0
    while time.monotonic() < stop_at:
        name, _, rows, features = pick.choices(mix, weights=[m[1] for m in mix])[0]
        X, y = make_table(rows, features, rng)
        encoded = encode_table(X, y)
        # A user works with a few tables at a time: re-fitting a slot
        # replaces its previous table, so the cache holds about
        # users x tables_per_user contexts -- "fit once, predict many",
        # not an ever-growing pile of abandoned contexts.
        dataset_id = f"wl-{user}-{n % args.tables_per_user}"
        n += 1
        t0 = time.monotonic()
        try:
            fit_table(client, encoded, dataset_id, args.model)
        except AdmissionRejected:
            with lock:
                stats[name].rejected += 1
            continue
        except (TabctxError, OSError) as e:
            with lock:
                stats[name].errors.append(f"fit: {type(e).__name__}: {e}"[:200])
            continue
        fit_s = time.monotonic() - t0
        X_test, _ = make_table(args.test_rows, features, rng)
        predicts = []
        for _ in range(args.predicts):
            t0 = time.monotonic()
            try:
                client.predict(dataset_id, X_test.tolist())
                predicts.append(time.monotonic() - t0)
            except (TabctxError, OSError) as e:
                with lock:
                    stats[name].errors.append(f"predict: {type(e).__name__}: {e}"[:200])
                break
        with lock:
            stats[name].fit_s.append(fit_s)
            stats[name].predict_s.extend(predicts)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--users", type=int, default=6)
    p.add_argument("--duration", type=float, default=120)
    p.add_argument("--predicts", type=int, default=5, help="per fitted table")
    p.add_argument("--tables-per-user", type=int, default=2)
    p.add_argument("--test-rows", type=int, default=200)
    p.add_argument("--model", default=None)
    p.add_argument("--tenant", default="bench")
    p.add_argument("--session-header", default="x-session-id")
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--mix",
        default=None,
        help="JSON list of [name, weight, rows, features], e.g. '[[\"s\",1,1000,10]]'",
    )
    p.add_argument("--save", default=None, help="write JSON results here")
    args = p.parse_args()
    mix = [tuple(m) for m in json.loads(args.mix)] if args.mix else DEFAULT_MIX

    stats = {m[0]: ClassStats() for m in mix}
    lock = threading.Lock()
    started = time.monotonic()
    stop_at = started + args.duration
    threads = [
        threading.Thread(target=user_loop, args=(u, args, mix, stats, lock, stop_at))
        for u in range(args.users)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - started

    try:
        ready = TabctxClient(args.base_url, timeout_s=args.timeout).ready()
    except (TabctxError, OSError) as e:
        ready = {"error": f"{type(e).__name__}: {e}"}
    total_predicts = sum(len(s.predict_s) for s in stats.values())
    result = {
        "timestamp": datetime.now(UTC).isoformat(),
        "base_url": args.base_url,
        "users": args.users,
        "duration_s": round(elapsed, 1),
        "mix": mix,
        "predict_throughput_per_s": round(total_predicts / elapsed, 2),
        "classes": {
            name: {
                "fits": len(s.fit_s),
                "fit_p50_s": pct(s.fit_s, 50),
                "fit_p95_s": pct(s.fit_s, 95),
                "predicts": len(s.predict_s),
                "predict_p50_s": pct(s.predict_s, 50),
                "predict_p95_s": pct(s.predict_s, 95),
                "rejected_413": s.rejected,
                "errors": len(s.errors),
                "first_errors": s.errors[:3],
            }
            for name, s in stats.items()
        },
        "devices": ready.get("devices"),
        "cache_stats": ready.get("cache_stats"),
        "gpu_capacity_bytes": ready.get("gpu_capacity_bytes"),
    }
    print(json.dumps(result, indent=2))
    if args.save:
        with open(args.save, "w") as f:
            json.dump(result, f, indent=2)
    n_errors = sum(len(s.errors) for s in stats.values())
    fit_times = [v for s in stats.values() for v in s.fit_s]
    print(
        f"\n{args.users} users, {elapsed:.0f}s: {sum(len(s.fit_s) for s in stats.values())} "
        f"fits (median {statistics.median(fit_times) if fit_times else 0:.2f}s), "
        f"{total_predicts} predicts ({result['predict_throughput_per_s']}/s), "
        f"{sum(s.rejected for s in stats.values())} rejected, {n_errors} errors",
        file=sys.stderr,
    )
    return 1 if n_errors else 0


if __name__ == "__main__":
    sys.exit(main())
