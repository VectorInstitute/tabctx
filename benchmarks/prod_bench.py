#!/usr/bin/env python3
"""The prod benchmark: what a gateway user experiences, in three phases.

1. latency -- one user at a time. For each model and table size, two fits
   of a new table (the first may pay one-time costs such as a compile on
   the GPU it lands on; the repeat shows the steady cost), then predicts.
2. mixed -- two users per model at once, each looping fit + predicts. The
   numbers that matter are the tails and the count of calls slower than
   STALL_S: a predict stuck behind another user's fit shows up here.
3. throughput -- tables fit once, then many concurrent predicts against
   them; predicts per minute and, when the Ray dashboard is reachable
   (--dashboard, e.g. a port-forward to localhost:8265), per-GPU
   utilization sampled every second from the tabctx node (NVML, the same
   counter nvidia-smi reports).

Every dataset id carries a run tag, so runs never reuse each other's
cached tables. Requests that hit the gateway's rate limit wait out its
Retry-After and are counted. Accuracy on real data is a separate script
(eval_real_datasets.py).

Usage:
  VP_API_KEY=... prod_bench.py --label 0.13.0 \\
      --save benchmarks/baselines/prod-bench-0.13.0.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import threading
import time
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np

MODELS = ("tabicl-v2", "tabpfn-3.5")
LATENCY_SHAPES = (
    (1_000, 10),
    (10_000, 10),
    (1_000, 100),
    (10_000, 100),
    (1_000, 500),
    (5_000, 500),
)
STALL_S = 5.0
MAX_RATE_LIMIT_RETRIES = 20


class Client:
    """fit/predict through the gateway, waiting out 429s."""

    def __init__(self, base_url: str, api_key: str, timeout_s: float = 900) -> None:
        self._base = base_url.rstrip("/") + "/tabctx"
        self._http = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout_s
        )
        self.rate_limited = 0
        self._lock = threading.Lock()

    def _post(self, path: str, body: dict) -> tuple[int, dict, float]:
        """(status, body, seconds). A network error is status 0, so it is
        counted rather than ending the calling thread; a 429 waits out the
        gateway's Retry-After and tries again, up to MAX_RATE_LIMIT_RETRIES."""
        for _ in range(MAX_RATE_LIMIT_RETRIES):
            t0 = time.monotonic()
            try:
                r = self._http.post(self._base + path, json=body)
            except httpx.HTTPError as e:
                return 0, {"detail": f"{type(e).__name__}: {e}"}, time.monotonic() - t0
            elapsed = time.monotonic() - t0
            if r.status_code != 429:
                try:
                    body = r.json()
                except ValueError:
                    body = {"detail": r.text[:200]}
                if isinstance(body, dict) and "server-timing" in r.headers:
                    body["server_timing"] = r.headers["server-timing"]
                return r.status_code, body, elapsed
            with self._lock:
                self.rate_limited += 1
            time.sleep(_retry_after_s(r.headers.get("retry-after")))
        return 429, {"detail": "still rate limited after retries"}, 0.0

    def fit(
        self, model: str, dataset_id: str, X: np.ndarray, y: list
    ) -> tuple[int, dict, float]:
        return self._post(
            "/fit",
            {
                "model": model,
                "dataset_id": dataset_id,
                "train_X": X.tolist(),
                "train_y": y,
            },
        )

    def predict(
        self, model: str, dataset_id: str, X: np.ndarray
    ) -> tuple[int, dict, float]:
        return self._post(
            "/predict", {"model": model, "dataset_id": dataset_id, "test_X": X.tolist()}
        )


def _retry_after_s(header: str | None) -> float:
    """Seconds to wait from a Retry-After header (seconds form; anything
    else, such as the HTTP-date form, gets the default), kept to [1, 30]."""
    try:
        seconds = float(header) if header is not None else 5.0
    except ValueError:
        seconds = 5.0
    return min(max(seconds, 1.0), 30.0)


def table(
    rng: np.random.Generator, rows: int, features: int
) -> tuple[np.ndarray, list[str]]:
    X = rng.normal(size=(rows, features)).round(3)
    return X, (X[:, 0] + X[:, 1 % features] > 0).astype(int).astype(str).tolist()


def summarize(seconds: list[float]) -> dict:
    if not seconds:
        return {"n": 0}
    xs = sorted(seconds)
    return {
        "n": len(xs),
        "p50_s": round(statistics.median(xs), 3),
        "p95_s": round(xs[math.ceil(0.95 * len(xs)) - 1], 3),  # nearest rank
        "max_s": round(xs[-1], 3),
        "over_stall": sum(x > STALL_S for x in xs),
    }


class GpuPoller:
    """Per-second GPU utilization of the tabctx node, from the Ray dashboard."""

    def __init__(self, dashboard: str | None) -> None:
        self._url = f"{dashboard.rstrip('/')}/nodes?view=summary" if dashboard else None
        self.samples: list[list[int]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def snapshot(self) -> tuple[list[int], list[float]] | None:
        if not self._url:
            return None
        try:
            data = json.load(urllib.request.urlopen(self._url, timeout=5))
            node = next(
                n for n in data["data"]["summary"] if "tabctx" in n.get("hostname", "")
            )
        except Exception:  # noqa: BLE001 -- the dashboard is optional
            return None
        return (
            [g["utilizationGpu"] for g in node["gpus"]],
            [round(g["memoryUsed"] / 1024, 1) for g in node["gpus"]],
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            snap = self.snapshot()
            if snap:
                self.samples.append(snap[0])
            time.sleep(1)

    def __enter__(self) -> GpuPoller:
        if self._url:
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def summary(self) -> dict | None:
        if not self.samples:
            return None
        n, gpus = len(self.samples), len(self.samples[0])
        cols = [[s[i] for s in self.samples] for i in range(gpus)]
        return {
            "samples": n,
            "mean_util_pct": [round(sum(c) / n, 1) for c in cols],
            "busy_share": [round(sum(v > 0 for v in c) / n, 3) for c in cols],
            "all_gpus_mean_util_pct": round(sum(map(sum, cols)) / (n * gpus), 1),
        }


CALL_LOG: list[dict] | None = None  # every call, when --log-calls is given
_CALL_LOG_LOCK = threading.Lock()
_START = time.monotonic()


class Slowest:
    """The slowest calls of a phase with their server timing breakdown
    (timings_ms, from tabctx 0.14), to say where the time went."""

    def __init__(self, keep: int = 10) -> None:
        self._keep = keep
        self._calls: list[tuple[float, str, dict | None, str | None]] = []
        self._lock = threading.Lock()

    def add(self, seconds: float, what: str, body: dict) -> None:
        """Record a call: its client-side seconds, what it was, and the
        server's breakdowns (tabctx's timings_ms; the gateway's
        Server-Timing, when it sends one)."""
        if CALL_LOG is not None:
            with _CALL_LOG_LOCK:
                CALL_LOG.append(
                    {
                        "end_s": round(time.monotonic() - _START, 2),
                        "s": round(seconds, 3),
                        "call": what,
                        "timings_ms": body.get("timings_ms"),
                        "server_timing": body.get("server_timing"),
                    }
                )
        with self._lock:
            self._calls.append(
                (seconds, what, body.get("timings_ms"), body.get("server_timing"))
            )
            self._calls.sort(key=lambda c: -c[0])
            del self._calls[self._keep :]

    def as_list(self) -> list[dict]:
        return [
            {"s": round(s, 2), "call": what, "timings_ms": t, "server_timing": st}
            for s, what, t, st in self._calls
        ]


class Tally:
    """Latencies per model and errors, shared by a phase's threads."""

    def __init__(self) -> None:
        self.seconds: dict[str, list[float]] = {m: [] for m in MODELS}
        self.errors: list[str] = []
        self.slowest = Slowest()
        self._lock = threading.Lock()

    def record(
        self, what: str, model: str, status: int, body: dict, seconds: float
    ) -> None:
        """One call's outcome, in the order Client returns it."""
        self.slowest.add(seconds, f"{what} {model}", body)
        with self._lock:
            if status == 200:
                self.seconds[model].append(seconds)
            else:
                self.errors.append(f"{what} {status}")


def phase_latency(client: Client, tag: str, rng: np.random.Generator) -> dict:
    out: dict = {}
    slowest = Slowest()
    for model in MODELS:
        for rows, features in LATENCY_SHAPES:
            X, y = table(rng, rows, features)
            Xt = rng.normal(size=(100, features)).round(3)
            point: dict = {}
            for attempt in ("first", "repeat"):
                ds = f"bench-{tag}-lat-{model}-{rows}x{features}-{attempt}"
                status, body, fit_s = client.fit(model, ds, X, y)
                slowest.add(fit_s, f"fit {model} {rows}x{features} {attempt}", body)
                point[f"{attempt}_fit_s"] = (
                    round(fit_s, 3) if status == 200 else f"{status}: {body}"
                )
            predicts = [client.predict(model, ds, Xt) for _ in range(3)]
            ok = [p[2] for p in predicts if p[0] == 200]
            point["predict_p50_s"] = (
                round(statistics.median(ok), 3)
                if ok
                else f"{predicts[0][0]}: {predicts[0][1]}"
            )
            out[f"{model} {rows}x{features}"] = point
            print(f"latency {model:11s} {rows:>6}x{features:<4} {point}", flush=True)
    out["slowest"] = slowest.as_list()
    return out


def phase_mixed(
    client: Client, tag: str, duration_s: float, users_per_model: int
) -> dict:
    fits, predicts = Tally(), Tally()
    stop = time.monotonic() + duration_s

    def user(model: str, seed: int) -> None:
        rng = np.random.default_rng(seed)
        n = 0
        while time.monotonic() < stop:
            X, y = table(rng, 10_000, 100)
            ds = f"bench-{tag}-mix-{model}-{seed}-{n % 2}"
            fits.record("fit", model, *client.fit(model, ds, X, y))
            for _ in range(5):
                Xt = rng.normal(size=(1_000, 100)).round(3)
                predicts.record("predict", model, *client.predict(model, ds, Xt))
            n += 1

    threads = [
        threading.Thread(target=user, args=(m, 100 * i + j))
        for i, m in enumerate(MODELS)
        for j in range(users_per_model)
    ]
    [t.start() for t in threads]
    [t.join() for t in threads]
    out = {
        m: {
            "fit": summarize(fits.seconds[m]),
            "predict": summarize(predicts.seconds[m]),
        }
        for m in MODELS
    } | {
        "errors": (fits.errors + predicts.errors)[:20],
        "n_errors": len(fits.errors) + len(predicts.errors),
        "slowest_fits": fits.slowest.as_list(),
        "slowest_predicts": predicts.slowest.as_list(),
    }
    print(f"mixed {json.dumps(out)}", flush=True)
    return out


def phase_throughput(
    client: Client, tag: str, duration_s: float, threads_n: int, poller: GpuPoller
) -> dict:
    rng = np.random.default_rng(7)
    tables = [(m, f"bench-{tag}-tp-{m}-{i}") for i in range(4) for m in MODELS]
    for model, ds in tables:
        X, y = table(rng, 10_000, 100)
        status, body, _ = client.fit(model, ds, X, y)
        if status != 200:
            raise RuntimeError(
                f"throughput setup: fit {model} {ds} got {status}: {body}"
            )
    tests = [rng.normal(size=(1_000, 100)).round(3) for _ in range(4)]
    tally = Tally()
    stop = time.monotonic() + duration_s

    def worker(i: int) -> None:
        rnd = random.Random(i)
        while time.monotonic() < stop:
            model, ds = rnd.choice(tables)
            tally.record(
                "predict", model, *client.predict(model, ds, tests[rnd.randrange(4)])
            )

    started = time.monotonic()
    with poller:
        ts = [threading.Thread(target=worker, args=(i,)) for i in range(threads_n)]
        [t.start() for t in ts]
        [t.join() for t in ts]
    wall_s = time.monotonic() - started
    out = {
        m: summarize(tally.seconds[m])
        | {"per_min": round(len(tally.seconds[m]) / wall_s * 60, 1)}
        for m in MODELS
    } | {
        "n_errors": len(tally.errors),
        "gpu": poller.summary(),
        "slowest": tally.slowest.as_list(),
    }
    print(f"throughput {json.dumps(out)}", flush=True)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--base-url", default="https://proxy.vectorinstitute.ai/v1")
    p.add_argument("--label", required=True, help="what is being measured, e.g. 0.13.0")
    p.add_argument("--phases", nargs="+", default=["latency", "mixed", "throughput"])
    p.add_argument("--mixed-s", type=float, default=240)
    p.add_argument("--users-per-model", type=int, default=2)
    p.add_argument("--throughput-s", type=float, default=180)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--dashboard", default="http://localhost:8265")
    p.add_argument("--save")
    p.add_argument(
        "--log-calls", action="store_true", help="save every call with its timings_ms"
    )
    args = p.parse_args()
    global CALL_LOG
    if args.log_calls:
        CALL_LOG = []

    tag = uuid.uuid4().hex[:6]
    client = Client(args.base_url, os.environ["VP_API_KEY"])
    poller = GpuPoller(args.dashboard)
    result: dict = {
        "label": args.label,
        "tag": tag,
        "timestamp": datetime.now(UTC).isoformat(),
        "base_url": args.base_url,
        "stall_s": STALL_S,
    }
    if "latency" in args.phases:
        result["latency"] = phase_latency(client, tag, np.random.default_rng(1))
    if "mixed" in args.phases:
        result["mixed"] = phase_mixed(client, tag, args.mixed_s, args.users_per_model)
    if "throughput" in args.phases:
        result["throughput"] = phase_throughput(
            client, tag, args.throughput_s, args.threads, poller
        )
    result["rate_limited"] = client.rate_limited
    if CALL_LOG is not None:
        result["calls"] = CALL_LOG
    if args.save:
        Path(args.save).write_text(json.dumps(result, indent=2) + "\n")
    print(f"done; rate-limited requests: {client.rate_limited}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
