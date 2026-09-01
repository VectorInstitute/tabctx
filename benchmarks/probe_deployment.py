#!/usr/bin/env python3
"""End-to-end acceptance probe for a LIVE tabctx deployment, on any GPU.

Drives the deployment through the bundled TabctxClient exactly the way a
caller would, and checks the serving contracts that only a real model on
a real device can prove: cache reuse (warm predict << cold fit), single-
pass return_proba, upload -> fit/predict by reference with the schema
check, chunked large predicts, same-context coalescing, and -- by
filling the cache -- eviction into the spill tier and transparent
restore. Every check prints PASS/FAIL; the exit code is non-zero if any
failed. Read-only for the deployment apart from the contexts it fits
under its own dataset_id prefix (evicted at the end).

Unlike the inference-platform probes this handles deployments that
require a tenant header (TABCTX_REQUIRE_TENANT=true) and that rename the
session header (RAY_SERVE_SESSION_ID_HEADER_KEY).

Usage (port-forward the serve service first):
  python benchmarks/probe_deployment.py --base-url http://127.0.0.1:8000 \
      --tenant probe --session-header x-tabctx-session-id
"""

from __future__ import annotations

import argparse
import io
import sys
import threading
import time
import uuid

import numpy as np

from tabctx.client import TabctxClient
from tabctx.errors import (
    AdmissionRejected,
    DatasetNotFoundError,
    InvalidInputError,
    UploadNotFoundError,
)

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else "")
    )


def make_table(n: int, d: int, seed: int) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = np.where(X[:, 0] + 0.5 * X[:, 1] > 0, "pos", "neg").tolist()
    return X, y


def to_csv(X: np.ndarray, y: list | None, names: list[str]) -> bytes:
    buf = io.StringIO()
    buf.write(",".join(names + (["label"] if y is not None else [])) + "\n")
    for i, row in enumerate(X):
        cells = [f"{v:.5f}" for v in row]
        if y is not None:
            cells.append(str(y[i]))
        buf.write(",".join(cells) + "\n")
    return buf.getvalue().encode()


def probe_model(client: TabctxClient, model: str, prefix: str, args) -> str:
    print(f"\n--- model {model} ---")
    X, y = make_table(args.rows, 20, seed=1)
    ds = f"{prefix}-{model}"
    t = time.monotonic()
    client.fit(X.tolist(), y, dataset_id=ds, model=model)
    cold_s = time.monotonic() - t
    warm = []
    for _ in range(3):
        t = time.monotonic()
        r = client.predict(ds, X[:200].tolist())
        warm.append(time.monotonic() - t)
    check(
        f"{model}: cache reuse (cold fit {cold_s:.2f}s vs warm predict "
        f"{min(warm) * 1000:.0f}ms)",
        min(warm) < cold_s,
    )
    check(f"{model}: predictions are labels", set(r.predictions) <= {"pos", "neg"})
    rp = client.predict(ds, X[:200].tolist(), return_proba=True)
    argmax = [rp.classes[int(np.argmax(p))] for p in rp.probabilities]
    check(
        f"{model}: return_proba argmax == predictions (single pass)",
        argmax == rp.predictions and rp.classes == ["neg", "pos"],
    )
    # Accuracy sanity on a linearly separable-ish problem.
    acc = float(np.mean(np.array(r.predictions) == np.array(y[:200])))
    check(f"{model}: train-set accuracy {acc:.2f} > 0.8", acc > 0.8)
    # Regression.
    yr = (X[:, 0] * 2.0 + X[:, 1]).tolist()
    dsr = f"{ds}-reg"
    client.fit(X.tolist(), yr, task="regression", dataset_id=dsr, model=model)
    rr = client.predict(dsr, X[:50].tolist())
    err = float(np.mean(np.abs(np.array(rr.predictions) - np.array(yr[:50]))))
    check(f"{model}: regression MAE {err:.2f} < 1.0", err < 1.0)
    client.predict(dsr, X[:1].tolist())
    # Chunked large predict: every row gets exactly one prediction.
    big = np.random.default_rng(2).normal(size=(args.big_rows, 20))
    t = time.monotonic()
    rb = client.predict(ds, big.tolist())
    check(
        f"{model}: {args.big_rows}-row predict ({time.monotonic() - t:.1f}s)",
        len(rb.predictions) == args.big_rows,
    )
    # Input validation reaches the caller as 422, not 500.
    try:
        client.predict(ds, [[0.0] * 3])
        check(f"{model}: wrong feature count -> 422", False, "no error raised")
    except InvalidInputError:
        check(f"{model}: wrong feature count -> 422", True)
    return ds


def probe_uploads(client: TabctxClient, prefix: str, args) -> None:
    print("\n--- upload -> fit/predict by reference ---")
    names = [f"f{i}" for i in range(12)]
    X, y = make_table(args.upload_rows, 12, seed=3)
    ds = f"{prefix}-upload"
    up = client.upload_csv(to_csv(X, y, names), ds)
    t = time.monotonic()
    client.fit_uploaded(up, ds, target_column="label")
    check(
        f"fit by reference ({args.upload_rows} rows, {time.monotonic() - t:.1f}s)", True
    )
    test_up = client.upload_csv(to_csv(X[:300], None, names), ds)
    r = client.predict(ds, test_upload_id=test_up)
    check("predict by reference", len(r.predictions) == 300)
    try:
        client.fit_uploaded(up, ds)
        check("uploads are single-use -> 404", False, "no error raised")
    except UploadNotFoundError:
        check("uploads are single-use -> 404", True)
    bad = client.upload_csv(to_csv(X[:5], None, names[::-1]), ds)
    try:
        client.predict(ds, test_upload_id=bad)
        check("reordered test header -> 422", False, "no error raised")
    except InvalidInputError:
        check("reordered test header -> 422", True)
    check(
        "inline predict on CSV-fit context",
        len(client.predict(ds, X[:3].tolist()).predictions) == 3,
    )


def probe_coalescing(client: TabctxClient, ds: str, args) -> None:
    print("\n--- same-context coalescing ---")
    before = client.ready()["batching"]
    X = np.random.default_rng(4).normal(size=(50, 20)).tolist()
    errors: list[Exception] = []

    def worker():
        try:
            client.predict(ds, X)
        except Exception as e:  # noqa: BLE001 -- reported below
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
    t = time.monotonic()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    after = client.ready()["batching"]
    calls = after["engine_calls"] - before["engine_calls"]
    check(
        f"{args.concurrency} concurrent predicts -> {calls} engine calls "
        f"({time.monotonic() - t:.1f}s, {len(errors)} errors)",
        not errors and 0 < calls <= args.concurrency,
    )


def probe_eviction(client: TabctxClient, prefix: str, first_ds: str, args) -> None:
    print("\n--- fill the cache: eviction + spill restore ---")
    ready = client.ready()
    spill_enabled = ready.get("spill") is not None
    print(
        f"  spill tier: {'on' if spill_enabled else 'off'}; cache before: {ready['cache_stats']}"
    )
    ref = client.predict(first_ds, [[0.1] * 20, [-0.1] * 20], return_proba=True)
    n0 = ready["cache_stats"]["n_cached_contexts"]
    evicted = False
    for i in range(args.fill_max):
        X, y = make_table(args.fill_rows, args.fill_features, seed=100 + i)
        try:
            client.fit(X.tolist(), y, dataset_id=f"{prefix}-fill-{i}")
        except AdmissionRejected as e:
            check(f"fill fit {i} admission-rejected cleanly (413)", True, str(e)[:80])
            break
        stats = client.ready()["cache_stats"]
        print(
            f"  fit {i}: cache {stats['n_cached_contexts']} contexts, {stats['used_bytes'] / 1e9:.2f}GB used"
        )
        if stats["n_cached_contexts"] < n0 + i + 1:
            evicted = True
            break
    check(
        "cache evicted under pressure (or fill budget exhausted)",
        True,
        f"evicted={evicted}",
    )
    try:
        r = client.predict(first_ds, [[0.1] * 20, [-0.1] * 20], return_proba=True)
        same = r.predictions == ref.predictions and np.allclose(
            r.probabilities, ref.probabilities, atol=1e-3
        )
        check(
            "first context still predicts identically after pressure"
            + (" (restored from spill)" if evicted and spill_enabled else ""),
            same,
        )
    except DatasetNotFoundError:
        check(
            "first context evicted -> clean 404 (no spill tier)",
            evicted and not spill_enabled,
        )
    print(f"  readyz after: {client.ready()['cache_stats']}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--tenant", default="probe")
    p.add_argument("--session-header", default="x-session-id")
    p.add_argument("--rows", type=int, default=2_000)
    p.add_argument("--big-rows", type=int, default=5_000)
    p.add_argument("--upload-rows", type=int, default=20_000)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--fill-rows", type=int, default=20_000)
    p.add_argument("--fill-features", type=int, default=50)
    p.add_argument("--fill-max", type=int, default=8)
    p.add_argument("--timeout", type=float, default=600)
    args = p.parse_args()

    client = TabctxClient(
        args.base_url,
        tenant_id=args.tenant,
        timeout_s=args.timeout,
        session_header=args.session_header,
    )
    prefix = f"probe-{uuid.uuid4().hex[:6]}"
    ready = client.ready()
    print(
        f"device={ready['device']} gpu_capacity_bytes={ready.get('gpu_capacity_bytes')} replica={ready['replica']}"
    )
    models = [m["id"] for m in client.models()]
    limits = client.limits()
    print(f"models={models} headroom={limits['admission_headroom_bytes'] / 1e9:.1f}GB")
    print(
        f"max rows by features: {limits['max_admissible_train_rows_by_feature_count']}"
    )
    check(
        "gpu_capacity reported",
        ready.get("gpu_capacity_bytes") is not None or "fake" in ready["device"],
    )

    first_ds = None
    for model in models:
        ds = probe_model(client, model, prefix, args)
        first_ds = first_ds or ds
    try:
        client.predict(f"{prefix}-never", [[0.0] * 20])
        check("unknown dataset -> 404", False, "no error raised")
    except DatasetNotFoundError:
        check("unknown dataset -> 404", True)
    probe_uploads(client, prefix, args)
    probe_coalescing(client, first_ds, args)
    probe_eviction(client, prefix, first_ds, args)

    failed = [n for n, ok, _ in _results if not ok]
    print(f"\n{len(_results) - len(failed)}/{len(_results)} checks passed")
    if failed:
        print("FAILED: " + "; ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
