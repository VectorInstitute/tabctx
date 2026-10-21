"""The same training tables sent through a gateway as JSON rows and as a
Parquet file (multipart), interleaved, to see whether the request's size
decides the time it spends in transit.

For each fit it prints the total time, the time in the hop to tabctx
(the gateway's Server-Timing `upstream`, less tabctx's own `worker_call`)
and tabctx's own work. Values are rounded to 4 decimals, as prod_bench's
tables are, so the JSON stays under the gateway's 30 MiB limit.

    set -a; . ./.env; set +a
    OUT=baselines/upload-formats.json uv run python benchmarks/bench_upload_formats.py
"""

import json
import os
import pathlib
import re
import time
import urllib.request

import numpy as np

from tabctx.client import _Multipart
from tabctx.table_files import columns, to_parquet

KEY = os.environ["VP_API_KEY"]
BASE = "https://proxy.vectorinstitute.ai"


def server_timing(headers):
    st = {}
    for part in (headers.get("Server-Timing") or "").split(","):
        m = re.match(r"\s*(\w+).*dur=([\d.]+)", part)
        if m:
            st[m.group(1)] = float(m.group(2)) / 1000
    return st


def send(path, data, ctype):
    req = urllib.request.Request(
        BASE + path, data, {"Authorization": f"Bearer {KEY}", "Content-Type": ctype}
    )
    t = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        body = json.loads(r.read())
        st = server_timing(r.headers)
    tm = body.get("timings_ms") or {}
    return time.time() - t, st.get("upstream", 0), tm.get("worker_call", 0) / 1000


rng = np.random.default_rng(0)
results = {"json": [], "parquet": []}
for rnd in range(6):
    for n, f in ((5000, 500), (10000, 100)):
        X = np.round(rng.normal(size=(n, f)), 4).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)
        for fmt in ("json", "parquet") if rnd % 2 == 0 else ("parquet", "json"):
            ds = f"jvp-{fmt}-{rnd}-{n}x{f}"
            if fmt == "json":
                data = json.dumps(
                    {
                        "model": "tabicl-v2",
                        "dataset_id": ds,
                        "train_X": np.round(X.astype(np.float64), 4).tolist(),
                        "train_y": y.tolist(),
                    }
                ).encode()
                ctype = "application/json"
            else:
                pq = to_parquet(columns(X), ("label", y))
                form = _Multipart(
                    {"model": "tabicl-v2", "dataset_id": ds, "target": "label"}, pq
                )
                data = b"".join(form.chunks())
                ctype = form.content_type
            total, up, work = send("/v1/tabctx/fit", data, ctype)
            results[fmt].append((n, f, len(data), total, up, work))
            print(
                f"{fmt:7s} {n}x{f} {len(data) / 2**20:5.1f} MiB total {total:6.1f}s hop {up - work:6.1f}s tabctx {work:5.1f}s",
                flush=True,
            )
pathlib.Path(os.environ["OUT"]).write_text(json.dumps(results))
for fmt, rows in results.items():
    hops = sorted(r[4] - r[5] for r in rows)
    tot = sorted(r[3] for r in rows)
    print(
        fmt,
        "MiB",
        round(np.mean([r[2] for r in rows]) / 2**20, 1),
        "hop p50",
        round(hops[len(hops) // 2], 1),
        "max",
        round(hops[-1], 1),
        "| total p50",
        round(tot[len(tot) // 2], 1),
        "max",
        round(tot[-1], 1),
        "| over 10s:",
        sum(t > 10 for t in tot),
        "/",
        len(tot),
    )
