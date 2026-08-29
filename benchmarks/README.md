# tabctx benchmarks

`bench_concurrency.py` runs a concurrency sweep against a live tabctx
deployment and reports throughput/latency using the same vocabulary this
org already uses for LLM benchmarking, adapted to what tabctx's workload
actually is (see the script's own docstring for the full mapping):

| LLM concept | tabctx analog |
|---|---|
| TTFT (time to first token) | `cold_fit_latency_ms`: one-time cost before a new tenant's context is usable at all |
| decode / inter-token latency | `warm_predict_latency_ms`: recurring cost of a request against an already-cached context |
| output tok/s | `warm_predict_ops_per_sec`: steady-state throughput against pre-warmed, distinct tenants |

## Usage

```bash
# Against a port-forwarded deployment:
python3 benchmarks/bench_concurrency.py \
    --base-url http://127.0.0.1:8000 \
    --concurrency 1 2 4 8 16 \
    --duration 20 \
    --save-baseline benchmarks/baselines/v0.5.0.json
```

## Why this exists

v1 serializes all GPU work per replica via one coarse lock (see the top-level
README's Gaps section). Unlike an LLM engine's continuous batching,
throughput here is **not** expected to scale with concurrency past c=1; if
anything it should plateau or degrade as concurrency grows. That's the
headline number this benchmark tracks: re-run it after any future
concurrency-model change and diff against a saved baseline in
`baselines/` to prove whether it actually improved, the same way
`inference-platform`'s LLM benchmarks compare confirmed before/after
numbers for a config change.

## Baselines

`baselines/` holds one JSON file per meaningfully-different version/config,
named after the tabctx version it was measured against (e.g.
`v0.5.0.json`). Each file records the shape used, the measured
`cold_fit_latency_ms` distribution, and the full concurrency sweep. Keep
these, since they're the only record of "was this actually faster" over time.
