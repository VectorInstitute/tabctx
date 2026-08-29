# Roadmap

This is the handoff document for whoever (human or AI) picks up `tabctx`
next. It assumes no prior context beyond what's in `README.md` and
`CHANGELOG.md`, so read those first. Everything below is ranked by priority,
with the reasoning that led to that ranking, what's already known, and
where to start. Update this file as priorities change; don't let it go stale.

## How we got here (context for the ranking below)

v1 (v0.1.0-v0.5.0) built and validated a working multi-tenant context cache
for TabICLv2 on a real A100-40GB: cache-reuse works, admission control
prevents the OOM that crashed the naive wrapper this replaced, and three
real bugs were found and fixed via extensive load testing (a ~14x
cache-accounting overestimate, a missing input-validation gate, and a
misleading throughput metric in the benchmark tool itself). Full detail in
`CHANGELOG.md`. The two benchmark baselines in `benchmarks/baselines/`
(`v0.5.0.json`, the concurrency sweep, and `v0.5.0-feature-sweep.json`, the
column-count sweep) are real, measured numbers, not estimates. Treat them
as ground truth for "did a change actually help," and re-run them after
any change that touches concurrency or the estimator.

## Priority 1: Multi-replica correctness (do this first)

**The problem:** tabctx's cache is in-process, per-replica. The benchmark
data shows single-replica throughput plateaus at ~9-9.4 ops/sec (see
`benchmarks/baselines/v0.5.0.json`), so the obvious next move is "add more
Ray Serve replicas." **That doesn't work today.** Ray Serve's default
request routing has no session affinity (confirmed: `request_router_config`
in this org's existing Helm charts only sets a stats timeout, no affinity
strategy). With 2+ replicas, a `predict()` call for an existing
`dataset_id` has no guarantee of landing on the replica that `fit()` it, so
callers would see intermittent, spurious 404s as traffic bounces between
replicas that don't share state. **The current architecture only works
correctly as exactly one replica.** This wasn't caught until deep into
scale-testing this session, so don't repeat that: any capacity-planning
advice involving replica count is wrong without this fixed first.

**Why this ranks above everything else:** it's not a missing feature, it's
a correctness bug hiding behind the obvious scaling path. Shipping this
today as "just add replicas" would cause real, confusing failures for real
users.

**Where to start:**
- Ray Serve supports custom request routing (`request_router_config` and
  `RequestRouter`; this org's charts already expose the config key, just
  unused for affinity). Investigate whether a custom router can hash
  `dataset_id` to a consistent replica (sticky routing) without needing
  Ray internals expertise from scratch.
- Alternative worth evaluating: don't route by `dataset_id` at the Serve
  layer at all. Instead, on a cache miss (`DatasetNotFoundError`), have
  the replica try fetching/re-fitting rather than immediately 404ing.
  Re-fitting defeats the caching benefit, but a request-level fallback
  might be simpler to ship correctly than custom routing, at the cost of
  the exact throughput this library exists to protect. Only sensible as a
  stopgap if custom routing turns out to be a bigger lift than expected.
- Whatever you build, add an explicit multi-replica test to
  `tests/gke-tabicl-test/` (in `inference-platform`, not this repo; see
  "Where things live" below) that deploys 2+ replicas and confirms
  `fit()` on one request followed by `predict()` on a *different* request
  against the same `dataset_id` succeeds regardless of which replica each
  lands on. This is the regression test that would have caught the gap
  immediately.

## Priority 2: Tenant / authz boundary

**The problem:** `dataset_id` is a flat, unauthenticated, guessable
namespace. Any caller who knows or guesses another tenant's `dataset_id`
can `predict()` against their cached model. This is a data-leakage risk,
not just a missing nicety, so rank it above pure performance work for that
reason.

**Where to start:** the minimal fix is namespacing, not a full auth system.
Thread an API-key or tenant-id through requests (header or request field),
and internally scope `dataset_id` as `f"{tenant_id}:{dataset_id}"` before
it ever touches `ContextCacheManager`. `TabctxEngine`'s public API
(`fit`/`predict`) would need a `tenant_id` parameter; `serve/app.py`'s
Pydantic request models would need a field for it, populated from a header
in practice. Decide whether tenant identity is caller-supplied (simple, but
trusts the caller) or verified against some external identity source
(more real security, more scope). That's a product decision, not
something to guess at.

## Priority 3: Cross-request batching, but investigate feasibility before committing

**The problem this targets:** the concurrency benchmark shows throughput
plateauing around c=2-4, not scaling further. Naively relaxing the coarse
lock to allow concurrent GPU calls is **unsafe as-is**: the memory
estimator's admission ceiling was derived assuming exactly one in-flight
backend call, so N concurrent calls near that ceiling could jointly exceed
real GPU capacity even though each individually passed admission control.
Don't do this without first re-deriving the ceiling as a function of max
concurrent in-flight calls.

**The more promising angle:** per-request latency (~200ms for a tiny
300-row table) looks dominated by fixed overhead, not GPU compute
saturation; an A100 is nowhere near busy processing one small table.
This suggests **batching multiple concurrent requests into fewer, larger
GPU calls** (the CRUMB-style batching this library has always listed as
out-of-scope, arXiv 2606.11473) is likely higher-leverage than raw
concurrency, and safer to reason about (one bigger call's memory need is
easier to bound than N overlapping ones).

**Before building anything:** check whether TabICL's API can even support
this. `TabICLClassifier.predict()` operates on one fitted context at a
time; batching predict() calls against the *same* cached context (pack
multiple requests' test rows into one call, split results back) is
straightforward and worth doing regardless. Batching across *different*
tenants' contexts in one GPU call is the harder, higher-value case and may
require the model to support batched/grouped attention across contexts,
which nothing so far has confirmed is possible. Spend a half-day
investigating TabICL's internals before committing to a design.

**How to validate whatever you build:** re-run
`benchmarks/bench_concurrency.py` and diff against `baselines/v0.5.0.json`.
If ops/sec doesn't meaningfully improve past c=4, the change didn't work.
Don't ship it based on intuition alone.

## Priority 4: Cache durability

**The problem:** a replica restart silently drops every cached context.
Lower urgency than #1/#2 because the failure mode is a clean 404 (caller
re-fits), not silent corruption or a security hole, but it's still real
for production reliability, especially once autoscaling or rolling
deploys are in play.

**Where to start:** this doesn't need full persistence. Even a simple
"has this replica restarted recently" signal in `/readyz`, or a
best-effort disk-backed spillover for evicted-but-still-referenced
contexts, would help. Full persistence (serializing GPU tensors to disk
and reloading) is a bigger, possibly not-worth-it lift, so scope this
carefully before overbuilding.

## Lower priority (don't start here)

These are real, documented gaps, but none of them block correctness or
security the way #1/#2 do, and none are proven to be the throughput
bottleneck the way #3 might be:

- **TabPFN backend.** `TabularICLBackend` (protocol in `backends/base.py`)
  is designed to support a second backend without touching the engine,
  cache, or estimator. Real work, but breadth, not depth, so do this once
  the multi-tenant story (1-3 above) is solid, not before.
- **Memory estimator's static fallback is low-confidence** (4 calibration
  points, one GPU type, one backend; see `memory/estimator.py`'s
  docstring). `AdaptiveMemoryEstimator` (v0.4.0) already reduces reliance
  on this for shapes the service has actually served; the fallback only
  matters for genuinely novel/larger shapes. More calibration data would
  help but isn't urgent.
- **Disk/CPU cache tiering** for contexts that would otherwise get
  evicted. Only matters once real deployments are pushing the ~25.7GB
  ceiling harder than anything tested so far.
- **Custom CUDA/Triton kernels.** Premature: nothing so far suggests
  kernel-level optimization is the bottleneck (the throughput plateau
  looks like a concurrency/overhead problem, not a raw-compute one; see
  the linear column-scaling result in `benchmarks/baselines/v0.5.0-feature-sweep.json`,
  which suggests the model's own compute is well-behaved).
- **PyPI publishing.** The GKE deploy still ships a hand-built wheel via
  ConfigMap (see "Where things live" below), which is fine for now, but a
  real distribution story matters once this has users outside this org.

## Where things live (so you don't have to rediscover this)

- This repo (`tabctx`): the library itself: engine, cache, estimator,
  backends, the Ray Serve app.
- `VectorInstitute/inference-platform`, branch `test/tabicl-gke-onboard`,
  directory `tests/gke-tabicl-test/`: the GKE deployment/test harness
  (`onboard.sh`, `probe.py`, `probe_cache.py`, `probe_extensive.py`,
  `probe_scale.py`) and the Helm values overlay
  (`helm-charts/ray-llm-app/values/tabicl-gke-test-values.yaml`) that
  deploys tabctx via a ConfigMap-mounted wheel plus `runtime_env.pip` (no
  custom image, no PyPI needed; see that overlay's comments for exactly
  why the wheel filename can't be renamed). That branch is **not merged to
  main**, it's a validated proof of concept, not production config.
  `onboard.sh` supports `KEEP_ALIVE=true` to leave the cluster up for
  interactive testing instead of auto-tearing-down.
- GCP project `agentic-ai-evaluation-bootcamp`, zone `us-central1-f`: where
  all real-hardware testing happened. Confirmed reliable on-demand
  A100-40GB capacity there as of 2026-08-28 (a different zone,
  `us-central1-c`, hit a capacity stockout, so `us-central1-f` is the
  known-good choice). **Check `gcloud container clusters list` before
  assuming nothing is running**, since ephemeral test clusters cost real
  money while alive.

## A note on process, not just content

Every fix in this project's history so far came from actually load-testing
against real hardware, not from reasoning about the code in the abstract:
the ~14x cache-accounting bug, the missing input validation, and the
misleading benchmark metric were all found this way. Keep doing that.
Before believing a change helped, deploy it and re-run the relevant
benchmark or probe script against a real GPU, and diff the result against
a saved baseline.
