<p align="center">
  <a href="https://github.com/VectorInstitute/tabctx/actions/workflows/tests.yml">
    <img src="https://github.com/VectorInstitute/tabctx/actions/workflows/tests.yml/badge.svg" alt="tests">
  </a>
  <img src="https://img.shields.io/badge/python-≥3.10-blue.svg" alt="Python ≥ 3.10">
  <img src="https://img.shields.io/badge/status-experimental%20(v0.x)-orange.svg" alt="status: experimental">
  <a href="LICENSE.md">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="license">
  </a>
</p>

# tabctx

Multi-tenant context caching and serving for tabular in-context-learning
(ICL) foundation models: [TabICL](https://github.com/soda-inria/tabicl)
and [TabPFN](https://github.com/PriorLabs/TabPFN) behind one engine,
cache, and serving stack (`TABCTX_BACKEND=tabicl|tabpfn`), with a
protocol seam (`backends/base.py`) for adding more.

## Why

TabICL and TabPFN both do a single forward pass over a labeled training
table plus a test table, with no autoregressive decoding and no
token-level KV cache. Both independently cache the *encoded training
context* to speed up repeated `.predict()` calls against the same training
set, and both call this a "KV cache" in their own docs. But it's a
single-process, single-estimator-object feature: nobody had built the
multi-tenant, evictable, memory-governed version of it. (Confirmed by
searching PyPI, GitHub, and arXiv: nothing like this existed before this
repo.)

**tabctx is that missing piece**, the analog of what PagedAttention did
for per-request LLM KV caches, applied to per-training-set tabular ICL
contexts.

It's deliberately **not** built on vLLM (its PagedAttention/continuous-
batching machinery targets autoregressive decode, which these models don't
do, and even vLLM's own maintainers caveat their closest precedent,
pooling models, as "not guaranteed to provide performance improvements")
and it's **not** written in a new systems language (vLLM's speed comes
from Python orchestration plus custom GPU kernels, not the host language;
the same recipe applies here). Pure Python, sitting behind a generic Ray
Serve `@serve.deployment`, with no engine-specific orchestration layer
needed.

## Quick wins

**See it work with zero setup** (no GPU, no model download: a fake
backend stands in for TabICL):

```bash
git clone https://github.com/VectorInstitute/tabctx.git && cd tabctx
pip install -e .
python examples/local_fit_predict.py
```

**Use it for real**, with TabICL doing the actual predicting:

```bash
pip install -e ".[tabicl]"
```

```python
from tabctx import TabctxEngine, ContextCacheManager
from tabctx.backends.tabicl import TabICLBackend
from tabctx.memory import AdaptiveMemoryEstimator, PowerLawMemoryEstimator, A100_40GB_TABICL_CALIBRATION

estimator = AdaptiveMemoryEstimator(fallback=PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION))
cache = ContextCacheManager(capacity_bytes=estimator.ceiling_bytes())
engine = TabctxEngine(backend=TabICLBackend(), cache=cache, estimator=estimator)

dataset_id = engine.fit(X_train, y_train, task="classification")
result = engine.predict(dataset_id, X_test, return_proba=True)

# The whole point: reuse the same cached context for a new test batch,
# with no re-fit cost.
result2 = engine.predict(dataset_id, other_X_test)
```

**Serve it over HTTP** with Ray Serve: `src/tabctx/serve/app.py` is a
ready-to-run deployment (`fit`/`predict` endpoints, health checks, live
memory-usage reporting). See `benchmarks/README.md` for how to measure it
once it's running.

**Talk to a deployment** with the bundled client (pure stdlib, sets the
routing/tenancy headers for you, raises the same tabctx exceptions the
in-process engine does):

```python
from tabctx.client import TabctxClient

client = TabctxClient("http://localhost:8000", tenant_id="acme")
client.fit(X_train, y_train, dataset_id="churn-v1")
result = client.predict("churn-v1", X_test, return_proba=True)
```

## Scaling out: multi-replica serving

The context cache is in-process and per-replica, so at `num_replicas >= 2`
requests for one dataset must consistently reach the replica that holds
its context. The deployment ships with Ray Serve's consistent-hash
request router (requires ray >= 2.58) configured for strict session
affinity, and the contract is simply: **send the dataset_id as the
`x-session-id` header on every `/v1/tabctx/fit` and `/v1/tabctx/predict`
call.**

```bash
curl -X POST $URL/v1/tabctx/fit \
  -H "content-type: application/json" -H "x-session-id: my-dataset" \
  -d '{"train_X": [[1,2],[3,4]], "train_y": [0,1], "dataset_id": "my-dataset"}'

curl -X POST $URL/v1/tabctx/predict \
  -H "content-type: application/json" -H "x-session-id: my-dataset" \
  -d '{"dataset_id": "my-dataset", "test_X": [[5,6]]}'
```

Details worth knowing:

- fit() adopts the header value as the dataset_id if the body omits one; a
  header/body mismatch is rejected 422 (a silent mismatch would mis-route).
- Omitting the header still works, but only routes correctly on
  single-replica deployments.
- Affinity is strict (no fallback replica): if the owning replica is
  backpressured, callers see retry-with-backoff/503 rather than a
  spurious 404 from a replica that doesn't hold the context.
- Responses report `served_by` (the replica tag) so affinity is
  observable and testable end to end.
- When replicas share one physical GPU (e.g. 2 replicas at
  `num_gpus: 0.5`), set `TABCTX_GPU_MEMORY_FRACTION` (e.g. `0.45`) so
  each replica budgets its share of GPU memory.
- `TABCTX_BACKEND=fake` runs the whole serve stack without a GPU or
  torch -- `tests/integration/test_multi_replica_affinity.py` uses it to
  prove the multi-replica contract on a laptop.

## Tenant isolation

`dataset_id` alone is a flat, guessable namespace -- without isolation,
any caller who knows another tenant's dataset_id could predict against
their cached model. The serving layer therefore scopes every dataset_id
by the `x-tabctx-tenant-id` header before it touches the cache: two
tenants can both call their dataset `mnist` and get fully separate
contexts, and a caller without the right tenant id sees a clean 404,
never another tenant's model.

- Set `TABCTX_REQUIRE_TENANT=true` in production: every `/v1/tabctx`
  request must then carry a tenant id (else 401), so there is no
  unscoped namespace at all.
- By default the header is optional (dev-friendly, backward compatible),
  but then unscoped ids coexist with scoped ones and the boundary is
  advisory, not enforced.
- Tenant identity is caller-supplied and unverified by design; put an
  authenticating proxy (API keys -> tenant id) in front for real
  security. See `src/tabctx/serve/tenancy.py` for the full trust-model
  notes.

## Installation

Requires Python ≥ 3.10.

```bash
pip install -e .                 # core library only (FakeBackend, no GPU deps)
pip install -e ".[tabicl]"       # + real TabICL backend (torch, tabicl)
pip install -e ".[tabpfn]"       # + TabPFN backend (weights license-gated:
                                 #   ux.priorlabs.ai + export TABPFN_TOKEN)
pip install -e ".[serve]"        # + Ray Serve deployment (ray[serve], fastapi)
pip install -e ".[dev]"          # + test dependencies
```

Not yet on PyPI; install from a clone for now (see [Roadmap](ROADMAP.md)).

## How it works

- **`TabctxEngine`**: `fit(X, y)` returns a `dataset_id`, `predict(dataset_id, X_test)`
  reuses the cached context, and `fit_predict()` is there for one-shot callers who don't
  need caching.
- **`ContextCacheManager`**: a multi-tenant, LRU-evictable cache of encoded
  training contexts, sized against a real memory budget.
- **`AdaptiveMemoryEstimator`**: admission control that starts from a
  conservative static formula and gets progressively less conservative as
  the service accumulates real per-`fit()` GPU measurements, safely (only
  ever using a real measurement to bound a *smaller-or-equal* future
  request, never to extrapolate upward).
- **Chunked prediction**: large test sets are automatically split against
  the memory budget so one oversized request can't crash the whole
  replica, the failure mode this library exists partly to prevent (the
  naive one-shot wrapper it replaces did crash this way; see
  [CHANGELOG](CHANGELOG.md)).
- **TabICL's kv-cache, actually on**: tabicl ships with its fit-time
  context cache disabled, which makes every `predict()` re-encode the
  whole training set. tabctx enables it by default
  (`TABCTX_KV_CACHE=kv|repr|off`), with predictions verified identical
  to the uncached path -- and loads the pretrained backbone once per
  process instead of per fit.
- **Same-context coalescing** (`CoalescingPredictor`): concurrent
  requests against one cached context are packed into a single GPU call
  within a ~5ms window and split back per caller, amortizing per-call
  overhead without ever running concurrent GPU work (the memory budget's
  single-in-flight-call assumption is preserved).

## Validated at scale (real A100-40GB, not simulated)

All numbers below are from a live GKE deployment of **two replicas
sharing one A100-40GB** (v0.7.0, 2026-08-29,
`benchmarks/baselines/v0.7.0-2replica.json`), except where noted:

- **Multi-replica routing is correct, not assumed**: 80 sticky predicts
  across 8 datasets on 2 replicas produced zero spurious 404s, every one
  pinned to the replica that fit its dataset; 428 concurrent predicts
  clean. The same probe run WITHOUT the affinity header 404s about half
  the time -- the gap the `x-session-id` contract closes.
- **Cache reuse works**: warm `predict()` at ~107ms server-side against a
  ~1,367ms `fit()` (~13x), with TabICL's kv-cache enabled (v0.7.0) so
  repeat predicts genuinely skip the training-set re-encode.
- **Throughput scales with concurrency now**: 7.6 -> 14.2 -> 22.1 -> 27.1
  ops/sec at c=1/2/4/8 (peak ~2.9x the v0.5.0 single-replica plateau of
  ~9.4), zero backpressure through c=16. Same-context coalescing verified
  live (48 concurrent requests -> 32 GPU calls).
- **Eviction under real pressure**: filling a replica to 95% of its
  fraction-scaled budget evicts oldest-first with clean 404s for evicted
  contexts and no memory leak across repeated fit/evict cycles.
- **Real bugs found by testing, not reasoning**: a ~14x cache-accounting
  overestimate (v0.3.0), malformed input surfacing as a raw 500 (v0.5.0),
  the multi-replica routing gap itself (v0.6.0), and an upstream Ray
  Serve quirk where `request_router_config` changes never reach live
  proxies (see ROADMAP.md). Full list in [CHANGELOG.md](CHANGELOG.md).
- **Column count scales linearly, not quadratically** (TabICLv2's
  inducing-point column attention, confirmed empirically out to 700
  columns; `benchmarks/baselines/v0.5.0-feature-sweep.json`, pre-kv-cache).

## Status & Roadmap

v0.6.0. Multi-replica deployments are now correct via session-sticky
routing (the former top roadmap item). **Read [ROADMAP.md](ROADMAP.md)
before starting new work** -- it ranks what's next and why.

## Contributing

Issues and PRs welcome. Run `pytest tests/unit/` before submitting (no GPU
required; the test suite runs entirely against a fake backend). See
[CHANGELOG.md](CHANGELOG.md) for the project's history and
[ROADMAP.md](ROADMAP.md) for where it's headed.

## License

[Apache 2.0](LICENSE.md)
