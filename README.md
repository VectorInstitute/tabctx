# tabctx

Multi-tenant context caching and serving for tabular in-context-learning
(ICL) foundation models — TabICL/TabICLv2 today, designed to add TabPFN and
similar models later without a rewrite.

## Why

TabICL and TabPFN both do a single forward pass over a labeled training
table plus a test table — no autoregressive decoding, no token-level KV
cache. Both independently cache the *encoded training context* to speed up
repeated `.predict()` calls against the same training set, and both call
this a "KV cache" in their own docs. But it's a single-process,
single-estimator-object feature — nobody has built the multi-tenant,
evictable, memory-governed version of it.

That's what `tabctx` is: the analog of what PagedAttention did for
per-request LLM KV caches, applied to per-training-set tabular ICL contexts.

This is deliberately **not** built on vLLM (its PagedAttention/continuous-
batching machinery targets autoregressive decode, which these models don't
do — even vLLM's own maintainers caveat their closest precedent, pooling
models, as "not guaranteed to provide performance improvements") and is
**not** written in a new systems language (vLLM's speed comes from Python
orchestration plus custom GPU kernels, not from the host language — same
recipe applies here). It's pure Python, designed to sit behind a generic
Ray Serve `@serve.deployment` (proven engine-agnostic; no Ray-side changes
needed).

## Status: v1

TabICL backend only. Core pieces:
- `TabctxEngine` — `fit()` → `dataset_id`, `predict(dataset_id, ...)` reusing
  the cached context, `fit_predict()` convenience for one-shot callers.
- `ContextCacheManager` — multi-tenant, LRU-evictable cache of encoded
  contexts, sized against a memory ceiling.
- `PowerLawMemoryEstimator` — admission control and eviction sizing,
  calibrated from real measurements on an A100-40GB (see
  `src/tabctx/memory/calibration_data.py`). **Read `confidence()` before
  trusting this anywhere else** — it's explicit about being low-confidence,
  single-GPU, single-backend calibration.
- Chunked test-row batching — large test sets are automatically split
  against the memory ceiling so one oversized request can't crash the whole
  replica (the failure mode this library exists partly to prevent).

## Out of scope for v1 (see Gaps below)

TabPFN backend · cross-request/heterogeneous-shape batching · disk/CPU cache
tiering · multi-replica cache-aware routing · custom CUDA/Triton kernels ·
PyPI/GitHub publishing.

## Validated at scale (2026-08-28, real A100-40GB)

Extensive multi-tenant load testing found and fixed a real bug, then
confirmed the fix under genuine sustained pressure: cache-capacity
accounting used to come from the same pre-fit formula used for admission
control, which ran ~14x higher than real GPU memory for a representative
shape (see CHANGELOG's v0.3.0 entry) — capping real usable capacity at ~2
concurrent contexts when the hardware could actually hold ~227. Fixed by
measuring real `torch.cuda.memory_allocated()` after `fit()` instead of
estimating before it. Re-validated by looping 260 fits to genuinely exceed
the cache ceiling: eviction fired correctly and continuously, the earliest
context was cleanly evicted (404) while the most recent stayed served
(200), accounted vs. real GPU memory tracked within <1% of each other
throughout, and 8 further cycles under continued full-capacity pressure
showed no memory leak (evicted contexts' GPU memory was genuinely
reclaimed). Multi-tenant isolation, concurrent-request correctness (no data
races, though GPU work itself still serializes per replica by design),
the regression task path, and `dataset_id` reuse/overwrite semantics were
all confirmed end-to-end on real hardware too.

## Quickstart

```python
from tabctx import TabctxEngine, ContextCacheManager
from tabctx.backends.tabicl import TabICLBackend
from tabctx.memory import PowerLawMemoryEstimator, A100_40GB_TABICL_CALIBRATION

estimator = PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION)
cache = ContextCacheManager(capacity_bytes=estimator.ceiling_bytes())
engine = TabctxEngine(backend=TabICLBackend(), cache=cache, estimator=estimator)

dataset_id = engine.fit(X_train, y_train, task="classification")
result = engine.predict(dataset_id, X_test, return_proba=True)
# Reuse the same cached context for a different test batch -- no re-fit:
result2 = engine.predict(dataset_id, other_X_test)
```

See `examples/local_fit_predict.py` for a runnable, no-GPU version using the
test-only `FakeBackend`, and `src/tabctx/serve/app.py` for the Ray Serve
deployment.

## Gaps / roadmap

- **Memory estimator (the pre-fit admission-control gate) confidence is
  LOW**: calibrated from 4 successful measurements + 1 known-OOM boundary,
  one A100-40GB card, one backend (TabICL). The OOM boundary itself is only
  bounded between 5.2M and 18.4M `(train+test)×features` cells — real
  calibration data collection (more shapes, other GPU types, other
  backends) is needed before trusting this elsewhere. (Cache *accounting*
  no longer uses this estimator when a real measurement is available — see
  "Validated at scale" above — this gap is specifically about the pre-fit
  safety gate, which must stay conservative since nothing has run yet.)
- **The estimator overestimates badly for tiny inputs** (found via unit
  testing, not just theorized): calibration only covers 6,000-5,200,000
  cells, and extrapolating below that range overestimates substantially — a
  20-row/2-feature table estimates to ~23MB. Safe (never risks OOM by
  under-estimating) but still relevant to the admission gate's precision
  for small requests. Needs a piecewise or additive-intercept model, not a
  pure power law, once more calibration data exists.
- **No tenant/authz boundary** on `dataset_id` — it's a flat, shared,
  unauthenticated cache namespace. Anyone who knows or guesses a
  `dataset_id` can `predict()` against someone else's cached context.
- **No cache durability** — a replica restart silently drops every cached
  context; callers only find out via a subsequent `DatasetNotFoundError`.
- **v1 serializes all GPU work per replica** via one coarse lock around the
  cache's fit/evict/predict critical section. Confirmed correct/necessary
  given how the admission ceiling is derived (it only accounts for one
  in-flight backend call) — not just an arbitrary v1 shortcut. A safe
  concurrency improvement requires re-deriving that ceiling as a function
  of the max number of concurrent in-flight backend calls first, then this
  lock can be relaxed accordingly.
- **No TabPFN backend yet** — the `TabularICLBackend` protocol is designed
  to support one, but it isn't implemented.
- **Checkpoint reload cost per `fit()`, confirmed empirically**: a fresh
  `TabICLClassifier`/`Regressor` instance reloads the checkpoint from disk
  every time (~0.1s on CPU, after the one-time download — `huggingface_hub`
  only avoids re-downloading, not re-parsing; confirmed by reading
  `tabicl/_sklearn/classifier.py`'s `_load_model()`). Required per v1's
  fresh-instance-per-`fit()` rule, but a real target for later: share loaded
  backbone weights across `fit()` calls in-process, keeping only the
  per-fit training-encoding state instance-specific.
- **No cross-request batching** — v1 only chunks *one request's* test rows
  against its own cached context; packing multiple different requests'
  tables into one GPU call (the problem CRUMB, arXiv 2606.11473, targets) is
  unsolved here.
- **No disk/CPU cache tiering** — contexts live in GPU memory only; both
  TabPFN's own cache and the TabICL paper's CPU-offload numbers suggest
  overflow-to-CPU is valuable for larger deployments.
- **No multi-replica, cache-aware routing** — NVIDIA Dynamo's KV-aware
  routing pattern (route to whichever replica already holds relevant cached
  state) is a good design to imitate for "route repeat-predict-on-same-
  training-set requests to the replica already caching it," but nothing here
  implements it; the architecture doesn't preclude adding it later.
- **No custom CUDA/Triton kernels** — v1 wraps TabICL's own eager PyTorch
  attention as-is.
- **ConfigMap+wheel delivery to GKE is a stopgap** for testing, not a real
  distribution story (no PyPI/GitHub publishing yet).
