# Roadmap

This is the handoff document for whoever (human or AI) picks up `tabctx`
next. It assumes no prior context beyond what's in `README.md` and
`CHANGELOG.md`, so read those first. Everything below is ranked by priority,
with the reasoning that led to that ranking, what's already known, and
where to start. Update this file as priorities change; don't let it go stale.

## How we got here (context for the ranking below)

v1 (v0.1.0-v0.5.0) built and validated a working multi-tenant context cache
for TabICLv2 on a real A100-40GB. The overnight session of 2026-08-28/29
(v0.6.0-v0.7.0) then closed the top three items of the previous roadmap:

- **Multi-replica correctness (was Priority 1): DONE.** Ray 2.58's
  experimental consistent-hash router is wired into the deployment with
  strict affinity, with the contract `x-session-id` header == dataset_id
  (`serve/affinity.py`). Proven by a local 2-replica integration test
  (`tests/integration/test_multi_replica_affinity.py`, also in CI) and by
  `probe_multi_replica.py` against a real 2-replica GKE deployment
  sharing one A100.
- **Tenant boundary (was Priority 2): DONE at the namespacing level.**
  `serve/tenancy.py` scopes dataset_ids by the `x-tabctx-tenant-id`
  header; `TABCTX_REQUIRE_TENANT=true` removes the unscoped namespace.
  Deliberately NOT done: verifying tenant identity — that belongs in an
  authenticating proxy in front (API key -> tenant id); the module
  docstring states the trust model precisely.
- **Batching feasibility (was Priority 3): INVESTIGATED, and the
  investigation found something bigger** — see "The kv-cache finding"
  below. Same-context coalescing is implemented (`batching.py`);
  cross-context batching feasibility is now precisely mapped (below).

## The kv-cache finding (v0.7.0, the important one)

Tracing tabicl's `predict_proba` internals revealed that tabicl ships
with `kv_cache=False`, so every predict was **re-encoding the entire
training set through all three transformer stages** — the exact repeat
cost this library exists to eliminate. tabctx now enables it by default
(`TABCTX_KV_CACHE`), verified prediction-identical to the uncached path.
Also: backbone weights now load once per process (tabicl's own
`_unsupervised`/`_finetune` sharing pattern) instead of per fit.

Consequence for anyone reading old numbers: **every pre-v0.7.0 benchmark
baseline (`benchmarks/baselines/v0.5.0*.json`) is historical**. The 3-10x
"cache reuse" speedup previously reported was measured with the model
secretly re-encoding training data per predict; re-baseline before
drawing any new conclusions.

## Priority 1: Memory recalibration for the kv-cache era

The re-baseline itself happened 2026-08-29 on the 2-replica A100
deployment (`benchmarks/baselines/v0.7.0-2replica.json`): peak 27 ops/s
at c=8 (2.9x the v0.5.0 plateau), warm predict ~107ms vs fit ~1.4s,
coalescing verified live. What remains is the memory side:

- **kv-cache contexts are much bigger**: a 4,000-row x 30-feature
  context costs ~850MB resident in "kv" mode (a replica held only ~13
  before evicting at a ~11.6GB fraction-scaled budget), vs ~110MB for a
  10,000x50 context pre-kv-cache. The adaptive estimator prices this
  automatically per-shape, but capacity planning intuition must change;
  "repr" mode (~24x smaller ICL cache, slower predicts) is the lever
  when tenant count matters more than latency — worth benchmarking as a
  config point.
- **The static formula is now doubly stale** (calibrated pre-kv-cache,
  and it gates ADMISSION for novel shapes): on 0.45-fraction replicas it
  413s shapes like 10,000x50 that would really fit. Recalibrate
  `memory/calibration_data.py` on kv-cache-enabled fits, ideally
  separating fit-time transient memory from resident context size.

## Priority 2: Cross-context batching (feasibility now precisely known)

A deep read of tabicl's model internals (2026-08-29) established:

- The model's batch dim is over **tables** (`TabICL.forward`, B = number
  of tables); the sklearn wrapper just happens to use it for ensemble
  members of one table. With a kv cache, test rows attend only to
  fit-time-cached K/V — **no test-to-test attention** — so batching
  different contexts' test rows into one forward is *numerically exact*.
- `TabICLCache.concat` works today for contexts that agree on
  `(n_feature_groups, train_size, dtype)`; `num_classes` must be handled
  by slicing per-context. So **wrapper-level cross-context batching is
  feasible if you bucket contexts by shape** — you must call
  `model_.forward_with_cache` directly (bypassing
  `_batch_forward_with_cache`, which re-splits by `batch_size=8`), and
  each context contributes `n_estimators` batch entries per norm method.
- **Cross-shape batching requires model changes**: the encoder stacks
  don't thread `key_padding_mask` (the leaf attention supports it), and
  — the trap — SSMax scales attention by a single scalar `src_len` for
  the whole batch, so padding `train_size` **silently degrades accuracy**
  rather than crashing. Do not pad the train dimension without fixing
  ssmax's per-element `n` first.

**Whether to build it:** only after Priority 1's re-baseline shows
cross-tenant traffic is still overhead-bound. Same-shape bucketing is a
real but narrow win (tenants must share n_features and train_size);
measure how often that actually co-occurs in-flight before building.

## Priority 3: Large-table ingestion (learned from PriorLabs' API shape)

Comparing tabctx's REST surface against PriorLabs' hosted TabPFN API
(2026-08-29; recipe captured in `benchmarks/bench_vs_hosted_api.py`)
showed the two converge on the same lifecycle — fit -> server-side
context handle -> repeated predicts (their `fitted_train_set_id` is
tabctx's `dataset_id`) — with two structural differences:

1. **They separate data upload from orchestration** (`prepare_*_upload`
   -> signed URLs -> PUT CSV, then fit/predict reference upload ids).
   **SHIPPED in v0.8.0**, adapted for self-hosting: `POST
   /v1/tabctx/upload` streams raw CSV to replica-local disk (no object
   store needed) and fit/predict consume it by reference; the
   x-session-id affinity contract is what makes replica-local uploads
   correct at 2+ replicas. Validated live on the A100 deployment
   (2026-08-29): a 91MB CSV streamed through the LB + Serve proxy at
   ~62 MB/s with no buffering issues, fit-by-reference 5,000x50 in
   1.3s, by-reference predicts pinned to the fit replica, reordered
   test-CSV headers 422. **The binding constraint on truly large FITS
   is now the admission gate, not transport** (200k x 60 uploads fine,
   then 413s off the stale conservative calibration) -- which is
   exactly Priority 1 above.
2. **They expose capability discovery** (`get_model_limits`). Adopted:
   `/v1/tabctx/limits` now reports the live admission boundary
   (max admissible train rows at representative feature counts, derived
   from the estimator, adaptive over time) so clients can validate
   shapes before uploading. `TabctxClient.limits()` wraps it.

Benchmark context (`benchmarks/baselines/v0.7.0-vs-hosted-api-cpu.json`):
on the fit-once-predict-many workload, tabctx warm predicts beat the
hosted API 6-14x at p50 **even with the local side on a laptop CPU**
(~139-233ms vs ~1.4-1.9s) because every hosted predict pays upload +
network round trips; the hosted side wins cold-fit on larger tables when
the local side lacks a GPU. This is a serving-architecture comparison,
not a model-speed one — the baseline file says so explicitly.

## Priority 4: Cache durability

Unchanged from before: a replica restart silently drops every cached
context; the failure mode is a clean 404 (caller re-fits), so this ranks
below performance truth-telling but is real for production (autoscaling,
rolling deploys). Sticky routing adds a wrinkle worth knowing: after a
replica set change, the hash ring remaps some dataset_ids, so a fraction
of tenants see one 404 + re-fit even without a restart. A "has this
replica restarted recently" signal in `/readyz`, or best-effort disk
spillover for evicted contexts, are still the right-sized first steps.
Full GPU-tensor persistence is likely not worth it; tabicl's kv cache
retains dtype and auto-upcasts on reload (see its docstring), so
serialization is *possible* if ever justified.

## Lower priority (don't start here)

- **TabPFN backend: built (2026-08-29), pending GPU validation.**
  `backends/tabpfn.py` exists and the protocol seam held (zero changes
  elsewhere); the kv-cache lesson applied verbatim (`fit_with_cache` is
  not TabPFN's default either). Blockers on full validation: the v8+
  checkpoint is license-gated by PriorLabs' own portal (a human must
  register at https://ux.priorlabs.ai, accept the license, and export
  TABPFN_TOKEN=<API key>; integration tests skip with instructions
  until then -- an HF token alone is NOT sufficient, confirmed), and the memory estimator's static calibration is
  TabICL-on-A100 data — run a TabPFN calibration pass on GPU before
  trusting capacity numbers.
- **Verified tenant identity** (API keys -> tenant id at a proxy, or in
  tabctx itself). The namespacing boundary exists; making identity
  trustworthy is product/deployment work.
- **Memory estimator's static fallback** — 4 calibration points, one GPU,
  one backend, and now pre-kv-cache. The adaptive estimator masks this in
  steady state; recalibrate when convenient (fold into Priority 1's runs).
- **Disk/CPU cache tiering** — only matters once deployments push the
  (per-replica, fraction-scaled) ceiling.
- **PyPI publishing + docs site.** The release workflow now exists
  (`.github/workflows/release.yml`, tag-triggered, Trusted Publishing)
  but needs the one-time human setup documented in its header (create
  the PyPI project + trusted publisher + GitHub `pypi` environment)
  before a `v*` tag actually publishes. A docs site is still unstarted.
  The GKE deploy still ships a hand-built wheel via ConfigMap; that's
  fine for testing, not for adoption.
- **Custom CUDA/Triton kernels.** Still premature; nothing measured so
  far implicates raw kernel time.

## Where things live (so you don't have to rediscover this)

- This repo (`tabctx`): the library — engine, cache, estimator, backends
  (`backends/`), affinity/tenancy/factory (`serve/`), coalescing
  (`batching.py`), the Ray Serve app (`serve/app.py`).
- `VectorInstitute/inference-platform`, branch `test/tabicl-gke-onboard`,
  directory `tests/gke-tabicl-test/`: the GKE deployment/test harness
  (`onboard.sh`, `probe.py`, `probe_cache.py`, `probe_extensive.py`,
  `probe_scale.py`, `probe_multi_replica.py`) and the Helm values overlay
  (`helm-charts/ray-llm-app/values/tabicl-gke-test-values.yaml`) that
  deploys tabctx via a ConfigMap-mounted wheel plus `runtime_env.pip`.
  The overlay runs `num_replicas: 2` at `num_gpus: 0.5` each with
  `TABCTX_GPU_MEMORY_FRACTION: "0.45"`. That branch is **not merged to
  main**. `onboard.sh` supports `KEEP_ALIVE=true`. Gotchas learned the
  hard way: the wheel ConfigMap must keep the real wheel filename; after
  replacing the ConfigMap in place, wait for kubelet propagation (compare
  sha256 inside the pod) AND bump `TABCTX_DEPLOY_NONCE` in the overlay —
  Serve won't retry a DEPLOY_FAILED runtime_env until the serve config
  changes. Two more, both found the hard way on 2026-08-29: (1) a rolling
  update deadlocks when resources exactly fit (old replicas hold the GPU
  the new ones need — scale to 1 and back to 2 to break it); (2) **Ray
  Serve does not apply `request_router_config` changes to live proxies**:
  the router property in `ray/serve/_private/router.py` only constructs a
  router `if not self._request_router`, so `update_deployment_config`
  swaps the class attribute but a proxy that predates the config keeps
  its old (power-of-two) router until restarted. Symptom: session
  affinity works on a fresh cluster and silently doesn't after an
  in-place upgrade from a pre-router version — ~half of sticky predicts
  404 on 2 replicas. Fix: restart the Ray pods (proxies rebuild with the
  config present). Worth filing upstream against Ray.
- GCP project `agentic-ai-evaluation-bootcamp`, zone `us-central1-f`:
  where all real-hardware testing happened (us-central1-c had an A100
  stockout). **Check `gcloud container clusters list` before assuming
  nothing is running** — ephemeral test clusters cost real money while
  alive, and their `expiry_epoch` label caps them at 8h.

## A note on process, not just content

Every significant fix in this project's history came from actually
testing against real systems, not reasoning in the abstract: the ~14x
cache-accounting bug, the missing input validation, the misleading
benchmark metric, the multi-replica routing gap — and now the kv-cache
finding, which came from reading the model's actual source rather than
trusting its API surface. Keep doing both: read the internals, then
deploy and measure against a saved baseline before believing anything
helped.
