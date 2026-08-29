# Changelog

## [Unreleased]

## [0.6.0] - 2026-08-28
### Fixed
- **Multi-replica deployments are now correct** (previously the architecture was only safe at exactly one replica -- ROADMAP.md's former Priority 1). The Ray Serve deployment now ships with Ray's consistent-hash request router (`ray.serve.experimental.consistent_hash_router.ConsistentHashRouter`, requires ray >= 2.58) configured for strict session affinity (`num_fallback_replicas=0`): requests carrying the `x-session-id` header always land on the same replica. The contract, enforced in the new `serve/affinity.py`, is that the session id IS the dataset_id -- clients send `x-session-id: <dataset_id>` on every `/v1/tabctx/fit` and `/v1/tabctx/predict` call (fit adopts the header value as the dataset_id when the body omits one; a mismatched header/body pair is rejected 422 loudly rather than silently mis-routing). Requests without the header behave as before, which only routes correctly on single-replica deployments.

### Security
- **Tenant isolation** (former ROADMAP.md Priority 2): the serving layer now scopes every dataset_id by the caller's `x-tabctx-tenant-id` header before it touches the cache (`serve/tenancy.py`), so a guessed/shared dataset_id no longer reaches another tenant's cached model. `TABCTX_REQUIRE_TENANT=true` makes the header mandatory (401 without it), eliminating the unscoped namespace entirely; the default keeps the header optional for dev/backward compatibility. Tenant identity is caller-supplied and unverified by design (pair with an authenticating proxy for real security -- trust model documented in the module). Malformed tenant ids are rejected 422 rather than silently unscoped; responses never leak the internal scoping prefix.

### Added
- `serve/factory.py`: environment-driven engine construction, shared by any host. `TABCTX_BACKEND=fake` runs the full serve app against the deterministic fake backend (no GPU/torch), which is what makes multi-replica routing testable on a laptop and in CI. `TABCTX_GPU_MEMORY_FRACTION` scales the estimator's admission ceiling and the cache's capacity budget for replicas sharing one physical GPU (e.g. two replicas at `num_gpus: 0.5` on one A100).
- `served_by` (replica tag) in fit/predict responses and `replica` in `/readyz`, so clients and probes can verify affinity actually pinned a tenant's traffic to one replica.
- `tests/integration/test_multi_replica_affinity.py`: boots a real local 2-replica Ray Serve cluster (fake backend) and asserts 16 datasets x 6 predicts see zero spurious 404s, every predict lands on its fit replica, and fits spread across both replicas. This is the regression test that would have caught the routing gap immediately; also run in CI (new `integration` job).
- `benchmarks/bench_concurrency.py` now sends the session header on all tabctx-native calls, so its numbers stay valid at `num_replicas >= 2`.

## [0.5.0] - 2026-08-28
### Fixed
- Malformed input (mismatched `train_X`/`train_y` lengths, empty tables, ragged rows, zero-feature rows, a `predict()` feature count mismatched against the cached context) now raises `InvalidInputError` -> HTTP 422, instead of reaching the backend as raw numpy/sklearn arrays and surfacing as an unhandled, untranslated exception (a bare 500 with no useful detail). Found via `probe_scale.py`'s malformed-input test against a real deployment.

### Context
Found during a dedicated race-condition/throughput/input-validation test pass (`tests/gke-tabicl-test/probe_scale.py` in inference-platform), run to validate the service is robust to one careless client's bad request -- a real multi-tenant requirement, not just a nice-to-have. That same pass confirmed no data corruption under concurrent `fit()`/`predict()` races on the same `dataset_id`, and measured real sustained throughput (~1.2 fit+predict cycles/sec across 8 concurrent users, p95 latency ~7.7s) -- concrete evidence of the coarse per-replica lock's real-world cost under load, which is already a documented gap.

## [0.4.0] - 2026-08-28
### Added
- `AdaptiveMemoryEstimator` (`memory/adaptive.py`): wraps the static `PowerLawMemoryEstimator` as a fallback, but the pre-fit admission gate now uses real per-fit measurements (fed back via `engine.fit()` -> `record_observation()`) whenever a past fit at least as large in both rows and features has been observed -- safe by construction (memory cost is assumed monotonic in table size, so a real measurement on a larger-or-equal shape is a valid upper bound for a smaller query), with a configurable safety margin on top. Falls back to the static formula for genuinely novel or larger-than-anything-seen shapes, and always for predict()-time chunking queries (a different, unmeasured quantity). Deployed as the default estimator in `serve/app.py`.
- `MemoryEstimator` protocol gained `record_observation()`; `PowerLawMemoryEstimator` implements it as a no-op.

### Context
Direct extension of the v0.3.0 cache-accounting fix: now that fit() calls report real measured GPU cost, this closes the loop by feeding that data back into the PRE-FIT admission gate too, so admission decisions get progressively less conservative for shapes the service has actually served -- compounding the multi-tenant capacity improvement from v0.3.0 over the life of a running replica.

## [0.3.0] - 2026-08-28
### Fixed
- Cache-accounting size for a fitted context now comes from the backend *after* `fit()` runs, not before. `TabICLBackend` now reports the real `torch.cuda.memory_allocated()` delta measured across `fit()` instead of the pre-fit formula-based estimate. Found via extensive multi-tenant load testing on a real A100-40GB: the formula-based estimate was ~14x higher than real GPU memory for a realistic shape (21.85GB accounted vs 1.56GB actually resident), needlessly throttling effective multi-tenant cache capacity to a fraction of what the hardware supports.
- The pre-fit admission-control gate (which must stay conservative, since nothing has run yet to measure) is unchanged -- this fix only affects how much of the cache's capacity budget a *successfully* fitted context is charged.

### Context
Found and fixed during an extensive multi-tenant test suite (concurrent users, cache eviction under pressure, leak detection) run against a real GKE A100-40GB deployment, in direct response to a requirement that this service support several concurrent users at scale.

## [0.2.0] - 2026-08-28
### Added
- `/readyz` now reports real GPU memory (`torch.cuda.memory_allocated`/`memory_reserved`) alongside the cache's own byte-accounting, so the two can be compared to catch the estimator drifting from reality or evicted contexts not actually releasing GPU memory.

### Context
Added while running an extensive multi-tenant test suite (concurrent users, cache eviction under pressure, regression task, dataset_id reuse, repeated fit/evict cycles) against a real GKE A100-40GB deployment, per the requirement that this service support several users at scale and be tested accordingly.

## [0.1.0] - 2026-08-28
Initial release. Multi-tenant context caching engine for tabular ICL models (TabICL backend), calibrated memory estimator, estimator-driven test-row chunking, and a Ray Serve deployment. See README.md for full v1 scope and known gaps.
