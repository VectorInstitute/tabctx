# Changelog

## [Unreleased]

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
