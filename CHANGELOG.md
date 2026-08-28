# Changelog

## [Unreleased]

## [0.2.0] - 2026-08-28
### Added
- `/readyz` now reports real GPU memory (`torch.cuda.memory_allocated`/`memory_reserved`) alongside the cache's own byte-accounting, so the two can be compared to catch the estimator drifting from reality or evicted contexts not actually releasing GPU memory.

### Context
Added while running an extensive multi-tenant test suite (concurrent users, cache eviction under pressure, regression task, dataset_id reuse, repeated fit/evict cycles) against a real GKE A100-40GB deployment, per the requirement that this service support several users at scale and be tested accordingly.

## [0.1.0] - 2026-08-28
Initial release. Multi-tenant context caching engine for tabular ICL models (TabICL backend), calibrated memory estimator, estimator-driven test-row chunking, and a Ray Serve deployment. See README.md for full v1 scope and known gaps.
