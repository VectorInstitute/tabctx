"""Unit tests for v0.9.0 peak-aware admission:

- backends report resident AND peak; the engine feeds PEAK into
  admission learning and RESIDENT into cache accounting (the v0.4.0
  conflation, fixed);
- preloaded calibration observations participate in estimates with
  their own (smaller) margin and are exempt from the runtime FIFO cap;
- admission headroom is usage-aware: peak-plus-resident is what OOMs.
"""

import math

import pytest

from tabctx.backends.fake import FakeBackend
from tabctx.cache.manager import ContextCacheManager
from tabctx.engine import TabctxEngine
from tabctx.errors import AdmissionRejected, DatasetNotFoundError
from tabctx.memory import (
    A100_40GB_TABICL_CALIBRATION,
    AdaptiveMemoryEstimator,
    PowerLawMemoryEstimator,
)
from tabctx.memory.adaptive import Observation

TRAIN_X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
TRAIN_Y = ["a", "b", "a"]


def _fallback():
    return PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION)


class TestEngineRecordsPeakNotResident:
    def test_peak_hint_feeds_admission_learning(self):
        estimator = AdaptiveMemoryEstimator(fallback=_fallback())
        backend = FakeBackend(bytes_hint=100, peak_bytes_hint=5000)
        engine = TabctxEngine(
            backend=backend,
            cache=ContextCacheManager(capacity_bytes=estimator.ceiling_bytes()),
            estimator=estimator,
        )
        engine.fit(TRAIN_X, TRAIN_Y, dataset_id="ds")
        # Admission estimate for a dominated shape = PEAK x runtime margin.
        assert estimator.estimate_bytes(2, 0, 2) == math.ceil(5000 * 1.5)
        # Cache accounting still charges the RESIDENT size.
        assert engine.stats().used_bytes == 100

    def test_without_peak_hint_falls_back_to_resident(self):
        estimator = AdaptiveMemoryEstimator(fallback=_fallback())
        backend = FakeBackend(bytes_hint=100)  # no peak measurement (CPU)
        engine = TabctxEngine(
            backend=backend,
            cache=ContextCacheManager(capacity_bytes=estimator.ceiling_bytes()),
            estimator=estimator,
        )
        engine.fit(TRAIN_X, TRAIN_Y, dataset_id="ds")
        assert estimator.estimate_bytes(2, 0, 2) == math.ceil(100 * 1.5)


class TestPreloadedObservations:
    def test_preloaded_used_with_calibration_margin(self):
        estimator = AdaptiveMemoryEstimator(
            fallback=_fallback(),
            preloaded=(Observation(100_000, 50, 31_000_000_000),),
        )
        # Dominated query -> preloaded peak x 1.1, not the formula.
        assert estimator.estimate_bytes(50_000, 0, 20) == math.ceil(31e9 * 1.1)

    def test_runtime_observation_tighter_when_smaller_cells(self):
        estimator = AdaptiveMemoryEstimator(
            fallback=_fallback(),
            preloaded=(Observation(100_000, 50, 31_000_000_000),),
        )
        estimator.record_observation(10_000, 30, 4_000_000_000)
        # Both dominate; the runtime one has fewer cells -> chosen, with
        # the runtime margin.
        assert estimator.estimate_bytes(5_000, 0, 20) == math.ceil(4e9 * 1.5)

    def test_preloaded_survive_runtime_fifo(self):
        estimator = AdaptiveMemoryEstimator(
            fallback=_fallback(),
            max_observations=3,
            preloaded=(Observation(1_000, 10, 999),),
        )
        for i in range(10):  # overflow the runtime FIFO many times over
            estimator.record_observation(5, 5, 10_000 + i)
        assert estimator.estimate_bytes(1_000, 0, 10) == math.ceil(999 * 1.1)

    def test_confidence_mentions_preloaded(self):
        estimator = AdaptiveMemoryEstimator(
            fallback=_fallback(), preloaded=(Observation(10, 10, 1),)
        )
        assert "1 preloaded calibration measurement" in estimator.confidence()


class TestUsageAwareHeadroom:
    def test_headroom_shrinks_with_usage(self):
        estimator = AdaptiveMemoryEstimator(fallback=_fallback())
        full = estimator.admission_headroom_bytes(0)
        assert full == int(_fallback().gpu_capacity_bytes * 0.85)
        assert (
            estimator.admission_headroom_bytes(10_000_000_000) == full - 10_000_000_000
        )
        assert estimator.admission_headroom_bytes(full + 1) == 0

    def test_static_estimator_headroom_ignores_usage(self):
        static = _fallback()
        assert (
            static.admission_headroom_bytes(0)
            == static.admission_headroom_bytes(10**12)
            == static.ceiling_bytes()
        )

    def test_evict_ahead_drains_cold_contexts_to_admit(self):
        """A warm cache must step aside for an admissible fit: cold
        contexts are evicted (spilled when a tier exists) until the new
        fit's transient peak fits -- proven necessary on a real A100
        (see engine.fit)."""
        estimator = AdaptiveMemoryEstimator(
            fallback=_fallback(),
            preloaded=(Observation(100_000, 50, 30_000_000_000),),
        )
        cache = ContextCacheManager(capacity_bytes=estimator.ceiling_bytes())
        # A prior fit left ~20GB resident: headroom is now ~14.5GB, the
        # next fit estimates 33GB -> the resident context must be
        # evicted ahead of the fit rather than the fit rejected.
        backend = FakeBackend(bytes_hint=20_000_000_000, peak_bytes_hint=20_000_000_001)
        engine = TabctxEngine(backend=backend, cache=cache, estimator=estimator)
        engine.fit(TRAIN_X, TRAIN_Y, dataset_id="big-resident")

        engine.fit(
            [[1.0] * 20 for _ in range(50_000)],
            ["a"] * 50_000,
            dataset_id="big-fit",
        )
        assert engine.predict("big-fit", [[1.0] * 20]).predictions == ["a"]
        # The cold context was drained to make transient room.
        with pytest.raises(DatasetNotFoundError):
            engine.predict("big-resident", TRAIN_X[:1])

    def test_infeasible_fit_rejected_without_draining_the_cache(self):
        """An oversized request that can NEVER fit must be rejected
        against the empty-cache bound BEFORE any eviction -- otherwise
        one bad request flushes every tenant's context on its way to a
        413 (observed live: a 200k x 60 request spilled 30 contexts and
        then failed anyway)."""
        estimator = AdaptiveMemoryEstimator(fallback=_fallback())
        cache = ContextCacheManager(capacity_bytes=estimator.ceiling_bytes())
        engine = TabctxEngine(
            backend=FakeBackend(bytes_hint=1), cache=cache, estimator=estimator
        )
        engine.fit(TRAIN_X, TRAIN_Y, dataset_id="innocent-bystander")
        # 200k x 50: formula estimates ~85GB -- infeasible on any A100.
        row = [1.0] * 50
        with pytest.raises(AdmissionRejected, match="empty cache"):
            engine.fit(
                [row] * 200_000,
                ["a"] * 200_000,
                dataset_id="never",
            )
        # The bystander must still be resident -- nothing was drained.
        assert engine.predict("innocent-bystander", TRAIN_X[:1]).predictions


class TestPredictChunkEstimates:
    def test_predict_estimate_uses_measured_predict_peaks(self):
        estimator = AdaptiveMemoryEstimator(
            fallback=_fallback(),
            preloaded=(Observation(50_000, 100, 32_000_000_000),),
            preloaded_predict=(Observation(50_000, 100, 1_300_000_000),),
        )
        # 1000 test rows (the measurement basis): predict peak x 1.1,
        # NOT the fit peak and NOT the formula (which for this shape
        # would exceed the whole ceiling and force 1-row chunks -- the
        # live timeout bug this fixes).
        est = estimator.estimate_bytes(50_000, 1_000, 100)
        assert est == math.ceil(1_300_000_000 * 1.1)
        # 2000 rows scale linearly from the basis.
        assert estimator.estimate_bytes(50_000, 2_000, 100) == math.ceil(
            1_300_000_000 * 2.0 * 1.1
        )
        # Fewer rows than the basis stay at the basis cost (conservative).
        assert estimator.estimate_bytes(50_000, 10, 100) == est

    def test_predict_estimate_falls_back_without_dominating_measurement(self):
        estimator = AdaptiveMemoryEstimator(
            fallback=_fallback(),
            preloaded_predict=(Observation(1_000, 10, 200_000_000),),
        )
        fallback = _fallback()
        assert estimator.estimate_bytes(50_000, 1_000, 100) == fallback.estimate_bytes(
            50_000, 1_000, 100
        )
