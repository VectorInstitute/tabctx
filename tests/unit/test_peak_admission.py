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
from tabctx.errors import AdmissionRejected
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
        assert full == int(_fallback().gpu_capacity_bytes * 0.9)
        assert estimator.admission_headroom_bytes(10_000_000_000) == full - 10_000_000_000
        assert estimator.admission_headroom_bytes(full + 1) == 0

    def test_static_estimator_headroom_ignores_usage(self):
        static = _fallback()
        assert (
            static.admission_headroom_bytes(0)
            == static.admission_headroom_bytes(10**12)
            == static.ceiling_bytes()
        )

    def test_engine_rejects_when_cache_usage_eats_headroom(self):
        # A shape whose measured peak fits an empty device but not one
        # already holding a huge resident cache.
        estimator = AdaptiveMemoryEstimator(
            fallback=_fallback(),
            preloaded=(Observation(100_000, 50, 30_000_000_000),),
        )
        cache = ContextCacheManager(capacity_bytes=estimator.ceiling_bytes())
        # Pretend a prior fit left ~20GB resident.
        backend = FakeBackend(bytes_hint=20_000_000_000, peak_bytes_hint=20_000_000_001)
        engine = TabctxEngine(backend=backend, cache=cache, estimator=estimator)
        engine.fit(TRAIN_X, TRAIN_Y, dataset_id="big-resident")

        # 50k x 20 is dominated by the preloaded point: estimate = 33GB.
        # Empty device: headroom = 0.9 x 40GB = ~36.5GB -> would admit.
        # With 20GB resident: headroom ~16.5GB -> must reject.
        with pytest.raises(AdmissionRejected, match="admission headroom"):
            engine.fit(
                [[1.0] * 20 for _ in range(50_000)],
                ["a"] * 50_000,
                dataset_id="big-fit",
            )