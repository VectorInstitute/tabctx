"""Unit tests for same-context predict coalescing (batching.py)."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tabctx.backends.fake import FakeBackend
from tabctx.batching import CoalescingPredictor
from tabctx.cache.manager import ContextCacheManager
from tabctx.engine import TabctxEngine
from tabctx.errors import DatasetNotFoundError, InvalidInputError
from tabctx.memory import (
    A100_40GB_TABICL_CALIBRATION,
    AdaptiveMemoryEstimator,
    PowerLawMemoryEstimator,
)


def _make_engine(predict_delay_s: float = 0.0) -> tuple[TabctxEngine, FakeBackend]:
    backend = FakeBackend(bytes_hint=1024, predict_delay_s=predict_delay_s)
    estimator = AdaptiveMemoryEstimator(
        fallback=PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION)
    )
    engine = TabctxEngine(
        backend=backend,
        cache=ContextCacheManager(capacity_bytes=estimator.ceiling_bytes()),
        estimator=estimator,
    )
    return engine, backend


def _fit(engine: TabctxEngine, dataset_id: str, majority: str = "yes") -> None:
    engine.fit(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [majority, majority, "no"],
        dataset_id=dataset_id,
    )


class TestCoalescing:
    def test_concurrent_same_context_requests_share_backend_calls(self):
        engine, backend = _make_engine(predict_delay_s=0.02)
        _fit(engine, "hot")
        predictor = CoalescingPredictor(engine, window_s=0.05)

        n_requests = 8
        barrier = threading.Barrier(n_requests)

        def call(i):
            barrier.wait()  # maximize overlap
            return predictor.predict("hot", [[float(i), 0.0]])

        with ThreadPoolExecutor(max_workers=n_requests) as pool:
            outcomes = list(pool.map(call, range(n_requests)))

        for outcome in outcomes:
            assert outcome.predictions == ["yes"]
        # 8 simultaneous requests must have coalesced into strictly fewer
        # backend calls (with a 50ms window and a barrier start, typically
        # 1-2 -- but the hard guarantee is "fewer than one per request").
        assert backend.predict_calls < n_requests, (
            f"no coalescing happened: {backend.predict_calls} backend calls "
            f"for {n_requests} requests"
        )

    def test_each_caller_gets_exactly_its_own_rows(self):
        engine, _ = _make_engine(predict_delay_s=0.01)
        _fit(engine, "ds")
        predictor = CoalescingPredictor(engine, window_s=0.05)

        sizes = [1, 3, 2, 5, 4]
        barrier = threading.Barrier(len(sizes))

        def call(n):
            barrier.wait()
            rows = [[float(n), float(j)] for j in range(n)]
            return predictor.predict("ds", rows, return_proba=True)

        with ThreadPoolExecutor(max_workers=len(sizes)) as pool:
            outcomes = list(pool.map(call, sizes))

        for n, outcome in zip(sizes, outcomes, strict=False):
            assert len(outcome.predictions) == n
            assert outcome.probabilities is not None
            assert len(outcome.probabilities) == n
            assert outcome.classes == ["no", "yes"]

    def test_different_contexts_are_not_mixed(self):
        engine, _ = _make_engine(predict_delay_s=0.01)
        _fit(engine, "ds-a", majority="aaa")
        _fit(engine, "ds-b", majority="bbb")
        predictor = CoalescingPredictor(engine, window_s=0.05)

        barrier = threading.Barrier(6)

        def call(i):
            barrier.wait()
            dataset_id, expected = ("ds-a", "aaa") if i % 2 == 0 else ("ds-b", "bbb")
            outcome = predictor.predict(dataset_id, [[1.0, 2.0]])
            assert outcome.predictions == [expected]

        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(call, range(6)))

    def test_zero_window_disables_coalescing(self):
        engine, backend = _make_engine()
        _fit(engine, "ds")
        predictor = CoalescingPredictor(engine, window_s=0)
        for _ in range(3):
            predictor.predict("ds", [[1.0, 2.0]])
        assert backend.predict_calls == 3

    def test_negative_window_rejected(self):
        engine, _ = _make_engine()
        with pytest.raises(ValueError):
            CoalescingPredictor(engine, window_s=-0.001)


class TestFailureIsolation:
    def test_missing_dataset_propagates_to_all_members(self):
        engine, _ = _make_engine()
        predictor = CoalescingPredictor(engine, window_s=0.05)
        with pytest.raises(DatasetNotFoundError):
            predictor.predict("never-fit", [[1.0, 2.0]])

    def test_one_malformed_member_fails_alone(self):
        engine, _ = _make_engine(predict_delay_s=0.01)
        _fit(engine, "ds")
        predictor = CoalescingPredictor(engine, window_s=0.08)

        barrier = threading.Barrier(3)
        results: dict[str, object] = {}

        def good(name):
            barrier.wait()
            results[name] = predictor.predict("ds", [[1.0, 2.0]])

        def bad():
            barrier.wait()
            try:
                predictor.predict("ds", [[1.0, 2.0, 3.0]])  # wrong feature count
                results["bad"] = "no-error"
            except InvalidInputError:
                results["bad"] = "rejected"

        threads = [
            threading.Thread(target=good, args=("g1",)),
            threading.Thread(target=good, args=("g2",)),
            threading.Thread(target=bad),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results["bad"] == "rejected"
        for name in ("g1", "g2"):
            outcome = results[name]
            assert outcome.predictions == ["yes"], (
                f"innocent request {name} was poisoned by another member's "
                "malformed input"
            )
