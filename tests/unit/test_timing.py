"""Per-request timing breakdowns (tabctx/timing.py) and where they're
recorded: the engine's phases, and across worker processes."""

import threading
import time

import pytest

from tabctx import timing


class TestCollector:
    def test_phases_and_counts_accumulate(self):
        with timing.collect() as t:
            with timing.phase("a"):
                time.sleep(0.01)
            with timing.phase("a"):
                pass
            timing.count("graphs_compiled", 2)
            timing.count("graphs_compiled", 0)  # zero doesn't add a key
        out = t.as_dict()
        assert out["a"] >= 10
        assert out["n_graphs_compiled"] == 2

    def test_no_collector_means_no_op(self):
        assert timing.current() is None
        with timing.phase("a"):
            pass
        timing.count("x")  # nothing to record into, nothing raises

    def test_collectors_nest_and_restore(self):
        with timing.collect() as outer:
            with timing.collect() as inner, timing.phase("inner"):
                pass
            assert timing.current() is outer
        assert "inner" in inner.as_dict() and "inner" not in outer.as_dict()

    def test_merge_folds_in_a_worker_breakdown(self):
        t = timing.Timings()
        t.add("fit_device", 0.5)
        t.merge({"fit_device": 250.0, "prepare": 10.0, "n_restores": 1})
        assert t.as_dict() == {"fit_device": 750.0, "prepare": 10.0, "n_restores": 1}


def _engine():
    from tabctx.backends.fake import FakeBackend
    from tabctx.cache.manager import ContextCacheManager
    from tabctx.engine import TabctxEngine
    from tabctx.memory.calibration_data import A100_40GB_TABICL_CALIBRATION
    from tabctx.memory.estimator import PowerLawMemoryEstimator

    return TabctxEngine(
        backend=FakeBackend(),
        cache=ContextCacheManager(capacity_bytes=200_000_000),
        estimator=PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION),
    )


def test_engine_records_fit_and_predict_phases():
    engine = _engine()
    X, y = [[float(i), 1.0] for i in range(20)], ["a", "b"] * 10
    with timing.collect() as fit_t:
        ds = engine.fit(X, y)
    assert {
        "prepare",
        "wait_host_memory",
        "wait_device",
        "make_room",
        "fit_device",
    } <= set(fit_t.as_dict())
    with timing.collect() as predict_t:
        engine.predict(ds, X[:3])
    out = predict_t.as_dict()
    assert {"wait_device", "predict_device"} <= set(out)
    assert out["n_predict_chunks"] == 1


def test_worker_process_breakdowns_reach_the_caller(tmp_path):
    from tabctx.serve.factory import ServeSettings, build_engine

    built = build_engine(
        ServeSettings(
            backends=("fake",), devices=("cuda:0", "cuda:1"), spill_dir=str(tmp_path)
        )
    )
    try:
        X, y = [[float(i), 1.0] for i in range(20)], ["a", "b"] * 10
        with timing.collect() as t:
            ds = built.engine.fit(X, y, dataset_id="timed")
        out = t.as_dict()
        assert "place" in out  # the router's own phase
        assert "fit_device" in out and "prepare" in out  # from the worker process
        with timing.collect() as t:
            built.engine.predict(ds, X[:2])
        assert "predict_device" in t.as_dict()
    finally:
        built.close()


def test_a_predict_of_a_spilled_context_records_its_restore(tmp_path):
    from tabctx.backends.fake import FakeBackend
    from tabctx.cache.manager import ContextCacheManager
    from tabctx.cache.spill import DiskSpillStore
    from tabctx.engine import TabctxEngine
    from tabctx.memory.calibration_data import A100_40GB_TABICL_CALIBRATION
    from tabctx.memory.estimator import PowerLawMemoryEstimator

    cache = ContextCacheManager(
        capacity_bytes=200_000_000, spill_store=DiskSpillStore(tmp_path)
    )
    engine = TabctxEngine(
        backend=FakeBackend(),
        cache=cache,
        estimator=PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION),
    )
    X, y = [[float(i), 1.0] for i in range(20)], ["a", "b"] * 10
    ds = engine.fit(X, y)
    assert cache.evict_one(ignore_pins=True) == ds  # spilled, not dropped
    with timing.collect() as t:
        engine.predict(ds, X[:2])
    out = t.as_dict()
    assert "restore" in out and out["n_restores"] == 1


def test_coalesced_predicts_each_report_their_batch_breakdown():
    from tabctx.batching import CoalescingPredictor

    engine = _engine()
    X, y = [[float(i), 1.0] for i in range(20)], ["a", "b"] * 10
    ds = engine.fit(X, y)
    predictor = CoalescingPredictor(engine, window_s=0.2)
    results: list[dict] = [{}, {}]

    def call(i: int) -> None:
        with timing.collect() as t:
            predictor.predict(ds, X[:2])
        results[i] = t.as_dict()

    threads = [threading.Thread(target=call, args=(i,)) for i in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert predictor.engine_calls == 1  # really one merged call
    for out in results:
        assert "predict_device" in out and "coalesce_wait" in out
        assert out["n_batched_with"] == 2


def test_a_fit_that_fails_mid_phase_releases_the_device():
    from tabctx.backends.fake import FakeBackend
    from tabctx.cache.manager import ContextCacheManager
    from tabctx.engine import TabctxEngine
    from tabctx.memory.calibration_data import A100_40GB_TABICL_CALIBRATION
    from tabctx.memory.estimator import PowerLawMemoryEstimator

    class FailsOnce(FakeBackend):
        failed = False

        def fit(self, X, y, task):
            if not self.failed:
                self.failed = True
                raise RuntimeError("boom")
            return super().fit(X, y, task)

    engine = TabctxEngine(
        backend=FailsOnce(),
        cache=ContextCacheManager(capacity_bytes=200_000_000),
        estimator=PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION),
    )
    X, y = [[float(i), 1.0] for i in range(20)], ["a", "b"] * 10
    with timing.collect(), pytest.raises(RuntimeError, match="boom"):
        engine.fit(X, y)
    done = threading.Event()
    threading.Thread(target=lambda: (engine.fit(X, y), done.set()), daemon=True).start()
    assert done.wait(10), "the failed fit left the device lock held"
