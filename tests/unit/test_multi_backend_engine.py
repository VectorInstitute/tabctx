"""Unit tests for multi-backend dispatch in TabctxEngine: several models
served over ONE shared cache and GPU budget, contexts remembering which
backend fit them (the chat-completions-style serving shape)."""

import pytest

from tabctx.backends.fake import FakeBackend
from tabctx.cache.manager import ContextCacheManager
from tabctx.engine import TabctxEngine
from tabctx.errors import DatasetNotFoundError, InvalidInputError
from tabctx.memory import (
    A100_40GB_TABICL_CALIBRATION,
    AdaptiveMemoryEstimator,
    PowerLawMemoryEstimator,
)

TRAIN_X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]


def _estimator():
    return AdaptiveMemoryEstimator(
        fallback=PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION)
    )


@pytest.fixture
def engine():
    icl = FakeBackend(bytes_hint=100, name="tabicl")
    pfn = FakeBackend(bytes_hint=100, name="tabpfn")
    est = _estimator()
    return (
        TabctxEngine(
            backends={"tabicl": icl, "tabpfn": pfn},
            estimators={"tabicl": est, "tabpfn": _estimator()},
            cache=ContextCacheManager(capacity_bytes=est.ceiling_bytes()),
            default_backend="tabicl",
        ),
        icl,
        pfn,
    )


class TestMultiBackend:
    def test_default_backend_used_when_unspecified(self, engine):
        eng, icl, pfn = engine
        eng.fit(TRAIN_X, ["a", "a", "b"], dataset_id="ds")
        assert (icl.fit_calls, pfn.fit_calls) == (1, 0)

    def test_predict_dispatches_to_fitting_backend(self, engine):
        eng, icl, pfn = engine
        eng.fit(TRAIN_X, ["icl-y"] * 3, dataset_id="ds-icl", backend="tabicl")
        eng.fit(TRAIN_X, ["pfn-y"] * 3, dataset_id="ds-pfn", backend="tabpfn")
        assert eng.predict("ds-icl", [[0.0, 0.0]]).predictions == ["icl-y"]
        assert eng.predict("ds-pfn", [[0.0, 0.0]]).predictions == ["pfn-y"]
        assert icl.predict_calls == 1 and pfn.predict_calls == 1

    def test_unknown_backend_rejected(self, engine):
        eng, _, _ = engine
        with pytest.raises(InvalidInputError, match="unknown backend"):
            eng.fit(TRAIN_X, ["a"] * 3, backend="xgboost")

    def test_shared_cache_budget_spans_backends(self, engine):
        eng, _, _ = engine
        eng.fit(TRAIN_X, ["a"] * 3, dataset_id="a", backend="tabicl")
        eng.fit(TRAIN_X, ["b"] * 3, dataset_id="b", backend="tabpfn")
        stats = eng.stats()
        assert stats.n_cached_contexts == 2
        assert stats.used_bytes == 200  # both charged to the ONE budget

    def test_single_backend_form_still_works(self):
        est = _estimator()
        eng = TabctxEngine(
            backend=FakeBackend(bytes_hint=10),
            cache=ContextCacheManager(capacity_bytes=est.ceiling_bytes()),
            estimator=est,
        )
        ds = eng.fit(TRAIN_X, ["a", "a", "b"])
        assert eng.predict(ds, [[1.0, 1.0]]).predictions == ["a"]
        assert eng.backend_names == ["fake"]

    def test_context_from_dropped_backend_is_clean_404(self):
        # A deployment reconfigured to drop a backend must not 500 on
        # contexts that backend left behind (e.g. restored from spill).
        est = _estimator()
        cache = ContextCacheManager(capacity_bytes=est.ceiling_bytes())
        both = TabctxEngine(
            backends={
                "a": FakeBackend(bytes_hint=1, name="a"),
                "b": FakeBackend(bytes_hint=1, name="b"),
            },
            estimators={"a": est, "b": _estimator()},
            cache=cache,
            default_backend="a",
        )
        both.fit(TRAIN_X, ["x"] * 3, dataset_id="orphan", backend="b")
        only_a = TabctxEngine(
            backends={"a": FakeBackend(bytes_hint=1, name="a")},
            estimators={"a": _estimator()},
            cache=cache,
            default_backend="a",
        )
        with pytest.raises(DatasetNotFoundError, match="no longer"):
            only_a.predict("orphan", [[0.0, 0.0]])
