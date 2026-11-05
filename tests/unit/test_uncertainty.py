"""Predictive quantiles end to end on the fake backend: validation, the
engine, chunking and OOM splitting, coalescing, the multi-GPU router, the
HTTP API, the client and the estimator. Each layer's test asks the same
question: does every caller get the quantiles of exactly its own rows, at
exactly the levels it asked for, in its order?"""

import statistics
import threading

import numpy as np
import pytest

from tabctx.backends.fake import FakeBackend
from tabctx.batching import CoalescingPredictor
from tabctx.cache.manager import ContextCacheManager
from tabctx.engine import TabctxEngine
from tabctx.errors import BackendComputeError, InvalidInputError
from tabctx.memory import A100_40GB_TABICL_CALIBRATION, PowerLawMemoryEstimator
from tabctx.types import PredictOutcome
from tabctx.uncertainty import MAX_QUANTILES, check_quantiles

Y = [1.0, 2.0, 3.0, 4.0, 10.0]
X = [[float(i), 1.0] for i in range(len(Y))]
DIST = statistics.NormalDist(statistics.fmean(Y), statistics.pstdev(Y))


def engine_with(backend=None) -> TabctxEngine:
    estimator = PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION)
    return TabctxEngine(
        backend=backend or FakeBackend(),
        cache=ContextCacheManager(estimator.ceiling_bytes()),
        estimator=estimator,
    )


class RowTagged(FakeBackend):
    """Each row's value and quantiles carry its first feature, so a
    misaligned row shows; optionally OOMs above `max_rows` rows."""

    def __init__(self, max_rows: int | None = None) -> None:
        super().__init__()
        self.max_rows = max_rows

    def predict(self, payload, X_test, return_proba=False, quantiles=None):
        if self.max_rows is not None and len(X_test) > self.max_rows:
            raise BackendComputeError("CUDA OOM")
        return PredictOutcome(
            predictions=[row[0] for row in X_test],
            quantiles=None
            if quantiles is None
            else [[row[0] + q for q in quantiles] for row in X_test],
        )


def tagged(rows, levels):
    return [[row[0] + q for q in levels] for row in rows]


@pytest.fixture
def engine():
    e = engine_with()
    e.fit(X, Y, task="regression", dataset_id="price")
    e.fit(X, ["a", "b", "a", "b", "a"], dataset_id="label")
    return e


class TestCheckQuantiles:
    def test_keeps_the_callers_order(self):
        assert check_quantiles([0.9, 0.1, 0.5]) == (0.9, 0.1, 0.5)
        assert check_quantiles(None) is None

    @pytest.mark.parametrize(
        "levels",
        [
            [],
            [0.0],
            [1.0],
            [-0.1],
            [1.5],
            [float("nan")],
            [0.5, 0.5],
            ["x"],
            "0.5",
            0.5,
        ],
    )
    def test_rejects(self, levels):
        with pytest.raises(InvalidInputError):
            check_quantiles(levels)

    def test_caps_the_count(self):
        check_quantiles([i / 100 for i in range(1, MAX_QUANTILES + 1)])
        with pytest.raises(InvalidInputError, match="at most"):
            check_quantiles([i / 1000 for i in range(1, MAX_QUANTILES + 2)])


class TestPredictOutcomeRows:
    def test_rows_and_concat_round_trip(self):
        whole = PredictOutcome(
            predictions=[1, 2, 3],
            probabilities=None,
            quantiles=[[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]],
        )
        parts = [whole.rows(0, 1), whole.rows(1, 3)]
        assert parts[1].quantiles == [[0.2, 0.8], [0.3, 0.7]]
        assert PredictOutcome.concat(parts) == whole


class TestEngine:
    def test_quantiles_of_each_row_at_each_level(self, engine):
        out = engine.predict("price", [[0.0, 1.0]] * 3, quantiles=[0.95, 0.05])
        expected = [DIST.inv_cdf(0.95), DIST.inv_cdf(0.05)]
        assert out.quantiles == [pytest.approx(expected)] * 3
        assert out.predictions == [pytest.approx(DIST.mean)] * 3

    def test_none_unless_asked(self, engine):
        assert engine.predict("price", [[0.0, 1.0]]).quantiles is None

    def test_classification_is_invalid(self, engine):
        with pytest.raises(InvalidInputError, match="regression"):
            engine.predict("label", [[0.0, 1.0]], quantiles=[0.5])

    def test_bad_levels_fail_before_the_device(self):
        backend = FakeBackend()
        e = engine_with(backend)
        e.fit(X, Y, task="regression", dataset_id="price")
        with pytest.raises(InvalidInputError):
            e.predict("price", [[0.0, 1.0]], quantiles=[2.0])
        assert backend.predict_calls == 0

    def test_rows_stay_aligned_when_a_device_oom_splits_them(self):
        """A device OOM halves the rows until they fit; the halves'
        quantiles must come back in row order."""
        e = engine_with(RowTagged(max_rows=2))
        e.fit(X, Y, task="regression", dataset_id="price")
        rows = [[float(i), 1.0] for i in range(7)]
        out = e.predict("price", rows, quantiles=[0.9, 0.1])
        assert out.quantiles == tagged(rows, [0.9, 0.1])

    def test_rows_stay_aligned_across_chunks(self, monkeypatch):
        monkeypatch.setattr("tabctx.engine.choose_chunk_rows", lambda *a: 3)
        e = engine_with(RowTagged())
        e.fit(X, Y, task="regression", dataset_id="price")
        rows = [[float(i), 1.0] for i in range(8)]
        out = e.predict("price", rows, quantiles=[0.5])
        assert out.quantiles == tagged(rows, [0.5])

    def test_a_model_without_quantiles_is_invalid_input(self):
        """A backend whose predict has no quantiles argument still serves
        plain predicts, and answers a request for quantiles with a 422."""

        class PointOnly(FakeBackend):
            def predict(self, payload, X_test, return_proba=False):
                return super().predict(payload, X_test, return_proba)

        e = engine_with(PointOnly())
        e.fit(X, Y, task="regression", dataset_id="price")
        assert e.predict("price", [[0.0, 1.0]]).predictions
        with pytest.raises(InvalidInputError, match="single value"):
            e.predict("price", [[0.0, 1.0]], quantiles=[0.5])

    def test_regression_asked_for_probabilities_has_an_empty_list(self, engine):
        assert (
            engine.predict("price", [[0.0, 1.0]], return_proba=True).probabilities == []
        )


class TestCoalescing:
    def test_requests_with_different_levels_are_not_merged(self, engine):
        predictor = CoalescingPredictor(engine, window_s=0.05)
        start = threading.Barrier(3)
        results = {}

        def call(name, levels):
            start.wait()
            results[name] = predictor.predict("price", [[0.0, 1.0]], quantiles=levels)

        threads = [
            threading.Thread(target=call, args=("narrow", [0.25, 0.75])),
            threading.Thread(target=call, args=("wide", [0.05, 0.95])),
            threading.Thread(target=call, args=("none", None)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
        assert results["narrow"].quantiles == [
            pytest.approx([DIST.inv_cdf(0.25), DIST.inv_cdf(0.75)])
        ]
        assert results["wide"].quantiles == [
            pytest.approx([DIST.inv_cdf(0.05), DIST.inv_cdf(0.95)])
        ]
        assert results["none"].quantiles is None

    def test_merged_requests_each_get_their_own_rows(self):
        e = engine_with(RowTagged())
        e.fit(X, Y, task="regression", dataset_id="price")
        predictor = CoalescingPredictor(e, window_s=0.3)
        start = threading.Barrier(4)
        rows = {i: [[float(10 * i + j), 1.0] for j in range(i + 1)] for i in range(4)}
        results = {}

        def call(i):
            start.wait()
            results[i] = predictor.predict("price", rows[i], quantiles=[0.1, 0.9])

        threads = [threading.Thread(target=call, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
        for i, out in results.items():
            assert out.quantiles == tagged(rows[i], [0.1, 0.9])
        assert predictor.engine_calls < 4  # they did share calls


def test_the_router_passes_quantiles_to_the_device():
    from tabctx.multidevice import MultiDeviceEngine

    shards = [engine_with(FakeBackend()) for _ in range(2)]
    router = MultiDeviceEngine(shards, ["cuda:0", "cuda:1"])
    router.fit(X, Y, task="regression", dataset_id="price")
    out = router.predict("price", [[0.0, 1.0]], quantiles=[0.5])
    assert out.quantiles == [[pytest.approx(DIST.mean)]]


class TestRegressionOutcome:
    def test_rows_by_levels(self):
        from tabctx.backends._outcome import regression_outcome

        out = regression_outcome([1.0, 2.0], [[0.5, 1.5], [1.5, 2.5]], 2)
        assert out.predictions == [1.0, 2.0]
        assert out.quantiles == [[0.5, 1.5], [1.5, 2.5]]

    def test_levels_by_rows_is_refused_not_scrambled(self):
        """3 rows x 2 levels arriving as (2, 3) would reshape silently."""
        from tabctx.backends._outcome import regression_outcome

        with pytest.raises(BackendComputeError, match="shape"):
            regression_outcome([1.0, 2.0, 3.0], np.zeros((2, 3)), 2)

    def test_point_only(self):
        from tabctx.backends._outcome import regression_outcome

        assert regression_outcome(np.array([1.0])).quantiles is None


def test_tabpfn_returns_the_mean_and_stacks_its_quantiles():
    """TabPFN's "main" output: the quantiles are one array per level."""
    from tabctx.backends.tabpfn import TabPFNBackend

    class Regressor:
        def predict(self, X, output_type="mean", quantiles=None):
            assert output_type == "main"
            n = len(X)
            return {
                "mean": np.full(n, 5.0),
                "quantiles": [np.full(n, q * 10) for q in quantiles],
            }

    backend = object.__new__(TabPFNBackend)
    out = backend._predict(Regressor(), np.zeros((2, 3)), False, (0.1, 0.9))
    assert out.predictions == [5.0, 5.0]
    assert out.quantiles == [[1.0, 9.0], [1.0, 9.0]]


class TestHttp:
    @pytest.fixture
    def app(self, monkeypatch):
        from tabctx.serve.factory import BACKEND_ENV_VAR
        from tabctx.serve.standalone import make_app
        from tabctx.serve.tenancy import REQUIRE_TENANT_ENV_VAR

        monkeypatch.setenv(BACKEND_ENV_VAR, "fake")
        monkeypatch.delenv(REQUIRE_TENANT_ENV_VAR, raising=False)
        return make_app()

    def _post(self, app, path, **kwargs):
        import asyncio

        import httpx

        async def go():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                return await c.post(path, **kwargs)

        return asyncio.run(go())

    def _fit(self, app):
        body = {"dataset_id": "price", "train_X": X, "train_y": Y, "task": "regression"}
        assert self._post(app, "/v1/tabctx/fit", json=body).status_code == 200

    def test_json(self, app):
        self._fit(app)
        body = {
            "dataset_id": "price",
            "test_X": [[0.0, 1.0]],
            "quantiles": [0.05, 0.95],
        }
        out = self._post(app, "/v1/tabctx/predict", json=body).json()
        assert out["quantile_levels"] == [0.05, 0.95]
        assert out["quantiles"] == [
            pytest.approx([DIST.inv_cdf(0.05), DIST.inv_cdf(0.95)])
        ]

    def test_a_form_field_lists_the_levels(self, app):
        self._fit(app)
        out = self._post(
            app,
            "/v1/tabctx/predict",
            data={"dataset_id": "price", "quantiles": "0.1, 0.9"},
            files={"file": ("t.csv", b"a,b\n0,1\n2,1\n", "text/csv")},
        ).json()
        assert out["quantile_levels"] == [0.1, 0.9] and len(out["quantiles"]) == 2

    @pytest.mark.parametrize("levels", [[1.5], [], ["x"]])
    def test_bad_levels_are_422(self, app, levels):
        self._fit(app)
        body = {"dataset_id": "price", "test_X": [[0.0, 1.0]], "quantiles": levels}
        assert self._post(app, "/v1/tabctx/predict", json=body).status_code == 422

    def test_bad_levels_fail_before_the_table_is_looked_up(self, app):
        body = {"dataset_id": "never-fit", "test_X": [[0.0]], "quantiles": [2]}
        response = self._post(app, "/v1/tabctx/predict", json=body)
        assert response.status_code == 422 and "quantiles" in response.text

    @pytest.mark.parametrize(
        "value", ["0.1,0.9", "0.1 0.9", "[0.1, 0.9]", " 0.1 , 0.9 "]
    )
    def test_form_spellings(self, app, value):
        self._fit(app)
        out = self._post(
            app,
            "/v1/tabctx/predict",
            data={"dataset_id": "price", "quantiles": value},
            files={"file": ("t.csv", b"a,b\n0,1\n", "text/csv")},
        ).json()
        assert out["quantile_levels"] == [0.1, 0.9]

    def test_a_repeated_form_field_says_how_to_send_a_list(self, app):
        self._fit(app)
        response = self._post(
            app,
            "/v1/tabctx/predict",
            data={"dataset_id": "price", "quantiles": ["0.1", "0.9"]},
            files={"file": ("t.csv", b"a,b\n0,1\n", "text/csv")},
        )
        assert response.status_code == 422 and "0.05,0.95" in response.text

    def test_json_takes_a_list_only(self, app):
        self._fit(app)
        body = {"dataset_id": "price", "test_X": [[0.0, 1.0]], "quantiles": "0.1,0.9"}
        assert self._post(app, "/v1/tabctx/predict", json=body).status_code == 422

    def test_the_form_schema_documents_one_value(self, app):
        body = app.openapi()["paths"]["/v1/tabctx/predict"]["post"]["requestBody"]
        form = body["content"]["multipart/form-data"]["schema"]["properties"]
        assert form["quantiles"]["type"] == "string"
        json_levels = body["content"]["application/json"]["schema"]["properties"]
        assert any(
            alt.get("type") == "array" for alt in json_levels["quantiles"]["anyOf"]
        )

    def test_none_unless_asked(self, app):
        self._fit(app)
        body = {"dataset_id": "price", "test_X": [[0.0, 1.0]]}
        out = self._post(app, "/v1/tabctx/predict", json=body).json()
        assert out["quantiles"] is None and out["quantile_levels"] is None
