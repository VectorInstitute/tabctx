"""Survival and causal fits over HTTP (the standalone service in-process,
fake backend): their second column inline or in a table file, the task it
implies, and a predict's survival probabilities."""

import asyncio

import httpx
import pytest

from tabctx.serve.factory import (
    BACKEND_ENV_VAR,
    DEVICES_ENV_VAR,
    WORKER_PROCESSES_ENV_VAR,
)
from tabctx.serve.standalone import make_app
from tabctx.serve.tenancy import REQUIRE_TENANT_ENV_VAR

X = [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]]
TIMES = [1.0, 2.0, 3.0, 4.0]
FLAGS = [1, 0, 1, 1]


def _request(app, method: str, path: str, **kwargs) -> httpx.Response:
    async def go() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            return await c.request(method, path, **kwargs)

    return asyncio.run(go())


@pytest.fixture(params=["one device", "two devices"])
def app(request, monkeypatch):
    """The service on one device, and on two (a multi-device engine, as
    on a multi-GPU replica; threads, so no worker processes to start)."""
    monkeypatch.setenv(BACKEND_ENV_VAR, "fake")
    monkeypatch.delenv(REQUIRE_TENANT_ENV_VAR, raising=False)
    if request.param == "two devices":
        monkeypatch.setenv(DEVICES_ENV_VAR, "cuda:0,cuda:1")
        monkeypatch.setenv(WORKER_PROCESSES_ENV_VAR, "0")
    return make_app()


def fit(app, **body) -> httpx.Response:
    return _request(app, "POST", "/v1/tabctx/fit", json=body)


def predict(app, **body) -> httpx.Response:
    return _request(app, "POST", "/v1/tabctx/predict", json=body)


class TestSurvival:
    def test_an_event_column_implies_survival(self, app):
        r = fit(app, train_X=X, train_y=TIMES, train_event=FLAGS, dataset_id="s")
        assert r.status_code == 200, r.text
        assert r.json()["task"] == "survival"

        p = predict(app, dataset_id="s", test_X=X[:2], times=[0, 2.5], quantiles=[0.5])
        assert p.status_code == 200, p.text
        body = p.json()
        assert body["predictions"] == [2.5, 2.5]
        assert body["survival"] == [[1.0, 0.5], [1.0, 0.5]]
        assert body["survival_times"] == [0, 2.5]
        assert body["quantiles"] == [[2.5], [2.5]]

    def test_from_a_table_file(self, app):
        csv = b"a,b,time,died\n0,1,1.0,1\n1,0,2.0,0\n2,1,3.0,1\n3,0,4.0,1\n"
        r = _request(
            app,
            "POST",
            "/v1/tabctx/fit",
            data={"dataset_id": "f", "target": "time", "event_column": "died"},
            files={"file": ("train.csv", csv, "text/csv")},
        )
        assert r.status_code == 200, r.text
        assert (r.json()["task"], r.json()["n_features"]) == ("survival", 2)
        test = b"b,a\n1,0\n"  # matched by name
        p = _request(
            app,
            "POST",
            "/v1/tabctx/predict",
            data={"dataset_id": "f", "times": "2.5,10"},
            files={"file": ("test.csv", test, "text/csv")},
        )
        assert p.status_code == 200, p.text
        assert p.json()["survival"] == [[0.5, 0.0]]

    def test_needs_its_event_column(self, app):
        r = fit(app, train_X=X, train_y=TIMES, task="survival")
        assert r.status_code == 422
        assert "needs its event column" in r.json()["detail"]

    def test_bad_times_are_a_422_before_any_table_is_read(self, app):
        r = predict(app, dataset_id="nope", test_X=X, times=[-1])
        assert r.status_code == 422

    def test_times_on_a_regression_context(self, app):
        fit(
            app,
            train_X=X,
            train_y=[0.5, 1.5, 2.5, 3.5],
            task="regression",
            dataset_id="r",
        )
        r = predict(app, dataset_id="r", test_X=X, times=[1.0])
        assert r.status_code == 422
        assert "times are for survival" in r.json()["detail"]


class TestCausal:
    def test_a_treatment_column_implies_causal(self, app):
        r = fit(
            app,
            train_X=X,
            train_y=[1.0, 2.0, 1.5, 3.5],
            train_treatment=[0, 1, 0, 1],
            dataset_id="c",
        )
        assert r.status_code == 200, r.text
        assert r.json()["task"] == "causal"
        p = predict(app, dataset_id="c", test_X=X[:1])
        assert p.json()["predictions"] == [pytest.approx(1.5)]

    def test_both_arms_are_needed(self, app):
        r = fit(app, train_X=X, train_y=TIMES, train_treatment=[1, 1, 1, 1])
        assert r.status_code == 422
        assert "both treated" in r.json()["detail"]


class TestRequestShape:
    def test_event_and_treatment_together(self, app):
        r = fit(app, train_X=X, train_y=TIMES, train_event=FLAGS, train_treatment=FLAGS)
        assert r.status_code == 422
        assert "not both" in r.json()["detail"]

    def test_a_second_column_that_contradicts_the_task(self, app):
        r = fit(app, train_X=X, train_y=TIMES, train_event=FLAGS, task="regression")
        assert r.status_code == 422
        assert "event column is for survival fits" in r.json()["detail"]

    def test_a_column_name_in_an_inline_fit(self, app):
        r = fit(app, train_X=X, train_y=TIMES, event_column="died")
        assert r.status_code == 422
        assert "inline fit sends train_event" in r.json()["detail"]

    def test_an_inline_second_column_with_a_file(self, app):
        r = _request(
            app,
            "POST",
            "/v1/tabctx/fit",
            data={"train_event": "1,0"},
            files={"file": ("t.csv", b"a,t\n1,2\n2,3\n", "text/csv")},
        )
        assert r.status_code == 422

    def test_two_values_per_row_without_a_task(self, app):
        r = fit(app, train_X=X, train_y=[[1.0, 1]] * 4)
        assert r.status_code == 422
