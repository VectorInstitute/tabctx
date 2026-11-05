"""Unit tests for the standalone host (serve/standalone.py).

The full HTTP service runs in-process over httpx's ASGI transport -- no
server, no ray. This suite runs in the unit environment, which installs
neither, so a pass is itself the proof that the standalone host needs
nothing of Ray's: the same service the integration suite exercises
through a live Ray Serve cluster answers here from a plain FastAPI app.
"""

import asyncio
import subprocess
import sys
import time
import urllib.parse

import httpx
import pytest
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from tabctx.errors import CacheUnavailableError
from tabctx.serve.factory import BACKEND_ENV_VAR
from tabctx.serve.service import ROUTE_SPECS, TabctxService, fastapi_app
from tabctx.serve.standalone import (
    MAX_ONGOING_ENV_VAR,
    MAX_QUEUED_ENV_VAR,
    _RequestLimiter,
    main,
    make_app,
)
from tabctx.serve.tenancy import REQUIRE_TENANT_ENV_VAR, TENANT_HEADER

TRAIN = {
    "train_X": [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.9, 0.1]],
    "train_y": ["a", "b", "a", "b"],
    "task": "classification",
}
TEST_X = [[0.1, 0.9], [0.8, 0.2]]


def _request(app, method: str, path: str, **kwargs) -> httpx.Response:
    async def go() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(go())


@pytest.fixture()
def app(monkeypatch):
    monkeypatch.setenv(BACKEND_ENV_VAR, "fake")
    monkeypatch.delenv(REQUIRE_TENANT_ENV_VAR, raising=False)
    return make_app()


class TestHttpContract:
    def test_fit_then_predict_round_trip(self, app):
        fit = _request(
            app, "POST", "/v1/tabctx/fit", json={**TRAIN, "dataset_id": "d1"}
        )
        assert fit.status_code == 200, fit.text
        body = fit.json()
        assert body["dataset_id"] == "d1"
        assert body["model"] == "fake"
        assert body["n_train"] == 4
        # Outside any Ray replica there is no replica tag, by design.
        assert body["served_by"] is None
        assert body["timings_ms"] is not None

        predict = _request(
            app,
            "POST",
            "/v1/tabctx/predict",
            json={"dataset_id": "d1", "test_X": TEST_X},
        )
        assert predict.status_code == 200, predict.text
        assert len(predict.json()["predictions"]) == len(TEST_X)

    def test_fit_with_multipart_table_file(self, app):
        csv = b"f1,f2,target\n0.0,1.0,a\n1.0,0.0,b\n0.5,0.5,a\n"
        fit = _request(
            app,
            "POST",
            "/v1/tabctx/fit",
            data={"dataset_id": "m1", "task": "classification"},
            files={"file": ("train.csv", csv, "text/csv")},
        )
        assert fit.status_code == 200, fit.text
        assert fit.json()["n_train"] == 3

        predict = _request(
            app,
            "POST",
            "/v1/tabctx/predict",
            json={"dataset_id": "m1", "test_X": [[0.2, 0.8]]},
        )
        assert predict.status_code == 200, predict.text

    def test_models_lists_the_fake_backend(self, app):
        r = _request(app, "GET", "/v1/models")
        assert r.status_code == 200
        assert r.json()["data"] == [{"id": "fake", "object": "model", "default": True}]

    def test_health_routes(self, app):
        assert _request(app, "GET", "/healthz").json() == {"status": "ok"}
        assert _request(app, "GET", "/livez").json() == {"status": "alive"}
        ready = _request(app, "GET", "/readyz")
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready"
        assert ready.json()["replica"] is None

    def test_predict_against_unknown_dataset_is_404(self, app):
        r = _request(
            app,
            "POST",
            "/v1/tabctx/predict",
            json={"dataset_id": "no", "test_X": TEST_X},
        )
        assert r.status_code == 404

    def test_malformed_json_body_is_422(self, app):
        r = _request(
            app,
            "POST",
            "/v1/tabctx/fit",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        assert r.status_code == 422

    def test_openapi_keeps_the_multipart_request_body(self, app):
        spec = _request(app, "GET", "/openapi.json").json()
        content = spec["paths"]["/v1/tabctx/fit"]["post"]["requestBody"]["content"]
        assert set(content) == {"application/json", "multipart/form-data"}


class TestTenancy:
    @pytest.fixture()
    def app(self, monkeypatch):
        monkeypatch.setenv(BACKEND_ENV_VAR, "fake")
        monkeypatch.setenv(REQUIRE_TENANT_ENV_VAR, "true")
        return make_app()

    def test_tenant_required_and_scoped(self, app):
        no_header = _request(
            app, "POST", "/v1/tabctx/fit", json={**TRAIN, "dataset_id": "d1"}
        )
        assert no_header.status_code == 401

        fit = _request(
            app,
            "POST",
            "/v1/tabctx/fit",
            json={**TRAIN, "dataset_id": "d1"},
            headers={TENANT_HEADER: "tenant-a"},
        )
        assert fit.status_code == 200, fit.text
        assert fit.json()["dataset_id"] == "d1"  # scoping never leaks

        own = _request(
            app,
            "POST",
            "/v1/tabctx/predict",
            json={"dataset_id": "d1", "test_X": TEST_X},
            headers={TENANT_HEADER: "tenant-a"},
        )
        assert own.status_code == 200

        other = _request(
            app,
            "POST",
            "/v1/tabctx/predict",
            json={"dataset_id": "d1", "test_X": TEST_X},
            headers={TENANT_HEADER: "tenant-b"},
        )
        assert other.status_code == 404


class TestLiveness:
    class _DeadShard:
        alive = False
        death_reason = "worker for cuda:1 died: CUDA error"

    def test_livez_fails_only_on_dead_workers(self, monkeypatch):
        monkeypatch.setenv(BACKEND_ENV_VAR, "fake")
        monkeypatch.delenv(REQUIRE_TENANT_ENV_VAR, raising=False)
        service = TabctxService()
        app = make_app(service)
        assert _request(app, "GET", "/livez").status_code == 200

        service._workers = (self._DeadShard(),)
        dead = _request(app, "GET", "/livez")
        assert dead.status_code == 503
        assert "cuda:1" in dead.json()["detail"]
        # /healthz stays a bare process-is-up probe.
        assert _request(app, "GET", "/healthz").status_code == 200

    def test_close_stops_worker_shards(self, monkeypatch):
        monkeypatch.setenv(BACKEND_ENV_VAR, "fake")
        service = TabctxService()
        closed = []

        class Shard:
            def close(self):
                closed.append(True)

        service._workers = (Shard(), Shard())
        service.close()
        assert closed == [True, True]


class TestRouteSpecs:
    def test_every_registered_route_is_rebound(self, app):
        want = {(spec.path, method) for spec in ROUTE_SPECS for method in spec.methods}
        source = {
            (r.path, m)
            for r in fastapi_app.routes
            if isinstance(r, APIRoute)
            for m in r.methods - {"HEAD"}
        }
        got = {
            (r.path, m)
            for r in app.routes
            if isinstance(r, APIRoute)
            for m in r.methods - {"HEAD"}
        }
        assert want == source
        assert want <= got  # the made app adds only its own docs routes


class TestRequestLimiter:
    def test_sheds_beyond_the_queue_and_never_sheds_probes(self):
        async def scenario():
            release = asyncio.Event()
            started = asyncio.Event()

            async def app(scope, receive, send):
                if scope["path"] == "/livez":
                    await JSONResponse({"status": "alive"})(scope, receive, send)
                    return
                started.set()
                await release.wait()
                await JSONResponse({"ok": True})(scope, receive, send)

            limiter = _RequestLimiter(app, max_ongoing=1, max_queued=1)
            transport = httpx.ASGITransport(app=limiter)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                first = asyncio.create_task(client.get("/work"))
                await started.wait()  # holds the one slot
                second = asyncio.create_task(client.get("/work"))
                for _ in range(200):  # until the second request is queued
                    if limiter._queued == 1:
                        break
                    await asyncio.sleep(0.005)
                assert limiter._queued == 1

                shed = await client.get("/work")  # over the queue cap
                assert shed.status_code == 503
                assert "retry" in shed.json()["detail"]

                probe = await client.get("/livez")  # bypasses the limiter
                assert probe.status_code == 200

                release.set()
                assert (await first).status_code == 200
                assert (await second).status_code == 200

        asyncio.run(scenario())

    def test_rejects_bad_limit_configuration(self, monkeypatch):
        monkeypatch.setenv(BACKEND_ENV_VAR, "fake")
        monkeypatch.setenv(MAX_ONGOING_ENV_VAR, "0")
        with pytest.raises(ValueError, match=MAX_ONGOING_ENV_VAR):
            make_app()
        monkeypatch.setenv(MAX_ONGOING_ENV_VAR, "8")
        monkeypatch.setenv(MAX_QUEUED_ENV_VAR, "many")
        with pytest.raises(ValueError, match=MAX_QUEUED_ENV_VAR):
            make_app()


class TestEntrypoint:
    def test_help_exits_before_importing_uvicorn(self):
        # argparse handles --help and exits; uvicorn (absent from the unit
        # environment) is only imported after argument parsing.
        with pytest.raises(SystemExit) as e:
            main(["--help"])
        assert e.value.code == 0

    def test_standalone_import_pulls_in_no_ray(self):
        # A fresh interpreter, because this test run itself may have ray
        # importable (e.g. a local env with the serve extra).
        code = (
            "import sys; import tabctx.serve.standalone; "
            "assert 'ray' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], check=True)


def _fit(app, dataset_id: str, headers: dict | None = None) -> None:
    fit = _request(
        app,
        "POST",
        "/v1/tabctx/fit",
        json={**TRAIN, "dataset_id": dataset_id},
        headers=headers or {},
    )
    assert fit.status_code == 200, fit.text


def _listed(app, headers: dict | None = None) -> list[str]:
    response = _request(app, "GET", "/v1/tabctx/datasets", headers=headers or {})
    assert response.status_code == 200, response.text
    return [d["dataset_id"] for d in response.json()["data"]]


def _delete(app, dataset_id: str, headers: dict | None = None) -> httpx.Response:
    path = f"/v1/tabctx/datasets/{urllib.parse.quote(dataset_id, safe='')}"
    return _request(app, "DELETE", path, headers=headers or {})


class TestDatasets:
    """GET /v1/tabctx/datasets and DELETE /v1/tabctx/datasets/{id}."""

    def test_lists_nothing_before_any_fit(self, app):
        assert _listed(app) == []

    def test_lists_each_context_with_its_metadata(self, app):
        before = time.time()
        _fit(app, "b")
        _fit(app, "a")
        data = _request(app, "GET", "/v1/tabctx/datasets").json()["data"]
        assert [d["dataset_id"] for d in data] == ["a", "b"]  # sorted
        first = data[0]
        assert {k: first[k] for k in ("model", "task", "n_train", "n_features")} == {
            "model": "fake",
            "task": "classification",
            "n_train": 4,
            "n_features": 2,
        }
        assert first["location"] == "resident" and first["device"]
        assert first["bytes"] > 0
        assert before <= first["fitted_at"] <= time.time()

    def test_delete_removes_the_context(self, app):
        _fit(app, "gone")
        _fit(app, "kept")
        response = _delete(app, "gone")
        assert response.status_code == 200, response.text
        assert response.json() == {"dataset_id": "gone", "deleted": True}
        assert _listed(app) == ["kept"]
        predict = _request(
            app,
            "POST",
            "/v1/tabctx/predict",
            json={"dataset_id": "gone", "test_X": TEST_X},
        )
        assert predict.status_code == 404

    def test_deleting_what_is_not_there_is_404(self, app):
        assert _delete(app, "never-fit").status_code == 404
        _fit(app, "twice")
        assert _delete(app, "twice").status_code == 200
        second = _delete(app, "twice")
        assert second.status_code == 404
        assert "already deleted" in second.json()["detail"]

    def test_ids_with_slashes_and_spaces(self, app):
        _fit(app, "team/churn v1")
        assert _listed(app) == ["team/churn v1"]
        assert _delete(app, "team/churn v1").status_code == 200
        assert _listed(app) == []

    def test_session_header_must_match_the_id(self, app):
        _fit(app, "d1")
        response = _delete(app, "d1", headers={"x-session-id": "other"})
        assert response.status_code == 422
        assert _listed(app) == ["d1"]

    def test_delete_forgets_the_schema(self, monkeypatch):
        monkeypatch.setenv(BACKEND_ENV_VAR, "fake")
        monkeypatch.delenv(REQUIRE_TENANT_ENV_VAR, raising=False)
        service = TabctxService()
        app = make_app(service)
        csv = b"f1,color,target\n0.0,red,a\n1.0,blue,b\n0.5,red,a\n"
        fit = _request(
            app,
            "POST",
            "/v1/tabctx/fit",
            data={"dataset_id": "csv", "target": "target"},
            files={"file": ("train.csv", csv, "text/csv")},
        )
        assert fit.status_code == 200, fit.text
        assert service._schemas.get("csv") is not None
        assert _delete(app, "csv").status_code == 200
        assert service._schemas.get("csv") is None

    def test_a_busy_worker_or_a_move_in_progress_is_503(self, monkeypatch):
        """Retryable: nothing was listed or deleted."""
        monkeypatch.setenv(BACKEND_ENV_VAR, "fake")
        monkeypatch.delenv(REQUIRE_TENANT_ENV_VAR, raising=False)
        service = TabctxService()
        app = make_app(service)

        def busy(*args):
            raise CacheUnavailableError("cuda:1 didn't list its contexts; retry")

        monkeypatch.setattr(service._engine, "contexts", busy)
        monkeypatch.setattr(service._engine, "evict", busy)
        assert _request(app, "GET", "/v1/tabctx/datasets").status_code == 503
        assert _delete(app, "d").status_code == 503

    def test_an_empty_id_is_invalid(self, app):
        assert _request(app, "DELETE", "/v1/tabctx/datasets/").status_code == 422

    def test_the_listing_names_the_replica(self, app):
        body = _request(app, "GET", "/v1/tabctx/datasets").json()
        assert body == {"data": [], "served_by": None}  # no Ray replica here

    def test_routes_are_in_the_openapi_spec(self, app):
        paths = _request(app, "GET", "/openapi.json").json()["paths"]
        assert "get" in paths["/v1/tabctx/datasets"]
        assert "delete" in paths["/v1/tabctx/datasets/{dataset_id}"]


class TestDatasetsPerTenant:
    @pytest.fixture()
    def app(self, monkeypatch):
        monkeypatch.setenv(BACKEND_ENV_VAR, "fake")
        monkeypatch.setenv(REQUIRE_TENANT_ENV_VAR, "true")
        return make_app()

    A = {TENANT_HEADER: "tenant-a"}
    B = {TENANT_HEADER: "tenant-b"}

    def test_a_tenant_lists_only_its_own(self, app):
        _fit(app, "shared-name", self.A)
        _fit(app, "a-only", self.A)
        _fit(app, "shared-name", self.B)
        assert _listed(app, self.A) == ["a-only", "shared-name"]
        assert _listed(app, self.B) == ["shared-name"]

    def test_a_tenant_deletes_only_its_own(self, app):
        _fit(app, "d", self.A)
        assert _delete(app, "d", self.B).status_code == 404
        _fit(app, "d", self.B)
        assert _delete(app, "d", self.B).status_code == 200
        assert _listed(app, self.A) == ["d"]
        assert _listed(app, self.B) == []

    def test_both_routes_need_a_tenant(self, app):
        assert _request(app, "GET", "/v1/tabctx/datasets").status_code == 401
        assert _delete(app, "d").status_code == 401

    def test_a_malformed_tenant_is_invalid(self, app):
        bad = {TENANT_HEADER: "no:colons"}
        assert (
            _request(app, "GET", "/v1/tabctx/datasets", headers=bad).status_code == 422
        )
        assert _delete(app, "d", bad).status_code == 422


class TestDatasetsWithoutTenancy:
    def test_unscoped_callers_never_see_tenant_contexts(self, app):
        _fit(app, "theirs", {TENANT_HEADER: "tenant-a"})
        _fit(app, "mine")
        assert _listed(app) == ["mine"]

    def test_unscoped_callers_cannot_name_a_tenant_context(self, app):
        """Not by its scoped id either: no delete, predict or overwrite."""
        tenant = {TENANT_HEADER: "acme"}
        _fit(app, "secret", tenant)
        scoped = "t:acme:secret"
        assert _delete(app, scoped).status_code == 422
        predict = _request(
            app,
            "POST",
            "/v1/tabctx/predict",
            json={"dataset_id": scoped, "test_X": TEST_X},
        )
        assert predict.status_code == 422
        fit = _request(
            app, "POST", "/v1/tabctx/fit", json={**TRAIN, "dataset_id": scoped}
        )
        assert fit.status_code == 422 and "reserved" in fit.json()["detail"]
        assert _listed(app, tenant) == ["secret"]


class TestOpenApi:
    def test_the_spec_needs_no_service(self, app, monkeypatch):
        """The docs build uses openapi_spec(): the same spec as a running
        server's, with no engine built (a real backend set in the
        environment must not matter)."""
        from tabctx.serve.standalone import openapi_spec

        expected = app.openapi()
        monkeypatch.setenv(BACKEND_ENV_VAR, "tabicl")  # would need torch
        assert openapi_spec() == expected

    def test_every_route_has_a_summary_and_one_known_tag(self, app):
        from tabctx.serve.service import OPENAPI_TAGS

        spec = app.openapi()
        assert spec["tags"] == OPENAPI_TAGS
        known = {t["name"] for t in OPENAPI_TAGS}
        for path, methods in spec["paths"].items():
            for op in methods.values():
                assert op.get("summary"), path
                assert len(op["tags"]) == 1 and op["tags"][0] in known, path

    def test_the_spec_names_the_service_and_its_version(self, app):
        info = app.openapi()["info"]
        assert info["title"] == "tabctx" and info["version"]


class TestReadyzNeverWaits:
    """/readyz answers from measurements it doesn't wait for (#51)."""

    @pytest.fixture()
    def service(self, monkeypatch):
        monkeypatch.setenv(BACKEND_ENV_VAR, "fake")
        monkeypatch.delenv(REQUIRE_TENANT_ENV_VAR, raising=False)
        service = TabctxService()
        yield service
        service.close()

    def test_a_stuck_device_measurement_doesnt_hold_it_up(self, service):
        """The last measurement is shown, and ages, while the next waits."""
        import threading

        from tabctx.serve.reporting import PeriodicReport

        calls, measuring, release = [], threading.Event(), threading.Event()

        def measure():
            calls.append(1)
            if len(calls) > 1:  # e.g. torch's allocator lock behind a fit
                measuring.set()
                release.wait(5)
            return {"gpu_memory": {}, "spill": {"n_spilled_contexts": len(calls)}}

        service._local_report.stop()
        service._local_report = PeriodicReport(measure, interval_s=0.01)
        app = make_app(service)
        try:
            assert measuring.wait(5)
            start = time.monotonic()
            first = _request(app, "GET", "/readyz").json()
            later = _request(app, "GET", "/readyz").json()
            assert time.monotonic() - start < 1.0
            assert first["spill"] == {"n_spilled_contexts": 1}
            assert later["device_report_age_s"] >= first["device_report_age_s"]
        finally:
            release.set()

    def test_a_held_host_memory_lock_doesnt_hold_it_up(self, service):
        from tabctx.memory.host import HostMemoryGuard

        guard = HostMemoryGuard(2**30, lock_timeout_s=30)
        service._host_guard = guard
        app = make_app(service)
        with guard._locked():  # another call holds the guard's lock
            start = time.monotonic()
            response = _request(app, "GET", "/readyz")
        assert response.status_code == 200
        assert time.monotonic() - start < 1.0
        assert response.json()["host_memory"]["budget_bytes"] == guard.budget_bytes

    class _Worker:
        alive = True
        death_reason = None
        device = "cuda:1"

        def __init__(self, age):
            self.age = age

        def close(self):
            pass

        def latest_report(self):
            report = {
                "gpu_memory": {"allocated_mb": 5.0, "reserved_mb": 8.0},
                "spill": None,
            }
            return report, self.age

    def test_worker_reports_and_their_age(self, service):
        service._workers = (self._Worker(age=3.2),)
        body = _request(make_app(service), "GET", "/readyz").json()
        assert body["real_gpu_memory"]["allocated_mb"] == 5.0
        assert body["device_report_age_s"] >= 3.2

    def test_a_worker_silent_for_too_long_is_not_ready(self, service):
        from tabctx.serve.reporting import REPORT_STALE_S

        service._workers = (self._Worker(age=REPORT_STALE_S + 1),)
        response = _request(make_app(service), "GET", "/readyz")
        assert response.status_code == 503 and "cuda:1" in response.json()["detail"]
