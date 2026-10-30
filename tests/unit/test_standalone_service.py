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

import httpx
import pytest
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

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
