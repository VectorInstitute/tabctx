"""Unit tests for the pure-stdlib HTTP client (client.py).

tabctx.client._open is monkeypatched throughout -- no real network I/O,
no running server required. This exercises exactly what the client
promises callers: automatic affinity/tenant headers, response parsing,
server-error -> tabctx-exception mapping, and 503 backpressure retry.
"""

from __future__ import annotations

import email.message
import io
import json
import urllib.error
import urllib.request

import pytest

from tabctx.client import PredictResult, TabctxBackpressureError, TabctxClient
from tabctx.errors import (
    AdmissionRejected,
    BackendComputeError,
    DatasetNotFoundError,
    InvalidInputError,
    TabctxError,
    UploadNotFoundError,
)


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._payload


def _json_response(body: dict) -> _FakeResponse:
    return _FakeResponse(json.dumps(body).encode())


def _http_error(
    code: int, detail_body: dict | bytes | None = None
) -> urllib.error.HTTPError:
    if isinstance(detail_body, bytes):
        payload = detail_body
    else:
        payload = json.dumps({} if detail_body is None else detail_body).encode()
    return urllib.error.HTTPError(
        url="http://example.test",
        code=code,
        msg="err",
        hdrs=email.message.Message(),
        fp=io.BytesIO(payload),
    )


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    """Retry backoff must never slow the test suite down."""
    sleeps: list[float] = []
    monkeypatch.setattr("tabctx.client.time.sleep", lambda s: sleeps.append(s))
    return sleeps


def _install(monkeypatch, handler):
    """handler(req) -> _FakeResponse, or raises urllib.error.HTTPError."""
    monkeypatch.setattr("tabctx.client._open", lambda req, timeout=None: handler(req))


class TestMapStatus:
    @pytest.mark.parametrize(
        "code,exc_type",
        [
            (422, InvalidInputError),
            (413, AdmissionRejected),
            (507, BackendComputeError),
            (401, PermissionError),
            (503, TabctxBackpressureError),
        ],
    )
    def test_maps_known_codes(self, code, exc_type):
        assert isinstance(TabctxClient._map_status(code, "detail"), exc_type)

    def test_404_without_upload_in_detail_is_dataset_not_found(self):
        assert isinstance(
            TabctxClient._map_status(404, "no cached context for dataset_id='x'"),
            DatasetNotFoundError,
        )

    def test_404_with_upload_in_detail_is_upload_not_found(self):
        assert isinstance(
            TabctxClient._map_status(404, "no upload 'x' on this replica"),
            UploadNotFoundError,
        )

    def test_unknown_code_falls_back_to_generic_error(self):
        err = TabctxClient._map_status(500, "boom")
        assert isinstance(err, TabctxError)
        assert "500" in str(err) and "boom" in str(err)


class TestDetail:
    def test_parses_detail_field(self):
        e = _http_error(422, {"detail": "train_X must be non-empty"})
        assert TabctxClient._detail(e) == "train_X must be non-empty"

    def test_falls_back_to_whole_payload_without_detail_key(self):
        e = _http_error(422, {"other": 1})
        assert TabctxClient._detail(e) == str({"other": 1})

    def test_falls_back_to_http_code_on_unparseable_body(self):
        e = _http_error(500, detail_body=b"not json")
        assert TabctxClient._detail(e) == "HTTP 500"


class TestFit:
    def test_sends_body_and_session_header(self, monkeypatch):
        captured = {}

        def handler(req):
            captured["url"] = req.full_url
            captured["method"] = req.get_method()
            captured["headers"] = req.headers
            captured["body"] = json.loads(req.data.decode())
            return _json_response({"dataset_id": "ds-1"})

        _install(monkeypatch, handler)
        client = TabctxClient("http://localhost:8000")
        result = client.fit([[1.0, 2.0]], ["a"], dataset_id="ds-1", model="tabicl-v2")

        assert result == "ds-1"
        assert captured["url"] == "http://localhost:8000/v1/tabctx/fit"
        assert captured["method"] == "POST"
        assert captured["headers"]["X-session-id"] == "ds-1"
        # No task: the server infers it from train_y.
        assert captured["body"] == {
            "train_X": [[1.0, 2.0]],
            "train_y": ["a"],
            "dataset_id": "ds-1",
            "model": "tabicl-v2",
        }

    def test_sends_task_when_given(self, monkeypatch):
        captured = {}

        def handler(req):
            captured["body"] = json.loads(req.data.decode())
            return _json_response({"dataset_id": "ds-1"})

        _install(monkeypatch, handler)
        TabctxClient("http://localhost:8000").fit([[1.0]], [2.5], task="regression")
        assert captured["body"]["task"] == "regression"

    def test_omits_dataset_id_and_model_when_not_given(self, monkeypatch):
        captured = {}

        def handler(req):
            captured["body"] = json.loads(req.data.decode())
            captured["headers"] = req.headers
            return _json_response({"dataset_id": "server-generated"})

        _install(monkeypatch, handler)
        client = TabctxClient("http://localhost:8000")
        client.fit([[1.0]], ["a"])

        assert "dataset_id" not in captured["body"]
        assert "model" not in captured["body"]
        assert "X-session-id" not in captured["headers"]

    def test_base_url_trailing_slash_stripped(self, monkeypatch):
        captured = {}
        _install(
            monkeypatch,
            lambda req: (
                captured.__setitem__("url", req.full_url),
                _json_response({"dataset_id": "x"}),
            )[1],
        )
        TabctxClient("http://localhost:8000/").fit([[1.0]], ["a"])
        assert captured["url"] == "http://localhost:8000/v1/tabctx/fit"


class TestSessionHeaderName:
    def test_custom_session_header_name(self, monkeypatch):
        # Deployments may rename the affinity header via
        # RAY_SERVE_SESSION_ID_HEADER_KEY; the client must follow suit or
        # affinity silently degrades to uniform routing.
        captured = {}
        _install(
            monkeypatch,
            lambda req: (
                captured.__setitem__("headers", req.headers),
                _json_response({"dataset_id": "x"}),
            )[1],
        )
        client = TabctxClient(
            "http://localhost:8000", session_header="x-tabctx-session-id"
        )
        client.fit([[1.0]], ["a"], dataset_id="ds-1")
        assert captured["headers"]["X-tabctx-session-id"] == "ds-1"
        assert "X-session-id" not in captured["headers"]


class TestTenantHeader:
    def test_tenant_header_sent_when_configured(self, monkeypatch):
        captured = {}
        _install(
            monkeypatch,
            lambda req: (
                captured.__setitem__("headers", req.headers),
                _json_response({"dataset_id": "x"}),
            )[1],
        )
        TabctxClient("http://localhost:8000", tenant_id="acme").fit([[1.0]], ["a"])
        assert captured["headers"]["X-tabctx-tenant-id"] == "acme"

    def test_tenant_header_absent_by_default(self, monkeypatch):
        captured = {}
        _install(
            monkeypatch,
            lambda req: (
                captured.__setitem__("headers", req.headers),
                _json_response({"dataset_id": "x"}),
            )[1],
        )
        TabctxClient("http://localhost:8000").fit([[1.0]], ["a"])
        assert "X-tabctx-tenant-id" not in captured["headers"]


class TestUpload:
    def test_upload_csv_sends_bytes_with_content_type_and_session_header(
        self, monkeypatch
    ):
        captured = {}

        def handler(req):
            captured["url"] = req.full_url
            captured["headers"] = req.headers
            captured["data"] = req.data
            captured["method"] = req.get_method()
            return _json_response({"upload_id": "up-1"})

        _install(monkeypatch, handler)
        client = TabctxClient("http://localhost:8000")
        upload_id = client.upload_csv(b"f0,f1\n1,2\n", dataset_id="ds-1")

        assert upload_id == "up-1"
        assert captured["url"] == "http://localhost:8000/v1/tabctx/upload"
        assert captured["method"] == "POST"
        assert captured["data"] == b"f0,f1\n1,2\n"
        assert captured["headers"]["Content-type"] == "text/csv"
        assert captured["headers"]["X-session-id"] == "ds-1"

    def test_upload_csv_file_streams_the_open_file(self, monkeypatch, tmp_path):
        # The by-reference path exists for tables too big for inline
        # JSON, so the client must hand urllib the open file (streamed
        # in blocks) with an explicit Content-Length, not f.read().
        p = tmp_path / "t.csv"
        p.write_bytes(b"a,b\n1,2\n")
        captured = {}

        def handler(req):
            captured["headers"] = req.headers
            captured["is_file"] = hasattr(req.data, "read")
            captured["data"] = req.data.read()
            return _json_response({"upload_id": "up-2"})

        _install(monkeypatch, handler)
        client = TabctxClient("http://localhost:8000")
        assert client.upload_csv_file(str(p), dataset_id="ds-1") == "up-2"
        assert captured["is_file"]
        assert captured["data"] == b"a,b\n1,2\n"
        assert captured["headers"]["Content-length"] == str(len(b"a,b\n1,2\n"))
        assert captured["headers"]["Content-type"] == "text/csv"

    def test_upload_csv_file_real_http_roundtrip(self, monkeypatch, tmp_path):
        """No urlopen stub: a real local HTTP server receives the
        streamed body, proving urllib actually accepts the file object
        + Content-Length combination end to end."""
        import http.server
        import threading

        monkeypatch.undo()  # drop the autouse urlopen/sleep patches
        body = b"f0,f1\n" + b"\n".join(b"%d,%d" % (i, i * 2) for i in range(5000))
        p = tmp_path / "big.csv"
        p.write_bytes(body)
        received = {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers["Content-Length"])
                received["body"] = self.rfile.read(n)
                received["path"] = self.path
                received["session"] = self.headers.get("x-session-id")
                out = json.dumps({"upload_id": "up-real"}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *args):  # keep pytest output clean
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = TabctxClient(f"http://127.0.0.1:{server.server_port}")
            assert client.upload_csv_file(str(p), dataset_id="ds-1") == "up-real"
        finally:
            server.shutdown()
        assert received["body"] == body
        assert received["path"] == "/v1/tabctx/upload"
        assert received["session"] == "ds-1"

    def test_upload_csv_maps_413_to_admission_rejected(self, monkeypatch):
        _install(
            monkeypatch,
            lambda req: (_ for _ in ()).throw(
                _http_error(413, {"detail": "upload too large"})
            ),
        )
        client = TabctxClient("http://localhost:8000")
        with pytest.raises(AdmissionRejected):
            client.upload_csv(b"x", dataset_id="ds-1")

    def test_fit_uploaded_body(self, monkeypatch):
        captured = {}
        _install(
            monkeypatch,
            lambda req: (
                captured.__setitem__("body", json.loads(req.data.decode())),
                _json_response({"dataset_id": "ds-1"}),
            )[1],
        )
        client = TabctxClient("http://localhost:8000")
        client.fit_uploaded(
            "up-1", "ds-1", task="regression", target_column="y", model="tabpfn-3"
        )
        assert captured["body"] == {
            "train_upload_id": "up-1",
            "target_column": "y",
            "task": "regression",
            "dataset_id": "ds-1",
            "model": "tabpfn-3",
        }


class TestPredict:
    def test_inline_predict_returns_predict_result(self, monkeypatch):
        _install(
            monkeypatch,
            lambda req: _json_response(
                {
                    "predictions": ["a", "b"],
                    "probabilities": [[0.9, 0.1], [0.2, 0.8]],
                    "classes": ["a", "b"],
                    "latency_ms": 12.5,
                    "served_by": "replica-0",
                }
            ),
        )
        client = TabctxClient("http://localhost:8000")
        result = client.predict("ds-1", [[1.0], [2.0]], return_proba=True)
        assert result == PredictResult(
            predictions=["a", "b"],
            probabilities=[[0.9, 0.1], [0.2, 0.8]],
            classes=["a", "b"],
            latency_ms=12.5,
            served_by="replica-0",
        )

    def test_predict_optional_fields_default_to_none(self, monkeypatch):
        _install(
            monkeypatch,
            lambda req: _json_response({"predictions": ["a"], "latency_ms": 1.0}),
        )
        result = TabctxClient("http://localhost:8000").predict("ds-1", [[1.0]])
        assert result.probabilities is None
        assert result.classes is None
        assert result.served_by is None
        assert result.cache_outcome is None

    def test_predict_by_reference_sends_test_upload_id(self, monkeypatch):
        captured = {}
        _install(
            monkeypatch,
            lambda req: (
                captured.__setitem__("body", json.loads(req.data.decode())),
                _json_response({"predictions": [], "latency_ms": 0.0}),
            )[1],
        )
        client = TabctxClient("http://localhost:8000")
        client.predict("ds-1", test_upload_id="up-9")
        assert captured["body"] == {
            "dataset_id": "ds-1",
            "return_proba": False,
            "test_upload_id": "up-9",
        }

    def test_predict_sends_session_header_matching_dataset_id(self, monkeypatch):
        captured = {}
        _install(
            monkeypatch,
            lambda req: (
                captured.__setitem__("headers", req.headers),
                _json_response({"predictions": [], "latency_ms": 0.0}),
            )[1],
        )
        TabctxClient("http://localhost:8000").predict("my-dataset", [[1.0]])
        assert captured["headers"]["X-session-id"] == "my-dataset"

    def test_predict_maps_404_to_dataset_not_found(self, monkeypatch):
        _install(
            monkeypatch,
            lambda req: (_ for _ in ()).throw(
                _http_error(404, {"detail": "no cached context for dataset_id='x'"})
            ),
        )
        with pytest.raises(DatasetNotFoundError):
            TabctxClient("http://localhost:8000").predict("x", [[1.0]])


class TestFitPredict:
    def test_uses_legacy_endpoint_with_no_session_header(self, monkeypatch):
        captured = {}

        def handler(req):
            captured["url"] = req.full_url
            captured["headers"] = req.headers
            return _json_response(
                {
                    "predictions": [1.0],
                    "probabilities": None,
                    "classes": None,
                    "latency_ms": 5.0,
                }
            )

        _install(monkeypatch, handler)
        client = TabctxClient("http://localhost:8000")
        result = client.fit_predict([[1.0]], [1.0], [[2.0]], task="regression")

        assert captured["url"] == "http://localhost:8000/v1/tabicl/predict"
        assert "X-session-id" not in captured["headers"]
        assert result.predictions == [1.0]
        assert result.served_by is None  # legacy endpoint has no affinity concept


class TestGetEndpoints:
    def test_get_maps_http_errors_like_post(self, monkeypatch):
        # GETs used to leak raw urllib.error.HTTPError; every call in the
        # client must speak tabctx's exception vocabulary.
        _install(
            monkeypatch,
            lambda req: (_ for _ in ()).throw(
                _http_error(503, {"detail": "replica busy"})
            ),
        )
        with pytest.raises(TabctxBackpressureError):
            TabctxClient("http://localhost:8000").ready()

    def test_get_uses_get_method_without_body(self, monkeypatch):
        captured = {}
        _install(
            monkeypatch,
            lambda req: (
                captured.__setitem__("method", req.get_method()),
                captured.__setitem__("data", req.data),
                _json_response({"status": "ready"}),
            )[2],
        )
        TabctxClient("http://localhost:8000").ready()
        assert captured["method"] == "GET"
        assert captured["data"] is None

    def test_ready(self, monkeypatch):
        captured = {}
        _install(
            monkeypatch,
            lambda req: (
                captured.__setitem__("url", req.full_url),
                _json_response({"status": "ready"}),
            )[1],
        )
        assert TabctxClient("http://localhost:8000").ready() == {"status": "ready"}
        assert captured["url"] == "http://localhost:8000/readyz"

    def test_models(self, monkeypatch):
        _install(
            monkeypatch,
            lambda req: _json_response(
                {"data": [{"id": "tabicl-v2", "object": "model", "default": True}]}
            ),
        )
        models = TabctxClient("http://localhost:8000").models()
        assert models == [{"id": "tabicl-v2", "object": "model", "default": True}]

    # A gateway in front of tabctx (Vector's) lists its whole catalog: LLMs
    # and the tabular models, none marked default.
    GATEWAY = [
        {"id": "accounts/fireworks/models/glm-5p2", "owned_by": "fireworks"},
        {"id": "gpt-5", "owned_by": "openai"},
        {"id": "tabicl-v2", "owned_by": "on_prem"},
        {"id": "tabpfn-3.5", "owned_by": "on_prem"},
    ]

    @pytest.mark.parametrize(
        "data, tabular, default",
        [
            # tabctx's own listing: the marked model, wherever it is listed.
            (
                [
                    {"id": "tabicl-v2", "default": False},
                    {"id": "tabpfn-3.5", "default": True},
                ],
                ["tabicl-v2", "tabpfn-3.5"],
                "tabpfn-3.5",
            ),
            # A tabctx-only deployment of one model.
            ([{"id": "tabicl-v2", "default": True}], ["tabicl-v2"], "tabicl-v2"),
            # A gateway: only the known tabular models, TabPFN preferred.
            (GATEWAY, ["tabicl-v2", "tabpfn-3.5"], "tabpfn-3.5"),
            # A gateway that offers only TabICL.
            (GATEWAY[:3], ["tabicl-v2"], "tabicl-v2"),
        ],
    )
    def test_tabular_and_default_model(self, monkeypatch, data, tabular, default):
        _install(monkeypatch, lambda req: _json_response({"data": data}))
        client = TabctxClient("http://localhost:8000")
        assert client.tabular_models() == tabular
        assert client.default_model() == default

    @pytest.mark.parametrize("data", [[], GATEWAY[:2]])
    def test_no_tabular_model_to_default_to(self, monkeypatch, data):
        _install(monkeypatch, lambda req: _json_response({"data": data}))
        client = TabctxClient("http://localhost:8000")
        assert client.tabular_models() == []
        with pytest.raises(TabctxError, match="no tabular model"):
            client.default_model()

    def test_limits(self, monkeypatch):
        _install(monkeypatch, lambda req: _json_response({"models": ["tabicl-v2"]}))
        assert TabctxClient("http://localhost:8000").limits() == {
            "models": ["tabicl-v2"]
        }


class TestDatasets:
    def _capture(self, monkeypatch, response):
        seen: list = []

        def handler(req):
            seen.append(req)
            if isinstance(response, Exception):
                raise response
            return _json_response(response)

        _install(monkeypatch, handler)
        return seen

    def test_lists_the_callers_contexts(self, monkeypatch):
        listed = [{"dataset_id": "d", "model": "tabicl-v2"}]
        seen = self._capture(monkeypatch, {"data": listed})
        client = TabctxClient("http://localhost:8000", tenant_id="acme")
        assert client.datasets() == listed
        (req,) = seen
        assert req.get_method() == "GET"
        assert req.full_url == "http://localhost:8000/v1/tabctx/datasets"
        assert req.headers["X-tabctx-tenant-id"] == "acme"

    def test_delete_sends_the_id_quoted_with_its_affinity_header(self, monkeypatch):
        seen = self._capture(monkeypatch, {"dataset_id": "a/b c", "deleted": True})
        assert TabctxClient("http://localhost:8000").delete("a/b c") is None
        (req,) = seen
        assert req.get_method() == "DELETE" and req.data is None
        assert req.full_url == "http://localhost:8000/v1/tabctx/datasets/a%2Fb%20c"
        assert req.headers["X-session-id"] == "a/b c"

    def test_deleting_a_missing_context_raises_not_found(self, monkeypatch):
        self._capture(monkeypatch, _http_error(404, {"detail": "no context"}))
        with pytest.raises(DatasetNotFoundError):
            TabctxClient("http://localhost:8000").delete("gone")

    def test_a_missing_id_that_says_upload_is_still_a_dataset(self, monkeypatch):
        detail = "no context for dataset_id='upload-batch-7' -- it was never fit"
        self._capture(monkeypatch, _http_error(404, {"detail": detail}))
        with pytest.raises(DatasetNotFoundError):
            TabctxClient("http://localhost:8000").delete("upload-batch-7")

    def test_a_retry_that_finds_it_gone_is_success(self, monkeypatch, no_real_sleep):
        """A 503 can follow a partial delete; the retry finding nothing left
        means it is deleted."""
        replies = [
            _http_error(503, {"detail": "deleting failed on cuda:1; retry"}),
            _http_error(404, {"detail": "no context for dataset_id='d'"}),
        ]

        def handler(req):
            raise replies.pop(0)

        _install(monkeypatch, handler)
        TabctxClient("http://localhost:8000").delete("d")
        assert replies == []

    def test_requests_without_a_body_send_no_content_type(self, monkeypatch):
        seen = self._capture(monkeypatch, {"data": []})
        client = TabctxClient("http://localhost:8000", api_key="k")
        client.datasets()
        client.delete("d")
        for req in seen:
            assert "Content-type" not in req.headers
            assert req.headers["Authorization"] == "Bearer k"

    def test_delete_retries_on_backpressure(self, monkeypatch, no_real_sleep):
        replies = [_http_error(503, {"detail": "busy"}), _json_response({})]

        def handler(req):
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply

        _install(monkeypatch, handler)
        TabctxClient("http://localhost:8000").delete("d")
        assert replies == [] and len(no_real_sleep) == 1


class TestBackpressureRetry:
    def test_retries_on_503_then_succeeds(self, monkeypatch, no_real_sleep):
        attempts = {"n": 0}

        def handler(req):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise _http_error(503, {"detail": "replica busy"})
            return _json_response({"dataset_id": "ds-1"})

        _install(monkeypatch, handler)
        client = TabctxClient(
            "http://localhost:8000", max_retries=3, retry_backoff_s=0.01
        )
        assert client.fit([[1.0]], ["a"]) == "ds-1"
        assert attempts["n"] == 3
        # Exponential backoff: 0.01 * 2**0, 0.01 * 2**1.
        assert no_real_sleep == [0.01, 0.02]

    def test_a_get_retries_on_503_too(self, monkeypatch, no_real_sleep):
        # The estimators ask /v1/models on every fit without a model; a
        # busy ingress must not fail that fit when a named model's would
        # have retried.
        attempts = {"n": 0}

        def handler(req):
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise _http_error(503, {"detail": "replica busy"})
            return _json_response({"data": [{"id": "tabicl-v2", "default": True}]})

        _install(monkeypatch, handler)
        client = TabctxClient("http://localhost:8000", retry_backoff_s=0.01)
        assert client.default_model() == "tabicl-v2"
        assert attempts["n"] == 2

    def test_exhausts_retries_raises_backpressure_error(
        self, monkeypatch, no_real_sleep
    ):
        attempts = {"n": 0}

        def handler(req):
            attempts["n"] += 1
            raise _http_error(503, {"detail": "replica busy"})

        _install(monkeypatch, handler)
        client = TabctxClient(
            "http://localhost:8000", max_retries=2, retry_backoff_s=0.01
        )
        with pytest.raises(TabctxBackpressureError):
            client.fit([[1.0]], ["a"])
        assert attempts["n"] == 3  # initial attempt + 2 retries

    def test_non_503_error_is_not_retried(self, monkeypatch, no_real_sleep):
        attempts = {"n": 0}

        def handler(req):
            attempts["n"] += 1
            raise _http_error(422, {"detail": "bad input"})

        _install(monkeypatch, handler)
        client = TabctxClient("http://localhost:8000", max_retries=3)
        with pytest.raises(InvalidInputError):
            client.fit([[1.0]], ["a"])
        assert attempts["n"] == 1
        assert no_real_sleep == []


class TestRedirectsAndKeys:
    def test_redirects_are_refused_and_the_key_stays_put(self, monkeypatch):
        import http.server
        import threading

        seen: list[dict] = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                seen.append(dict(self.headers))
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(302)
                self.send_header("Location", "http://127.0.0.1:1/elsewhere")
                self.end_headers()

            do_POST = do_GET  # noqa: N815

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            client = TabctxClient(
                f"http://127.0.0.1:{server.server_port}", api_key="secret-key"
            )
            with pytest.raises(TabctxError, match="redirect") as err:
                client.models()
            assert "secret-key" not in str(err.value)
            with pytest.raises(TabctxError, match="redirect"):
                client.fit_file(b"a,b\n1,2\n", dataset_id="d")
            assert len(seen) == 2  # each request went to this server only
        finally:
            server.shutdown()

    def test_a_401_without_a_key_says_how_to_send_one(self, monkeypatch):
        monkeypatch.delenv("TABCTX_API_KEY", raising=False)

        def handler(req):
            raise _http_error(401, {"detail": "Missing API key"})

        _install(monkeypatch, handler)
        with pytest.raises(PermissionError, match="TABCTX_API_KEY"):
            TabctxClient("http://example.test").models()


@pytest.mark.parametrize("make", [bytes, bytearray, memoryview])
def test_multipart_bodies_from_bytes_like_files(make):
    from tabctx.client import _Multipart

    form = _Multipart({"dataset_id": "d"}, make(b"a,b\n1,2\n"))
    body = b"".join(form.chunks())
    assert len(body) == form.length and b"a,b\n1,2\n" in body


class TestTablesAsJson:
    """fit and predict take DataFrames and numpy arrays, not just lists."""

    def test_dataframes_and_arrays_become_rows_with_nulls(self, monkeypatch):
        pd = pytest.importorskip("pandas")
        np = pytest.importorskip("numpy")
        sent = []

        def handler(req):
            sent.append(json.loads(req.data))
            return _json_response({"dataset_id": "d"})

        _install(monkeypatch, handler)
        X = pd.DataFrame(
            {"a": [1.5, np.nan], "b": ["x", None], "c": pd.array([1, pd.NA], "Int64")}
        )
        TabctxClient("http://x").fit(X, pd.Series(["y", "n"]), dataset_id="d")
        assert sent[0]["train_X"] == [[1.5, "x", 1], [None, None, None]]
        assert sent[0]["train_y"] == ["y", "n"]
        TabctxClient("http://x").fit(np.array([[1.0, np.nan]]), np.array([3]))
        assert sent[1]["train_X"] == [[1.0, None]] and sent[1]["train_y"] == [3]


class TestQuantiles:
    def _capture(self, monkeypatch, response):
        seen: list = []

        def handler(req):
            seen.append(req)
            return _json_response(response)

        _install(monkeypatch, handler)
        return seen

    RESPONSE = {
        "predictions": [2.0],
        "quantiles": [[1.0, 3.0]],
        "quantile_levels": [0.1, 0.9],
        "n_test": 1,
        "latency_ms": 1.0,
    }

    def test_json_body_and_result(self, monkeypatch):
        seen = self._capture(monkeypatch, self.RESPONSE)
        result = TabctxClient("http://x").predict("d", [[1.0]], quantiles=(0.1, 0.9))
        assert json.loads(seen[0].data)["quantiles"] == [0.1, 0.9]
        assert result.quantiles == [[1.0, 3.0]]

    def test_left_out_unless_asked(self, monkeypatch):
        seen = self._capture(monkeypatch, {**self.RESPONSE, "quantiles": None})
        result = TabctxClient("http://x").predict("d", [[1.0]])
        assert "quantiles" not in json.loads(seen[0].data)
        assert result.quantiles is None

    def test_numpy_levels_are_sent_as_plain_floats(self, monkeypatch):
        np = pytest.importorskip("numpy")
        seen = self._capture(monkeypatch, self.RESPONSE)
        levels = np.array([0.1, 0.9])
        result = TabctxClient("http://x").predict("d", [[1.0]], quantiles=levels)
        assert json.loads(seen[0].data)["quantiles"] == [0.1, 0.9]
        assert result.quantile_levels == [0.1, 0.9]

    @pytest.mark.parametrize("levels", [0.5, "0.5", [1.5]])
    def test_bad_levels_fail_before_sending(self, monkeypatch, levels):
        seen = self._capture(monkeypatch, self.RESPONSE)
        with pytest.raises(InvalidInputError):
            TabctxClient("http://x").predict("d", [[1.0]], quantiles=levels)
        assert seen == []

    def test_a_server_without_quantiles_is_an_error_not_silence(self, monkeypatch):
        old = {k: v for k, v in self.RESPONSE.items() if not k.startswith("quantile")}
        self._capture(monkeypatch, old)
        with pytest.raises(TabctxError, match="returned no quantiles"):
            TabctxClient("http://x").predict("d", [[1.0]], quantiles=[0.5])

    def test_a_file_predict_sends_them_as_a_form_field(self, monkeypatch):
        seen = self._capture(monkeypatch, self.RESPONSE)
        TabctxClient("http://x").predict_file("d", b"a\n1\n", quantiles=[0.1, 0.9])
        body = b"".join(seen[0].data)
        assert b'name="quantiles"\r\n\r\n0.1,0.9\r\n' in body


@pytest.mark.parametrize("cache_outcome", ["hit", "restored"])
def test_predict_cache_outcome(monkeypatch, cache_outcome):
    _install(
        monkeypatch,
        lambda req: _json_response(
            {
                "predictions": ["a"],
                "latency_ms": 1.0,
                "cache_outcome": cache_outcome,
            }
        ),
    )
    result = TabctxClient("http://localhost:8000").predict("ds-1", [[1.0]])
    assert result.cache_outcome == cache_outcome


class TestSurvivalAndCausal:
    def _capture(self, monkeypatch, response):
        seen: list = []

        def handler(req):
            seen.append(req)
            return _json_response(response)

        _install(monkeypatch, handler)
        return seen

    def test_a_survival_fit_sends_its_event_column(self, monkeypatch):
        seen = self._capture(monkeypatch, {"dataset_id": "s"})
        TabctxClient("http://x").fit([[1.0], [2.0]], [3.0, 4.0], event=[1, 0])
        body = json.loads(seen[0].data)
        assert (body["train_y"], body["train_event"]) == ([3.0, 4.0], [1, 0])
        assert "train_treatment" not in body
        # Named, so a server that predates survival refuses the fit rather
        # than ignoring train_event and fitting a regression.
        assert body["task"] == "survival"

    def test_a_causal_fit_from_a_file_names_its_treatment_column(self, monkeypatch):
        seen = self._capture(monkeypatch, {"dataset_id": "c"})
        TabctxClient("http://x").fit_file(
            b"a,t,y\n1,0,2\n", target="y", treatment_column="t", dataset_id="c"
        )
        body = b"".join(seen[0].data)
        assert b'name="treatment_column"\r\n\r\nt\r\n' in body
        assert b'name="task"\r\n\r\ncausal\r\n' in body

    def test_survival_times_and_result(self, monkeypatch):
        response = {
            "predictions": [2.0],
            "survival": [[0.9, 0.4]],
            "survival_times": [1.0, 5.0],
            "latency_ms": 1.0,
        }
        seen = self._capture(monkeypatch, response)
        result = TabctxClient("http://x").predict("s", [[1.0]], times=[1, 5])
        assert json.loads(seen[0].data)["times"] == [1.0, 5.0]
        assert (result.survival, result.survival_times) == ([[0.9, 0.4]], [1.0, 5.0])

    def test_rmst_and_potential_outcomes(self, monkeypatch):
        response = {
            "predictions": [2.0],
            "rmst": [1.5],
            "potential_outcomes": [[1.0, 3.0]],
            "latency_ms": 1.0,
        }
        seen = self._capture(monkeypatch, response)
        client = TabctxClient("http://x")
        result = client.predict("s", [[1.0]], rmst_horizon=30, potential_outcomes=True)
        body = json.loads(seen[0].data)
        assert (body["rmst_horizon"], body["potential_outcomes"]) == (30.0, True)
        assert (result.rmst, result.potential_outcomes) == ([1.5], [[1.0, 3.0]])
        client.predict_file("s", b"a\n1\n", rmst_horizon=30, potential_outcomes=True)
        form = b"".join(seen[-1].data)
        assert b'name="rmst_horizon"\r\n\r\n30.0\r\n' in form
        assert b'name="potential_outcomes"\r\n\r\ntrue\r\n' in form

    def test_a_server_without_rmst_is_an_error(self, monkeypatch):
        self._capture(monkeypatch, {"predictions": [2.0], "latency_ms": 1.0})
        with pytest.raises(TabctxError, match="no rmst"):
            TabctxClient("http://x").predict("s", [[1.0]], rmst_horizon=5)

    def test_a_server_without_survival_is_an_error_not_silence(self, monkeypatch):
        self._capture(monkeypatch, {"predictions": [2.0], "latency_ms": 1.0})
        with pytest.raises(TabctxError, match="no survival probabilities"):
            TabctxClient("http://x").predict("s", [[1.0]], times=[1])

    def test_bad_times_fail_before_sending(self, monkeypatch):
        seen = self._capture(monkeypatch, {})
        with pytest.raises(InvalidInputError, match="non-negative"):
            TabctxClient("http://x").predict("s", [[1.0]], times=[-1])
        assert seen == []

    LISTING = [
        {"id": "tabicl-v2", "default": True, "tasks": ["classification", "regression"]},
        {"id": "survivalpfn-v0.1", "default": False, "tasks": ["survival"]},
    ]
    GATEWAY = [
        {"id": "gpt-5", "owned_by": "openai"},
        {"id": "causalpfn-v0", "owned_by": "on_prem"},
        {"id": "tabicl-v2", "owned_by": "on_prem"},
    ]

    @pytest.mark.parametrize(
        "data, task, tabular, default",
        [
            (LISTING, None, ["tabicl-v2", "survivalpfn-v0.1"], "tabicl-v2"),
            (LISTING, "regression", ["tabicl-v2"], "tabicl-v2"),
            # The default doesn't fit survival: the model that does.
            (LISTING, "survival", ["survivalpfn-v0.1"], "survivalpfn-v0.1"),
            # A gateway lists no tasks: the client knows the models'.
            (GATEWAY, "causal", ["causalpfn-v0"], "causalpfn-v0"),
            (GATEWAY, "classification", ["tabicl-v2"], "tabicl-v2"),
            # A tabctx older than `tasks`: its models are predictive.
            ([{"id": "x", "default": True}], "regression", ["x"], "x"),
        ],
    )
    def test_models_for_a_task(self, monkeypatch, data, task, tabular, default):
        _install(monkeypatch, lambda req: _json_response({"data": data}))
        client = TabctxClient("http://x")
        assert client.tabular_models(task) == tabular
        assert client.default_model(task) == default

    def test_no_model_for_a_task(self, monkeypatch):
        _install(monkeypatch, lambda req: _json_response({"data": self.LISTING}))
        with pytest.raises(TabctxError, match="no tabular model for causal"):
            TabctxClient("http://x").default_model("causal")
