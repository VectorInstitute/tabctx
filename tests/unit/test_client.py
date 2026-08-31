"""Unit tests for the pure-stdlib HTTP client (client.py).

urllib.request.urlopen is monkeypatched throughout -- no real network I/O,
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
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda req, timeout=None: handler(req)
    )


class TestMapStatus:
    @pytest.mark.parametrize(
        "code,exc_type",
        [
            (422, InvalidInputError),
            (413, AdmissionRejected),
            (507, BackendComputeError),
            (401, PermissionError),
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
        assert captured["body"] == {
            "train_X": [[1.0, 2.0]],
            "train_y": ["a"],
            "task": "classification",
            "dataset_id": "ds-1",
            "model": "tabicl-v2",
        }

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

    def test_upload_csv_file_reads_file_bytes(self, monkeypatch, tmp_path):
        p = tmp_path / "t.csv"
        p.write_bytes(b"a,b\n1,2\n")
        captured = {}
        _install(
            monkeypatch,
            lambda req: (
                captured.__setitem__("data", req.data),
                _json_response({"upload_id": "up-2"}),
            )[1],
        )
        client = TabctxClient("http://localhost:8000")
        assert client.upload_csv_file(str(p), dataset_id="ds-1") == "up-2"
        assert captured["data"] == b"a,b\n1,2\n"

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

    def test_limits(self, monkeypatch):
        _install(monkeypatch, lambda req: _json_response({"models": ["tabicl-v2"]}))
        assert TabctxClient("http://localhost:8000").limits() == {
            "models": ["tabicl-v2"]
        }


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
