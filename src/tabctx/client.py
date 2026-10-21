"""Minimal HTTP client for a tabctx deployment, directly or through a
gateway (Vector's: https://proxy.vectorinstitute.ai, with an API key).

Exists mainly so callers can't get the serving contracts wrong: the
session-affinity header (`x-session-id` == dataset_id, required for
correct routing at num_replicas >= 2; see serve/affinity.py) and the
tenant header (`x-tabctx-tenant-id`; see serve/tenancy.py) are set
automatically on every call. Pure stdlib (urllib) -- no dependency
beyond tabctx's own errors, so `pip install tabctx` is enough to talk
to a deployment. For scikit-learn-style estimators over this client, see
tabctx.estimators.

Usage:

    from tabctx.client import TabctxClient

    # Or set TABCTX_BASE_URL and TABCTX_API_KEY and call TabctxClient().
    client = TabctxClient("https://proxy.vectorinstitute.ai", api_key=KEY)
    dataset_id = client.fit(X_train, y_train, dataset_id="churn-v1", model="tabicl-v2")
    result = client.predict("churn-v1", X_test, return_proba=True, model="tabicl-v2")
    result.predictions, result.probabilities, result.classes

    # A table file (CSV, gzip/zstd CSV or Parquet; path or bytes) in one call:
    client.fit_file("train.parquet", target="churned", dataset_id="churn-v1", model="tabicl-v2")
    client.predict_file("churn-v1", "test.csv", model="tabicl-v2")

The API key is sent as `Authorization: Bearer`. It never appears in the
client's repr and is left out when the client is pickled: an unpickled
client (a saved model, a joblib worker) reads TABCTX_API_KEY instead.

Server-side errors come back as the same tabctx exceptions the engine
raises locally (InvalidInputError for 422, DatasetNotFoundError for 404,
AdmissionRejected for 413, BackendComputeError for 507), so code can be
written once against tabctx's error types and run against either the
in-process engine or a remote deployment. 401 (tenant required) raises
PermissionError; 503 backpressure raises TabctxBackpressureError, which
is retryable by design (strict replica affinity queues on the owning
replica rather than mis-routing -- see serve/app.py).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from tabctx.errors import (
    AdmissionRejected,
    BackendComputeError,
    DatasetNotFoundError,
    InvalidInputError,
    TabctxError,
    UploadNotFoundError,
)
from tabctx.serve.affinity import DEFAULT_SESSION_ID_HEADER
from tabctx.serve.tenancy import TENANT_HEADER
from tabctx.types import ArrayLike, Task

BASE_URL_ENV = "TABCTX_BASE_URL"
API_KEY_ENV = "TABCTX_API_KEY"

# A table file: a path, or its bytes.
TableFile = str | os.PathLike | bytes | bytearray | memoryview


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    """urllib would follow a redirect with the Authorization header, to any
    host, and turn a POST into a body-less GET; tabctx follows none."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # urllib raises the 3xx as an HTTPError


_OPENER = urllib.request.build_opener(_NoRedirects)


def _open(req: urllib.request.Request, timeout: float):
    return _OPENER.open(req, timeout=timeout)


class TabctxBackpressureError(TabctxError):
    """The owning replica is at capacity (HTTP 503). Retryable: back off
    and try again; affinity guarantees the retry reaches the replica
    that holds the context."""


def _without_none(body: dict) -> dict:
    """Leave out fields the caller didn't set, so the server's defaults
    apply (and servers that predate a field still accept the request)."""
    return {k: v for k, v in body.items() if v is not None}


@dataclass(frozen=True)
class PredictResult:
    predictions: list
    probabilities: list[list[float]] | None
    classes: list[str] | None
    latency_ms: float
    served_by: str | None


class TabctxClient:
    def __init__(
        self,
        base_url: str | None = None,
        tenant_id: str | None = None,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        retry_backoff_s: float = 0.25,
        session_header: str = DEFAULT_SESSION_ID_HEADER,
        api_key: str | None = None,
    ) -> None:
        """base_url and api_key default to TABCTX_BASE_URL and
        TABCTX_API_KEY. A gateway needs the key (and sets the tenant
        itself); a deployment reached directly needs neither.

        max_retries applies only to 503 backpressure (safe to retry by
        construction); every other error propagates immediately.

        session_header: the affinity header name. Must match the
        deployment's RAY_SERVE_SESSION_ID_HEADER_KEY (default
        "x-session-id"); a deployment that renames it would otherwise
        silently lose replica affinity."""
        base_url = base_url or os.environ.get(BASE_URL_ENV)
        if not base_url:
            raise ValueError(
                f"no deployment URL: pass base_url or set {BASE_URL_ENV} "
                "(e.g. https://proxy.vectorinstitute.ai)"
            )
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or os.environ.get(API_KEY_ENV)
        self._tenant_id = tenant_id
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._retry_backoff_s = retry_backoff_s
        self._session_header = session_header

    def __repr__(self) -> str:
        return f"TabctxClient({self._base_url!r})"

    def __getstate__(self) -> dict:
        # Keys stay out of pickles (saved models, joblib workers).
        return {**self.__dict__, "_api_key": None}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._api_key = os.environ.get(API_KEY_ENV)

    def __copy__(self) -> TabctxClient:
        # Stateless between calls, so copies (sklearn's clone) share it --
        # and keep the key, which a pickle round trip would drop.
        return self

    def __deepcopy__(self, memo: dict) -> TabctxClient:
        return self

    # ---- public API ------------------------------------------------------

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        task: Task | None = None,
        dataset_id: str | None = None,
        model: str | None = None,
    ) -> str:
        """Encode and cache a training context; returns its dataset_id.

        Omit task to let the server infer it from y (non-integer numbers
        mean regression). Supply a stable dataset_id when deploying
        multi-replica: a server-generated id can't be used to route the fit
        itself, so the context would land on an arbitrary replica.
        """
        body = _without_none(
            {
                "train_X": X,
                "train_y": y,
                "task": task,
                "dataset_id": dataset_id,
                "model": model,
            }
        )
        resp = self._post("/v1/tabctx/fit", body, session_id=dataset_id)
        return resp["dataset_id"]

    def upload_csv(self, data: bytes | IO[bytes], dataset_id: str) -> str:
        """Upload a CSV for later fit/predict-by-reference; returns the
        upload_id. `data` is the raw CSV as bytes, or an open binary
        file object (streamed -- see upload_csv_file). dataset_id is
        required because it is the affinity routing key: the upload must
        land on the same replica the fit/predict for that dataset will
        reach."""
        headers = {**self._headers(dataset_id), "Content-Type": "text/csv"}
        if not isinstance(data, bytes | bytearray):
            # urllib streams a file object in blocks, but only if it
            # knows the length up front (else it refuses non-bytes bodies).
            headers["Content-Length"] = str(os.fstat(data.fileno()).st_size)
        return self._request("/v1/tabctx/upload", data, headers)["upload_id"]

    def upload_csv_file(self, path: str, dataset_id: str) -> str:
        """upload_csv from a file path, streamed from disk in blocks
        rather than read into memory -- the path exists for tables too
        big to hold as inline JSON, so it must not buffer them either."""
        with open(path, "rb") as f:
            return self.upload_csv(f, dataset_id)

    def fit_uploaded(
        self,
        train_upload_id: str,
        dataset_id: str,
        task: Task | None = None,
        target_column: str | None = None,
        model: str | None = None,
    ) -> str:
        """Fit-by-reference: consume a prior upload_csv() of a CSV whose
        columns are the features plus one target column (target_column,
        default: the last column)."""
        resp = self._post(
            "/v1/tabctx/fit",
            _without_none(
                {
                    "train_upload_id": train_upload_id,
                    "target_column": target_column,
                    "task": task,
                    "dataset_id": dataset_id,
                    "model": model,
                }
            ),
            session_id=dataset_id,
        )
        return resp["dataset_id"]

    def fit_file(
        self,
        file: TableFile,
        target: str | None = None,
        dataset_id: str | None = None,
        model: str | None = None,
        task: Task | None = None,
    ) -> str:
        """fit() from a table file sent with the request: a path (streamed
        from disk) or the file's bytes. Columns are the features plus the
        label column, `target` (default: the last). Returns the dataset_id."""
        fields = {
            "target": target,
            "dataset_id": dataset_id,
            "model": model,
            "task": task,
        }
        resp = self._post_file("/v1/tabctx/fit", fields, file, session_id=dataset_id)
        return resp["dataset_id"]

    def predict(
        self,
        dataset_id: str,
        X_test: ArrayLike | None = None,
        return_proba: bool = False,
        test_upload_id: str | None = None,
        model: str | None = None,
    ) -> PredictResult:
        """model: needed through a gateway, which routes predicts by it;
        a deployment reached directly knows it from the fit."""
        body: dict = _without_none(
            {
                "dataset_id": dataset_id,
                "return_proba": return_proba,
                "test_X": X_test,
                "test_upload_id": test_upload_id,
                "model": model,
            }
        )
        resp = self._post("/v1/tabctx/predict", body, session_id=dataset_id)
        return _predict_result(resp)

    def predict_file(
        self,
        dataset_id: str,
        file: TableFile,
        return_proba: bool = False,
        model: str | None = None,
    ) -> PredictResult:
        """predict() on a table file sent with the request (path or bytes).
        When the training table was a file, columns are matched by name."""
        fields = {
            "dataset_id": dataset_id,
            "return_proba": "true" if return_proba else "false",
            "model": model,
        }
        resp = self._post_file(
            "/v1/tabctx/predict", fields, file, session_id=dataset_id
        )
        return _predict_result(resp)

    def fit_predict(
        self,
        X: ArrayLike,
        y: ArrayLike,
        X_test: ArrayLike,
        task: Task | None = None,
        return_proba: bool = False,
    ) -> PredictResult:
        """One-shot fit+predict+evict via the legacy endpoint. Prefer
        fit()+predict() when the same training set is queried again."""
        resp = self._post(
            "/v1/tabicl/predict",
            _without_none(
                {
                    "train_X": X,
                    "train_y": y,
                    "test_X": X_test,
                    "task": task,
                    "return_proba": return_proba,
                }
            ),
            session_id=None,  # no cached state -> any replica may serve it
        )
        return PredictResult(
            predictions=resp["predictions"],
            probabilities=resp.get("probabilities"),
            classes=resp.get("classes"),
            latency_ms=resp["latency_ms"],
            served_by=None,
        )

    def ready(self) -> dict:
        """The deployment's /readyz payload (device, cache stats, ...)."""
        return self._get("/readyz")

    def models(self) -> list[dict]:
        """The models this deployment serves; pass an id as fit(model=...)."""
        return self._get("/v1/models")["data"]

    def limits(self) -> dict:
        """Capability discovery: what shapes will this deployment admit?
        Use it to validate a table client-side before uploading it."""
        return self._get("/v1/tabctx/limits")

    # ---- internals -------------------------------------------------------

    def _get(self, path: str) -> dict:
        return self._request(path, None, self._auth())

    def _headers(
        self, session_id: str | None, content_type: str = "application/json"
    ) -> dict[str, str]:
        headers = {"Content-Type": content_type}
        if session_id is not None:
            headers[self._session_header] = session_id
        if self._tenant_id is not None:
            headers[TENANT_HEADER] = self._tenant_id
        return {**headers, **self._auth()}

    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def _request(
        self, path: str, data: bytes | IO[bytes] | None, headers: dict[str, str]
    ) -> dict:
        """One HTTP round trip (POST when data is given, else GET), JSON
        response decoded, HTTP errors mapped to tabctx exceptions. No
        retry: 503 is raised as TabctxBackpressureError for _post to
        handle, since only idempotent-by-construction calls may retry."""
        req = urllib.request.Request(
            f"{self._base_url}{path}",
            data=data,
            headers=headers,
            method="POST" if data is not None else "GET",
        )
        try:
            with _open(req, timeout=self._timeout_s) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if 300 <= e.code < 400:
                raise TabctxError(
                    f"HTTP {e.code}: the deployment redirected to "
                    f"{e.headers.get('Location')!r}; redirects aren't followed (they "
                    "would carry the API key elsewhere): set base_url to the "
                    "deployment's final address"
                ) from e
            error = self._map_status(e.code, self._detail(e))
            if isinstance(error, PermissionError) and not self._api_key:
                error = PermissionError(
                    f"{error} -- no API key was sent: pass api_key or set "
                    f"{API_KEY_ENV} (a client unpickled in a worker, e.g. with "
                    "n_jobs > 1, reads it from the environment)"
                )
            raise error from e

    def _post(self, path: str, body: dict, session_id: str | None) -> dict:
        data = json.dumps(body).encode()
        return self._with_retries(
            lambda: self._request(path, data, self._headers(session_id))
        )

    def _post_file(
        self, path: str, fields: dict, file: TableFile, session_id: str | None
    ) -> dict:
        form = _Multipart({k: v for k, v in fields.items() if v is not None}, file)
        headers = {
            **self._headers(session_id, form.content_type),
            "Content-Length": str(form.length),
        }
        # A fresh body per attempt: a path is streamed again from the start.
        return self._with_retries(lambda: self._request(path, form.chunks(), headers))

    def _with_retries(self, send):
        for attempt in range(self._max_retries + 1):
            try:
                return send()
            except TabctxBackpressureError:
                if attempt >= self._max_retries:
                    raise
                time.sleep(self._retry_backoff_s * (2**attempt))
        raise AssertionError("unreachable")  # pragma: no cover

    @staticmethod
    def _detail(e: urllib.error.HTTPError) -> str:
        try:
            payload = json.loads(e.read().decode())
            return str(payload.get("detail", payload))
        except Exception:  # noqa: BLE001 -- any unparseable body
            return f"HTTP {e.code}"

    @staticmethod
    def _map_status(code: int, detail: str) -> Exception:
        # Inverse of serve/app.py's _map_error, so remote callers catch
        # the same exception types local engine users do.
        if code == 422:
            return InvalidInputError(detail)
        if code == 404:
            # Both not-found kinds share the status; the server's upload
            # message always says "upload" (see UploadStore.consume), so
            # this sniff only ever swaps between two same-semantics
            # not-found types.
            if "upload" in detail.lower():
                return UploadNotFoundError(detail)
            return DatasetNotFoundError(detail)
        if code == 413:
            return AdmissionRejected(detail)
        if code == 507:
            return BackendComputeError(detail)
        if code in (401, 403):
            return PermissionError(detail)
        if code == 503:
            return TabctxBackpressureError(detail)
        return TabctxError(f"HTTP {code}: {detail}")


def _predict_result(resp: dict) -> PredictResult:
    return PredictResult(
        predictions=resp["predictions"],
        probabilities=resp.get("probabilities"),
        classes=resp.get("classes"),
        latency_ms=resp["latency_ms"],
        served_by=resp.get("served_by"),
    )


class _Multipart:
    """A multipart/form-data body of text fields plus one `file` part,
    with its length known up front, so urllib streams it (a path is read
    in blocks, never whole into memory)."""

    _BLOCK = 1 << 20

    def __init__(self, fields: dict[str, str], file: TableFile) -> None:
        boundary = uuid.uuid4().hex
        self.content_type = f"multipart/form-data; boundary={boundary}"
        head = b"".join(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"'
            f"\r\n\r\n{value}\r\n".encode()
            for name, value in fields.items()
        )
        self._head = (
            head
            + (
                f"--{boundary}\r\nContent-Disposition: form-data; "
                f'name="file"; filename="table"\r\n'
                "Content-Type: application/octet-stream\r\n\r\n"
            ).encode()
        )
        self._tail = f"\r\n--{boundary}--\r\n".encode()
        if isinstance(file, bytearray | memoryview):
            file = bytes(file)
        self._file = file
        size = len(file) if isinstance(file, bytes) else os.path.getsize(file)
        self.length = len(self._head) + size + len(self._tail)

    def chunks(self) -> Iterator[bytes]:
        yield self._head
        if isinstance(self._file, bytes):
            yield self._file
        else:
            with open(Path(self._file), "rb") as f:
                while block := f.read(self._BLOCK):
                    yield block
        yield self._tail
