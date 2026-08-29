"""Minimal HTTP client for a running tabctx deployment.

Exists mainly so callers can't get the serving contracts wrong: the
session-affinity header (`x-session-id` == dataset_id, required for
correct routing at num_replicas >= 2; see serve/affinity.py) and the
tenant header (`x-tabctx-tenant-id`; see serve/tenancy.py) are set
automatically on every call. Pure stdlib (urllib) -- no dependency
beyond tabctx's own errors, so `pip install tabctx` is enough to talk
to a deployment.

Usage:

    from tabctx.client import TabctxClient

    client = TabctxClient("http://localhost:8000", tenant_id="acme")
    dataset_id = client.fit(X_train, y_train, dataset_id="churn-v1")
    result = client.predict("churn-v1", X_test, return_proba=True)
    result.predictions, result.probabilities, result.classes

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
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from tabctx.errors import (
    AdmissionRejected,
    BackendComputeError,
    DatasetNotFoundError,
    InvalidInputError,
    TabctxError,
)
from tabctx.types import ArrayLike, Task


class TabctxBackpressureError(TabctxError):
    """The owning replica is at capacity (HTTP 503). Retryable: back off
    and try again; affinity guarantees the retry reaches the replica
    that holds the context."""


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
        base_url: str,
        tenant_id: str | None = None,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        retry_backoff_s: float = 0.25,
    ) -> None:
        """max_retries applies only to 503 backpressure (safe to retry by
        construction); every other error propagates immediately."""
        self._base_url = base_url.rstrip("/")
        self._tenant_id = tenant_id
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._retry_backoff_s = retry_backoff_s

    # ---- public API ------------------------------------------------------

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        task: Task = "classification",
        dataset_id: str | None = None,
    ) -> str:
        """Encode and cache a training context; returns its dataset_id.

        Supply a stable dataset_id when deploying multi-replica: a
        server-generated id can't be used to route the fit itself, so
        the context would land on an arbitrary replica.
        """
        body = {"train_X": X, "train_y": y, "task": task}
        if dataset_id is not None:
            body["dataset_id"] = dataset_id
        resp = self._post("/v1/tabctx/fit", body, session_id=dataset_id)
        return resp["dataset_id"]

    def predict(
        self, dataset_id: str, X_test: ArrayLike, return_proba: bool = False
    ) -> PredictResult:
        resp = self._post(
            "/v1/tabctx/predict",
            {"dataset_id": dataset_id, "test_X": X_test, "return_proba": return_proba},
            session_id=dataset_id,
        )
        return PredictResult(
            predictions=resp["predictions"],
            probabilities=resp.get("probabilities"),
            classes=resp.get("classes"),
            latency_ms=resp["latency_ms"],
            served_by=resp.get("served_by"),
        )

    def fit_predict(
        self,
        X: ArrayLike,
        y: ArrayLike,
        X_test: ArrayLike,
        task: Task = "classification",
        return_proba: bool = False,
    ) -> PredictResult:
        """One-shot fit+predict+evict via the legacy endpoint. Prefer
        fit()+predict() when the same training set is queried again."""
        resp = self._post(
            "/v1/tabicl/predict",
            {
                "train_X": X,
                "train_y": y,
                "test_X": X_test,
                "task": task,
                "return_proba": return_proba,
            },
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

    def limits(self) -> dict:
        """Capability discovery: what shapes will this deployment admit?
        Use it to validate a table client-side before uploading it."""
        return self._get("/v1/tabctx/limits")

    def _get(self, path: str) -> dict:
        req = urllib.request.Request(f"{self._base_url}{path}")
        with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
            return json.loads(resp.read().decode())

    # ---- internals -------------------------------------------------------

    def _headers(self, session_id: str | None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if session_id is not None:
            headers["x-session-id"] = session_id
        if self._tenant_id is not None:
            headers["x-tabctx-tenant-id"] = self._tenant_id
        return headers

    def _post(self, path: str, body: dict, session_id: str | None) -> dict:
        data = json.dumps(body).encode()
        last_backpressure: TabctxBackpressureError | None = None
        for attempt in range(self._max_retries + 1):
            req = urllib.request.Request(
                f"{self._base_url}{path}",
                data=data,
                headers=self._headers(session_id),
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                detail = self._detail(e)
                if e.code == 503:
                    last_backpressure = TabctxBackpressureError(detail)
                    if attempt < self._max_retries:
                        time.sleep(self._retry_backoff_s * (2**attempt))
                        continue
                    raise last_backpressure from e
                raise self._map_status(e.code, detail) from e
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
            return DatasetNotFoundError(detail)
        if code == 413:
            return AdmissionRejected(detail)
        if code == 507:
            return BackendComputeError(detail)
        if code == 401:
            return PermissionError(detail)
        return TabctxError(f"HTTP {code}: {detail}")
