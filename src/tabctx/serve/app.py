"""Ray Serve deployment wrapping TabctxEngine behind FastAPI.

Two endpoint families:

1. /v1/tabicl/predict -- kept byte-for-byte compatible with the schema of
   the hand-rolled wrapper this library replaces (inference-platform's
   tests/gke-tabicl-test/app.py), implemented as a thin call to
   engine.fit_predict(). This means that repo's existing probe.py and its
   5-shape latency/memory table keep passing unchanged, as a regression
   check that switching to tabctx didn't make things slower or less
   reliable.
2. /v1/tabctx/fit + /v1/tabctx/predict -- the actual new capability this
   library exists to prove out: fit once, predict many times against the
   same cached context, with no re-fit cost on repeat calls.

Multi-replica routing (v0.6.0): the deployment ships with Ray Serve's
consistent-hash request router so requests carrying the session-affinity
header (`x-session-id`, configurable via RAY_SERVE_SESSION_ID_HEADER_KEY)
pin to a consistent replica. The contract -- the session id IS the
dataset_id -- lives in serve/affinity.py; engine construction (backend
selection, GPU memory budgeting) lives in serve/factory.py; tenant
namespacing (`x-tabctx-tenant-id` scoping dataset_ids per tenant, see
serve/tenancy.py) closes the guessable-dataset_id data-leakage gap.
"""

import logging
import time
import uuid
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from ray import serve
from ray.serve.config import RequestRouterConfig
from starlette.concurrency import run_in_threadpool

from tabctx import (
    AdmissionRejected,
    BackendComputeError,
    CacheCapacityError,
    DatasetNotFoundError,
    InvalidInputError,
)
from tabctx.serve.affinity import resolve_dataset_id, session_id_from_headers
from tabctx.serve.factory import ServeSettings, build_engine
from tabctx.serve.tenancy import (
    TenantRequiredError,
    resolve_tenant_id,
    scope_dataset_id,
)

logger = logging.getLogger("tabctx.serve")
logging.basicConfig(level=logging.INFO)

fastapi_app = FastAPI(title="tabctx Serve API")


def _torch_or_none():
    """The fake-backend configuration must run without torch installed;
    everything torch-dependent here (GPU memory reporting) degrades to
    None rather than failing at import."""
    try:
        import torch

        return torch
    except ImportError:
        return None


class GpuMemory(BaseModel):
    allocated_mb: float
    reserved_mb: float


class LegacyPredictRequest(BaseModel):
    train_X: list[list[float]]
    train_y: list[float | str]
    test_X: list[list[float]]
    task: Literal["classification", "regression"] = "classification"
    return_proba: bool = False


class LegacyPredictResponse(BaseModel):
    request_id: str
    predictions: list
    probabilities: list[list[float]] | None = None
    classes: list[str] | None = None
    n_train: int
    n_test: int
    n_features: int
    latency_ms: float
    gpu_memory: GpuMemory | None = None


class FitRequest(BaseModel):
    train_X: list[list[float]]
    train_y: list[float | str]
    task: Literal["classification", "regression"] = "classification"
    dataset_id: str | None = None


class FitResponse(BaseModel):
    dataset_id: str
    n_train: int
    n_features: int
    # Which replica served this -- lets clients and probes verify that
    # session affinity actually pinned fit() and later predict() calls to
    # the same replica (the multi-replica correctness contract).
    served_by: str | None = None


class TabctxPredictRequest(BaseModel):
    dataset_id: str
    test_X: list[list[float]]
    return_proba: bool = False


class TabctxPredictResponse(BaseModel):
    predictions: list
    probabilities: list[list[float]] | None = None
    classes: list[str] | None = None
    n_test: int
    latency_ms: float
    served_by: str | None = None


_AUTH_ERRORS = (TenantRequiredError,)
_INVALID_INPUT_ERRORS = (InvalidInputError,)
_ADMISSION_ERRORS = (AdmissionRejected, CacheCapacityError)
_NOT_FOUND_ERRORS = (DatasetNotFoundError,)
_COMPUTE_ERRORS = (BackendComputeError,)


def _map_error(e: Exception) -> HTTPException:
    if isinstance(e, _AUTH_ERRORS):
        return HTTPException(401, str(e))
    if isinstance(e, _INVALID_INPUT_ERRORS):
        return HTTPException(422, str(e))
    if isinstance(e, _ADMISSION_ERRORS):
        return HTTPException(413, str(e))
    if isinstance(e, _NOT_FOUND_ERRORS):
        return HTTPException(404, str(e))
    if isinstance(e, _COMPUTE_ERRORS):
        return HTTPException(507, str(e))
    raise e


def _replica_tag() -> str | None:
    try:
        return serve.get_replica_context().replica_tag
    except Exception:  # not running inside a Serve replica (e.g. tests)
        return None


@serve.deployment(
    max_ongoing_requests=2,
    max_queued_requests=8,
    health_check_period_s=30,
    health_check_timeout_s=60,
    # Consistent-hash routing on the session-affinity header: requests for
    # the same dataset_id (sent as `x-session-id`) always land on the same
    # replica, which is what makes the per-replica context cache correct at
    # num_replicas > 1 at all. num_fallback_replicas=0 is deliberate:
    # falling back to a *different* replica under backpressure would land
    # predict() calls on a replica without the cached context -- the exact
    # spurious-404 failure this router exists to prevent. Strict affinity
    # means backpressure surfaces as retry-with-backoff (and eventually
    # 503) on the owning replica instead, which is honest and retryable.
    request_router_config=RequestRouterConfig(
        request_router_class=(
            "ray.serve.experimental.consistent_hash_router.ConsistentHashRouter"
        ),
        request_router_kwargs={"num_fallback_replicas": 0},
    ),
)
@serve.ingress(fastapi_app)
class TabctxService:
    def __init__(self) -> None:
        settings = ServeSettings.from_env()
        built = build_engine(settings)
        self._engine = built.engine
        self._estimator = built.estimator
        self._device = built.device
        if "cuda" not in self._device and settings.backend == "tabicl":
            logger.warning("No CUDA device visible -- running tabctx on CPU")
        logger.info(
            "TabctxService initialized (backend=%s, device=%s, "
            "gpu_memory_fraction=%s, replica=%s). %s",
            settings.backend,
            self._device,
            settings.gpu_memory_fraction,
            _replica_tag(),
            self._estimator.confidence(),
        )

    # ---- legacy one-shot endpoint (regression-compatible with the old app.py) ----

    @fastapi_app.post("/v1/tabicl/predict", response_model=LegacyPredictResponse)
    async def legacy_predict(self, req: LegacyPredictRequest) -> LegacyPredictResponse:
        return await run_in_threadpool(self._legacy_predict_sync, req)

    def _legacy_predict_sync(self, req: LegacyPredictRequest) -> LegacyPredictResponse:
        torch = _torch_or_none()
        cuda = torch is not None and torch.cuda.is_available()

        request_id = str(uuid.uuid4())
        n_train, n_test = len(req.train_X), len(req.test_X)
        n_features = len(req.train_X[0]) if n_train else 0

        if cuda:
            torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        try:
            outcome = self._engine.fit_predict(
                req.train_X,
                req.train_y,
                req.test_X,
                task=req.task,
                return_proba=req.return_proba,
            )
        except (*_INVALID_INPUT_ERRORS, *_ADMISSION_ERRORS, *_NOT_FOUND_ERRORS, *_COMPUTE_ERRORS) as e:
            logger.error("[%s] %s: %s", request_id, type(e).__name__, e)
            raise _map_error(e) from e
        latency_ms = (time.monotonic() - start) * 1000

        gpu_memory = None
        if cuda:
            gpu_memory = GpuMemory(
                allocated_mb=torch.cuda.max_memory_allocated() / 1e6,
                reserved_mb=torch.cuda.max_memory_reserved() / 1e6,
            )
        logger.info(
            "[%s] legacy predict train=%d test=%d feats=%d latency_ms=%.1f gpu=%s",
            request_id, n_train, n_test, n_features, latency_ms, gpu_memory,
        )
        return LegacyPredictResponse(
            request_id=request_id,
            predictions=outcome.predictions,
            probabilities=outcome.probabilities,
            classes=outcome.classes,
            n_train=n_train,
            n_test=n_test,
            n_features=n_features,
            latency_ms=latency_ms,
            gpu_memory=gpu_memory,
        )

    # ---- new tabctx-native endpoints: the actual capability this library adds ----

    @fastapi_app.post("/v1/tabctx/fit", response_model=FitResponse)
    async def tabctx_fit(self, req: FitRequest, request: Request) -> FitResponse:
        session_id = session_id_from_headers(request.headers)
        return await run_in_threadpool(self._fit_sync, req, dict(request.headers), session_id)

    def _fit_sync(
        self, req: FitRequest, headers: dict[str, str], session_id: str | None
    ) -> FitResponse:
        n_train = len(req.train_X)
        n_features = len(req.train_X[0]) if n_train else 0
        try:
            tenant_id = resolve_tenant_id(headers)
            # The caller-visible dataset_id: header/body reconciliation
            # first (affinity contract), server-generated as a last
            # resort. Tenant scoping is applied only on the cache-facing
            # id -- responses always show the caller's own id.
            dataset_id = resolve_dataset_id(session_id, req.dataset_id) or str(
                uuid.uuid4()
            )
            self._engine.fit(
                req.train_X,
                req.train_y,
                task=req.task,
                dataset_id=scope_dataset_id(tenant_id, dataset_id),
            )
        except (
            *_AUTH_ERRORS,
            *_INVALID_INPUT_ERRORS,
            *_ADMISSION_ERRORS,
            *_COMPUTE_ERRORS,
        ) as e:
            raise _map_error(e) from e
        return FitResponse(
            dataset_id=dataset_id,
            n_train=n_train,
            n_features=n_features,
            served_by=_replica_tag(),
        )

    @fastapi_app.post("/v1/tabctx/predict", response_model=TabctxPredictResponse)
    async def tabctx_predict(
        self, req: TabctxPredictRequest, request: Request
    ) -> TabctxPredictResponse:
        session_id = session_id_from_headers(request.headers)
        return await run_in_threadpool(
            self._predict_sync, req, dict(request.headers), session_id
        )

    def _predict_sync(
        self,
        req: TabctxPredictRequest,
        headers: dict[str, str],
        session_id: str | None,
    ) -> TabctxPredictResponse:
        start = time.monotonic()
        try:
            tenant_id = resolve_tenant_id(headers)
            resolve_dataset_id(session_id, req.dataset_id)
            outcome = self._engine.predict(
                scope_dataset_id(tenant_id, req.dataset_id),
                req.test_X,
                return_proba=req.return_proba,
            )
        except (
            *_AUTH_ERRORS,
            *_INVALID_INPUT_ERRORS,
            *_NOT_FOUND_ERRORS,
            *_COMPUTE_ERRORS,
        ) as e:
            raise _map_error(e) from e
        latency_ms = (time.monotonic() - start) * 1000
        return TabctxPredictResponse(
            predictions=outcome.predictions,
            probabilities=outcome.probabilities,
            classes=outcome.classes,
            n_test=len(req.test_X),
            latency_ms=latency_ms,
            served_by=_replica_tag(),
        )

    @fastapi_app.get("/healthz")
    def healthz(self):
        return {"status": "ok"}

    @fastapi_app.get("/readyz")
    def readyz(self):
        torch = _torch_or_none()

        stats = self._engine.stats()
        real_gpu_memory = None
        if torch is not None and torch.cuda.is_available():
            # Real device memory, independent of our own byte-accounting --
            # cache_stats.used_bytes is the estimator's *prediction* for what
            # cached contexts should cost; this is what the device actually
            # reports. Comparing the two across a long-running replica is how
            # you'd catch the estimator drifting from reality or the cache
            # failing to actually release GPU memory on eviction (Python
            # dropping a reference doesn't guarantee PyTorch's caching
            # allocator returns the memory immediately).
            real_gpu_memory = {
                "allocated_mb": torch.cuda.memory_allocated() / 1e6,
                "reserved_mb": torch.cuda.memory_reserved() / 1e6,
            }
        return {
            "status": "ready",
            "device": self._device,
            "cuda_available": torch is not None and torch.cuda.is_available(),
            "replica": _replica_tag(),
            "estimator_confidence": self._estimator.confidence(),
            "cache_stats": {
                "n_cached_contexts": stats.n_cached_contexts,
                "used_bytes": stats.used_bytes,
                "free_bytes": stats.free_bytes,
                "capacity_bytes": stats.capacity_bytes,
            },
            "real_gpu_memory": real_gpu_memory,
        }


app = TabctxService.bind()
