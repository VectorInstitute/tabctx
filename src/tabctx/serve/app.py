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
"""

import logging
import time
import uuid
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ray import serve
from starlette.concurrency import run_in_threadpool

from tabctx import (
    AdmissionRejected,
    BackendComputeError,
    CacheCapacityError,
    ContextCacheManager,
    DatasetNotFoundError,
    InvalidInputError,
    TabctxEngine,
)
from tabctx.backends.tabicl import TabICLBackend
from tabctx.memory import (
    A100_40GB_TABICL_CALIBRATION,
    AdaptiveMemoryEstimator,
    PowerLawMemoryEstimator,
)

logger = logging.getLogger("tabctx.serve")
logging.basicConfig(level=logging.INFO)

fastapi_app = FastAPI(title="tabctx Serve API")


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


_INVALID_INPUT_ERRORS = (InvalidInputError,)
_ADMISSION_ERRORS = (AdmissionRejected, CacheCapacityError)
_NOT_FOUND_ERRORS = (DatasetNotFoundError,)
_COMPUTE_ERRORS = (BackendComputeError,)


def _map_error(e: Exception) -> HTTPException:
    if isinstance(e, _INVALID_INPUT_ERRORS):
        return HTTPException(422, str(e))
    if isinstance(e, _ADMISSION_ERRORS):
        return HTTPException(413, str(e))
    if isinstance(e, _NOT_FOUND_ERRORS):
        return HTTPException(404, str(e))
    if isinstance(e, _COMPUTE_ERRORS):
        return HTTPException(507, str(e))
    raise e


@serve.deployment(
    max_ongoing_requests=2,
    max_queued_requests=8,
    health_check_period_s=30,
    health_check_timeout_s=60,
)
@serve.ingress(fastapi_app)
class TabctxService:
    def __init__(self) -> None:
        import torch

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._device != "cuda":
            logger.warning("No CUDA device visible -- running tabctx on CPU")

        # AdaptiveMemoryEstimator wraps the static formula as a fallback for
        # shapes never seen before, but uses real per-fit measurements (fed
        # back via engine.fit() -> record_observation()) for the admission
        # gate whenever a past fit at least as large has been observed --
        # see memory/adaptive.py. This is why confidence() is queried fresh
        # per /readyz call below rather than cached at startup: it changes
        # as the replica accumulates real operational observations.
        self._estimator = AdaptiveMemoryEstimator(
            fallback=PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION)
        )
        cache = ContextCacheManager(capacity_bytes=self._estimator.ceiling_bytes())
        self._engine = TabctxEngine(
            backend=TabICLBackend(device=self._device), cache=cache, estimator=self._estimator
        )
        logger.info(
            "TabctxService initialized (device=%s). %s",
            self._device,
            self._estimator.confidence(),
        )

    # ---- legacy one-shot endpoint (regression-compatible with the old app.py) ----

    @fastapi_app.post("/v1/tabicl/predict", response_model=LegacyPredictResponse)
    async def legacy_predict(self, req: LegacyPredictRequest) -> LegacyPredictResponse:
        return await run_in_threadpool(self._legacy_predict_sync, req)

    def _legacy_predict_sync(self, req: LegacyPredictRequest) -> LegacyPredictResponse:
        import torch

        request_id = str(uuid.uuid4())
        n_train, n_test = len(req.train_X), len(req.test_X)
        n_features = len(req.train_X[0]) if n_train else 0

        if torch.cuda.is_available():
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
        if torch.cuda.is_available():
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
    async def tabctx_fit(self, req: FitRequest) -> FitResponse:
        return await run_in_threadpool(self._fit_sync, req)

    def _fit_sync(self, req: FitRequest) -> FitResponse:
        n_train = len(req.train_X)
        n_features = len(req.train_X[0]) if n_train else 0
        try:
            dataset_id = self._engine.fit(
                req.train_X, req.train_y, task=req.task, dataset_id=req.dataset_id
            )
        except (*_INVALID_INPUT_ERRORS, *_ADMISSION_ERRORS, *_COMPUTE_ERRORS) as e:
            raise _map_error(e) from e
        return FitResponse(dataset_id=dataset_id, n_train=n_train, n_features=n_features)

    @fastapi_app.post("/v1/tabctx/predict", response_model=TabctxPredictResponse)
    async def tabctx_predict(self, req: TabctxPredictRequest) -> TabctxPredictResponse:
        return await run_in_threadpool(self._predict_sync, req)

    def _predict_sync(self, req: TabctxPredictRequest) -> TabctxPredictResponse:
        start = time.monotonic()
        try:
            outcome = self._engine.predict(
                req.dataset_id, req.test_X, return_proba=req.return_proba
            )
        except (*_INVALID_INPUT_ERRORS, *_NOT_FOUND_ERRORS, *_COMPUTE_ERRORS) as e:
            raise _map_error(e) from e
        latency_ms = (time.monotonic() - start) * 1000
        return TabctxPredictResponse(
            predictions=outcome.predictions,
            probabilities=outcome.probabilities,
            classes=outcome.classes,
            n_test=len(req.test_X),
            latency_ms=latency_ms,
        )

    @fastapi_app.get("/healthz")
    def healthz(self):
        return {"status": "ok"}

    @fastapi_app.get("/readyz")
    def readyz(self):
        import torch

        stats = self._engine.stats()
        real_gpu_memory = None
        if torch.cuda.is_available():
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
            "cuda_available": torch.cuda.is_available(),
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
