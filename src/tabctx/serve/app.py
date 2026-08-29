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
    UploadNotFoundError,
    UploadTooLargeError,
)
from tabctx.batching import CoalescingPredictor
from tabctx.serve.affinity import resolve_dataset_id, session_id_from_headers
from tabctx.serve.csv_io import parse_features_csv, parse_train_csv
from tabctx.serve.factory import ServeSettings, build_engine
from tabctx.serve.uploads import UploadStore
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
    # Inline path (small tables) ...
    train_X: list[list[float]] | None = None
    train_y: list[float | str] | None = None
    # ... or by-reference path (large tables): a prior POST /v1/tabctx/upload
    # of a CSV whose columns are features plus one target column.
    train_upload_id: str | None = None
    target_column: str | None = None  # default: the CSV's last column
    task: Literal["classification", "regression"] = "classification"
    dataset_id: str | None = None


class UploadResponse(BaseModel):
    upload_id: str
    n_bytes: int
    served_by: str | None = None


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
    # Inline path ...
    test_X: list[list[float]] | None = None
    # ... or by-reference: an uploaded CSV of feature columns only, with
    # the same header (names and order) the training CSV had.
    test_upload_id: str | None = None
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
_ADMISSION_ERRORS = (AdmissionRejected, CacheCapacityError, UploadTooLargeError)
_NOT_FOUND_ERRORS = (DatasetNotFoundError, UploadNotFoundError)
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
    # 8 concurrent requests per replica (was 2): safe because the engine
    # serializes all GPU work behind its cache lock regardless -- extra
    # in-flight requests wait on the lock (or coalesce, see batching.py)
    # rather than running concurrent GPU calls, so the memory ceiling's
    # one-in-flight-call assumption still holds. More in-flight requests
    # = more same-context coalescing opportunity.
    max_ongoing_requests=8,
    max_queued_requests=16,
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
        self._settings = settings
        built = build_engine(settings)
        self._engine = built.engine
        self._estimator = built.estimator
        self._backend_name = built.backend.name
        self._device = built.device
        # Same-context predict coalescing (see batching.py): concurrent
        # requests against one cached context share a single backend call.
        self._predictor = CoalescingPredictor(
            self._engine, window_s=settings.batch_window_ms / 1000.0
        )
        # Large-table ingestion (fit/predict-by-reference): replica-local
        # streamed CSV uploads; see serve/uploads.py for the affinity and
        # tenancy contracts that make this correct at num_replicas >= 2.
        self._uploads = UploadStore(
            ttl_s=settings.upload_ttl_s,
            max_upload_bytes=settings.max_upload_bytes,
        )
        # dataset_id (scoped) -> training CSV feature names, so a test
        # CSV's header can be checked against the training schema (a
        # silently reordered column would mean garbage predictions).
        # Entries are dropped when a predict finds the dataset evicted.
        self._feature_names: dict[str, list[str]] = {}
        # Restart visibility (ROADMAP "cache durability", first step): a
        # replica restart silently drops every cached context, and the
        # caller-visible symptom (404 -> re-fit) is indistinguishable
        # from eviction. Exposing when this replica started lets
        # operators and probes correlate a burst of 404s with a restart
        # instead of chasing a phantom eviction/routing bug.
        self._started_at = time.time()
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

    @fastapi_app.post("/v1/tabctx/upload", response_model=UploadResponse)
    async def tabctx_upload(self, request: Request) -> UploadResponse:
        """Streamed CSV upload for fit/predict-by-reference. Send the raw
        CSV as the request body. In multi-replica deployments the request
        MUST carry `x-session-id: <dataset_id>` so the upload lands on
        the replica that the fit/predict for that dataset will reach."""
        try:
            tenant_id = resolve_tenant_id(dict(request.headers))
        except _AUTH_ERRORS as e:
            raise _map_error(e) from e
        writer = self._uploads.begin()
        try:
            async for chunk in request.stream():
                if chunk:
                    writer.write(chunk)
            record = writer.commit(tenant_id)
        except UploadTooLargeError as e:
            raise _map_error(e) from e
        except BaseException:
            writer.abort()
            raise
        return UploadResponse(
            upload_id=record.upload_id,
            n_bytes=record.n_bytes,
            served_by=_replica_tag(),
        )

    @fastapi_app.post("/v1/tabctx/fit", response_model=FitResponse)
    async def tabctx_fit(self, req: FitRequest, request: Request) -> FitResponse:
        session_id = session_id_from_headers(request.headers)
        return await run_in_threadpool(self._fit_sync, req, dict(request.headers), session_id)

    def _resolve_train_table(
        self, req: FitRequest, tenant_id: str | None
    ) -> tuple[object, object, list[str] | None]:
        """Returns (X, y, feature_names_or_None) from whichever of the
        inline / by-reference paths the request used -- exactly one."""
        inline = req.train_X is not None or req.train_y is not None
        if inline and req.train_upload_id is not None:
            raise InvalidInputError(
                "provide either inline train_X/train_y or train_upload_id, not both"
            )
        if req.train_upload_id is not None:
            path = self._uploads.consume(req.train_upload_id, tenant_id)
            try:
                X, y, feature_names = parse_train_csv(
                    path, req.task, req.target_column
                )
            finally:
                self._uploads.discard(path)
            return X, y, feature_names
        if req.train_X is None or req.train_y is None:
            raise InvalidInputError(
                "fit needs either train_X and train_y (inline) or "
                "train_upload_id (an uploaded CSV)"
            )
        return req.train_X, req.train_y, None

    def _fit_sync(
        self, req: FitRequest, headers: dict[str, str], session_id: str | None
    ) -> FitResponse:
        try:
            tenant_id = resolve_tenant_id(headers)
            X, y, feature_names = self._resolve_train_table(req, tenant_id)
            n_train = len(X)
            n_features = len(X[0]) if n_train else 0
            # The caller-visible dataset_id: header/body reconciliation
            # first (affinity contract), server-generated as a last
            # resort. Tenant scoping is applied only on the cache-facing
            # id -- responses always show the caller's own id.
            dataset_id = resolve_dataset_id(session_id, req.dataset_id) or str(
                uuid.uuid4()
            )
            scoped_id = scope_dataset_id(tenant_id, dataset_id)
            self._engine.fit(X, y, task=req.task, dataset_id=scoped_id)
            if feature_names is not None:
                self._feature_names[scoped_id] = feature_names
            else:
                self._feature_names.pop(scoped_id, None)
        except (
            *_AUTH_ERRORS,
            *_INVALID_INPUT_ERRORS,
            *_ADMISSION_ERRORS,
            *_NOT_FOUND_ERRORS,
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

    def _resolve_test_table(
        self, req: TabctxPredictRequest, tenant_id: str | None, scoped_id: str
    ) -> object:
        if req.test_X is not None and req.test_upload_id is not None:
            raise InvalidInputError(
                "provide either inline test_X or test_upload_id, not both"
            )
        if req.test_upload_id is not None:
            path = self._uploads.consume(req.test_upload_id, tenant_id)
            try:
                # Schema check against the training CSV's header when the
                # context was fit by reference; inline-fit contexts fall
                # back to the engine's feature-count check.
                return parse_features_csv(path, self._feature_names.get(scoped_id))
            finally:
                self._uploads.discard(path)
        if req.test_X is None:
            raise InvalidInputError(
                "predict needs either test_X (inline) or test_upload_id "
                "(an uploaded CSV)"
            )
        return req.test_X

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
            scoped_id = scope_dataset_id(tenant_id, req.dataset_id)
            X_test = self._resolve_test_table(req, tenant_id, scoped_id)
            try:
                outcome = self._predictor.predict(
                    scoped_id, X_test, return_proba=req.return_proba
                )
            except DatasetNotFoundError:
                # The context is gone (evicted/restart) -- drop the stale
                # training-schema entry so it can't mismatch a future
                # re-fit of the same id via a different path.
                self._feature_names.pop(scoped_id, None)
                raise
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
            n_test=len(X_test),
            latency_ms=latency_ms,
            served_by=_replica_tag(),
        )

    @fastapi_app.get("/healthz")
    def healthz(self):
        return {"status": "ok"}

    @fastapi_app.get("/v1/tabctx/limits")
    def limits(self):
        """Capability discovery (inspired by PriorLabs' get_model_limits):
        what will this deployment admit? Lets clients validate table
        shapes BEFORE uploading data, instead of discovering a 413 the
        expensive way. max_admissible_train_rows is derived live from
        the admission gate at representative feature counts, so it
        tightens/loosens as the adaptive estimator learns real costs."""
        used_bytes = self._engine.stats().used_bytes
        headroom = self._estimator.admission_headroom_bytes(used_bytes)
        max_rows_by_features = {}
        for n_features in (10, 50, 100, 200, 500):
            lo, hi = 0, 2_000_000
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if self._estimator.estimate_bytes(mid, 0, n_features) <= headroom:
                    lo = mid
                else:
                    hi = mid - 1
            max_rows_by_features[str(n_features)] = lo
        return {
            "backend": self._backend_name,
            "cache_mode": self._settings.kv_cache,
            "memory_ceiling_bytes": self._estimator.ceiling_bytes(),
            "admission_headroom_bytes": headroom,
            "cache_used_bytes": used_bytes,
            "gpu_memory_fraction": self._settings.gpu_memory_fraction,
            "max_admissible_train_rows_by_feature_count": max_rows_by_features,
            "note": (
                "Admission limits are per-replica, usage-aware (they shrink "
                "as this replica's cache fills and recover as contexts are "
                "evicted), and adaptive: they loosen for shapes similar to "
                "ones this replica has measured. A rejected fit returns 413 "
                "with the same numbers."
            ),
        }

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
            "replica_started_at_unix": self._started_at,
            "replica_uptime_s": round(time.time() - self._started_at, 1),
            "estimator_confidence": self._estimator.confidence(),
            "cache_stats": {
                "n_cached_contexts": stats.n_cached_contexts,
                "used_bytes": stats.used_bytes,
                "free_bytes": stats.free_bytes,
                "capacity_bytes": stats.capacity_bytes,
            },
            "batching": {
                "batched_requests": self._predictor.batched_requests,
                "engine_calls": self._predictor.engine_calls,
            },
            "uploads": self._uploads.stats(),
            "real_gpu_memory": real_gpu_memory,
        }


app = TabctxService.bind()
