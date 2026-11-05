"""The tabctx HTTP service: TabctxEngine behind FastAPI, host-agnostic.

This module defines the service class and its routes but binds them to no
host. Two hosts wrap it:

- serve/app.py: the Ray Serve deployment (`tabctx.serve.app:app`), which
  adds Ray's HTTP proxy, replica management and -- at num_replicas >= 2 --
  the consistent-hash session-affinity router described below.
- serve/standalone.py: a plain uvicorn process (`python -m tabctx.serve`),
  for hosts without Ray. One replica only: nothing routes requests for a
  dataset back to the process that fit it.

Nothing here may import ray: the unit-test environment doesn't have it
(ray lives in app.py, covered by the integration tests), and the
standalone host exists for deployments where Ray isn't installed at all.

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

Multi-replica routing (v0.6.0): the Ray deployment ships with Ray Serve's
consistent-hash request router (see serve/app.py for the wiring) so
requests carrying the session-affinity header (`x-session-id`,
configurable via RAY_SERVE_SESSION_ID_HEADER_KEY) pin to a consistent
replica. The contract -- the session id IS the dataset_id -- lives in
serve/affinity.py; engine construction (backend selection, GPU memory
budgeting) lives in serve/factory.py; tenant namespacing
(`x-tabctx-tenant-id` scoping dataset_ids per tenant, see
serve/tenancy.py) closes the guessable-dataset_id data-leakage gap.
"""

import dataclasses
import functools
import json
import logging
import math
import re
import time
import uuid
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal, NamedTuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import AliasChoices, BaseModel, Field, ValidationError, field_validator
from starlette.concurrency import run_in_threadpool

from tabctx import (
    AdmissionRejected,
    BackendComputeError,
    CacheCapacityError,
    CacheUnavailableError,
    DatasetNotFoundError,
    HostMemoryUnavailableError,
    InvalidInputError,
    UploadNotFoundError,
    UploadTooLargeError,
    timing,
)
from tabctx.batching import CoalescingPredictor
from tabctx.events import RecentEvents
from tabctx.labels import resolve_task
from tabctx.multidevice import MultiDeviceEngine
from tabctx.schema import TableSchema
from tabctx.serve.affinity import resolve_dataset_id, session_id_from_headers
from tabctx.serve.factory import EXPIRY_SWEEP_S, ServeSettings, build_engine
from tabctx.serve.files import read_table, read_training_table
from tabctx.serve.multipart import FILE_FIELD, MAX_FIELD_BYTES, MAX_FIELDS, read_form
from tabctx.serve.reporting import REPORT_STALE_S, PeriodicReport, spill_stats
from tabctx.serve.schemas import SchemaCache
from tabctx.serve.tables import encode_rows, learn_rows
from tabctx.serve.tenancy import (
    TenantRequiredError,
    resolve_tenant_id,
    scope_dataset_id,
    visible_dataset_id,
)
from tabctx.serve.uploads import UploadStore
from tabctx.uncertainty import MAX_QUANTILES, check_quantiles

logger = logging.getLogger("tabctx.serve")
logging.basicConfig(level=logging.INFO)


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("tabctx")
    except Exception:  # noqa: BLE001 -- a source checkout without metadata
        return "dev"


# Tags group the routes in the OpenAPI docs, in this order.
OPENAPI_TAGS = [
    {
        "name": "contexts",
        "description": "Fit a table, predict against it, list and delete.",
    },
    {"name": "deployment", "description": "What this deployment serves and admits."},
    {"name": "health", "description": "Liveness and readiness probes."},
    {
        "name": "legacy",
        "description": "One-shot fit and predict, kept for old callers.",
    },
]

fastapi_app = FastAPI(
    title="tabctx",
    summary="The serving engine for tabular foundation models.",
    version=_version(),
    openapi_tags=OPENAPI_TAGS,
)


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


_TABLE_DOC = (
    "Rows of cells: numbers, text (a category: the server encodes text "
    "columns at fit and reuses that encoding at predict), or null for a "
    "missing value."
)
Cell = float | str | None

_TIMINGS_DOC = (
    "Where this request's time went on the server, in ms per phase (waits "
    "for placement, host memory or the device; preprocessing; making room; "
    "restoring a spilled context; device work), plus counts as n_<name> "
    "(e.g. n_graphs_compiled, which counts the whole worker process, so "
    "concurrent requests can share it). Phases nest and overlap (worker_call "
    "contains the worker's phases); coalesced predicts report their batch's "
    "breakdown. Diagnostic: keys may change."
)

_TASK_DOC = (
    "classification or regression. Omit it to infer from train_y: non-integer "
    "numbers, or whole numbers with more distinct values than both 30 and 5% "
    "of the rows, mean regression; anything else (text, class codes) "
    "classification."
)


class LegacyPredictRequest(BaseModel):
    train_X: list[list[Cell]] = Field(description=_TABLE_DOC)
    train_y: list[float | str]
    test_X: list[list[Cell]] = Field(description=_TABLE_DOC)
    task: Literal["classification", "regression"] | None = Field(
        default=None, description=_TASK_DOC
    )
    return_proba: bool = False


_PREDICTIONS_DOC = (
    "One per test row. Classification: the predicted class as a string, one of "
    "the training labels (the `classes` returned with return_proba), whatever "
    "return_proba is. Regression: numbers."
)


class LegacyPredictResponse(BaseModel):
    task: Literal["classification", "regression"] | None = Field(
        default=None, description="The task fit: the request's, or inferred."
    )
    request_id: str
    predictions: list = Field(description=_PREDICTIONS_DOC)
    probabilities: list[list[float]] | None = None
    classes: list[str] | None = None
    n_train: int
    n_test: int
    n_features: int
    latency_ms: float
    gpu_memory: GpuMemory | None = None


class FitRequest(BaseModel):
    # Inline path (small tables) ...
    train_X: list[list[Cell]] | None = Field(default=None, description=_TABLE_DOC)
    train_y: list[float | str] | None = None
    # ... or a table file: sent with the request (multipart `file`), or a
    # prior POST /v1/tabctx/upload referenced by id. Its columns are the
    # features plus one target column.
    train_upload_id: str | None = None
    target_column: str | None = Field(
        default=None,
        validation_alias=AliasChoices("target_column", "target"),
        description="The table file's label column (also accepted as `target`); "
        "default: its last column.",
    )
    task: Literal["classification", "regression"] | None = Field(
        default=None, description=_TASK_DOC
    )
    dataset_id: str | None = None
    # Which model serves this dataset: an exact
    # id from GET /v1/models (e.g. "tabicl-v2", "tabpfn-3.5"). Default:
    # the deployment's first-listed model.
    model: str | None = None
    ttl_s: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
        description="Delete this context after this many seconds, from every "
        "tier, whatever else happens. The server's TABCTX_MAX_CONTEXT_AGE_S, "
        "if set, caps it (and applies when this is omitted).",
    )


class UploadResponse(BaseModel):
    upload_id: str
    n_bytes: int
    served_by: str | None = None


class FitResponse(BaseModel):
    dataset_id: str
    model: str
    task: Literal["classification", "regression"] = Field(
        description="The task the context was fit for: the request's, or inferred."
    )
    n_train: int
    n_features: int
    expires_at: float | None = Field(
        default=None,
        description="Unix time the context is deleted at (its retention "
        "limit), or null if it has none.",
    )
    # Which replica served this -- lets clients and probes verify that
    # session affinity actually pinned fit() and later predict() calls to
    # the same replica (the multi-replica correctness contract).
    served_by: str | None = None
    timings_ms: dict[str, float | int] | None = Field(
        default=None, description=_TIMINGS_DOC
    )


class TabctxPredictRequest(BaseModel):
    dataset_id: str
    # Inline path ...
    test_X: list[list[Cell]] | None = Field(default=None, description=_TABLE_DOC)
    # ... or a table file (multipart `file`, or an upload's id). When the
    # training table was a file, columns are matched by name, so their
    # order doesn't matter and extra columns (such as the target) are
    # ignored; otherwise by position.
    test_upload_id: str | None = None
    return_proba: bool = False
    quantiles: (
        Annotated[
            list[Annotated[float, Field(gt=0, lt=1)]],
            Field(min_length=1, max_length=MAX_QUANTILES),
        ]
        | None
    ) = Field(
        default=None,
        description="Regression only: also return each row's predicted "
        "distribution at these levels in (0, 1), e.g. [0.05, 0.5, 0.95] for "
        "the median and a 90% interval. Each level at most once.",
    )

    @field_validator("quantiles")
    @classmethod
    def _valid_levels(cls, value: list[float] | None) -> list[float] | None:
        """Checked with the request, before any table is read."""
        try:
            check_quantiles(value)
        except InvalidInputError as e:
            raise ValueError(str(e)) from e
        return value


class TabctxPredictResponse(BaseModel):
    predictions: list = Field(description=_PREDICTIONS_DOC)
    probabilities: list[list[float]] | None = None
    classes: list[str] | None = None
    quantiles: list[list[float]] | None = Field(
        default=None,
        description="With quantiles requested: one row per test row, the value "
        "at each requested level, in the order of quantile_levels.",
    )
    quantile_levels: list[float] | None = Field(
        default=None, description="The levels the quantiles are at, as asked."
    )
    n_test: int
    latency_ms: float
    served_by: str | None = None
    timings_ms: dict[str, float | int] | None = Field(
        default=None, description=_TIMINGS_DOC
    )


class DatasetInfo(BaseModel):
    dataset_id: str
    model: str
    task: Literal["classification", "regression"]
    n_train: int
    n_features: int
    bytes: int = Field(
        description="The context's size in the cache: the device memory it "
        "occupies when resident."
    )
    location: Literal["resident", "spilled"] = Field(
        description="resident: in device memory. spilled: in this replica's "
        "disk tier (TABCTX_SPILL_DIR), restored on its next predict."
    )
    device: str | None = Field(
        default=None, description="Which GPU holds it, on a multi-GPU replica."
    )
    fitted_at: float | None = Field(
        default=None,
        description="Unix time of the fit (null for a context spilled by an "
        "older tabctx).",
    )
    expires_at: float | None = Field(
        default=None,
        description="Unix time it is deleted at (its retention limit), or null.",
    )


class DatasetList(BaseModel):
    data: list[DatasetInfo]
    served_by: str | None = Field(
        default=None,
        description="The replica that answered: with several, each lists only "
        "its own contexts.",
    )


class DeleteResponse(BaseModel):
    dataset_id: str
    deleted: Literal[True] = True


_AUTH_ERRORS = (TenantRequiredError,)
_INVALID_INPUT_ERRORS = (InvalidInputError,)
_ADMISSION_ERRORS = (AdmissionRejected, CacheCapacityError, UploadTooLargeError)
_NOT_FOUND_ERRORS = (DatasetNotFoundError, UploadNotFoundError)
_COMPUTE_ERRORS = (BackendComputeError,)
_UNAVAILABLE_ERRORS = (HostMemoryUnavailableError, CacheUnavailableError)


def _map_error(e: Exception) -> HTTPException:
    if isinstance(e, _AUTH_ERRORS):
        return HTTPException(401, str(e))
    if isinstance(e, _INVALID_INPUT_ERRORS):
        return HTTPException(422, str(e))
    if isinstance(e, _ADMISSION_ERRORS):
        return HTTPException(413, str(e))
    if isinstance(e, _NOT_FOUND_ERRORS):
        return HTTPException(404, str(e))
    if isinstance(e, _UNAVAILABLE_ERRORS):
        return HTTPException(503, str(e))
    if isinstance(e, _COMPUTE_ERRORS):
        return HTTPException(507, str(e))
    raise e


def _list_fields(schema: dict) -> set[str]:
    """A model's fields that take a list (optional or not)."""

    def is_list(prop: dict) -> bool:
        return prop.get("type") == "array" or any(
            alt.get("type") == "array" for alt in prop.get("anyOf", ())
        )

    return {name for name, prop in schema["properties"].items() if is_list(prop)}


@functools.cache
def _model_list_fields(model: type[BaseModel]) -> frozenset[str]:
    return frozenset(_list_fields(model.model_json_schema()))


def _form_lists(fields: dict[str, str], names: frozenset[str]) -> dict[str, object]:
    """A form field carries a list as one value: "0.05,0.95", "0.05 0.95"
    or a JSON array. (Repeating the field is refused: see multipart.py.)"""
    out: dict[str, object] = dict(fields)
    for name in names & fields.keys():
        text = fields[name].strip()
        if text.startswith("["):
            try:
                out[name] = json.loads(text)
            except ValueError:
                out[name] = text  # the model's validation names the field
        else:
            out[name] = [part for part in re.split(r"[,\s]+", text) if part]
    return out


def _request_body(model: type[BaseModel]) -> dict:
    """OpenAPI for a body that is JSON, or multipart/form-data with the
    same fields as form fields plus the table as a `file` part. A list
    field is one form value there: comma-separated, or a JSON array."""
    schema = model.model_json_schema()
    form_lists = {
        name: {
            "type": "string",
            "description": f"{schema['properties'][name].get('description', '')} "
            "As one form value: comma-separated, or a JSON array.".strip(),
            "examples": ["0.05,0.95"],
        }
        for name in _list_fields(schema)
    }
    form = {
        **schema,
        "properties": {
            "file": {
                "type": "string",
                "format": "binary",
                "description": "The table: CSV (plain, gzip or zstd) or Parquet, "
                "with a header row.",
            },
            **schema["properties"],
            **form_lists,
        },
        "required": ["file", *schema.get("required", [])],
    }
    return {
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {"schema": schema},
                "multipart/form-data": {"schema": form},
            },
        }
    }


def _body_error(e: ValidationError) -> RequestValidationError:
    """FastAPI's own 422 shape for a body it validated: locations under
    "body", and the rejected input shown as text when JSON can't carry it
    (infinity, NaN), else the error response itself would fail."""
    return RequestValidationError(
        [
            {**err, "loc": ("body", *err["loc"]), "input": _json_safe(err.get("input"))}
            for err in e.errors()
        ]
    )


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _weakly(
    service: "TabctxService",
    task: Callable[["TabctxService"], object],
    interval_s: float | None = None,
) -> PeriodicReport:
    """task(service) on a timer (a PeriodicReport), on a thread that holds
    the service weakly: it never keeps one alive, and ends once it is gone."""
    ref = weakref.ref(service)

    def run() -> object:
        current = ref()
        return PeriodicReport.STOP if current is None else task(current)

    return PeriodicReport(run, interval_s=interval_s)


def _combine_spill(per_device: list[dict]) -> dict | None:
    """Spill stats summed over devices, with the per-device breakdown."""
    if not per_device:
        return None
    totals: dict = {}
    for stats in per_device:
        for key, value in stats.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] = totals.get(key, 0) + value
    return {**totals, "per_device": per_device}


def _replica_tag() -> str | None:
    try:
        from ray import serve  # imported lazily: absent on standalone hosts

        return serve.get_replica_context().replica_tag
    except Exception:  # no ray, or not inside a Serve replica (e.g. tests)
        return None


class TabctxService:
    """The service itself, undecorated. serve/app.py applies the Ray Serve
    deployment and ingress decorators; serve/standalone.py instantiates it
    directly and rebinds the routes (ROUTE_SPECS below)."""

    def __init__(self) -> None:
        settings = ServeSettings.from_env()
        self._settings = settings
        built = build_engine(settings)
        self._engine = built.engine
        self._estimator = built.estimator
        self._backend_name = built.default  # default model id
        self._spill_store = built.spill_store
        self._spill_stores = built.spill_stores
        self._workers = built.worker_shards
        self._host_guard = built.host_guard
        self._devices = built.devices
        self._device = built.device
        self._gpu_capacity_bytes = built.gpu_capacity_bytes
        # Residency events from the multi-device router (events.py): the
        # latest, for /v1/tabctx/residency; a platform adds its own sinks.
        self._residency = RecentEvents()
        bus = getattr(self._engine, "events", None)
        if bus is not None:
            bus.subscribe(self._residency)
        # Same-context predict coalescing (see batching.py): concurrent
        # requests against one cached context share a single backend call.
        self._predictor = CoalescingPredictor(
            self._engine, window_s=settings.batch_window_ms / 1000.0
        )
        # Large-table ingestion (fit/predict-by-reference): replica-local
        # streamed CSV uploads; see serve/uploads.py for the affinity and
        # tenancy contracts that make this correct at num_replicas >= 2.
        # Each fitted context's schema, kept here too so a predict needn't
        # ask the engine (a round trip to a worker process) for it.
        self._schemas = SchemaCache()
        self._uploads = UploadStore(
            ttl_s=settings.upload_ttl_s,
            max_upload_bytes=settings.max_upload_bytes,
        )
        # Restart visibility (ROADMAP "cache durability", first step): a
        # replica restart silently drops every cached context, and the
        # caller-visible symptom (404 -> re-fit) is indistinguishable
        # from eviction. Exposing when this replica started lets
        # operators and probes correlate a burst of 404s with a restart
        # instead of chasing a phantom eviction/routing bug.
        self._started_at = time.time()
        # Device measurements /readyz reads without waiting (#51): worker
        # processes send their own; devices served in this process are
        # measured here.
        self._local_report = (
            None if self._workers else _weakly(self, TabctxService._measure_local)
        )
        # Contexts' engines delete them at their retention deadline; their
        # schemas here (text categories of the training table) go too.
        self._schema_purge = _weakly(
            self,
            lambda service: service._schemas.purge_expired(),
            interval_s=EXPIRY_SWEEP_S,
        )
        if "cuda" not in self._device and "fake" not in settings.backends:
            logger.warning("No CUDA device visible -- running tabctx on CPU")
        logger.info(
            "TabctxService initialized (backend=%s, device=%s, "
            "gpu_memory_fraction=%s, replica=%s). %s",
            ",".join(settings.backends),
            self._device,
            settings.gpu_memory_fraction,
            _replica_tag(),
            self._estimator.confidence(),
        )

    # ---- legacy one-shot endpoint (regression-compatible with the old app.py) ----

    @fastapi_app.post(
        "/v1/tabicl/predict",
        response_model=LegacyPredictResponse,
        summary="Fit, predict once and discard",
        tags=["legacy"],
    )
    async def legacy_predict(self, req: LegacyPredictRequest) -> LegacyPredictResponse:
        return await run_in_threadpool(self._legacy_predict_sync, req)

    def _legacy_predict_sync(self, req: LegacyPredictRequest) -> LegacyPredictResponse:
        torch = _torch_or_none()
        cuda = torch is not None and torch.cuda.is_available()

        request_id = str(uuid.uuid4())
        n_train, n_test = len(req.train_X), len(req.test_X)
        n_features = len(req.train_X[0]) if n_train else 0

        # Peak GPU memory of this call, measured in-process. With worker
        # processes the call ran elsewhere: this process holds no models,
        # and touching each GPU would only create CUDA contexts here.
        gpus = self._cuda_devices() if cuda and not self._workers else []
        for d in gpus:
            torch.cuda.reset_peak_memory_stats(d)
        start = time.monotonic()
        try:
            task = resolve_task(req.train_y, req.task)
            X, schema = learn_rows(req.train_X)
            outcome = self._engine.fit_predict(
                X,
                req.train_y,
                encode_rows(req.test_X, schema),
                task=task,
                return_proba=req.return_proba,
            )
        except (
            *_INVALID_INPUT_ERRORS,
            *_ADMISSION_ERRORS,
            *_NOT_FOUND_ERRORS,
            *_COMPUTE_ERRORS,
            *_UNAVAILABLE_ERRORS,
        ) as e:
            logger.error("[%s] %s: %s", request_id, type(e).__name__, e)
            raise _map_error(e) from e
        latency_ms = (time.monotonic() - start) * 1000

        gpu_memory = None
        if gpus:
            # The fit ran on one of the devices; the peak over all of them
            # is that device's (the others had no new work from this call).
            gpu_memory = GpuMemory(
                allocated_mb=max(torch.cuda.max_memory_allocated(d) for d in gpus)
                / 1e6,
                reserved_mb=max(torch.cuda.max_memory_reserved(d) for d in gpus) / 1e6,
            )
        logger.info(
            "[%s] legacy predict train=%d test=%d feats=%d latency_ms=%.1f gpu=%s",
            request_id,
            n_train,
            n_test,
            n_features,
            latency_ms,
            gpu_memory,
        )
        return LegacyPredictResponse(
            task=task,
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

    @fastapi_app.post(
        "/v1/tabctx/upload",
        response_model=UploadResponse,
        summary="Upload a table ahead of a fit or predict",
        tags=["contexts"],
    )
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

    # ---- request bodies: JSON, or multipart/form-data with a table file ----

    async def _read_request[R: BaseModel](
        self, request: Request, model: type[R]
    ) -> tuple[R, Path | None]:
        """The request's fields, and its table file if it sent one. A JSON
        body carries the fields; a multipart/form-data body carries them as
        form fields and the table as a `file` part, which streams to disk
        under the upload size cap (serve/multipart.py) for the caller to
        read and discard."""
        # Refused callers and oversized bodies are turned away from the
        # headers alone, before any of the body is read.
        try:
            resolve_tenant_id(dict(request.headers))
        except (*_AUTH_ERRORS, *_INVALID_INPUT_ERRORS) as e:
            raise _map_error(e) from e
        content_type = request.headers.get("content-type", "")
        if not content_type.lower().startswith("multipart/form-data"):
            try:
                return model.model_validate_json(await request.body()), None
            except ValidationError as e:
                raise _body_error(e) from e
        self._check_length(request)
        try:
            fields, writer = await read_form(
                request.stream(), content_type, self._uploads.begin
            )
        except (InvalidInputError, UploadTooLargeError) as e:
            raise _map_error(e) from e
        if writer is None:
            raise HTTPException(
                422, f"a multipart request carries its table as a `{FILE_FIELD}` part"
            )
        path = writer.finish()
        try:
            values = _form_lists(fields, _model_list_fields(model))
            return model.model_validate(values), path
        except ValidationError as e:
            self._uploads.discard(path)
            raise _body_error(e) from e

    def _check_length(self, request: Request) -> None:
        declared = request.headers.get("content-length", "")
        cap = self._uploads.max_upload_bytes + MAX_FIELDS * MAX_FIELD_BYTES
        if declared.isdigit() and int(declared) > cap:
            raise _map_error(
                UploadTooLargeError(
                    f"the request is {int(declared)} bytes; tables are capped at "
                    f"{self._uploads.max_upload_bytes} (TABCTX_MAX_UPLOAD_BYTES)"
                )
            )

    def _table_file(
        self,
        sent: Path | None,
        upload_id: str | None,
        inline: bool,
        tenant_id: str | None,
        rows: str,
    ) -> Path | None:
        """The request's table file -- sent with it, or an earlier upload --
        or None for inline rows. Exactly one of the three, checked before
        an upload is used up (uploads are single-use)."""
        given = (sent is not None) + (upload_id is not None) + inline
        if given != 1:
            raise InvalidInputError(
                f"send the table one way: inline {rows}, a table file, or an "
                f"upload id ({'got several' if given else 'got none'})"
            )
        if upload_id is not None:
            return self._uploads.consume(upload_id, tenant_id)
        return sent

    def _schema(self, scoped_id: str) -> TableSchema | None:
        """The context's schema: this replica recorded it at fit (affinity
        sends a dataset's fits and predicts here), else -- after a restart --
        the engine's copy. Raises DatasetNotFoundError for no context.
        (A re-fit racing a predict of the same id can encode the predict's
        rows with the old schema; re-fitting an id while predicting
        against it has no defined result anyway.)"""
        schema = self._schemas.get(scoped_id)
        return schema if schema is not None else self._engine.schema(scoped_id)

    @fastapi_app.post(
        "/v1/tabctx/fit",
        response_model=FitResponse,
        summary="Fit a table into a context",
        tags=["contexts"],
        openapi_extra=_request_body(FitRequest),
    )
    async def tabctx_fit(self, request: Request) -> FitResponse:
        req, file = await self._read_request(request, FitRequest)
        session_id = session_id_from_headers(request.headers)
        try:
            return await run_in_threadpool(
                self._fit_sync, req, file, dict(request.headers), session_id
            )
        finally:
            if file is not None:
                self._uploads.discard(file)

    def _resolve_train_table(
        self, req: FitRequest, file: Path | None, tenant_id: str | None
    ) -> tuple[object, object, TableSchema | None]:
        """Returns (X, y, schema) from whichever of inline rows or a table
        file the request used -- exactly one."""
        inline = req.train_X is not None or req.train_y is not None
        file = self._table_file(
            file, req.train_upload_id, inline, tenant_id, "train_X and train_y"
        )
        if file is None:
            if req.train_X is None or req.train_y is None:
                raise InvalidInputError("an inline fit needs both train_X and train_y")
            X, schema = learn_rows(req.train_X)
            return X, req.train_y, schema
        try:
            features, y = read_training_table(file, req.target_column, req.task)
        finally:
            self._uploads.discard(file)
        schema, X = TableSchema.learn(features.columns, features.names)
        return X, y, schema

    def _fit_sync(
        self,
        req: FitRequest,
        file: Path | None,
        headers: dict[str, str],
        session_id: str | None,
    ) -> FitResponse:
        with timing.collect() as timings:
            response = self._fit_timed(req, file, headers, session_id)
        response.timings_ms = timings.as_dict()
        return response

    def _fit_timed(
        self,
        req: FitRequest,
        file: Path | None,
        headers: dict[str, str],
        session_id: str | None,
    ) -> FitResponse:
        try:
            tenant_id = resolve_tenant_id(headers)
            with timing.phase("read_table"):
                X, y, schema = self._resolve_train_table(req, file, tenant_id)
            task = resolve_task(y, req.task)
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
            model_id = req.model or self._engine.default_backend
            # The schema (column names, text categories) is recorded on
            # the cached context so test tables are encoded the same way;
            # it lives and dies with the context, spill included.
            with self._schemas.fitting(scoped_id) as record:
                expires_at = self._expires_at(req.ttl_s)
                self._engine.fit(
                    X,
                    y,
                    task=task,
                    dataset_id=scoped_id,
                    backend=model_id,
                    schema=schema,
                    expires_at=expires_at,
                )
                if schema is not None:
                    record(schema, expires_at)
        except (
            *_AUTH_ERRORS,
            *_INVALID_INPUT_ERRORS,
            *_ADMISSION_ERRORS,
            *_NOT_FOUND_ERRORS,
            *_COMPUTE_ERRORS,
            *_UNAVAILABLE_ERRORS,
        ) as e:
            raise _map_error(e) from e
        return FitResponse(
            dataset_id=dataset_id,
            model=model_id,
            task=task,
            n_train=n_train,
            n_features=n_features,
            expires_at=expires_at,
            served_by=_replica_tag(),
        )

    def _expires_at(self, ttl_s: float | None) -> float | None:
        """When a context fit now is deleted: after the request's ttl_s,
        capped by the server's maximum age (which applies alone when the
        request names none); None for neither."""
        limits = [t for t in (ttl_s, self._settings.max_context_age_s) if t]
        return time.time() + min(limits) if limits else None

    @fastapi_app.post(
        "/v1/tabctx/predict",
        response_model=TabctxPredictResponse,
        summary="Predict against a context",
        tags=["contexts"],
        openapi_extra=_request_body(TabctxPredictRequest),
    )
    async def tabctx_predict(self, request: Request) -> TabctxPredictResponse:
        req, file = await self._read_request(request, TabctxPredictRequest)
        session_id = session_id_from_headers(request.headers)
        try:
            return await run_in_threadpool(
                self._predict_sync, req, file, dict(request.headers), session_id
            )
        finally:
            if file is not None:
                self._uploads.discard(file)

    def _resolve_test_table(
        self,
        req: TabctxPredictRequest,
        file: Path | None,
        tenant_id: str | None,
        scoped_id: str,
    ) -> object:
        file = self._table_file(
            file, req.test_upload_id, req.test_X is not None, tenant_id, "test_X"
        )
        try:
            # Encoded as the training table was (text categories, column
            # names); contexts from before schemas existed have none.
            schema = self._schema(scoped_id)
            if file is None:
                return encode_rows(req.test_X, schema)
            text = schema.text_names if schema is not None else ()
            table = read_table(file, text_columns=text)
            if schema is None:
                schema = TableSchema((None,) * len(table.columns))
            return schema.encode(schema.select(table.names, table.columns))
        finally:
            if file is not None:
                self._uploads.discard(file)

    def _predict_sync(
        self,
        req: TabctxPredictRequest,
        file: Path | None,
        headers: dict[str, str],
        session_id: str | None,
    ) -> TabctxPredictResponse:
        with timing.collect() as timings:
            response = self._predict_timed(req, file, headers, session_id)
        response.timings_ms = timings.as_dict()
        return response

    def _predict_timed(
        self,
        req: TabctxPredictRequest,
        file: Path | None,
        headers: dict[str, str],
        session_id: str | None,
    ) -> TabctxPredictResponse:
        start = time.monotonic()
        try:
            tenant_id = resolve_tenant_id(headers)
            resolve_dataset_id(session_id, req.dataset_id)
            scoped_id = scope_dataset_id(tenant_id, req.dataset_id)
            with timing.phase("read_table"):
                X_test = self._resolve_test_table(req, file, tenant_id, scoped_id)
            outcome = self._predictor.predict(
                scoped_id,
                X_test,
                return_proba=req.return_proba,
                quantiles=req.quantiles,
            )
        except (
            *_AUTH_ERRORS,
            *_INVALID_INPUT_ERRORS,
            *_ADMISSION_ERRORS,
            *_NOT_FOUND_ERRORS,
            *_COMPUTE_ERRORS,
            *_UNAVAILABLE_ERRORS,
        ) as e:
            raise _map_error(e) from e
        latency_ms = (time.monotonic() - start) * 1000
        return TabctxPredictResponse(
            predictions=outcome.predictions,
            probabilities=outcome.probabilities,
            classes=outcome.classes,
            quantiles=outcome.quantiles,
            quantile_levels=req.quantiles if outcome.quantiles is not None else None,
            n_test=len(X_test),
            latency_ms=latency_ms,
            served_by=_replica_tag(),
        )

    # ---- the caller's contexts: list and delete ----

    @fastapi_app.get(
        "/v1/tabctx/datasets",
        response_model=DatasetList,
        summary="List your contexts",
        tags=["contexts"],
    )
    def datasets(self, request: Request) -> DatasetList:
        """The contexts this replica holds for the caller's tenant, resident
        or spilled: metadata only, never their contents. With several
        replicas, only the one that answers (served_by) is listed."""
        try:
            tenant_id = resolve_tenant_id(dict(request.headers))
            contexts = self._engine.contexts()
        except (*_AUTH_ERRORS, *_INVALID_INPUT_ERRORS, *_UNAVAILABLE_ERRORS) as e:
            raise _map_error(e) from e
        data = []
        for info in contexts:
            shown = visible_dataset_id(tenant_id, info.dataset_id)
            if shown is None:
                continue
            fields = dataclasses.asdict(info)
            fields.update(dataset_id=shown, device=info.device or self._single_device())
            data.append(DatasetInfo(**fields))
        data.sort(key=lambda d: (d.dataset_id, d.device or ""))
        return DatasetList(data=data, served_by=_replica_tag())

    @fastapi_app.delete(
        "/v1/tabctx/datasets/{dataset_id:path}",
        response_model=DeleteResponse,
        summary="Delete a context",
        tags=["contexts"],
    )
    def delete_dataset(self, dataset_id: str, request: Request) -> DeleteResponse:
        """Delete a context: from device memory, host memory and the disk
        tier, on every GPU of this replica. 404 if there is none (never fit,
        already deleted, evicted without a spill tier, or lost to a
        restart); 503 if that can't be done right now (retry). Uploaded
        tables are already gone: they are deleted once the fit or predict
        that used them has read them. In multi-replica deployments, send
        `x-session-id: <dataset_id>` so the request reaches the replica
        that holds it."""
        try:
            if not dataset_id:
                raise InvalidInputError("name the dataset_id to delete")
            tenant_id = resolve_tenant_id(dict(request.headers))
            resolve_dataset_id(session_id_from_headers(request.headers), dataset_id)
            scoped_id = scope_dataset_id(tenant_id, dataset_id)
            # Ordered against fits of this id on this replica, so a fit
            # finishing meanwhile can't record its schema after the delete.
            with self._schemas.deleting(scoped_id):
                existed = self._engine.evict(scoped_id)
            if not existed:
                raise DatasetNotFoundError(
                    f"no context for dataset_id={dataset_id!r} -- it was never "
                    "fit, was already deleted, was evicted, or was lost to a "
                    "replica restart"
                )
        except (
            *_AUTH_ERRORS,
            *_INVALID_INPUT_ERRORS,
            *_NOT_FOUND_ERRORS,
            *_UNAVAILABLE_ERRORS,
        ) as e:
            raise _map_error(e) from e
        logger.info("deleted context %s", scoped_id)
        return DeleteResponse(dataset_id=dataset_id)

    def _single_device(self) -> str | None:
        """The device of a single-device replica (a multi-device engine tags
        each context itself)."""
        return None if isinstance(self._engine, MultiDeviceEngine) else self._device

    @fastapi_app.get("/healthz", summary="Process is up", tags=["health"])
    def healthz(self):
        return {"status": "ok"}

    def check_health(self) -> None:
        """Ray Serve's replica health probe: a dead per-GPU worker loses that
        device's contexts and capacity for good, so fail the replica and
        let Serve restart it whole."""
        dead = [w.death_reason for w in self._workers if not w.alive]
        if dead:
            raise RuntimeError("; ".join(r for r in dead if r))

    def close(self) -> None:
        """Stop the per-GPU worker processes, if any. Standalone hosts call
        this at shutdown; a Ray replica's teardown kills its process tree
        (and workers.py's atexit hook backstops both)."""
        self._schema_purge.stop()
        if self._local_report is not None:
            self._local_report.stop()
        for shard in self._workers:
            shard.close()

    @fastapi_app.get("/livez", summary="Liveness probe", tags=["health"])
    def livez(self):
        """check_health as an HTTP route, for hosts without Ray (e.g. a
        Kubernetes livenessProbe): 503 only on the condition a restart
        fixes, a dead per-GPU worker. /readyz is the wrong liveness probe
        -- it also 503s on transient pressure (host memory, a slow worker
        report), which would restart a healthy replica under load."""
        try:
            self.check_health()
        except RuntimeError as e:
            raise HTTPException(503, detail=str(e)) from e
        return {"status": "alive"}

    @fastapi_app.get(
        "/v1/models", summary="List the models served", tags=["deployment"]
    )
    def models(self):
        """Which models this deployment serves: pass one of these ids
        as `model` in fit requests.
        The first is the default."""
        return {
            "data": [
                {
                    "id": name,
                    "object": "model",
                    "default": name == self._engine.default_backend,
                }
                for name in self._engine.backend_names
            ]
        }

    @fastapi_app.get(
        "/v1/tabctx/residency", summary="Where your contexts are", tags=["contexts"]
    )
    def residency(self, request: Request):
        """Where the caller's contexts are now, per device, and the router's
        latest placements and removals of them (multi-GPU replicas). Scoped
        like every /v1/tabctx route: a tenant sees only its own datasets.
        Events are advisory (contexts dropped by capacity eviction without
        spill aren't reported); `devices` is the current truth to
        reconcile against. A single-GPU replica reports neither."""
        try:
            tenant_id = resolve_tenant_id(dict(request.headers))
        except _AUTH_ERRORS as e:
            raise _map_error(e) from e

        def mine(ids: list[str]) -> list[str]:
            visible = (visible_dataset_id(tenant_id, i) for i in ids)
            return [i for i in visible if i is not None]

        residency = getattr(self._engine, "residency", None)
        devices = residency() if callable(residency) else {}
        events = []
        for event in self._residency.snapshot():
            shown = visible_dataset_id(tenant_id, event["dataset_id"])
            if shown is not None:
                events.append({**event, "dataset_id": shown})
        return {
            "devices": {
                device: {kind: mine(ids) for kind, ids in view.items()}
                for device, view in devices.items()
            },
            "events": events,
        }

    @fastapi_app.get(
        "/v1/tabctx/limits", summary="Table shapes it admits now", tags=["deployment"]
    )
    def limits(self):
        """Capability discovery (inspired by PriorLabs' get_model_limits):
        what will this deployment admit? Lets clients validate table
        shapes BEFORE uploading data, instead of discovering a 413 the
        expensive way. max_admissible_train_rows is derived live from
        the admission gate at representative feature counts, so it
        tightens/loosens as the adaptive estimator learns real costs."""
        used_bytes = self._engine.stats_nowait().used_bytes
        # The best any one device offers: a fit lands on a single device.
        headroom = self._engine.admission_headroom_bytes()
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
            "models": self._engine.backend_names,
            "default_model": self._engine.default_backend,
            "cache_mode": self._settings.kv_cache,
            "memory_ceiling_bytes": self._estimator.ceiling_bytes(),
            "admission_headroom_bytes": headroom,
            "cache_used_bytes": used_bytes,
            "gpu_memory_fraction": self._settings.gpu_memory_fraction,
            # Retention: no context kept longer than this after its fit
            # (null: no limit); a fit's ttl_s is capped by it.
            "max_context_age_s": self._settings.max_context_age_s,
            "gpu_capacity_bytes": self._gpu_capacity_bytes,
            "devices": list(self._devices),
            # Scopes: cache_used_bytes is the replica's total;
            # memory_ceiling_bytes and the headrooms are per device.
            "device_admission_headroom_bytes": (
                self._engine.device_headroom_bytes()
                if isinstance(self._engine, MultiDeviceEngine)
                else [headroom]
            ),
            "max_admissible_train_rows_by_feature_count": max_rows_by_features,
            "note": (
                "Admission limits are per-replica, usage-aware (they shrink "
                "as this replica's cache fills and recover as contexts are "
                "evicted), and adaptive: they loosen for shapes similar to "
                "ones this replica has measured. A rejected fit returns 413 "
                "with the same numbers. With several GPUs, a table must fit "
                "on one of them: the limits are the best single device's."
            ),
        }

    @fastapi_app.get("/readyz", summary="Readiness and cache state", tags=["health"])
    def readyz(self):
        """Readiness, from measurements it never waits for (#51): the cache's
        lock-free counters, the host-memory guard's lock-free read, and the
        device reports measured on their own threads (see
        serve/reporting.py). 503 for a dead GPU worker, or one silent for
        REPORT_STALE_S (it is stuck). This process's own report is shown,
        not judged: its age only says how fresh the numbers are."""
        torch = _torch_or_none()
        stats = self._engine.stats_nowait()
        dead = [w.death_reason for w in self._workers if not w.alive]
        if dead:
            raise HTTPException(503, detail="; ".join(r for r in dead if r))
        # Worker processes measure their own device (this process holds no
        # CUDA context and must not create one on every GPU) and send it.
        reports = [(w.device, *w.latest_report()) for w in self._workers]
        stale = [
            f"{device} ({age:.0f} s)"
            for device, _, age in reports
            if age is not None and age > REPORT_STALE_S
        ]
        if stale:
            raise HTTPException(
                503, detail=f"no device report from {', '.join(stale)}: stuck"
            )
        local, local_age = (
            self._local_report.latest() if self._local_report else (None, None)
        )
        per_device: dict[str, dict] = {
            device: report["gpu_memory"]
            for device, report, _ in reports
            if report is not None and report["gpu_memory"] is not None
        }
        if local is not None:
            per_device.update(local["gpu_memory"])
        real_gpu_memory = None
        if per_device:
            # Same flat shape as ever (summed over devices when several),
            # plus the per-device breakdown on a multi-GPU replica.
            real_gpu_memory = {
                "allocated_mb": sum(m["allocated_mb"] for m in per_device.values()),
                "reserved_mb": sum(m["reserved_mb"] for m in per_device.values()),
            }
            if len(per_device) > 1:
                real_gpu_memory["per_device"] = per_device
        ages = [age for _, _, age in reports if age is not None]
        if local_age is not None:
            ages.append(local_age)
        spill = (
            local["spill"]
            if local is not None and local["spill"] is not None
            else _combine_spill(
                [
                    report["spill"]
                    for _, report, _ in reports
                    if report is not None and report["spill"] is not None
                ]
            )
        )
        return {
            "status": "ready",
            "device": self._device,
            "cuda_available": torch is not None and torch.cuda.is_available(),
            "gpu_capacity_bytes": self._gpu_capacity_bytes,
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
            "spill": spill,
            "real_gpu_memory": real_gpu_memory,
            # How old the device measurements above are (the oldest).
            "device_report_age_s": round(max(ages), 1) if ages else None,
            "devices": self._device_breakdown(),
            "host_memory": self._host_guard.stats_nowait()
            if self._host_guard
            else None,
        }

    def _measure_local(self) -> dict:
        """This process's device report: GPU memory torch holds on each
        device it serves in-process, and its spill tier. Run on the report
        thread (both can wait: torch's allocator lock, a directory walk)."""
        torch = _torch_or_none()
        gpu: dict[str, dict] = {}
        if torch is not None and torch.cuda.is_available():
            # Real device memory, independent of our own byte-accounting --
            # cache_stats.used_bytes is the estimator's *prediction* for what
            # cached contexts should cost; this is what the device actually
            # reports. Comparing the two across a long-running replica is how
            # you'd catch the estimator drifting from reality or the cache
            # failing to actually release GPU memory on eviction.
            gpu = {
                d: {
                    "allocated_mb": torch.cuda.memory_allocated(d) / 1e6,
                    "reserved_mb": torch.cuda.memory_reserved(d) / 1e6,
                }
                for d in self._cuda_devices()
            }
        spill = spill_stats(self._spill_store)
        if spill is None and self._spill_stores:
            spill = _combine_spill([spill_stats(s) for s in self._spill_stores])
        return {"gpu_memory": gpu, "spill": spill}

    def _cuda_devices(self) -> list[str]:
        """The CUDA devices this replica serves from (just "cuda", the
        current device, on an auto single-GPU replica)."""
        return [d for d in self._devices if d.startswith("cuda")]

    def _device_breakdown(self) -> list[dict] | None:
        """Per-device cache use and activity on a multi-GPU replica (None
        on a single device, where cache_stats already says it all)."""
        if not isinstance(self._engine, MultiDeviceEngine):
            return None
        return [
            {
                "device": device,
                "n_cached_contexts": stats.n_cached_contexts,
                "used_bytes": stats.used_bytes,
                "capacity_bytes": stats.capacity_bytes,
                **activity,
            }
            for device, stats, activity in zip(
                self._engine.devices,
                self._engine.device_stats(),
                self._engine.device_activity(),
                strict=True,
            )
        ]


class RouteSpec(NamedTuple):
    """One route as registered on fastapi_app above, captured at import
    time -- before a Ray host's serve.ingress(fastapi_app) rewrites the
    routes' endpoints for its own dispatch -- so a standalone host can
    rebind each handler name to a real instance (serve/standalone.py)."""

    path: str
    methods: tuple[str, ...]
    endpoint_name: str
    response_model: object | None
    openapi_extra: dict | None
    summary: str | None
    tags: list


ROUTE_SPECS: tuple[RouteSpec, ...] = tuple(
    RouteSpec(
        path=route.path,
        methods=tuple(sorted(route.methods - {"HEAD"})),
        endpoint_name=route.endpoint.__name__,
        response_model=route.response_model,
        openapi_extra=route.openapi_extra,
        summary=route.summary,
        tags=list(route.tags),
    )
    for route in fastapi_app.routes
    if isinstance(route, APIRoute)
)
