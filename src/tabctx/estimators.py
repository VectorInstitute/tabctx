"""scikit-learn estimators backed by a tabctx deployment.

    from tabctx import TabctxClassifier

    clf = TabctxClassifier(model="tabpfn-3.5")   # TABCTX_BASE_URL, TABCTX_API_KEY
    clf.fit(X_train, y_train)                    # DataFrame or array; text and NaN fine
    clf.predict_proba(X_test)

They behave like any scikit-learn estimator (clone, pipelines,
cross_val_score), except that the work happens on the deployment: fit
sends the training table, which the server encodes once and keeps on a
GPU; predict sends only the new rows. X may be a pandas DataFrame, a
numpy array or a list of rows, with text columns, dates and missing
values (NaN/None) as they are: the server encodes text and handles
missing values itself. Test columns are matched to the training columns
by name when both tables have names (DataFrames), else by position.

Each table travels as one Parquet file. Predicts larger than
max_request_bytes are split into several requests; a training table
larger than it is refused here, before it is sent. The training
context's id comes from the table, the labels, the model and the task,
so fitting the same data again reuses the id; if the server has dropped
the context (eviction, a restart), predict fits it again from the
training file the estimator keeps and retries once.

With n_jobs > 1 (joblib workers) or a saved model, the client reaches
the worker without its API key, by design: set TABCTX_API_KEY in the
environment.

Requires scikit-learn and pyarrow (`pip install "tabctx[client]"`).
"""

from __future__ import annotations

import hashlib
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, column_or_1d

from tabctx.client import PredictResult, TabctxClient
from tabctx.errors import DatasetNotFoundError
from tabctx.table_files import (
    Columns,
    _missing,
    columns,
    is_frame,
    take_rows,
    to_parquet,
)
from tabctx.types import Task

DEFAULT_MODEL = "tabpfn-3.5"
# The Vector gateway takes request bodies up to 30 MiB; this leaves room
# for the multipart envelope around the file.
MAX_REQUEST_BYTES = 29 * 1024 * 1024
_TARGET = "__tabctx_target__"
# How the server says a context isn't there (engine._not_found); other
# 404s (an unknown model or route) must not trigger a re-fit.
_CONTEXT_GONE = "no cached context"


class _TabctxEstimator(BaseEstimator):
    _task: Task

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        client: TabctxClient | None = None,
        max_request_bytes: int | None = MAX_REQUEST_BYTES,
    ) -> None:
        """model: a model id from client.models() ("tabpfn-3.5",
        "tabicl-v2"). client: where to send the tables; default
        TabctxClient(), configured by TABCTX_BASE_URL and TABCTX_API_KEY.
        max_request_bytes: the largest table file to send in one request
        (the Vector gateway takes 30 MiB bodies); None for no limit."""
        self.model = model
        self.client = client
        self.max_request_bytes = max_request_bytes

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True
        tags.input_tags.categorical = True
        tags.non_deterministic = True  # served, not computed here
        return tags

    def _client(self) -> TabctxClient:
        return self.client if self.client is not None else TabctxClient()

    # ---- fit -----------------------------------------------------------

    def _fit(self, X, labels: np.ndarray):
        cols = columns(X)
        if cols.n_rows != len(labels):
            raise ValueError(f"X has {cols.n_rows} rows but y has {len(labels)} labels")
        data = to_parquet(cols, (_TARGET, labels))
        limit = self.max_request_bytes
        if limit is not None and len(data) > limit:
            raise ValueError(
                f"the training table is {len(data):,} bytes as Parquet; a request "
                f"can be at most {limit:,} (max_request_bytes). "
                "Fit on fewer rows."
            )
        digest = hashlib.sha256(data)
        digest.update(f"\0{self.model}\0{self._task}".encode())
        self.dataset_id_ = f"sklearn-{digest.hexdigest()[:32]}"
        self.model_ = self.model
        self.n_features_in_ = len(cols.names)
        self._names = cols.names
        self._named = cols.named
        self._text = cols.text_names
        if cols.named and all(isinstance(c, str) for c in X.columns):
            self.feature_names_in_ = np.asarray(cols.names, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        self._train_file = data
        self._send_fit()
        return self

    def _send_fit(self) -> None:
        self._client().fit_file(
            self._train_file,
            target=_TARGET,
            dataset_id=self.dataset_id_,
            model=self.model_,
            task=self._task,
        )

    # ---- predict -------------------------------------------------------

    def _predict(self, X, return_proba: bool) -> list[PredictResult]:
        check_is_fitted(self, "dataset_id_")
        cols = self._test_columns(X)
        return [self._predict_file(data, return_proba) for data in self._files(cols)]

    def _test_columns(self, X) -> Columns:
        if is_frame(X) and self._named:
            # By name, as the server matches them; training text stays text.
            return columns(X, text=self._text)
        positions = [self._names.index(n) for n in self._text]
        cols = columns(X, positions=positions)
        if len(cols.names) != self.n_features_in_:
            raise ValueError(
                f"X has {len(cols.names)} features, but {type(self).__name__} is "
                f"expecting {self.n_features_in_} features as input."
            )
        if is_frame(X) or self._named:
            warnings.warn(
                "X and the training table don't both have column names; their "
                "columns are matched by position",
                UserWarning,
                stacklevel=3,
            )
        return cols.renamed(self._names)

    def _files(self, cols: Columns) -> list[bytes]:
        """The test table as Parquet files, each under max_request_bytes."""
        data = to_parquet(cols)
        limit = self.max_request_bytes
        if limit is None or len(data) <= limit:
            return [data]
        if cols.n_rows == 1:
            raise ValueError(
                f"one test row is {len(data):,} bytes as Parquet, over "
                f"max_request_bytes ({limit:,})"
            )
        half = cols.n_rows // 2
        return [
            *self._files(take_rows(cols, 0, half)),
            *self._files(take_rows(cols, half, cols.n_rows)),
        ]

    def _predict_file(self, data: bytes, return_proba: bool) -> PredictResult:
        def send():
            return self._client().predict_file(
                self.dataset_id_, data, return_proba=return_proba, model=self.model_
            )

        try:
            return send()
        except DatasetNotFoundError as e:
            if _CONTEXT_GONE not in str(e):
                raise
        # Evicted or lost in a restart: fit again from the kept file.
        self._send_fit()
        return send()


class TabctxClassifier(ClassifierMixin, _TabctxEstimator):
    """Classification on a tabctx deployment. Labels may be of any one
    type (strings, ints, bools); predict returns them as given to fit, and
    predict_proba's columns follow classes_."""

    _task = "classification"

    def fit(self, X, y):
        if any(_missing(v) for v in np.ravel(np.asarray(y, dtype=object))):
            raise ValueError("y has missing labels; every training row needs one")
        y = column_or_1d(np.asarray(y, dtype=object) if _mixed(y) else y, warn=True)
        check_classification_targets(y)
        self.classes_, codes = np.unique(y, return_inverse=True)
        # Labels travel as class indices, so they come back exactly as given.
        return self._fit(X, codes.astype(np.int64))

    def predict(self, X) -> np.ndarray:
        codes = [int(p) for r in self._predict(X, False) for p in r.predictions]
        return self.classes_[codes]

    def predict_proba(self, X) -> np.ndarray:
        blocks = []
        for result in self._predict(X, True):
            proba = np.zeros((len(result.probabilities), len(self.classes_)))
            proba[:, [int(c) for c in result.classes]] = result.probabilities
            blocks.append(proba)
        return np.vstack(blocks)


class TabctxRegressor(RegressorMixin, _TabctxEstimator):
    """Regression on a tabctx deployment."""

    _task = "regression"

    def fit(self, X, y):
        y = column_or_1d(y, warn=True).astype(np.float64)
        if not np.isfinite(y).all():
            raise ValueError("y has missing or infinite values")
        return self._fit(X, y)

    def predict(self, X) -> np.ndarray:
        return np.array(
            [p for r in self._predict(X, False) for p in r.predictions],
            dtype=np.float64,
        )


def _mixed(y) -> bool:
    """Labels of more than one type (1 and "a"), which numpy would turn
    into text; kept as objects so np.unique refuses them."""
    values = list(np.ravel(np.asarray(y, dtype=object)))
    return len({type(v) for v in values if v is not None}) > 1 and any(
        isinstance(v, str) for v in values
    )
