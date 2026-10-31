"""The tabular tools an agent calls. Every method takes plain arguments
(paths, names, lists, dicts) and returns a JSON-friendly dict, so the MCP
server and the CLI can hand them over as they are. Errors are ValueErrors
with a message that says what to change."""

from __future__ import annotations

import datetime as dt
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

from tabctx.estimators import DEFAULT_MODEL, TabctxClassifier, TabctxRegressor
from tabctx.labels import infer_task
from tabctx.mcp_tools.store import ModelInfo, ModelStore, check_name
from tabctx.mcp_tools.tables import column_kind, profile, read_table, require_columns

MODELS = ("tabpfn-3.5", "tabicl-v2")
BASELINE = "gradient-boosting"
# Most HistGradientBoosting accepts as one categorical feature.
_MAX_CATEGORIES = 255


class TabularTools:
    def __init__(self, store: ModelStore | None = None, client: Any = None) -> None:
        """store: where fitted models are kept (default ModelStore()).
        client: the tabctx deployment (default TabctxClient(), from
        TABCTX_BASE_URL and TABCTX_API_KEY); tests pass a fake."""
        self.store = store or ModelStore()
        self.client = client

    # ---- describe ------------------------------------------------------

    def inspect_table(self, path: str, target: str | None = None) -> dict[str, Any]:
        """Rows, columns, each column's kind, missing values and typical
        values; with target, the task it implies and its class balance."""
        out = profile(read_table(path), target)
        out["file"] = str(path)
        return out

    def list_models(self) -> dict[str, Any]:
        return {
            "store": str(self.store.root),
            "models": [i.as_dict() for i in self.store.list()],
        }

    # ---- fit and predict -----------------------------------------------

    def fit_model(
        self,
        path: str,
        target: str,
        model: str = DEFAULT_MODEL,
        name: str | None = None,
        drop: Sequence[str] = (),
        task: str | None = None,
    ) -> dict[str, Any]:
        """Fit a model on every labelled row of the table and keep it
        under a name for predict and what_if."""
        frame = read_table(path)
        X, y, task = _features_and_labels(frame, target, drop, task)
        name = name or _default_name(path, target, model)
        check_name(name)
        estimator = _estimator(task, model, self.client)
        start = time.perf_counter()
        estimator.fit(X, y)
        seconds = time.perf_counter() - start
        info = ModelInfo(
            name=name,
            model=model,
            task=task,
            target=target,
            features=[str(c) for c in X.columns],
            numeric_features=[
                str(c) for c in X.columns if column_kind(X[c]) == "numeric"
            ],
            train_file=str(path),
            train_rows=len(X),
            classes=_classes(estimator),
            dropped=list(drop),
            created=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        )
        estimator.set_params(client=None)  # the store must not pickle a client
        self.store.save(info, estimator)
        out = info.as_dict()
        out["fit_seconds"] = round(seconds, 2)
        skipped = len(frame) - len(X)
        if skipped:
            out["note"] = f"{skipped} rows without a {target!r} value were left out"
        return out

    def predict(
        self,
        name: str,
        path: str,
        output: str | None = None,
        top: int = 10,
        rank_by: str | None = None,
        where: dict[str, Any] | None = None,
        show: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Predict every row of a table with a fitted model. Writes the
        table with prediction (and probability) columns added, and returns
        the top rows: highest probability of class rank_by (default the
        rarer class), or highest prediction for regression, among the rows
        matching where ({column: value}), with the id column and the show
        columns. If the table has the target column, also scores the
        predictions against it."""
        info, estimator = self._load(name)
        frame = read_table(path)
        require_columns(frame, info.features)
        X = _coerce(frame[info.features], info)
        start = time.perf_counter()
        predicted = _predict(estimator, info, X)
        seconds = time.perf_counter() - start

        out_path = Path(output) if output else _output_path(path, name)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([frame, predicted], axis=1).to_csv(out_path, index=False)

        result: dict[str, Any] = {
            "model": name,
            "rows": len(frame),
            "output": str(out_path),
            "added_columns": list(predicted.columns),
            "predict_seconds": round(seconds, 2),
            "summary": _summary(predicted, info),
            **_top_rows(frame, predicted, info, top, rank_by, where, show),
        }
        if info.target in frame.columns:
            labelled = frame[info.target].notna()
            if labelled.any():
                result["score_against_target"] = _score(
                    info.task,
                    frame.loc[labelled, info.target],
                    predicted[labelled],
                    info,
                )
        return result

    def what_if(
        self,
        name: str,
        rows: list[dict[str, Any]],
        base_file: str | None = None,
        base_where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Predict a few hand-made rows, to see how the prediction moves
        when a column changes. Each row is a {column: value} dict; columns
        left out are missing. With base_file and base_where ({column:
        value} matching exactly one row of that file, e.g. a customer id),
        each row instead holds only the changes to that base row, and {}
        is the base row as it is."""
        if not rows:
            raise ValueError("rows is empty: give at least one {column: value} dict")
        info, estimator = self._load(name)
        unknown = sorted({k for row in rows for k in row} - set(info.features))
        if unknown:
            raise ValueError(
                f"unknown column(s) {unknown}; the model's features are {info.features}"
            )
        base = _base_row(base_file, base_where, info)
        cases = [{**base, **row} for row in rows]
        X = _coerce(pd.DataFrame(cases, columns=info.features), info)
        predicted = _predict(estimator, info, X)
        answers = []
        for row, case, (_, p) in zip(rows, cases, predicted.iterrows(), strict=True):
            answer: dict[str, Any] = {"prediction": _plain(p[_pred_col(info)])}
            if info.task == "classification":
                answer["probabilities"] = {
                    c: round(float(p[_proba_col(info, c)]), 4) for c in info.classes
                }
            if base:
                answer["changes"] = row
            left_out = [f for f in info.features if f not in case]
            if left_out:
                answer["missing_columns"] = left_out
            answers.append(answer)
        out: dict[str, Any] = {"model": name, "results": answers}
        if base:
            out["base_row"] = base
        return out

    # ---- measure -------------------------------------------------------

    def evaluate(
        self,
        path: str,
        target: str,
        models: Sequence[str] = MODELS,
        baseline: bool = True,
        test_size: float = 0.25,
        drop: Sequence[str] = (),
        task: str | None = None,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Hold out part of the table, fit each model on the rest, and
        score it on the held-out rows; with baseline, a scikit-learn
        gradient-boosting model as well, for comparison."""
        frame = read_table(path)
        X, y, task = _features_and_labels(frame, target, drop, task)
        Xtr, Xte, ytr, yte = _split(X, y, task, test_size, seed)
        results = []
        for model in models:
            estimator = _estimator(task, model, self.client)
            results.append(_fit_and_score(model, estimator, task, Xtr, ytr, Xte, yte))
        if baseline:
            Xtr_b, Xte_b = _for_gradient_boosting(Xtr, Xte)
            results.append(
                _fit_and_score(
                    BASELINE,
                    _gradient_boosting(task, seed),
                    task,
                    Xtr_b,
                    ytr,
                    Xte_b,
                    yte,
                )
            )
        key, higher = ("roc_auc", True) if task == "classification" else ("r2", True)
        scored = [r for r in results if r["metrics"].get(key) is not None]
        if task == "classification" and not scored:
            key, scored = "accuracy", results
        best = max(scored, key=lambda r: r["metrics"][key] * (1 if higher else -1))
        return {
            "task": task,
            "train_rows": len(Xtr),
            "test_rows": len(Xte),
            "results": results,
            "best_by": key,
            "best": best["model"],
        }

    def explain(
        self,
        path: str,
        target: str,
        model: str = DEFAULT_MODEL,
        drop: Sequence[str] = (),
        task: str | None = None,
        test_size: float = 0.25,
        repeats: int = 3,
        max_rows: int = 500,
        top: int = 20,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Which columns the model relies on (permutation importance): fit
        on part of the table, then shuffle one column at a time in the
        held-out rows and measure how much the score drops. All shuffled
        copies go to the server in one predict."""
        frame = read_table(path)
        X, y, task = _features_and_labels(frame, target, drop, task)
        Xtr, Xte, ytr, yte = _split(X, y, task, test_size, seed)
        if len(Xte) > max_rows:
            Xte, yte = Xte.iloc[:max_rows], yte.iloc[:max_rows]
        estimator = _estimator(task, model, self.client).fit(Xtr, ytr)
        name, score = _importance_score(task, estimator, yte)

        rng = np.random.default_rng(seed)
        blocks = [Xte]
        for column in Xte.columns:
            for _ in range(repeats):
                block = Xte.copy()
                block[column] = rng.permutation(block[column].to_numpy())
                blocks.append(block)
        start = time.perf_counter()
        outputs = _raw_outputs(estimator, task, pd.concat(blocks, ignore_index=True))
        seconds = time.perf_counter() - start

        n = len(Xte)
        base = score(outputs[:n])
        drops = []
        for i, column in enumerate(Xte.columns):
            scores = [
                score(outputs[n * (1 + i * repeats + r) : n * (2 + i * repeats + r)])
                for r in range(repeats)
            ]
            drops.append((column, base - float(np.mean(scores)), float(np.std(scores))))
        drops.sort(key=lambda d: d[1], reverse=True)
        return {
            "model": model,
            "task": task,
            "metric": name,
            "score_with_all_columns": round(base, 4),
            "held_out_rows": n,
            "rows_predicted": len(outputs),
            "predict_seconds": round(seconds, 2),
            "importance": [
                {"column": c, "score_drop": round(m, 4), "spread": round(s, 4)}
                for c, m, s in drops[:top]
            ],
            "how_to_read": (
                f"score_drop: how much {name} falls when the column is shuffled; "
                "near 0 means the model does not use it, negative is noise."
            ),
        }

    # ---- internals -----------------------------------------------------

    def _load(self, name: str) -> tuple[ModelInfo, Any]:
        info, estimator = self.store.load(name)
        estimator.set_params(client=self.client)
        return info, estimator


# ---- tables to estimators ----------------------------------------------


def _features_and_labels(
    frame: pd.DataFrame, target: str, drop: Sequence[str], task: str | None
) -> tuple[pd.DataFrame, pd.Series, str]:
    require_columns(frame, [target, *drop])
    if target in drop:
        raise ValueError(f"{target!r} is the target; it cannot also be dropped")
    labelled = frame[frame[target].notna()]
    if len(labelled) < 2:
        raise ValueError(f"the target {target!r} has fewer than 2 values")
    X = labelled.drop(columns=[target, *drop]).reset_index(drop=True)
    if X.shape[1] == 0:
        raise ValueError("no feature columns left after dropping the target")
    y = labelled[target].reset_index(drop=True)
    if task is None:
        task = infer_task(y.to_numpy())
    elif task not in ("classification", "regression"):
        raise ValueError(f"task {task!r}: use 'classification' or 'regression'")
    if task == "classification":
        y = y.astype(str)
    return X, y, task


def _estimator(task: str, model: str, client: Any):
    cls = TabctxClassifier if task == "classification" else TabctxRegressor
    return cls(model=model, client=client)


def _split(X, y, task, test_size, seed):
    stratify = None
    if task == "classification" and y.value_counts().min() >= 2:
        stratify = y
    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )


def _base_row(
    path: str | None, where: dict[str, Any] | None, info: ModelInfo
) -> dict[str, Any]:
    """The one row of path matching every {column: value} of where, as a
    {feature: value} dict (missing values left out); {} without a file."""
    if path is None and where is None:
        return {}
    if path is None or not where:
        raise ValueError("give base_file and base_where together")
    frame = read_table(path)
    require_columns(frame, list(where))
    match = pd.Series(True, index=frame.index)
    for column, value in where.items():
        match &= frame[column].astype(str) == str(value)
    if match.sum() != 1:
        raise ValueError(
            f"base_where {where} matches {int(match.sum())} rows of {path}; it must match one"
        )
    row = frame.loc[match].iloc[0]
    return {
        f: _plain(row[f]) for f in info.features if f in row.index and pd.notna(row[f])
    }


def _coerce(X: pd.DataFrame, info: ModelInfo) -> pd.DataFrame:
    """Columns numeric at fit time as numbers, whatever the caller sent
    ("42" from an agent's JSON is 42); text the server could not read as
    a number is refused here, with the column's name."""
    X = X.copy()
    for name in info.numeric_features:
        values = pd.to_numeric(X[name], errors="coerce")
        bad = X[name].notna() & values.isna()
        if bad.any():
            raise ValueError(
                f"column {name!r} is numeric, but got {X[name][bad].iloc[0]!r}"
            )
        X[name] = values
    return X


# ---- outputs -----------------------------------------------------------


def _pred_col(info: ModelInfo) -> str:
    return f"predicted_{info.target}"


def _proba_col(info: ModelInfo, cls: str) -> str:
    return f"p_{info.target}={cls}"


def _classes(estimator) -> list[str] | None:
    classes = getattr(estimator, "classes_", None)
    return None if classes is None else [str(c) for c in classes]


def _predict(estimator, info: ModelInfo, X: pd.DataFrame) -> pd.DataFrame:
    """The prediction columns for X: the prediction, and for
    classification one probability column per class."""
    if info.task == "regression":
        return pd.DataFrame({_pred_col(info): estimator.predict(X)})
    proba = estimator.predict_proba(X)
    out = {_pred_col(info): [info.classes[i] for i in proba.argmax(axis=1)]}
    for i, cls in enumerate(info.classes):
        out[_proba_col(info, cls)] = np.round(proba[:, i], 4)
    return pd.DataFrame(out)


def _summary(predicted: pd.DataFrame, info: ModelInfo) -> dict[str, Any]:
    col = predicted[_pred_col(info)]
    if info.task == "regression":
        return {
            "mean": round(float(col.mean()), 4),
            "min": round(float(col.min()), 4),
            "max": round(float(col.max()), 4),
        }
    return {
        "predicted_counts": {str(k): int(v) for k, v in col.value_counts().items()},
        "mean_probability": {
            c: round(float(predicted[_proba_col(info, c)].mean()), 4)
            for c in info.classes
        },
    }


def _top_rows(
    frame: pd.DataFrame,
    predicted: pd.DataFrame,
    info: ModelInfo,
    top: int,
    rank_by: str | None,
    where: dict[str, Any] | None,
    show: Sequence[str],
) -> dict[str, Any]:
    """The top rows of the predictions, as {"ranked_by", "top_rows"}."""
    if top < 0:
        raise ValueError(f"top must be 0 or more, got {top}")
    require_columns(frame, [*(where or {}), *show])
    if info.task == "classification":
        if rank_by is None:  # the rarer class: churn, fraud, default...
            rank_by = min(
                info.classes, key=lambda c: predicted[_proba_col(info, c)].mean()
            )
        if rank_by not in info.classes:
            raise ValueError(
                f"rank_by {rank_by!r}: use one of the classes {info.classes}"
            )
        key = _proba_col(info, rank_by)
    else:
        key = _pred_col(info)
    match = pd.Series(True, index=frame.index)
    for column, value in (where or {}).items():
        match &= frame[column].astype(str) == str(value)
    ids = [
        c
        for c in frame.columns
        if c not in info.features and c != info.target and _is_id(frame[c])
    ]
    columns = list(dict.fromkeys([*ids[:1], *show]))
    rows = pd.concat([frame[columns], predicted], axis=1)[match]
    rows = rows.sort_values(key, ascending=False, kind="stable").head(top)
    out: dict[str, Any] = {"ranked_by": key, "top_rows": _records(rows)}
    if where:
        out["rows_matching_where"] = int(match.sum())
    return out


def _is_id(col: pd.Series) -> bool:
    return col.notna().all() and col.nunique() == len(col)


def _score(task, y_true, predicted, info: ModelInfo) -> dict[str, Any]:
    if task == "regression":
        return _regression_metrics(y_true, predicted[_pred_col(info)].to_numpy())
    proba = predicted[[_proba_col(info, c) for c in info.classes]].to_numpy()
    return _classification_metrics(
        y_true.astype(str), predicted[_pred_col(info)], proba, info.classes
    )


def _output_path(path: str, name: str) -> Path:
    src = Path(path)
    stem = src.name.split(".")[0]
    return src.with_name(f"{stem}.{name}.predictions.csv")


def _default_name(path: str, target: str, model: str) -> str:
    raw = f"{Path(path).name.split('.')[0]}-{target}-{model}"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-.")[:100]


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{k: _plain(v) for k, v in row.items()} for row in frame.to_dict("records")]


def _plain(v: Any) -> Any:
    return v.item() if isinstance(v, np.generic) else v


# ---- scoring -------------------------------------------------------------


def _fit_and_score(name, estimator, task, Xtr, ytr, Xte, yte) -> dict[str, Any]:
    start = time.perf_counter()
    estimator.fit(Xtr, ytr)
    fit_s = time.perf_counter() - start
    start = time.perf_counter()
    if task == "regression":
        scores = _regression_metrics(yte, estimator.predict(Xte))
    else:
        proba = estimator.predict_proba(Xte)
        classes = [str(c) for c in estimator.classes_]
        pred = [classes[i] for i in proba.argmax(axis=1)]
        scores = _classification_metrics(yte, pred, proba, classes)
    predict_s = time.perf_counter() - start
    return {
        "model": name,
        "metrics": scores,
        "fit_seconds": round(fit_s, 2),
        "predict_seconds": round(predict_s, 2),
    }


def _classification_metrics(y_true, y_pred, proba, classes) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=str)
    out: dict[str, Any] = {
        "accuracy": round(
            float(metrics.accuracy_score(y_true, np.asarray(y_pred, dtype=str))), 4
        ),
        "roc_auc": None,
        "log_loss": None,
    }
    proba = np.clip(np.asarray(proba, dtype=float), 1e-15, 1)
    proba = proba / proba.sum(axis=1, keepdims=True)
    try:
        if len(classes) == 2:
            auc = metrics.roc_auc_score(y_true == classes[1], proba[:, 1])
        else:
            auc = metrics.roc_auc_score(
                y_true, proba, multi_class="ovr", labels=classes
            )
        out["roc_auc"] = round(float(auc), 4)
    except ValueError:
        pass  # a class missing from y_true
    try:
        out["log_loss"] = round(
            float(metrics.log_loss(y_true, proba, labels=classes)), 4
        )
    except ValueError:
        pass
    return out


def _regression_metrics(y_true, y_pred) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "r2": round(float(metrics.r2_score(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(metrics.mean_squared_error(y_true, y_pred))), 4),
        "mae": round(float(metrics.mean_absolute_error(y_true, y_pred)), 4),
    }


def _importance_score(task, estimator, y_true):
    """(metric name, score(outputs) for a block of held-out rows)."""
    y = np.asarray(y_true, dtype=str if task == "classification" else float)
    if task == "regression":
        return "r2", lambda pred: float(metrics.r2_score(y, pred))
    classes = [str(c) for c in estimator.classes_]
    if len(classes) == 2 and len(set(y)) == 2:
        return "roc_auc", lambda proba: float(
            metrics.roc_auc_score(y == classes[1], proba[:, 1])
        )
    return "accuracy", lambda proba: float(
        metrics.accuracy_score(y, np.asarray(classes)[proba.argmax(axis=1)])
    )


def _raw_outputs(estimator, task, X):
    return estimator.predict(X) if task == "regression" else estimator.predict_proba(X)


# ---- the gradient-boosting baseline ------------------------------------


def _gradient_boosting(task: str, seed: int):
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )

    cls = (
        HistGradientBoostingClassifier
        if task == "classification"
        else HistGradientBoostingRegressor
    )
    return cls(categorical_features="from_dtype", random_state=seed)


def _for_gradient_boosting(Xtr: pd.DataFrame, Xte: pd.DataFrame):
    """Text and boolean columns as pandas categoricals (categories from the
    training rows; unseen values become missing), dates as numbers; text
    with too many values for a categorical becomes its category code."""
    Xtr, Xte = Xtr.copy(), Xte.copy()
    for name in Xtr.columns:
        kind = column_kind(Xtr[name])
        if kind == "numeric":
            continue
        if kind == "datetime":
            Xtr[name] = _epoch(Xtr[name])
            Xte[name] = _epoch(Xte[name])
            continue
        train, test = _as_text(Xtr[name]), _as_text(Xte[name])
        categories = sorted(train.dropna().unique())
        if len(categories) > _MAX_CATEGORIES:
            codes = {c: float(i) for i, c in enumerate(categories)}
            Xtr[name], Xte[name] = train.map(codes), test.map(codes)
        else:
            Xtr[name] = pd.Categorical(train, categories=categories)
            Xte[name] = pd.Categorical(
                test.where(test.isin(categories)), categories=categories
            )
    return Xtr, Xte


def _as_text(col: pd.Series) -> pd.Series:
    """Values as plain str, missing as None (scikit-learn's encoders
    refuse pandas' NA)."""
    return col.astype(object).map(lambda v: None if pd.isna(v) else str(v))


def _epoch(col: pd.Series) -> pd.Series:
    return pd.to_datetime(col).astype("int64").where(col.notna()).astype(float)
