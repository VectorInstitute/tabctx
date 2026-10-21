#!/usr/bin/env python3
"""Accuracy on real datasets, through the gateway: does serving give the
predictions the models are known for?

The sweeps time synthetic random tables; they can't tell a model that
predicts well from one returning noise. This fits each served model on
well-known public datasets (the Kaggle classics, from OpenML and
scikit-learn, so no Kaggle account is needed), predicts a held-out split,
and scores it next to a gradient-boosting baseline trained on the same
features and a trivial one (majority class / mean). A served model should
beat trivial by a wide margin and match or beat boosting on these sizes.

Categorical columns become integer codes (missing -> -1). Missing numbers
are filled with the training median (--missing impute, as servers before
tabctx 0.14 require) or sent as null for the model to handle itself
(--missing null). The share of missing cells is reported per dataset.

Usage:
  VP_API_KEY=... eval_real_datasets.py --models tabicl-v2 tabpfn-3.5 \\
      --save benchmarks/baselines/eval-real-gateway.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn import datasets as skd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import accuracy_score, log_loss, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_gateway import Gateway  # noqa: E402

Task = Literal["classification", "regression"]


@dataclass(frozen=True)
class Dataset:
    name: str
    task: Task
    source: str
    load: Callable[[], tuple[pd.DataFrame, pd.Series]] = field(repr=False)


def _openml(data_id: int, target: str, drop: tuple[str, ...] = ()):
    def load() -> tuple[pd.DataFrame, pd.Series]:
        frame = skd.fetch_openml(data_id=data_id, as_frame=True, parser="auto").frame
        return frame.drop(columns=[target, *drop]), frame[target]

    return load


def _sklearn(loader: Callable):
    def load() -> tuple[pd.DataFrame, pd.Series]:
        bunch = loader(as_frame=True)
        return bunch.data, bunch.target

    return load


DATASETS = [
    # Titanic: boat and body are recorded after the outcome (they leak it).
    Dataset(
        "titanic",
        "classification",
        "OpenML 40945 (Kaggle Titanic)",
        _openml(
            40945, "survived", ("name", "ticket", "cabin", "boat", "body", "home.dest")
        ),
    ),
    Dataset(
        "adult-income",
        "classification",
        "OpenML 1590 (UCI/Kaggle Adult)",
        _openml(1590, "class"),
    ),
    Dataset(
        "bank-marketing",
        "classification",
        "OpenML 1461 (UCI/Kaggle Bank Marketing)",
        _openml(1461, "Class"),
    ),
    Dataset(
        "credit-g", "classification", "OpenML 31 (German credit)", _openml(31, "class")
    ),
    Dataset(
        "breast-cancer",
        "classification",
        "scikit-learn (Wisconsin)",
        _sklearn(skd.load_breast_cancer),
    ),
    Dataset(
        "digits",
        "classification",
        "scikit-learn (10 classes, 64 features)",
        _sklearn(skd.load_digits),
    ),
    Dataset(
        "california-housing",
        "regression",
        "scikit-learn (StatLib)",
        _sklearn(skd.fetch_california_housing),
    ),
    Dataset(
        "house-prices",
        "regression",
        "OpenML 42165 (Kaggle House Prices, Ames)",
        _openml(42165, "SalePrice", ("Id",)),
    ),
]


@dataclass
class Split:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    missing_share: float


def prepare(
    X: pd.DataFrame,
    y: pd.Series,
    task: Task,
    max_train: int,
    max_test: int,
    seed: int,
    impute: bool = True,
) -> Split:
    """Split, then encode (and fill, if impute) using training data only."""
    y = y.astype(str) if task == "classification" else y.astype(float)
    stratify = y if task == "classification" else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=stratify
    )
    rng = np.random.default_rng(seed)
    if len(X_tr) > max_train:
        keep = rng.choice(len(X_tr), max_train, replace=False)
        X_tr, y_tr = X_tr.iloc[keep], y_tr.iloc[keep]
    if len(X_te) > max_test:
        keep = rng.choice(len(X_te), max_test, replace=False)
        X_te, y_te = X_te.iloc[keep], y_te.iloc[keep]

    def encode(frame: pd.DataFrame, fit_on: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)
        for col in frame.columns:
            if frame[col].dtype.kind in "biuf":
                out[col] = frame[col].astype(float)
            else:  # categorical / text / bool: codes learned on training data
                cats = pd.Index(fit_on[col].dropna().astype(str).unique()).sort_values()
                codes = pd.Categorical(
                    frame[col].astype("string"), categories=cats
                ).codes
                out[col] = np.where(codes < 0, -1, codes).astype(float)
        return out

    enc_tr, enc_te = encode(X_tr, X_tr), encode(X_te, X_tr)
    cells = enc_tr.size + enc_te.size
    missing = int(enc_tr.isna().sum().sum() + enc_te.isna().sum().sum())
    if impute:
        medians = enc_tr.median().fillna(0.0)
        enc_tr, enc_te = enc_tr.fillna(medians), enc_te.fillna(medians)
    return Split(
        enc_tr.to_numpy(np.float64),
        y_tr.to_numpy(),
        enc_te.to_numpy(np.float64),
        y_te.to_numpy(),
        missing / cells if cells else 0.0,
    )


def _json_rows(arr: np.ndarray) -> list[list[float | None]]:
    """JSON has no NaN: a missing cell goes as null."""
    return [[None if np.isnan(v) else float(v) for v in row] for row in arr]


def score(
    task: Task,
    y_true: np.ndarray,
    pred: np.ndarray,
    proba: np.ndarray | None = None,
    classes: list | None = None,
) -> dict:
    if task == "regression":
        pred = pred.astype(float)
        return {
            "r2": round(float(r2_score(y_true, pred)), 4),
            "rmse": round(float(np.sqrt(np.mean((y_true - pred) ** 2))), 4),
        }
    out = {"accuracy": round(float(accuracy_score(y_true, pred.astype(str))), 4)}
    if proba is not None and classes is not None:
        labels = [str(c) for c in classes]
        p = np.clip(np.asarray(proba, dtype=float), 1e-12, 1)
        p = p / p.sum(axis=1, keepdims=True)
        if len(labels) == 2:
            out["roc_auc"] = round(
                float(roc_auc_score(y_true == labels[1], p[:, 1])), 4
            )
        else:
            out["roc_auc"] = round(
                float(roc_auc_score(y_true, p, multi_class="ovr", labels=labels)), 4
            )
        out["log_loss"] = round(float(log_loss(y_true, p, labels=labels)), 4)
    return out


def trivial(task: Task, split: Split) -> dict:
    if task == "regression":
        return score(
            task, split.y_test, np.full(len(split.y_test), split.y_train.mean())
        )
    labels, counts = np.unique(split.y_train, return_counts=True)
    prior = counts / counts.sum()
    proba = np.tile(prior, (len(split.y_test), 1))
    return score(
        task,
        split.y_test,
        np.full(len(split.y_test), labels[counts.argmax()]),
        proba,
        list(labels),
    )


def boosting(task: Task, split: Split, seed: int) -> dict:
    t0 = time.monotonic()
    if task == "regression":
        model = HistGradientBoostingRegressor(random_state=seed).fit(
            split.X_train, split.y_train
        )
        result = score(task, split.y_test, model.predict(split.X_test))
    else:
        model = HistGradientBoostingClassifier(random_state=seed).fit(
            split.X_train, split.y_train
        )
        result = score(
            task,
            split.y_test,
            model.predict(split.X_test),
            model.predict_proba(split.X_test),
            list(model.classes_),
        )
    return result | {"fit_predict_s": round(time.monotonic() - t0, 2)}


def served(gateway: Gateway, ds: Dataset, split: Split) -> dict:
    dataset_id = f"eval-real-{ds.name}"
    y_train = (
        [str(v) for v in split.y_train]
        if ds.task == "classification"
        else [float(v) for v in split.y_train]
    )
    t0 = time.monotonic()
    status, out = gateway.fit(
        ("json", (_json_rows(split.X_train), y_train)), dataset_id, ds.task
    )
    fit_s = time.monotonic() - t0
    if status != 200:
        return {"error": f"fit {status}: {str(out.get('detail'))[:200]}"}
    t0 = time.monotonic()
    status, out = gateway.predict(
        dataset_id, _json_rows(split.X_test), return_proba=ds.task == "classification"
    )
    predict_s = time.monotonic() - t0
    if status != 200:
        return {"error": f"predict {status}: {str(out.get('detail'))[:200]}"}
    result = score(
        ds.task,
        split.y_test,
        np.asarray(out["predictions"]),
        out.get("probabilities"),
        out.get("classes"),
    )
    return result | {"fit_s": round(fit_s, 2), "predict_s": round(predict_s, 2)}


def main() -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--base-url", default="https://proxy.vectorinstitute.ai/v1")
    p.add_argument("--models", nargs="+", default=["tabicl-v2", "tabpfn-3.5"])
    p.add_argument("--datasets", nargs="+", default=[d.name for d in DATASETS])
    p.add_argument("--max-train", type=int, default=10_000)
    p.add_argument("--max-test", type=int, default=2_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--missing",
        choices=["impute", "null"],
        default="impute",
        help="fill missing numbers with the training median, or send them as null",
    )
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--save", default=None)
    args = p.parse_args()

    api_key = os.environ["VP_API_KEY"]
    results = []
    for ds in (d for d in DATASETS if d.name in args.datasets):
        X, y = ds.load()
        split = prepare(
            X,
            y,
            ds.task,
            args.max_train,
            args.max_test,
            args.seed,
            impute=args.missing == "impute",
        )
        row = {
            "dataset": ds.name,
            "source": ds.source,
            "task": ds.task,
            "n_train": len(split.y_train),
            "n_test": len(split.y_test),
            "n_features": split.X_train.shape[1],
            "n_classes": len(np.unique(split.y_train))
            if ds.task == "classification"
            else None,
            "missing_share": round(split.missing_share, 4),
            "trivial": trivial(ds.task, split),
            "boosting": boosting(ds.task, split, args.seed),
        }
        for model in args.models:
            row[model] = served(
                Gateway(args.base_url, api_key, model, args.timeout), ds, split
            )
        results.append(row)
        print(json.dumps(row), flush=True)

    if args.save:
        Path(args.save).write_text(
            json.dumps(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "base_url": args.base_url,
                    "seed": args.seed,
                    "missing": args.missing,
                    "max_train": args.max_train,
                    "max_test": args.max_test,
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
