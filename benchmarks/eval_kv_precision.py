#!/usr/bin/env python3
"""Accuracy, memory and latency of TabICL KV-cache precisions (#34).

For each dataset, fits TabICLBackend once per precision (native / fp16 /
int8) on the same split and compares against native: prediction
agreement, max and mean probability drift, the task metric (accuracy and
ROC AUC, or R^2), resident context bytes, and warm predict latency (what
decompression costs).

Datasets: the ones bundled with scikit-learn (no download) plus larger
synthetic tables, where KV size actually matters.

Usage: eval_kv_precision.py --device cuda:0 --out kv-precision-a100.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_wine,
    make_classification,
    make_regression,
)
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tabctx.backends.tabicl import TabICLBackend

PRECISIONS = ("native", "fp16", "int8")


def datasets():
    yield "breast_cancer", "classification", *load_breast_cancer(return_X_y=True)
    yield "digits(10cls)", "classification", *load_digits(return_X_y=True)
    yield "wine(3cls)", "classification", *load_wine(return_X_y=True)
    yield "diabetes", "regression", *load_diabetes(return_X_y=True)
    X, y = make_classification(
        20_000,
        40,
        n_informative=20,
        n_classes=4,
        n_clusters_per_class=2,
        random_state=0,
    )
    yield "synth 20k x 40 (4cls)", "classification", X, y
    X, y = make_classification(50_000, 10, n_informative=6, random_state=1)
    yield "synth 50k x 10", "classification", X, y
    X, y = make_regression(20_000, 30, n_informative=15, noise=5.0, random_state=2)
    yield "synth 20k x 30 (reg)", "regression", X, y


def metric(task: str, y_true, outcome) -> dict:
    if task == "regression":
        return {"r2": round(float(r2_score(y_true, outcome.predictions)), 5)}
    pred = np.array(outcome.predictions)
    out = {
        "accuracy": round(float(accuracy_score(np.array(y_true).astype(str), pred)), 5)
    }
    proba = np.array(outcome.probabilities)
    classes = outcome.classes
    y_idx = np.array([classes.index(str(v)) for v in y_true])
    if proba.shape[1] == 2:
        out["auc"] = round(float(roc_auc_score(y_idx, proba[:, 1])), 5)
    else:
        out["auc"] = round(float(roc_auc_score(y_idx, proba, multi_class="ovr")), 5)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    backends = {
        prec: TabICLBackend(device=args.device, kv_precision=prec)
        for prec in PRECISIONS
    }
    for b in backends.values():
        b.warmup()
    results = []
    for name, task, X, y in datasets():
        X = np.asarray(X, dtype=np.float32)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
        row = {
            "dataset": name,
            "task": task,
            "train_rows": len(X_tr),
            "features": X.shape[1],
        }
        base = None
        for prec, b in backends.items():
            payload = b.fit(X_tr, y_tr, task)
            resident = b.context_bytes_hint(len(X_tr), X.shape[1]) or 0
            out = b.predict(payload, X_te, return_proba=task == "classification")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                b.predict(payload, X_te[:100], False)
            torch.cuda.synchronize()
            entry = {
                "resident_mib": round(resident / 2**20, 1),
                "predict100_ms": round((time.perf_counter() - t0) / 5 * 1000, 1),
                **metric(task, y_te, out),
            }
            if base is None:
                base = out
            elif task == "classification":
                pa, pb = np.array(base.probabilities), np.array(out.probabilities)
                entry["label_agreement"] = float(
                    np.mean(np.array(base.predictions) == np.array(out.predictions))
                )
                entry["max_prob_diff"] = float(np.abs(pa - pb).max())
                entry["mean_prob_diff"] = float(np.abs(pa - pb).mean())
            else:
                a, c = np.array(base.predictions), np.array(out.predictions)
                entry["max_rel_diff"] = float(
                    np.abs(a - c).max() / (np.abs(a).max() + 1e-9)
                )
            row[prec] = entry
            del payload
            torch.cuda.empty_cache()
        results.append(row)
        print(json.dumps(row), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
