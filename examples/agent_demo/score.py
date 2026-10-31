"""Score an agent's predictions against what actually happened to the
current rows (answers/, which the agent never sees). task.py says which
files and columns.

    python examples/agent_demo/score.py runs/claude
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import task
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent


def score(run_dir: Path) -> dict:
    # The newest CSV with the probability column: the tools' default output
    # (*.predictions.csv) or wherever the agent asked predict to write.
    files = sorted(run_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    files = [f for f in files if task.PROBA in pd.read_csv(f, nrows=0).columns]
    if not files:
        raise SystemExit(
            f"no predictions CSV with a {task.PROBA!r} column in {run_dir}"
        )
    predicted = pd.read_csv(files[-1])
    truth = pd.read_csv(HERE / "answers" / task.ANSWERS)
    merged = truth.merge(predicted[[task.ID, task.PROBA]], on=task.ID)
    if len(merged) != len(truth):
        raise SystemExit(f"{files[-1]} covers {len(merged)} of {len(truth)} rows")
    positive = merged[task.TARGET] == task.POSITIVE
    ranked = merged.sort_values(task.PROBA, ascending=False, kind="stable")
    out = {
        "file": files[-1].name,
        "rows": len(merged),
        "roc_auc": round(roc_auc_score(positive, merged[task.PROBA]), 4),
        "base_rate": round(float(positive.mean()), 3),
    }
    for k in (10, 100):
        out[f"positive_in_top_{k}"] = round(
            float((ranked.head(k)[task.TARGET] == task.POSITIVE).mean()), 3
        )
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:] or ["workspace"]:
        print(arg, score(Path(arg)))
