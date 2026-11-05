"""Score an agent's predictions against what actually happened to the
current rows (answers/<scenario>/, which the agent never sees). The
scenario's task.py says which files and columns (DEMO_SCENARIO picks it;
see scenario.py).

    python examples/agent_demo/score.py runs/readmission/claude
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import scenario as scenarios
from sklearn.metrics import roc_auc_score


def score(run_dir: Path, scenario: scenarios.Scenario | None = None) -> dict:
    scenario = scenario or scenarios.load()
    task = scenario.task
    # The newest CSV with the probability column: the tools' default output
    # (*.predictions.csv) or wherever the agent asked predict to write.
    files = sorted(run_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    files = [f for f in files if task.PROBA in pd.read_csv(f, nrows=0).columns]
    if not files:
        raise SystemExit(
            f"no predictions CSV with a {task.PROBA!r} column in {run_dir}"
        )
    # One row per id: a raw table can list a row twice; its copies match.
    predicted = pd.read_csv(files[-1]).drop_duplicates(task.ID)
    truth = pd.read_csv(scenario.answers / task.ANSWERS)
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
