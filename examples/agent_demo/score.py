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
    """Score a run's predictions; the scenario is the run folder's
    (runs/<scenario>/<agent>) unless given."""
    scenario = scenario or scenarios.of_run(run_dir)
    task = scenario.task
    truth = pd.read_csv(scenario.answers / task.ANSWERS)
    predicted, name = _predictions(run_dir, task, set(truth[task.ID]))
    merged = truth.merge(predicted[[task.ID, task.PROBA]], on=task.ID)
    positive = merged[task.TARGET] == task.POSITIVE
    ranked = merged.sort_values(task.PROBA, ascending=False, kind="stable")
    out = {
        "file": name,
        "rows": len(merged),
        "roc_auc": round(roc_auc_score(positive, merged[task.PROBA]), 4),
        "base_rate": round(float(positive.mean()), 3),
    }
    for k in (10, 100):
        out[f"positive_in_top_{k}"] = round(
            float((ranked.head(k)[task.TARGET] == task.POSITIVE).mean()), 3
        )
    return out


def _predictions(run_dir: Path, task, ids: set) -> tuple[pd.DataFrame, str]:
    """The newest CSV with the probability column that covers every row:
    the tools' default output (*.predictions.csv) or wherever the agent
    asked predict to write. Newer partial ones (a what-if table, a subset)
    are passed over. One row per id: a raw table can list a row twice, and
    its copies match."""
    files = sorted(run_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    files = [f for f in files if task.PROBA in pd.read_csv(f, nrows=0).columns]
    if not files:
        raise SystemExit(
            f"no predictions CSV with a {task.PROBA!r} column in {run_dir}"
        )
    best = None
    for f in reversed(files):
        table = pd.read_csv(f)
        if task.ID not in table.columns:
            raise SystemExit(
                f"{f.name} has no {task.ID!r} column to match the answers on"
            )
        table = table.drop_duplicates(task.ID)
        covered = len(ids & set(table[task.ID]))
        if covered == len(ids):
            return table, f.name
        best = best or (f.name, covered)
    raise SystemExit(f"{best[0]} covers {best[1]} of {len(ids)} rows")


if __name__ == "__main__":
    for arg in sys.argv[1:] or [
        str(p) for p in sorted(scenarios.load().runs.glob("*"))
    ]:
        print(arg, score(Path(arg)))
