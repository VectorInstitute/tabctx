"""Score an agent's churn predictions against what the current customers
actually did (answers/current_churn.csv, which the agent never sees).

    python examples/agent_demo/score.py runs/claude
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
PROBA = "p_Churn=Yes"


def score(run_dir: Path) -> dict:
    # The newest CSV with churn probabilities: the tools' default output
    # (*.predictions.csv) or wherever the agent asked predict to write.
    files = sorted(run_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    files = [f for f in files if PROBA in pd.read_csv(f, nrows=0).columns]
    if not files:
        raise SystemExit(f"no predictions CSV with a {PROBA!r} column in {run_dir}")
    predicted = pd.read_csv(files[-1])
    truth = pd.read_csv(HERE / "answers" / "current_churn.csv")
    merged = truth.merge(predicted[["customer_id", PROBA]], on="customer_id")
    if len(merged) != len(truth):
        raise SystemExit(f"{files[-1]} covers {len(merged)} of {len(truth)} customers")
    churned = merged["Churn"] == "Yes"
    ranked = merged.sort_values(PROBA, ascending=False)
    out = {
        "file": files[-1].name,
        "customers": len(merged),
        "roc_auc": round(roc_auc_score(churned, merged[PROBA]), 4),
        "base_rate": round(float(churned.mean()), 3),
    }
    for k in (10, 100):
        out[f"churned_in_top_{k}"] = f"{(ranked.head(k)['Churn'] == 'Yes').mean():.0%}"
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:] or ["workspace"]:
        print(arg, score(Path(arg)))
