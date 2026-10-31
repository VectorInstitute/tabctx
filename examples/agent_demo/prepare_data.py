"""Build the demo's workspace from the public Telco customer churn table
(IBM, via OpenML #42178: 7,043 customers, 19 features).

    uv run --extra mcp python examples/agent_demo/prepare_data.py

Writes, under examples/agent_demo/:

- workspace/customers_history.csv  past customers, with their Churn outcome
- workspace/customers_current.csv  current customers; Churn is withheld
- answers/current_churn.csv        what the current customers did (for
                                   score.py; outside the agent's workspace)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
OPENML_ID = 42178
CURRENT_ROWS = 1500
SEED = 7


def load() -> pd.DataFrame:
    frame = fetch_openml(data_id=OPENML_ID, as_frame=True).frame
    for name in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[name]):
            # OpenML's ARFF keeps quotes around values with spaces.
            frame[name] = frame[name].astype(str).str.strip().str.strip("'")
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    frame.insert(0, "customer_id", [f"C{i:05d}" for i in range(len(frame))])
    return frame


def main() -> None:
    frame = load()
    history, current = train_test_split(
        frame, test_size=CURRENT_ROWS, random_state=SEED, stratify=frame["Churn"]
    )
    workspace, answers = HERE / "workspace", HERE / "answers"
    workspace.mkdir(exist_ok=True)
    answers.mkdir(exist_ok=True)
    history.to_csv(workspace / "customers_history.csv", index=False)
    current.drop(columns=["Churn"]).to_csv(
        workspace / "customers_current.csv", index=False
    )
    current[["customer_id", "Churn"]].to_csv(answers / "current_churn.csv", index=False)
    print(
        f"{len(history)} past customers ({(history.Churn == 'Yes').mean():.0%} churned), "
        f"{len(current)} current -> {workspace}"
    )


if __name__ == "__main__":
    main()
