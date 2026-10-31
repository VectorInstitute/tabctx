"""What the demo is about, in one place: the files, the id and outcome
columns, and how the page names them. prepare_data.py, score.py, run.sh
(through `python task.py files`) and the web page all read it, so another
dataset means changing this file, prepare_data.py and PROMPT.md.
"""

from __future__ import annotations

import sys

HISTORY = "patients_history.csv"  # past patients, with the outcome
CURRENT = "patients_current.csv"  # current patients, outcome withheld
ANSWERS = "current_outcomes.csv"  # under answers/: the withheld outcome
ID = "patient_id"
TARGET = "readmitted_30d"
POSITIVE = "Yes"
NEGATIVE = "No"
PROBA = f"p_{TARGET}={POSITIVE}"  # the column predict writes

# For the page: what a row is, what the positive outcome is called, and
# PROMPT.md's steps as icons (a tool's icon, or "report").
LABELS = {
    "entity": "patients",
    "outcome": "readmitted",
    "window": "within 30 days",
    "history": "past discharges",
    "current": "today's discharges",
    "steps": [
        ["evaluate", "Compare", "Benchmark both models against gradient boosting"],
        ["predict", "Rank", "Predict every patient's readmission risk"],
        ["explain", "Drivers", "Find what drives readmission"],
        ["what_if", "What-if", "Test care changes on the highest-risk patients"],
        ["report", "Report", "Write report.md"],
    ],
}

if __name__ == "__main__" and sys.argv[1:] == ["files"]:
    print(HISTORY, CURRENT)
