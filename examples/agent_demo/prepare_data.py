"""Build the demo's workspace from the public Diabetes 130-US Hospitals
table (UCI; OpenML #4541): 101,766 hospital stays of diabetic patients,
1999-2008, and whether the patient came back within 30 days.

    uv run --extra mcp python examples/agent_demo/prepare_data.py

Prepared as in most work on this table (after Strack et al., 2014): no
stays that ended in death or hospice, diagnosis codes grouped into ICD-9
chapters, admission and discharge codes named, and columns that are almost all empty or constant left out.
Current patients (one stay each) are kept out of the history, so
the model never sees the same person on both sides. Writes, under
examples/agent_demo/:

- workspace/patients_history.csv  past stays, with readmitted_30d
- workspace/patients_current.csv  one stay for each current patient; the
                                  outcome withheld
- answers/current_outcomes.csv    what happened to them (for score.py;
                                  outside the agent's workspace)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import task
from sklearn.datasets import fetch_openml

HERE = Path(__file__).resolve().parent
OPENML_ID = 4541
HISTORY_ROWS = 30_000
CURRENT_ROWS = 2_000
SEED = 7

# Discharged to hospice or expired: they cannot be readmitted.
NOT_AT_RISK = {11, 13, 14, 19, 20, 21}

# The table's three coded columns, named (from its IDs_mapping.csv, cut
# short). Codes not listed, and "NULL" / "Not Mapped" ones, are Unknown.
ADMISSION_TYPE = {
    1: "Emergency",
    2: "Urgent",
    3: "Elective",
    4: "Newborn",
    7: "Trauma center",
}
DISCHARGE = {
    1: "Home",
    2: "Short-term hospital",
    3: "Skilled nursing facility",
    4: "Intermediate care facility",
    5: "Other inpatient care",
    6: "Home health service",
    7: "Left against advice",
    8: "Home IV provider",
    9: "Admitted as inpatient",
    10: "Neonatal aftercare",
    12: "Outpatient return",
    15: "Swing bed",
    16: "Outpatient elsewhere",
    17: "Outpatient here",
    22: "Rehab facility",
    23: "Long-term care hospital",
    24: "Medicaid nursing facility",
    27: "Federal facility",
    28: "Psychiatric hospital",
    29: "Critical access hospital",
    30: "Other facility",
}
ADMISSION_SOURCE = {
    1: "Physician referral",
    2: "Clinic referral",
    3: "HMO referral",
    4: "Hospital transfer",
    5: "Nursing facility transfer",
    6: "Other facility transfer",
    7: "Emergency room",
    8: "Court or law enforcement",
    10: "Critical access hospital transfer",
    11: "Normal delivery",
    13: "Sick baby",
    14: "Extramural birth",
    18: "Home health transfer",
    22: "Same-facility transfer",
    25: "Ambulatory surgery transfer",
    26: "Hospice transfer",
}
CODED = {
    "admission_type_id": ("admission_type", ADMISSION_TYPE),
    "discharge_disposition_id": ("discharge_disposition", DISCHARGE),
    "admission_source_id": ("admission_source", ADMISSION_SOURCE),
}
# Almost entirely missing (weight) or one value (most rare drugs).
DROP = ["encounter_id", "readmitted", "weight"]
MIN_MINORITY = 0.005


def icd9_group(code: object) -> str:
    """An ICD-9 code's chapter, as grouped by Strack et al."""
    text = str(code)
    if text in ("", "?", "nan", "None"):
        return "Missing"
    if text[0] in "VE":
        return "Other"
    value = float(text)
    if 390 <= value <= 459 or value == 785:
        return "Circulatory"
    if 460 <= value <= 519 or value == 786:
        return "Respiratory"
    if 520 <= value <= 579 or value == 787:
        return "Digestive"
    if int(value) == 250:
        return "Diabetes"
    if 800 <= value <= 999:
        return "Injury"
    if 710 <= value <= 739:
        return "Musculoskeletal"
    if 580 <= value <= 629 or value == 788:
        return "Genitourinary"
    if 140 <= value <= 239:
        return "Neoplasms"
    return "Other"


def load() -> pd.DataFrame:
    frame = fetch_openml(data_id=OPENML_ID, as_frame=True).frame
    for name in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[name]):
            # OpenML's ARFF keeps quotes; "?" marks a missing value.
            frame[name] = (
                frame[name]
                .astype(str)
                .str.strip("'")
                .replace({"?": pd.NA, "nan": pd.NA})
            )
    frame = frame.sort_values("encounter_id")
    frame = frame[~frame["discharge_disposition_id"].isin(NOT_AT_RISK)]
    for code, (_, names) in CODED.items():
        frame[code] = frame[code].map(names).fillna("Unknown")
    frame = frame.rename(columns={code: name for code, (name, _) in CODED.items()})
    for name in ("diag_1", "diag_2", "diag_3"):
        frame[name] = frame[name].map(icd9_group)
    # "None" means the test was not done; CSV readers take it for missing.
    for name in ("A1Cresult", "max_glu_serum"):
        frame[name] = frame[name].replace({"None": "Not measured"})
    frame[task.TARGET] = (frame["readmitted"] == "<30").map({True: "Yes", False: "No"})
    frame = frame.drop(columns=DROP)
    keep = [
        name
        for name in frame.columns
        if 1 - frame[name].value_counts(normalize=True, dropna=False).iloc[0]
        >= MIN_MINORITY
    ]
    frame = frame[keep].reset_index(drop=True)
    frame.insert(0, task.ID, [f"P{i:06d}" for i in range(len(frame))])
    return frame


def split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One stay each of CURRENT_ROWS patients, and HISTORY_ROWS stays of
    everyone else.

    The history is a sample of stays, so a patient with many stays (the
    ones most often readmitted) shows up more. Current patients are
    picked the same way, with a chance in proportion to their number of
    stays, so both sides have the same mix; picking patients uniformly
    gives a current set that is readmitted less often than the history.
    """
    stays = frame["patient_nbr"].value_counts()
    chosen = stays.sample(n=CURRENT_ROWS, weights=stays, random_state=SEED).index
    aside = frame["patient_nbr"].isin(chosen)
    current = (
        frame[aside]
        .groupby("patient_nbr")
        .sample(n=1, random_state=SEED)
        .sample(frac=1, random_state=SEED)
    )
    history = frame[~aside].sample(n=HISTORY_ROWS, random_state=SEED)
    return history.drop(columns="patient_nbr"), current.drop(columns="patient_nbr")


def main() -> None:
    frame = load()
    history, current = split(frame)
    workspace, answers = HERE / "workspace", HERE / "answers"
    workspace.mkdir(exist_ok=True)
    answers.mkdir(exist_ok=True)
    history.to_csv(workspace / task.HISTORY, index=False)
    current.drop(columns=[task.TARGET]).to_csv(workspace / task.CURRENT, index=False)
    current[[task.ID, task.TARGET]].to_csv(answers / task.ANSWERS, index=False)
    rate = (history[task.TARGET] == task.POSITIVE).mean()
    print(
        f"{len(history)} past stays ({rate:.1%} readmitted), "
        f"{len(current)} current, {len(current.columns) - 1} features -> {workspace}"
    )


if __name__ == "__main__":
    main()
