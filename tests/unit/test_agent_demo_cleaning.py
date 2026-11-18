"""The agent demo page's "messy -> tidy" view (examples/agent_demo/ui/
cleaning.py): a raw export set beside the clean table an agent wrote,
and the words the page uses for each scenario. No agent, no model."""

import json
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

DEMO = Path(__file__).resolve().parents[2] / "examples" / "agent_demo"
sys.path[:0] = [str(DEMO), str(DEMO / "ui")]
import cleaning  # noqa: E402
import scenario as scenarios  # noqa: E402

RAW = """Policy ID,Prov,Premium,Age,Status
A-1,ON,"$1,200.50",40,Renewed
A-2,ont,1300,999,RENEWED
A-2,ont,1300,999,RENEWED
A-3,Ontario,CAD 900,n/a,cxl
A-4, on,800,52,Cancelled
A-5,QC,700,33,Renewed
TEST-1,ON,1,1,Renewed
TOTAL,,5001,,
"""
CLEAN = """policy_id,province,premium,age,cancelled
A-1,ON,1200.5,40,No
A-2,ON,1300,,No
A-3,ON,900,,Yes
A-4,ON,800,52,Yes
A-5,QC,700,33,No
"""
COLUMNS = {
    "Prov": "province",
    "Premium": "premium",
    "Age": "age",
    "Status": "cancelled",
    "Missing": "not_there",
}


@pytest.fixture
def run(tmp_path):
    (tmp_path / "raw.csv").write_text(RAW)
    (tmp_path / "clean.csv").write_text(CLEAN)
    return tmp_path


def view(run):
    return cleaning.summarize(
        run / "raw.csv", run / "clean.csv", "Policy ID", "policy_id", COLUMNS
    )


def test_rows_removed_are_split_into_duplicates_and_non_records(run):
    v = view(run)
    assert (v["rows_before"], v["rows_after"]) == (8, 5)
    assert v["removed"] == [
        {"n": 1, "what": "duplicate rows"},
        {"n": 2, "what": "rows that aren't records (tests, totals)"},
    ]


def test_categories_report_spellings_and_the_most_spelled_value(run):
    by = {c["clean"]: c for c in view(run)["columns"]}
    assert "not_there" not in by  # a mapped column the files lack is skipped
    prov = by["province"]
    assert (prov["kind"], prov["before"], prov["after"]) == ("category", 5, 2)
    assert prov["example"]["value"] == "ON"
    assert set(prov["example"]["spellings"]) == {"ON", "ont", "Ontario", " on"}
    status = by["cancelled"]
    assert (status["before"], status["after"]) == (4, 2)


def test_numbers_report_text_converted_and_impossible_values_emptied(run):
    by = {c["clean"]: c for c in view(run)["columns"]}
    assert by["premium"] == {
        "raw": "Premium",
        "clean": "premium",
        "kind": "numeric",
        "before": 2,  # "$1,200.50" and "CAD 900"
        "after": 0,
    }
    assert (by["age"]["before"], by["age"]["after"]) == (0, 1)  # 999 emptied


def test_the_sample_puts_the_most_fixed_rows_first_and_shows_cells_plainly(run):
    sample = view(run)["sample"]
    assert sample["columns"][0] == ["Prov", "province"]
    first = sample["rows"][0]
    # A-3: province, premium, age (n/a) and status all fixed.
    assert first == [["Ontario", "ON"], ["CAD 900", "900"], ["n/a", ""], ["cxl", "Yes"]]
    # 1200.5 shows as written, not as 1200.50 or 1200.5000.
    a1 = next(
        r for r in sample["rows"] if r[0] == ["ON", "ON"] and r[1][0].startswith("$")
    )
    assert a1[1] == ["$1,200.50", "1200.5"]
    assert len(sample["rows"]) == cleaning.SAMPLE_ROWS


def test_a_run_prefers_its_saved_view_then_its_files(run):
    insurance = scenarios.load("insurance")
    assert cleaning.for_run(run, insurance) is None  # not the scenario's files
    (run / cleaning.SAVED).write_text(json.dumps({"saved": True}))
    assert cleaning.for_run(run, insurance) == {"saved": True}


def test_a_run_whose_agent_wrote_a_different_table_has_no_view(tmp_path):
    insurance = scenarios.load("insurance")
    task = insurance.task
    (tmp_path / task.HISTORY).write_text(RAW.replace("Policy ID", insurance.raw_id))
    (tmp_path / task.CLEAN_HISTORY).write_text("something,else\n1,2\n")
    assert cleaning.for_run(tmp_path, insurance) is None


@pytest.mark.parametrize("name", scenarios.names())
def test_every_scenario_tells_the_page_its_question_and_column_names(name):
    labels = scenarios.load(name).task.LABELS
    assert labels["question"].strip() and labels["audience"] and labels["org"]
    names = labels["column_names"]
    assert all(1 <= len(v) <= 2 for v in names.values())
    assert all("{}" in v[1] for v in names.values() if len(v) == 2)
    task = scenarios.load(name).task
    assert task.ID in names and task.TARGET in names
    for clean in getattr(task, "RAW_COLUMNS", {}).values():
        assert clean in names  # the tidy view names every column it shows
