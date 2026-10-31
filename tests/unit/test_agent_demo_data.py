"""The agent demo's data preparation and scoring (examples/agent_demo/),
on small synthetic tables: no download, no agent."""

import json
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

DEMO = Path(__file__).resolve().parents[2] / "examples" / "agent_demo"
sys.path.insert(0, str(DEMO))
import prepare_data  # noqa: E402
import score as score_module  # noqa: E402
import task  # noqa: E402


@pytest.mark.parametrize(
    ("code", "group"),
    [
        ("428", "Circulatory"),
        ("785", "Circulatory"),
        ("486", "Respiratory"),
        ("250.83", "Diabetes"),
        ("250", "Diabetes"),
        ("V57", "Other"),
        ("E888", "Other"),
        ("996", "Injury"),
        ("715", "Musculoskeletal"),
        ("599", "Genitourinary"),
        ("174", "Neoplasms"),
        ("578", "Digestive"),
        ("?", "Missing"),
        (None, "Missing"),
        ("38", "Other"),
    ],
)
def test_icd9_chapters(code, group):
    assert prepare_data.icd9_group(code) == group


def stays(patients: int, per_patient: int) -> pd.DataFrame:
    rows = [
        {
            "patient_nbr": p,
            task.ID: f"S{p}-{k}",
            task.TARGET: "Yes" if k % 3 == 0 else "No",
        }
        for p in range(patients)
        for k in range(per_patient)
    ]
    return pd.DataFrame(rows)


def test_split_keeps_current_patients_out_of_the_history(monkeypatch):
    monkeypatch.setattr(prepare_data, "CURRENT_ROWS", 20)
    monkeypatch.setattr(prepare_data, "HISTORY_ROWS", 300)
    frame = stays(patients=200, per_patient=4)
    history, current = prepare_data.split(frame)
    assert len(current) == 20 and len(history) == 300
    owner = frame.set_index(task.ID)["patient_nbr"]
    current_patients = set(owner[current[task.ID]])
    assert len(current_patients) == 20  # one stay each
    assert current_patients.isdisjoint(owner[history[task.ID]])
    assert "patient_nbr" not in history.columns and "patient_nbr" not in current.columns


def test_split_picks_patients_by_their_number_of_stays(monkeypatch):
    """Like the history, which samples stays: frequent patients (most often
    readmitted) are as common among current patients as in the history."""
    monkeypatch.setattr(prepare_data, "CURRENT_ROWS", 300)
    monkeypatch.setattr(prepare_data, "HISTORY_ROWS", 1000)
    frequent = stays(patients=200, per_patient=10)  # 2,000 stays
    rare = stays(patients=2000, per_patient=1)  # 2,000 stays
    rare["patient_nbr"] += 1000
    rare[task.ID] = "R" + rare[task.ID]
    _, current = prepare_data.split(pd.concat([frequent, rare]))
    share = current[task.ID].str.startswith("S").mean()
    assert 0.3 < share < 0.7  # uniform over patients would give about 0.09


def test_score(tmp_path, monkeypatch):
    answers = tmp_path / "answers"
    answers.mkdir()
    ids = [f"P{i}" for i in range(200)]
    truth = ["Yes" if i < 20 else "No" for i in range(200)]
    pd.DataFrame({task.ID: ids, task.TARGET: truth}).to_csv(
        answers / task.ANSWERS, index=False
    )
    monkeypatch.setattr(score_module, "HERE", tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    # Ranks the 20 positives first except for two swapped into places 9-10.
    proba = [1 - i / 200 for i in range(200)]
    proba[8], proba[40] = proba[40], proba[8]
    proba[9], proba[41] = proba[41], proba[9]
    pd.DataFrame({task.ID: ids, task.PROBA: proba}).to_csv(
        run / "x.predictions.csv", index=False
    )
    out = score_module.score(run)
    assert out["rows"] == 200
    assert out["base_rate"] == 0.1
    assert out["positive_in_top_10"] == 0.8
    assert out["positive_in_top_100"] == 0.2
    assert 0.95 < out["roc_auc"] < 1


def test_score_needs_every_row(tmp_path, monkeypatch):
    (tmp_path / "answers").mkdir()
    pd.DataFrame({task.ID: ["a", "b"], task.TARGET: ["Yes", "No"]}).to_csv(
        tmp_path / "answers" / task.ANSWERS, index=False
    )
    monkeypatch.setattr(score_module, "HERE", tmp_path)
    pd.DataFrame({task.ID: ["a"], task.PROBA: [0.5]}).to_csv(
        tmp_path / "p.csv", index=False
    )
    with pytest.raises(SystemExit, match="covers 1 of 2"):
        score_module.score(tmp_path)


def test_showcase_runs_match_the_task():
    """The replayable runs were made with this task's data and scoring."""
    prompt = (DEMO / "PROMPT.md").read_text()
    assert task.HISTORY in prompt and task.CURRENT in prompt and task.TARGET in prompt
    for agent in ("claude", "codex", "pi"):
        folder = DEMO / "showcase" / agent
        stored = json.loads((folder / "score.json").read_text())
        assert {
            "roc_auc",
            "base_rate",
            "positive_in_top_10",
            "positive_in_top_100",
        } <= set(stored)
        assert task.CURRENT in (folder / "transcript.jsonl").read_text()


def test_run_sh_reads_the_file_names():
    from subprocess import run

    out = run(
        [sys.executable, "task.py", "files"], cwd=DEMO, capture_output=True, text=True
    )
    assert out.stdout.split() == [task.HISTORY, task.CURRENT]


def test_showcase_transcripts_hide_the_machine(tmp_path, monkeypatch):
    import make_showcase

    monkeypatch.setattr(make_showcase.getpass, "getuser", lambda: "jdoe")
    run_dir = tmp_path / "runs" / "codex"
    line = json.dumps(
        {
            "type": "item.completed",
            "item": {
                "command": f"cat {run_dir}/report.md",
                "aggregated_output": "-rw-r--r--  1 jdoe  staff  10 a.csv\n/Users/jdoe/x",
            },
        }
    )
    out = make_showcase.sanitize("codex", line, run_dir)
    assert "jdoe" not in out and str(tmp_path) not in out
    assert "cat report.md" in out and "1 user  staff" in out
    assert (
        make_showcase.sanitize("codex", json.dumps({"type": "thread.started"}), run_dir)
        is None
    )
