"""The fit endpoint infers the task when it's omitted, reports it, and
rejects a classification of continuous labels with the fix in the
message -- over HTTP, against the real 2-replica deployment."""

import pytest

pytest.importorskip("ray")

import httpx  # noqa: E402

X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]


def _fit(base_url, body):
    dataset_id = body.setdefault("dataset_id", "task-inference")
    return httpx.post(
        f"{base_url}/v1/tabctx/fit",
        json={"train_X": X, **body},
        headers={"x-session-id": dataset_id},
        timeout=60,
    )


@pytest.mark.parametrize(
    "train_y, task",
    [
        ([0.5, 1.25, 2.0, 3.75], "regression"),
        (["a", "b", "a", "b"], "classification"),
        ([0, 1, 0, 1], "classification"),
        (["250.01", "401.9", "250.01", "401.9"], "classification"),
    ],
)
def test_omitted_task_is_inferred_and_reported(two_replica_service, train_y, task):
    resp = _fit(two_replica_service, {"train_y": train_y})
    assert resp.status_code == 200, resp.text
    assert resp.json()["task"] == task


def test_explicit_task_wins(two_replica_service):
    resp = _fit(two_replica_service, {"train_y": [0, 1, 0, 1], "task": "regression"})
    assert resp.json()["task"] == "regression"


def test_continuous_labels_as_classification_is_a_422_naming_the_fix(
    two_replica_service,
):
    resp = _fit(
        two_replica_service,
        {"train_y": [0.5, 1.25, 2.0, 3.75], "task": "classification"},
    )
    assert resp.status_code == 422
    assert "task='regression'" in resp.json()["detail"]


def test_legacy_predict_infers_the_task_too(two_replica_service):
    resp = httpx.post(
        f"{two_replica_service}/v1/tabicl/predict",
        json={"train_X": X, "train_y": [0.5, 1.25, 2.0, 3.75], "test_X": X[:1]},
        timeout=60,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["classes"] is None  # regression has none
    assert resp.json()["task"] == "regression"
