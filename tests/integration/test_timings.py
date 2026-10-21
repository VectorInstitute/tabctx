"""Fit and predict responses carry the server-side timing breakdown."""

import pytest

pytest.importorskip("ray")

import httpx  # noqa: E402


def test_fit_and_predict_report_where_their_time_went(two_replica_service):
    ds = "timings"
    headers = {"x-session-id": ds}
    X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    fit = httpx.post(
        f"{two_replica_service}/v1/tabctx/fit",
        json={"dataset_id": ds, "train_X": X, "train_y": ["a", "b", "a"]},
        headers=headers,
        timeout=60,
    ).json()
    assert {"read_table", "prepare", "wait_device", "fit_device"} <= set(
        fit["timings_ms"]
    )
    predict = httpx.post(
        f"{two_replica_service}/v1/tabctx/predict",
        json={"dataset_id": ds, "test_X": X[:1]},
        headers=headers,
        timeout=60,
    ).json()
    assert {"read_table", "wait_device", "predict_device"} <= set(predict["timings_ms"])
    assert predict["timings_ms"]["n_predict_chunks"] == 1
    assert isinstance(predict["timings_ms"]["n_predict_chunks"], int)
