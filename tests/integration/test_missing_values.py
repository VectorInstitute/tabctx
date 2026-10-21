"""Missing values (JSON null) through the HTTP API, against the real
2-replica deployment: fit, predict and the legacy one-shot endpoint."""

import pytest

pytest.importorskip("ray")

import httpx  # noqa: E402

X = [[1.0, None], [None, 4.0], [5.0, 6.0], [7.0, None]]


def _post(base_url, path, body, dataset_id=None):
    headers = {"x-session-id": dataset_id} if dataset_id else {}
    return httpx.post(f"{base_url}{path}", json=body, headers=headers, timeout=60)


def test_fit_and_predict_accept_null_cells(two_replica_service):
    ds = "missing-values"
    resp = _post(
        two_replica_service,
        "/v1/tabctx/fit",
        {"dataset_id": ds, "train_X": X, "train_y": ["a", "b", "a", "b"]},
        ds,
    )
    assert resp.status_code == 200, resp.text
    resp = _post(
        two_replica_service,
        "/v1/tabctx/predict",
        {"dataset_id": ds, "test_X": [[None, None], [2.0, 3.0]]},
        ds,
    )
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["predictions"]) == 2


def test_legacy_endpoint_accepts_null_cells(two_replica_service):
    resp = _post(
        two_replica_service,
        "/v1/tabicl/predict",
        {"train_X": X, "train_y": ["a", "b", "a", "b"], "test_X": [[None, 1.0]]},
    )
    assert resp.status_code == 200, resp.text


def test_ragged_rows_with_nulls_are_still_a_422(two_replica_service):
    resp = _post(
        two_replica_service,
        "/v1/tabctx/fit",
        {
            "dataset_id": "ragged",
            "train_X": [[1.0, None], [2.0]],
            "train_y": ["a", "b"],
        },
        "ragged",
    )
    assert resp.status_code == 422
