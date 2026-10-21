"""A table file in the fit/predict request itself (multipart/form-data),
and text columns, on a real local 2-replica deployment -- the curl path:

    curl $URL/v1/tabctx/fit -F dataset_id=churn -F target=churned -F file=@train.csv
"""

import gzip
import io

import pytest

pytest.importorskip("ray")
httpx = pytest.importorskip("httpx")
pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402

TRAIN_CSV = (
    "age,plan,churned\n"
    + "".join(
        f"{20 + i},{'gold' if i % 2 else 'basic'},{'yes' if i < 7 else 'no'}\n"
        for i in range(10)
    )
).encode()


def post(url, path, dataset_id, **kwargs):
    return httpx.post(
        f"{url}{path}", headers={"x-session-id": dataset_id}, timeout=60, **kwargs
    )


def fit_file(url, dataset_id, content, **fields):
    return post(
        url,
        "/v1/tabctx/fit",
        dataset_id,
        data={"dataset_id": dataset_id, **fields},
        files={"file": ("train", content)},
    )


def predict_file(url, dataset_id, content, **fields):
    return post(
        url,
        "/v1/tabctx/predict",
        dataset_id,
        data={"dataset_id": dataset_id, **fields},
        files={"file": ("test", content)},
    )


def parquet(table) -> bytes:
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    return buf.getvalue()


def test_fit_and_predict_with_a_file_in_one_call(two_replica_service):
    url, ds = two_replica_service, "file-csv"
    fit = fit_file(url, ds, TRAIN_CSV, target="churned")
    assert fit.status_code == 200, fit.text
    body = fit.json()
    assert (body["dataset_id"], body["n_train"], body["n_features"]) == (ds, 10, 2)
    assert body["task"] == "classification"

    # Columns by name, any order; text encoded as at fit; unseen -> missing.
    test = b"plan,age\ngold,30\nplatinum,\n"
    pred = predict_file(url, ds, test, return_proba="true")
    assert pred.status_code == 200, pred.text
    assert pred.json()["predictions"] == ["yes", "yes"]
    assert pred.json()["classes"] == ["no", "yes"]


def test_gzip_csv_and_parquet_files(two_replica_service):
    url, ds = two_replica_service, "file-formats"
    assert fit_file(url, ds, gzip.compress(TRAIN_CSV)).status_code == 200
    table = pa.table({"age": [41.0, None], "plan": ["basic", None]})
    pred = predict_file(url, ds, parquet(table))
    assert pred.status_code == 200, pred.text
    assert pred.json()["n_test"] == 2


def test_text_cells_in_json_rows(two_replica_service):
    url, ds = two_replica_service, "json-text"
    fit = post(
        url,
        "/v1/tabctx/fit",
        ds,
        json={
            "dataset_id": ds,
            "train_X": [[1.5, "gold", None], [2.0, "basic", 3.0], [None, "gold", 1.0]],
            "train_y": ["yes", "no", "yes"],
        },
    )
    assert fit.status_code == 200, fit.text
    pred = post(
        url,
        "/v1/tabctx/predict",
        ds,
        json={"dataset_id": ds, "test_X": [[1.0, "basic", 2.0], [None, "new", None]]},
    )
    assert pred.status_code == 200, pred.text
    assert pred.json()["predictions"] == ["yes", "yes"]
    # Text where training had numbers is a clear 422.
    bad = post(
        url,
        "/v1/tabctx/predict",
        ds,
        json={"dataset_id": ds, "test_X": [["x", "gold", 1.0]]},
    )
    assert bad.status_code == 422 and "held numbers" in bad.text


@pytest.mark.parametrize(
    ("request_kwargs", "status", "match"),
    [
        (
            {"data": {"dataset_id": "bad"}, "files": {"other": ("x", b"a\n1\n")}},
            422,
            "`file` part",
        ),
        (
            {
                "data": {"dataset_id": "bad", "task": "nope"},
                "files": {"file": ("x", TRAIN_CSV)},
            },
            422,
            "task",
        ),
        (
            {
                "data": {"dataset_id": "bad"},
                "files": {"file": ("x", b"\x00\x01garbage")},
            },
            422,
            "table",
        ),
        (
            {
                "data": {"dataset_id": "bad", "train_upload_id": "abc"},
                "files": {"file": ("x", TRAIN_CSV)},
            },
            422,
            "not both",
        ),
    ],
)
def test_bad_multipart_fits_are_422(two_replica_service, request_kwargs, status, match):
    resp = post(two_replica_service, "/v1/tabctx/fit", "bad", **request_kwargs)
    assert resp.status_code == status, resp.text
    assert match in resp.text


def test_the_openapi_schema_documents_both_body_types(two_replica_service):
    spec = httpx.get(f"{two_replica_service}/openapi.json").json()
    content = spec["paths"]["/v1/tabctx/fit"]["post"]["requestBody"]["content"]
    assert set(content) == {"application/json", "multipart/form-data"}
    assert "file" in content["multipart/form-data"]["schema"]["required"]
