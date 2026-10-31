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
            "'file' part",
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
            "got several",
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


def test_an_unknown_context_is_404_whatever_the_rows(two_replica_service):
    # 404 is what tells a client to fit again; text rows must not turn it
    # into a 422 by being encoded against a schema that isn't there.
    url = two_replica_service
    assert predict_file(url, "never-fit", b"plan,age\ngold,1\n").status_code == 404
    inline = post(
        url,
        "/v1/tabctx/predict",
        "never-fit",
        json={"dataset_id": "never-fit", "test_X": [[1.0, "gold"]]},
    )
    assert inline.status_code == 404


def test_an_invalid_request_leaves_its_upload_unused(two_replica_service):
    from tabctx.client import TabctxClient

    url, ds = two_replica_service, "upload-kept"
    upload_id = TabctxClient(url).upload_csv(TRAIN_CSV, ds)
    bad = post(
        url,
        "/v1/tabctx/fit",
        ds,
        json={"dataset_id": ds, "train_upload_id": upload_id, "train_y": ["a"]},
    )
    assert bad.status_code == 422 and "got several" in bad.text
    good = post(
        url, "/v1/tabctx/fit", ds, json={"dataset_id": ds, "train_upload_id": upload_id}
    )
    assert good.status_code == 200, good.text


def test_an_oversized_body_is_refused_from_its_headers(two_replica_service):
    import http.client
    from urllib.parse import urlparse

    target = urlparse(two_replica_service)
    conn = http.client.HTTPConnection(target.hostname, target.port, timeout=30)
    conn.putrequest("POST", "/v1/tabctx/fit")
    conn.putheader("Content-Type", "multipart/form-data; boundary=x")
    conn.putheader("Content-Length", str(64 * 1024**3))
    conn.putheader("x-session-id", "huge")
    conn.endheaders()
    conn.send(b"--x\r\n")  # a few bytes of 64 GiB: answered without the rest
    resp = conn.getresponse()
    assert resp.status == 413
    conn.close()


def test_validation_errors_keep_fastapis_shape(two_replica_service):
    resp = post(
        two_replica_service,
        "/v1/tabctx/fit",
        "shape",
        json={"dataset_id": "shape", "task": "nope", "train_X": [[1]], "train_y": [1]},
    )
    assert resp.status_code == 422
    assert resp.json()["detail"][0]["loc"] == ["body", "task"]


def test_a_pandas_parquet_files_index_is_not_a_column(two_replica_service):
    """A DataFrame from train_test_split, saved with to_parquet(), carries
    its shuffled index as a column; fitting it and predicting JSON rows
    without that column used to be a 422 ("the table has 2 columns; the
    training table had 3"), and with no target the index was the labels."""
    pd = pytest.importorskip("pandas")
    url, ds = two_replica_service, "file-pandas-index"
    frame = pd.DataFrame(
        {
            "age": [20 + i for i in range(10)],
            "plan": ["gold" if i % 2 else "basic" for i in range(10)],
            "churned": ["yes" if i < 7 else "no" for i in range(10)],
        },
        index=[9, 2, 7, 0, 5, 3, 8, 1, 6, 4],
    )
    buf = io.BytesIO()
    frame.to_parquet(buf)
    fit = fit_file(url, ds, buf.getvalue())  # no target: the last column
    assert fit.status_code == 200, fit.text
    assert fit.json()["n_features"] == 2 and fit.json()["task"] == "classification"
    pred = post(
        url,
        "/v1/tabctx/predict",
        ds,
        json={"dataset_id": ds, "test_X": [[30, "gold"]], "return_proba": True},
    )
    assert pred.status_code == 200, pred.text
    assert sorted(pred.json()["classes"]) == ["no", "yes"]

    # A predict file from the other split carries its own (different) index.
    test = pd.DataFrame({"plan": ["gold", "basic"], "age": [31, 44]}, index=[12, 10])
    buf = io.BytesIO()
    test.to_parquet(buf)
    pred = predict_file(url, ds, buf.getvalue())
    assert pred.status_code == 200, pred.text
    assert pred.json()["n_test"] == 2
