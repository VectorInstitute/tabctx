"""End-to-end tests for large-table ingestion (upload -> fit/predict by
reference) on a real local 2-replica deployment.

The critical property under test: uploads are replica-LOCAL, so the
whole flow only works because the session-affinity header routes the
upload, the fit, and every predict for one dataset_id to the same
replica. A large-ish table (50k rows) exercises the streamed path with
data that would be obnoxious as inline JSON.
"""

import numpy as np
import pytest

pytest.importorskip("ray")

from tabctx.client import TabctxClient  # noqa: E402
from tabctx.errors import (  # noqa: E402
    InvalidInputError,
    UploadNotFoundError,
)


def _train_csv(n_rows: int, n_features: int, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_features))
    # Deterministic labels so predictions are verifiable end to end:
    # the fake backend predicts the majority class.
    y = ["hot"] * (n_rows * 2 // 3) + ["cold"] * (n_rows - n_rows * 2 // 3)
    header = ",".join([f"f{i}" for i in range(n_features)] + ["label"])
    lines = [header]
    for row, label in zip(X, y):
        lines.append(",".join(f"{v:.4f}" for v in row) + f",{label}")
    return ("\n".join(lines) + "\n").encode()


def _test_csv(n_rows: int, n_features: int, seed: int = 1) -> bytes:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_features))
    header = ",".join(f"f{i}" for i in range(n_features))
    lines = [header] + [",".join(f"{v:.4f}" for v in row) for row in X]
    return ("\n".join(lines) + "\n").encode()


def test_upload_fit_predict_50k_rows(two_replica_service):
    client = TabctxClient(two_replica_service)
    dataset_id = "upload-large"

    upload_id = client.upload_csv(_train_csv(50_000, 12), dataset_id)
    assert client.fit_uploaded(upload_id, dataset_id, target_column="label") == dataset_id

    # Predict by reference too, repeatedly -- affinity must hold across
    # upload -> fit -> predict on 2 replicas (zero 404s).
    for i in range(3):
        test_upload = client.upload_csv(_test_csv(500, 12, seed=i), dataset_id)
        result = client.predict(dataset_id, test_upload_id=test_upload)
        assert result.predictions == ["hot"] * 500

    # Inline predict against the same uploaded-fit context also works.
    assert client.predict(dataset_id, [[0.0] * 12]).predictions == ["hot"]


def test_uploads_are_single_use(two_replica_service):
    client = TabctxClient(two_replica_service)
    dataset_id = "upload-once"
    upload_id = client.upload_csv(_train_csv(30, 3), dataset_id)
    client.fit_uploaded(upload_id, dataset_id)
    with pytest.raises(UploadNotFoundError):
        client.fit_uploaded(upload_id, dataset_id)


def test_test_csv_schema_mismatch_is_422(two_replica_service):
    client = TabctxClient(two_replica_service)
    dataset_id = "upload-schema"
    upload_id = client.upload_csv(_train_csv(30, 3), dataset_id)
    client.fit_uploaded(upload_id, dataset_id)

    # Wrong column ORDER with the right names -- silently accepting this
    # would produce garbage predictions.
    bad = b"f1,f0,f2\n1,2,3\n"
    bad_upload = client.upload_csv(bad, dataset_id)
    with pytest.raises(InvalidInputError, match="do not match"):
        client.predict(dataset_id, test_upload_id=bad_upload)


def test_uploads_are_tenant_scoped(two_replica_service):
    acme = TabctxClient(two_replica_service, tenant_id="acme")
    globex = TabctxClient(two_replica_service, tenant_id="globex")
    dataset_id = "upload-tenant"
    upload_id = acme.upload_csv(_train_csv(30, 3), dataset_id)
    with pytest.raises(UploadNotFoundError):
        globex.fit_uploaded(upload_id, dataset_id)
    # Not consumed by the failed cross-tenant attempt:
    acme.fit_uploaded(upload_id, dataset_id)


def test_both_paths_in_one_request_rejected(two_replica_service):
    client = TabctxClient(two_replica_service)
    dataset_id = "upload-both"
    upload_id = client.upload_csv(_train_csv(30, 3), dataset_id)
    with pytest.raises(InvalidInputError, match="not both"):
        client._post(
            "/v1/tabctx/fit",
            {
                "train_X": [[1.0]],
                "train_y": ["a"],
                "train_upload_id": upload_id,
                "dataset_id": dataset_id,
            },
            session_id=dataset_id,
        )


def test_neither_path_rejected(two_replica_service):
    client = TabctxClient(two_replica_service)
    with pytest.raises(InvalidInputError, match="either"):
        client._post(
            "/v1/tabctx/fit",
            {"dataset_id": "upload-neither"},
            session_id="upload-neither",
        )
