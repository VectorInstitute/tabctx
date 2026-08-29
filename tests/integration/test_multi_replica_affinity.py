"""Multi-replica session-affinity integration test.

THE regression test for the v0.6.0 multi-replica correctness fix: deploys
TabctxService with TWO replicas (fake backend -- no GPU, no torch) on a
local Ray cluster and verifies that a fit() on one HTTP request followed
by predict() calls on *different* HTTP requests against the same
dataset_id succeeds regardless of which replica each would naively land
on. Before the consistent-hash router was wired in, this failed with
intermittent 404s at num_replicas >= 2 (each predict had a ~1/2 chance of
landing on the replica that never fit the dataset).

Statistical strength: with 16 datasets x 6 predicts each and no affinity,
the chance of zero 404s is ~(1/2)^96 -- a pass is not luck.

Requires ray[serve]; skipped automatically when it isn't installed (unit
CI installs core deps only, the integration CI job installs `.[serve]`).
"""

import pytest

ray = pytest.importorskip("ray")
httpx = pytest.importorskip("httpx")

NUM_REPLICAS = 2  # keep in sync with conftest.py's deployment
NUM_DATASETS = 16
PREDICTS_PER_DATASET = 6


def _train_body(seed: int, dataset_id: str | None = None) -> dict:
    body = {
        "train_X": [[float(seed), 2.0], [3.0, 4.0], [5.0, float(seed) + 1]],
        "train_y": ["a", "b", "a"],
        "task": "classification",
    }
    if dataset_id is not None:
        body["dataset_id"] = dataset_id
    return body


def test_fit_then_predict_across_replicas_never_404s(two_replica_service):
    base = two_replica_service
    fit_served_by: dict[str, str] = {}

    with httpx.Client(base_url=base, timeout=60.0) as client:
        for i in range(NUM_DATASETS):
            dataset_id = f"affinity-ds-{i}"
            headers = {"x-session-id": dataset_id}
            r = client.post(
                "/v1/tabctx/fit",
                json=_train_body(i, dataset_id),
                headers=headers,
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["dataset_id"] == dataset_id
            assert body["served_by"], "replica tag must be reported"
            fit_served_by[dataset_id] = body["served_by"]

        # The heart of the test: every predict must reach the replica
        # that holds the context -- zero spurious 404s.
        for dataset_id, fit_replica in fit_served_by.items():
            for _ in range(PREDICTS_PER_DATASET):
                r = client.post(
                    "/v1/tabctx/predict",
                    json={"dataset_id": dataset_id, "test_X": [[1.0, 2.0]]},
                    headers={"x-session-id": dataset_id},
                )
                assert r.status_code == 200, (
                    f"{dataset_id}: expected 200, got {r.status_code} -- "
                    f"affinity broke: {r.text}"
                )
                assert r.json()["served_by"] == fit_replica, (
                    f"{dataset_id}: predict served by "
                    f"{r.json()['served_by']}, but fit ran on {fit_replica}"
                )

    # Sanity check that this test actually exercised MULTIPLE replicas
    # (otherwise a silently single-replica deployment would vacuously
    # pass). With 16 keys consistent-hashed over 2 replicas and 100
    # vnodes each, all 16 landing on one replica is ~0.8% likely per
    # direction -- tolerate nothing, since a flake here means the ring
    # itself is broken or num_replicas was ignored.
    assert len(set(fit_served_by.values())) == NUM_REPLICAS, (
        f"expected fits spread over {NUM_REPLICAS} replicas, saw "
        f"{set(fit_served_by.values())}"
    )


def test_fit_adopts_session_header_as_dataset_id(two_replica_service):
    base = two_replica_service
    with httpx.Client(base_url=base, timeout=60.0) as client:
        r = client.post(
            "/v1/tabctx/fit",
            json=_train_body(99),  # no dataset_id in the body
            headers={"x-session-id": "header-named-ds"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["dataset_id"] == "header-named-ds"

        r = client.post(
            "/v1/tabctx/predict",
            json={"dataset_id": "header-named-ds", "test_X": [[1.0, 2.0]]},
            headers={"x-session-id": "header-named-ds"},
        )
        assert r.status_code == 200, r.text


def test_session_header_dataset_id_mismatch_is_422(two_replica_service):
    base = two_replica_service
    with httpx.Client(base_url=base, timeout=60.0) as client:
        r = client.post(
            "/v1/tabctx/fit",
            json=_train_body(7, "body-ds"),
            headers={"x-session-id": "other-ds"},
        )
        assert r.status_code == 422, r.text

        r = client.post(
            "/v1/tabctx/predict",
            json={"dataset_id": "body-ds", "test_X": [[1.0, 2.0]]},
            headers={"x-session-id": "other-ds"},
        )
        assert r.status_code == 422, r.text


def test_tenant_namespacing_isolates_same_dataset_id(two_replica_service):
    """Two tenants using the SAME dataset_id must get their own contexts
    (v0.6.0 Priority-2 fix: dataset_id alone was a flat, guessable
    namespace). The fake backend predicts the training majority class, so
    each tenant's predictions reveal exactly whose context served them."""
    base = two_replica_service
    dataset_id = "shared-name"
    headers = {"x-session-id": dataset_id}

    def fit_body(majority_class: str) -> dict:
        return {
            "train_X": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            "train_y": [majority_class, majority_class, "other"],
            "task": "classification",
            "dataset_id": dataset_id,
        }

    with httpx.Client(base_url=base, timeout=60.0) as client:
        for tenant, label in (("acme", "acme-class"), ("globex", "globex-class")):
            r = client.post(
                "/v1/tabctx/fit",
                json=fit_body(label),
                headers={**headers, "x-tabctx-tenant-id": tenant},
            )
            assert r.status_code == 200, r.text
            # The scoping prefix must never leak into responses.
            assert r.json()["dataset_id"] == dataset_id

        for tenant, label in (("acme", "acme-class"), ("globex", "globex-class")):
            r = client.post(
                "/v1/tabctx/predict",
                json={"dataset_id": dataset_id, "test_X": [[9.0, 9.0]]},
                headers={**headers, "x-tabctx-tenant-id": tenant},
            )
            assert r.status_code == 200, r.text
            assert r.json()["predictions"] == [label], (
                f"tenant {tenant} got another tenant's model output"
            )

        # No tenant header -> unscoped namespace -> nothing cached there.
        r = client.post(
            "/v1/tabctx/predict",
            json={"dataset_id": dataset_id, "test_X": [[9.0, 9.0]]},
            headers=headers,
        )
        assert r.status_code == 404, r.text

        # Malformed tenant id is rejected loudly, not silently unscoped.
        r = client.post(
            "/v1/tabctx/predict",
            json={"dataset_id": dataset_id, "test_X": [[9.0, 9.0]]},
            headers={**headers, "x-tabctx-tenant-id": "not:allowed"},
        )
        assert r.status_code == 422, r.text


def test_unknown_dataset_is_a_clean_404(two_replica_service):
    base = two_replica_service
    with httpx.Client(base_url=base, timeout=60.0) as client:
        r = client.post(
            "/v1/tabctx/predict",
            json={"dataset_id": "never-fit", "test_X": [[1.0, 2.0]]},
            headers={"x-session-id": "never-fit"},
        )
        assert r.status_code == 404, r.text
