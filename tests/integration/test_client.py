"""End-to-end tests for tabctx.client.TabctxClient against a real local
2-replica deployment -- proving the client's automatic headers satisfy
both serving contracts (session affinity + tenant scoping) so its users
can't get them wrong."""

import pytest

pytest.importorskip("ray")

from tabctx.client import TabctxClient  # noqa: E402
from tabctx.errors import DatasetNotFoundError, InvalidInputError  # noqa: E402

TRAIN_X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
TEST_X = [[9.0, 9.0]]


def test_fit_predict_roundtrip_multi_replica(two_replica_service):
    client = TabctxClient(two_replica_service)
    dataset_id = client.fit(TRAIN_X, ["y", "y", "n"], dataset_id="client-ds")
    assert dataset_id == "client-ds"
    # Many predicts: the client's automatic session header must pin every
    # one to the replica holding the context (zero 404s on 2 replicas).
    served = set()
    for _ in range(10):
        result = client.predict("client-ds", TEST_X, return_proba=True)
        assert result.predictions == ["y"]
        assert result.classes == ["n", "y"]
        served.add(result.served_by)
    assert len(served) == 1, f"affinity not pinned: {served}"


def test_tenant_isolation_via_client(two_replica_service):
    acme = TabctxClient(two_replica_service, tenant_id="acme")
    globex = TabctxClient(two_replica_service, tenant_id="globex")
    acme.fit(TRAIN_X, ["acme-y", "acme-y", "n"], dataset_id="shared-id")
    globex.fit(TRAIN_X, ["globex-y", "globex-y", "n"], dataset_id="shared-id")
    assert acme.predict("shared-id", TEST_X).predictions == ["acme-y"]
    assert globex.predict("shared-id", TEST_X).predictions == ["globex-y"]
    # A client with no tenant sees nothing under that id.
    with pytest.raises(DatasetNotFoundError):
        TabctxClient(two_replica_service).predict("shared-id", TEST_X)


def test_errors_map_to_tabctx_exceptions(two_replica_service):
    client = TabctxClient(two_replica_service)
    with pytest.raises(DatasetNotFoundError):
        client.predict("no-such-dataset", TEST_X)
    with pytest.raises(InvalidInputError):
        client.fit([[1.0]], ["a", "b"])  # X/y length mismatch -> 422


def test_one_shot_fit_predict(two_replica_service):
    client = TabctxClient(two_replica_service)
    result = client.fit_predict(TRAIN_X, ["y", "y", "n"], TEST_X)
    assert result.predictions == ["y"]


def test_ready(two_replica_service):
    ready = TabctxClient(two_replica_service).ready()
    assert ready["status"] == "ready"
    assert "cache_stats" in ready
    # Restart visibility: operators must be able to tell "this replica
    # restarted and dropped its cache" apart from eviction.
    assert ready["replica_uptime_s"] >= 0
    assert ready["replica_started_at_unix"] > 0
