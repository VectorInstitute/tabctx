"""The serving layer on a multi-device replica (fake backend: no GPU).

TABCTX_DEVICES=cuda:0,cuda:1 with the fake backend builds a real
MultiDeviceEngine without touching CUDA, so this exercises the exact
HTTP paths a multi-GPU replica serves: placement across devices, routing
predicts back to them, and the per-device /readyz and /limits views.
"""

import pytest

ray = pytest.importorskip("ray")
httpx = pytest.importorskip("httpx")

PREFIX = "/multi"


@pytest.fixture(scope="module")
def multi_device_service(two_replica_service):
    from ray import serve

    from tabctx.serve.app import TabctxService

    serve.run(
        TabctxService.options(
            num_replicas=1,
            ray_actor_options={
                "runtime_env": {
                    "env_vars": {
                        "TABCTX_BACKEND": "fake",
                        "TABCTX_DEVICES": "cuda:0,cuda:1",
                    }
                }
            },
        ).bind(),
        name="multi-device",
        route_prefix=PREFIX,
    )
    yield two_replica_service + PREFIX
    serve.delete("multi-device")


def test_contexts_spread_over_devices_and_predict_from_them(multi_device_service):
    with httpx.Client(base_url=multi_device_service, timeout=60.0) as client:
        ready = client.get("/readyz").json()
        assert [d["device"] for d in ready["devices"]] == ["cuda:0", "cuda:1"]

        for i in range(6):
            r = client.post(
                "/v1/tabctx/fit",
                json={
                    "train_X": [[float(i), 1.0], [2.0, 3.0]],
                    "train_y": [f"y{i}", f"y{i}"],
                    "dataset_id": f"md-{i}",
                },
            )
            assert r.status_code == 200, r.text
        for i in range(6):
            r = client.post(
                "/v1/tabctx/predict",
                json={"dataset_id": f"md-{i}", "test_X": [[0.0, 0.0]]},
            )
            assert r.status_code == 200, r.text
            assert r.json()["predictions"] == [f"y{i}"]

        devices = client.get("/readyz").json()["devices"]
        assert sum(d["n_cached_contexts"] for d in devices) == 6
        assert all(d["n_cached_contexts"] > 0 for d in devices)

        limits = client.get("/v1/tabctx/limits").json()
        assert limits["devices"] == ["cuda:0", "cuda:1"]
