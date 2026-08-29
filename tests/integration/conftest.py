"""Shared fixture: a local 2-replica Ray Serve deployment (fake backend).

Module-scoped in each test module would boot Ray repeatedly; session
scope boots it once for the whole integration run.
"""

import pytest

ray = pytest.importorskip("ray")

SERVE_PORT = 18123
BASE_URL = f"http://127.0.0.1:{SERVE_PORT}"
NUM_REPLICAS = 2


@pytest.fixture(scope="session")
def two_replica_service():
    from ray import serve

    from tabctx.serve.app import TabctxService

    ray.init(num_cpus=4, include_dashboard=False)
    serve.start(http_options={"host": "127.0.0.1", "port": SERVE_PORT})
    serve.run(
        TabctxService.options(
            num_replicas=NUM_REPLICAS,
            ray_actor_options={
                # Replicas are separate processes; the driver's env vars
                # don't reach them implicitly. runtime_env is the
                # explicit, reliable channel.
                "runtime_env": {"env_vars": {"TABCTX_BACKEND": "fake"}},
            },
        ).bind()
    )
    yield BASE_URL
    serve.shutdown()
    ray.shutdown()
