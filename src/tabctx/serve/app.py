"""Ray Serve host for the tabctx service (import path tabctx.serve.app:app).

The service itself -- the FastAPI routes, request/response models and
engine wiring -- lives in serve/service.py, which is host-agnostic; this
module is the only one in the serve package that imports ray. It applies
the deployment settings and the consistent-hash session-affinity router
that make the per-replica context cache correct at num_replicas >= 2
(the contract lives in serve/affinity.py). For hosts without Ray, see
serve/standalone.py (`python -m tabctx.serve`).
"""

from ray import cloudpickle, serve
from ray.serve.config import RequestRouterConfig

import tabctx.serve.service
from tabctx.serve.service import TabctxService as _TabctxService
from tabctx.serve.service import fastapi_app

# serve.ingress freezes fastapi_app by cloudpickling it, AFTER rewriting
# each route's `self` parameter into a Depends -- a rewrite made on the
# handler functions themselves (their __signature__). cloudpickle ships an
# importable function by reference, so without this line the replica would
# re-import the pristine handlers from serve/service.py, and FastAPI would
# demand `self` as a query parameter (a 422 on every route). Before the
# service moved out of this module the handlers were not importable in
# pristine form (this module's `TabctxService` name is the Deployment, not
# the class), so cloudpickle fell back to by-value implicitly; registering
# the module keeps exactly those semantics.
cloudpickle.register_pickle_by_value(tabctx.serve.service)

TabctxService = serve.deployment(
    # 8 concurrent requests per replica (was 2): safe because each device's
    # engine schedules its GPU work itself (one fit, plus one light predict
    # beside it; see engine.py) -- extra in-flight requests queue there (or
    # coalesce, see batching.py) rather than running unbounded concurrent
    # GPU calls. More in-flight requests = more same-context coalescing
    # opportunity.
    max_ongoing_requests=8,
    max_queued_requests=16,
    health_check_period_s=30,
    health_check_timeout_s=60,
    # Consistent-hash routing on the session-affinity header: requests for
    # the same dataset_id (sent as `x-session-id`) always land on the same
    # replica, which is what makes the per-replica context cache correct at
    # num_replicas > 1 at all. num_fallback_replicas=0 is deliberate:
    # falling back to a *different* replica under backpressure would land
    # predict() calls on a replica without the cached context -- the exact
    # spurious-404 failure this router exists to prevent. Strict affinity
    # means backpressure surfaces as retry-with-backoff (and eventually
    # 503) on the owning replica instead, which is honest and retryable.
    request_router_config=RequestRouterConfig(
        request_router_class=(
            "ray.serve.experimental.consistent_hash_router.ConsistentHashRouter"
        ),
        request_router_kwargs={"num_fallback_replicas": 0},
    ),
)(serve.ingress(fastapi_app)(_TabctxService))

app = TabctxService.bind()
