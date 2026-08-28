#!/usr/bin/env python3
"""No-GPU, no-Ray smoke example using FakeBackend -- run with plain python,
no extras installed. For a real prediction using TabICL, see the README
Quickstart (needs the `tabicl` extra) or serve/app.py for the Ray Serve
deployment.
"""

from tabctx import ContextCacheManager, TabctxEngine
from tabctx.backends.fake import FakeBackend
from tabctx.memory import A100_40GB_TABICL_CALIBRATION, PowerLawMemoryEstimator


def main() -> None:
    estimator = PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION)
    print(estimator.confidence())

    cache = ContextCacheManager(capacity_bytes=estimator.ceiling_bytes())
    engine = TabctxEngine(backend=FakeBackend(), cache=cache, estimator=estimator)

    X_train = [[float(i), float(i) * 2] for i in range(20)]
    y_train = ["a" if i % 2 == 0 else "b" for i in range(20)]
    dataset_id = engine.fit(X_train, y_train, task="classification")
    print(f"fit -> dataset_id={dataset_id}")

    result = engine.predict(dataset_id, [[1.0, 2.0], [3.0, 4.0]], return_proba=True)
    print(f"predict #1: {result.predictions} (proba shape reused cache)")

    # Reuse the same cached context for a different test batch -- no re-fit.
    result2 = engine.predict(dataset_id, [[5.0, 10.0]])
    print(f"predict #2 (cache reused): {result2.predictions}")

    print(f"stats: {engine.stats()}")
    engine.evict(dataset_id)


if __name__ == "__main__":
    main()
