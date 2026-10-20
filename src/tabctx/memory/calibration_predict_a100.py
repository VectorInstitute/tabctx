"""GENERATED predict calibration -- do not hand-edit.

Produced by benchmarks/gen_predict_grid.py from
benchmarks/calibrate_predict.py on real hardware: each PredictPoint is
one predict's peak allocation (above what was allocated before it)
against a kv-cached context, under the serving configuration. Read by
serve/factory.py into PredictCostModel (memory/predict.py).

Measured on NVIDIA A100-SXM4-40GB, torch 2.13.0+cu130 (kv_cache=kv, kv_precision=int8, compile_models=True).
"""

from __future__ import annotations

from tabctx.memory.predict import PredictPoint

A100_40GB_TABICL_PREDICT_POINTS: tuple[PredictPoint, ...] = (
    PredictPoint(n_train=1000, n_test=100, n_features=10, peak_bytes=28756480),
    PredictPoint(n_train=1000, n_test=1000, n_features=10, peak_bytes=290993664),
    PredictPoint(n_train=5000, n_test=100, n_features=10, peak_bytes=46926848),
    PredictPoint(n_train=5000, n_test=1000, n_features=10, peak_bytes=114848256),
    PredictPoint(n_train=20000, n_test=100, n_features=10, peak_bytes=168823808),
    PredictPoint(n_train=20000, n_test=1000, n_features=10, peak_bytes=189512192),
    PredictPoint(n_train=1000, n_test=100, n_features=50, peak_bytes=110932480),
    PredictPoint(n_train=1000, n_test=1000, n_features=50, peak_bytes=1112098304),
    PredictPoint(n_train=5000, n_test=100, n_features=50, peak_bytes=47056384),
    PredictPoint(n_train=5000, n_test=1000, n_features=50, peak_bytes=443168256),
    PredictPoint(n_train=20000, n_test=100, n_features=50, peak_bytes=168887808),
    PredictPoint(n_train=20000, n_test=1000, n_features=50, peak_bytes=443168256),
    PredictPoint(n_train=1000, n_test=100, n_features=100, peak_bytes=85357056),
    PredictPoint(n_train=1000, n_test=1000, n_features=100, peak_bytes=855861760),
    PredictPoint(n_train=5000, n_test=100, n_features=100, peak_bytes=85520896),
    PredictPoint(n_train=5000, n_test=1000, n_features=100, peak_bytes=855687168),
    PredictPoint(n_train=20000, n_test=100, n_features=100, peak_bytes=168967680),
    PredictPoint(n_train=20000, n_test=1000, n_features=100, peak_bytes=853568000),
    PredictPoint(n_train=1000, n_test=100, n_features=200, peak_bytes=167436800),
    PredictPoint(n_train=1000, n_test=1000, n_features=200, peak_bytes=1678824448),
    PredictPoint(n_train=5000, n_test=100, n_features=200, peak_bytes=167436800),
    PredictPoint(n_train=5000, n_test=1000, n_features=200, peak_bytes=1679818752),
    PredictPoint(n_train=20000, n_test=100, n_features=200, peak_bytes=169127424),
    PredictPoint(n_train=20000, n_test=1000, n_features=200, peak_bytes=1679818752),
    PredictPoint(n_train=1000, n_test=100, n_features=500, peak_bytes=413677056),
    PredictPoint(n_train=1000, n_test=1000, n_features=500, peak_bytes=4136960000),
    PredictPoint(n_train=5000, n_test=100, n_features=500, peak_bytes=413677056),
    PredictPoint(n_train=5000, n_test=1000, n_features=500, peak_bytes=4136768000),
    PredictPoint(n_train=20000, n_test=100, n_features=500, peak_bytes=413677056),
    PredictPoint(n_train=20000, n_test=1000, n_features=500, peak_bytes=4136768000),
    PredictPoint(n_train=1000, n_test=100, n_features=1000, peak_bytes=824076800),
    PredictPoint(n_train=1000, n_test=1000, n_features=1000, peak_bytes=8241390592),
    PredictPoint(n_train=5000, n_test=100, n_features=1000, peak_bytes=825769984),
    PredictPoint(n_train=5000, n_test=1000, n_features=1000, peak_bytes=8241390592),
)

A100_40GB_TABPFN_PREDICT_POINTS: tuple[PredictPoint, ...] = (
    PredictPoint(n_train=1000, n_test=100, n_features=10, peak_bytes=62734336),
    PredictPoint(n_train=1000, n_test=1000, n_features=10, peak_bytes=605032448),
    PredictPoint(n_train=5000, n_test=100, n_features=10, peak_bytes=106308096),
    PredictPoint(n_train=5000, n_test=1000, n_features=10, peak_bytes=605608960),
    PredictPoint(n_train=20000, n_test=100, n_features=10, peak_bytes=389089792),
    PredictPoint(n_train=20000, n_test=1000, n_features=10, peak_bytes=606250496),
    PredictPoint(n_train=1000, n_test=100, n_features=50, peak_bytes=205309440),
    PredictPoint(n_train=1000, n_test=1000, n_features=50, peak_bytes=1918574592),
    PredictPoint(n_train=5000, n_test=100, n_features=50, peak_bytes=205099008),
    PredictPoint(n_train=5000, n_test=1000, n_features=50, peak_bytes=1918835712),
    PredictPoint(n_train=20000, n_test=100, n_features=50, peak_bytes=389824000),
    PredictPoint(n_train=20000, n_test=1000, n_features=50, peak_bytes=1920798208),
    PredictPoint(n_train=1000, n_test=100, n_features=100, peak_bytes=405837824),
    PredictPoint(n_train=1000, n_test=1000, n_features=100, peak_bytes=3575545856),
    PredictPoint(n_train=5000, n_test=100, n_features=100, peak_bytes=405239296),
    PredictPoint(n_train=5000, n_test=1000, n_features=100, peak_bytes=3575804928),
    PredictPoint(n_train=20000, n_test=100, n_features=100, peak_bytes=407184384),
    PredictPoint(n_train=20000, n_test=1000, n_features=100, peak_bytes=3576409600),
    PredictPoint(n_train=1000, n_test=100, n_features=200, peak_bytes=807400960),
    PredictPoint(n_train=1000, n_test=1000, n_features=200, peak_bytes=6754677248),
    PredictPoint(n_train=5000, n_test=100, n_features=200, peak_bytes=807410688),
    PredictPoint(n_train=5000, n_test=1000, n_features=200, peak_bytes=6754936320),
    PredictPoint(n_train=20000, n_test=100, n_features=200, peak_bytes=807975936),
    PredictPoint(n_train=20000, n_test=1000, n_features=200, peak_bytes=6755895296),
    PredictPoint(n_train=1000, n_test=100, n_features=500, peak_bytes=2008236544),
    PredictPoint(n_train=1000, n_test=1000, n_features=500, peak_bytes=6675193856),
    PredictPoint(n_train=5000, n_test=100, n_features=500, peak_bytes=2008526336),
    PredictPoint(n_train=5000, n_test=1000, n_features=500, peak_bytes=6675992576),
    PredictPoint(n_train=20000, n_test=100, n_features=500, peak_bytes=2009817088),
    PredictPoint(n_train=20000, n_test=1000, n_features=500, peak_bytes=6676935680),
    PredictPoint(n_train=1000, n_test=100, n_features=1000, peak_bytes=3082464768),
    PredictPoint(n_train=1000, n_test=1000, n_features=1000, peak_bytes=6683149824),
    PredictPoint(n_train=5000, n_test=100, n_features=1000, peak_bytes=3082723840),
    PredictPoint(n_train=5000, n_test=1000, n_features=1000, peak_bytes=6683408896),
)
