"""GENERATED calibration grids -- do not hand-edit.

Produced by benchmarks/gen_calibration_data.py from
benchmarks/calibrate_memory.py sweeps on real hardware. Each Observation
records the measured PEAK fit bytes (transient high-water allocation
delta during fit -- the admission-relevant quantity; see
backends/base.py) for one (n_train, n_features) shape; resident context
size and fit time ride along as comments. OOM boundary shapes are listed
per grid: nothing beyond them has ever succeeded on this hardware.

These grids are preloaded into AdaptiveMemoryEstimator by
serve/factory.py so admission decisions rest on measurements from the
first request onward. They are A100-40GB + TabICLv2 measurements:
different GPUs/models need their own sweep (same script).
"""

from __future__ import annotations

from tabctx.memory.adaptive import Observation

# mode='kv': measured 2026-08-29T14:20:30.255378+00:00 on NVIDIA A100-SXM4-40GB (42404806656 bytes), torch 2.9.1+cu129.
# First OOM per feature count: 200000x10, 200000x50.
A100_40GB_TABICL_KV_PEAK_GRID: tuple[Observation, ...] = (
    Observation(
        n_train=1000, n_features=10, real_bytes=670151168
    ),  # fit 0.19s, resident 446627840
    Observation(
        n_train=5000, n_features=10, real_bytes=1311535104
    ),  # fit 0.66s, resident 797310976
    Observation(
        n_train=20000, n_features=10, real_bytes=5424395776
    ),  # fit 1.41s, resident 3443785728
    Observation(
        n_train=50000, n_features=10, real_bytes=12826331136
    ),  # fit 4.94s, resident 7891320832
    Observation(
        n_train=100000, n_features=10, real_bytes=24628596224
    ),  # fit 15.09s, resident 14760148992
    Observation(
        n_train=1000, n_features=50, real_bytes=1607126016
    ),  # fit 0.48s, resident 552653824
    Observation(
        n_train=5000, n_features=50, real_bytes=3946299392
    ),  # fit 1.02s, resident 796131328
    Observation(
        n_train=20000, n_features=50, real_bytes=15991590400
    ),  # fit 3.65s, resident 3480748032
    Observation(
        n_train=50000, n_features=50, real_bytes=27332870656
    ),  # fit 10.45s, resident 7921738752
    Observation(
        n_train=100000, n_features=50, real_bytes=31023448576
    ),  # fit 25.57s, resident 14792916992
    Observation(
        n_train=1000, n_features=100, real_bytes=1676122624
    ),  # fit 0.7s, resident 328499200
    Observation(
        n_train=5000, n_features=100, real_bytes=7492955136
    ),  # fit 1.84s, resident 971374592
    Observation(
        n_train=20000, n_features=100, real_bytes=25935502848
    ),  # fit 6.99s, resident 3522428928
    Observation(
        n_train=50000, n_features=100, real_bytes=32045499904
    ),  # fit 17.69s, resident 7959085056
    Observation(
        n_train=100000, n_features=100, real_bytes=24915988992
    ),  # fit 54.74s, resident 14832762880
    Observation(
        n_train=1000, n_features=200, real_bytes=3259352064
    ),  # fit 1.65s, resident 522267648
    Observation(
        n_train=5000, n_features=200, real_bytes=14302022144
    ),  # fit 3.66s, resident 1042306048
    Observation(
        n_train=20000, n_features=200, real_bytes=30111151616
    ),  # fit 13.0s, resident 3601072128
    Observation(
        n_train=50000, n_features=200, real_bytes=21795407360
    ),  # fit 66.37s, resident 8037728256
)

# Measured PREDICT peaks for the same shapes, at n_test=1000 test rows (chunking's quantity; see memory/adaptive.py).
A100_40GB_TABICL_KV_PREDICT_PEAK_GRID: tuple[Observation, ...] = (
    Observation(n_train=1000, n_features=10, real_bytes=290993664),
    Observation(n_train=5000, n_features=10, real_bytes=176049152),
    Observation(n_train=20000, n_features=10, real_bytes=173296640),
    Observation(n_train=50000, n_features=10, real_bytes=175361024),
    Observation(n_train=100000, n_features=10, real_bytes=173296640),
    Observation(n_train=1000, n_features=50, real_bytes=1112127488),
    Observation(n_train=5000, n_features=50, real_bytes=669047808),
    Observation(n_train=20000, n_features=50, real_bytes=667376640),
    Observation(n_train=50000, n_features=50, real_bytes=670161920),
    Observation(n_train=100000, n_features=50, real_bytes=670161920),
    Observation(n_train=1000, n_features=100, real_bytes=1290022400),
    Observation(n_train=5000, n_features=100, real_bytes=1288646144),
    Observation(n_train=20000, n_features=100, real_bytes=1290257408),
    Observation(n_train=50000, n_features=100, real_bytes=1289705984),
    Observation(n_train=100000, n_features=100, real_bytes=1288646144),
    Observation(n_train=1000, n_features=200, real_bytes=2521649664),
    Observation(n_train=5000, n_features=200, real_bytes=2520384512),
    Observation(n_train=20000, n_features=200, real_bytes=2522380288),
    Observation(n_train=50000, n_features=200, real_bytes=2521649664),
)

# mode='repr': measured 2026-08-29T15:08:02.156104+00:00 on NVIDIA A100-SXM4-40GB (42404806656 bytes), torch 2.9.1+cu129.
A100_40GB_TABICL_REPR_PEAK_GRID: tuple[Observation, ...] = (
    Observation(
        n_train=20000, n_features=10, real_bytes=3858774528
    ),  # fit 1.07s, resident 185860096
    Observation(
        n_train=100000, n_features=10, real_bytes=19070516736
    ),  # fit 3.32s, resident 748814336
    Observation(
        n_train=300000, n_features=10, real_bytes=28429872640
    ),  # fit 10.46s, resident 4516528640
    Observation(
        n_train=500000, n_features=10, real_bytes=32279461376
    ),  # fit 34.11s, resident 5745410048
    Observation(
        n_train=20000, n_features=50, real_bytes=10537553408
    ),  # fit 2.75s, resident -3858235392
    Observation(
        n_train=100000, n_features=50, real_bytes=32657339904
    ),  # fit 13.8s, resident 1598947328
    Observation(
        n_train=200000, n_features=50, real_bytes=20197451776
    ),  # fit 64.24s, resident 2500067328
    Observation(
        n_train=20000, n_features=100, real_bytes=24121433600
    ),  # fit 6.33s, resident -1187512320
    Observation(
        n_train=100000, n_features=100, real_bytes=21172241408
    ),  # fit 43.35s, resident 1556348928
    Observation(
        n_train=1000, n_features=200, real_bytes=2266317824
    ),  # fit 1.22s, resident -563740672
    Observation(
        n_train=20000, n_features=200, real_bytes=29807771136
    ),  # fit 11.6s, resident 479920128
    Observation(
        n_train=50000, n_features=200, real_bytes=21090801152
    ),  # fit 42.34s, resident 815792128
)

# Measured PREDICT peaks for the same shapes, at n_test=1000 test rows (chunking's quantity; see memory/adaptive.py).
A100_40GB_TABICL_REPR_PREDICT_PEAK_GRID: tuple[Observation, ...] = (
    Observation(n_train=20000, n_features=10, real_bytes=1321876992),
    Observation(n_train=100000, n_features=10, real_bytes=6243958272),
    Observation(n_train=300000, n_features=10, real_bytes=14568504320),
    Observation(n_train=500000, n_features=10, real_bytes=10885301248),
    Observation(n_train=20000, n_features=50, real_bytes=1322516992),
    Observation(n_train=100000, n_features=50, real_bytes=6244598272),
    Observation(n_train=200000, n_features=50, real_bytes=12382601728),
    Observation(n_train=20000, n_features=100, real_bytes=1323316736),
    Observation(n_train=100000, n_features=100, real_bytes=6243431936),
    Observation(n_train=1000, n_features=200, real_bytes=2520830464),
    Observation(n_train=20000, n_features=200, real_bytes=2520176640),
    Observation(n_train=50000, n_features=200, real_bytes=3168116736),
)
