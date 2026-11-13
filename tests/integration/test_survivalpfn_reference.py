"""SurvivalPFNBackend against SurvivalPFN's own published benchmark.

The SurvivalPFN authors' reference: their paper's held-out C-index per
dataset (results/paper/benchmark_results.csv in rgklab/SurvivalPFN), from
10 random 70/30 splits (seeds 0-9) of each dataset, scored on the
predicted median event time, without temperature calibration. Their own
split code (survivalpfn.inference.data.SurvivalBenchmarkDataset) makes the
same splits here, from the public SurvSet datasets.

On every split:
- tabctx's median is SurvivalPFN's median, wherever tabctx gives one,
  and tabctx gives none (null) exactly where that median is past the
  longest training time (nothing is estimated past the follow-up);
- SurvivalPFN's median reproduces the published C-index (so the
  checkpoint, the package and the splits are the paper's);
- ranking rows by tabctx's RMST up to the longest training time, which
  is always defined, scores within a small margin of it. With heavy
  censoring most medians are null (zinc: 81% censored), so RMST is how
  a caller ranks rows.

Needs the model packages and the reference data (`uv sync --group
survivalpfn --group survivalpfn-reference`) and the pinned weights; it
runs on CPU in a few seconds.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("survivalpfn")
pytest.importorskip("torch")
pytest.importorskip("SurvSet")
pytest.importorskip("h5py")
lifelines_utils = pytest.importorskip("lifelines.utils")

from survivalpfn import SurvivalEstimator  # noqa: E402
from survivalpfn.inference.data import SurvivalBenchmarkDataset  # noqa: E402

from tabctx.backends.survivalpfn import SurvivalPFNBackend  # noqa: E402
from tabctx.tasks import pack_targets  # noqa: E402

# Published mean C-index (rgklab/SurvivalPFN@960f8c6,
# results/paper/benchmark_results.csv). Small public datasets with
# little to heavy censoring: veteran 7%, mgus 7%, whas500 62%,
# stagec 63%, Melanoma 72%, zinc 81%.
PUBLISHED_CI = {
    "veteran": 0.7274,
    "mgus": 0.7163,
    "whas500": 0.7296,
    "stagec": 0.7015,
    "Melanoma": 0.7000,
    "zinc": 0.7768,
}
SPLITS = 10
# The published value is a mean over 10 splits: SurvivalPFN's own run
# reproduces it to rounding; RMST is a different score than the median,
# so it may differ a little (measured: at most 0.008).
SAME = 0.002
CLOSE = 0.015


@pytest.fixture(scope="module")
def backend():
    b = SurvivalPFNBackend(device="cpu", context_size=1_000_000)
    try:
        b.warmup()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"SurvivalPFN weights unavailable: {e}")
    return b


def splits(name: str):
    try:
        data = SurvivalBenchmarkDataset(name, 0.7, 0.0, 0.3, preprocess=False, seed=0)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"benchmark dataset {name} unavailable: {e}")
    return (data() for _ in range(SPLITS))  # each call: the next seed


@pytest.mark.parametrize("name", sorted(PUBLISHED_CI))
def test_matches_the_published_benchmark(backend, name):
    concordance = lifelines_utils.concordance_index
    theirs, ours = [], []
    for split in splits(name):
        upstream = SurvivalEstimator(
            device="cpu", model_path=None, icl_model=backend._shared_network()
        )
        upstream.fit(X=split["X_train"], delta=split["delta_train"], T=split["T_train"])
        median = (
            upstream.predict_event_distribution(split["X_test"]).median().cpu().numpy()
        )

        longest = float(split["T_train"].max())
        context = backend.fit(
            split["X_train"],
            pack_targets("survival", split["T_train"], split["delta_train"]),
            "survival",
        )
        out = backend.predict(context, split["X_test"], rmst_horizon=longest)

        given = np.array([v is not None for v in out.predictions])
        np.testing.assert_allclose(
            np.array([v for v in out.predictions if v is not None], dtype=float),
            median[given],
            rtol=1e-5,
        )
        assert (median[~given] >= longest * (1 - 1e-6)).all()
        assert (median[given] <= longest * (1 + 1e-6)).all()

        theirs.append(concordance(split["T_test"], median, split["delta_test"]))
        rmst = np.array(out.rmst, dtype=float)
        assert np.isfinite(rmst).all()
        ours.append(concordance(split["T_test"], rmst, split["delta_test"]))

    assert np.mean(theirs) == pytest.approx(PUBLISHED_CI[name], abs=SAME)
    assert np.mean(ours) == pytest.approx(PUBLISHED_CI[name], abs=CLOSE)
