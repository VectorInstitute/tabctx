from tabctx.memory.adaptive import AdaptiveMemoryEstimator, Observation
from tabctx.memory.calibration_data import A100_40GB_TABICL_CALIBRATION
from tabctx.memory.estimator import PowerLawMemoryEstimator


def make_adaptive(**kwargs):
    fallback = PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION)
    return AdaptiveMemoryEstimator(fallback=fallback, **kwargs)


def test_observation_dominates():
    obs = Observation(n_train=100, n_features=10, real_bytes=1000)
    assert obs.dominates(50, 5)
    assert obs.dominates(100, 10)  # equal counts as dominating
    assert not obs.dominates(200, 5)
    assert not obs.dominates(50, 20)


def test_falls_back_to_static_with_no_observations():
    est = make_adaptive()
    fallback = PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION)
    assert est.estimate_bytes(10_000, 0, 50) == fallback.estimate_bytes(10_000, 0, 50)


def test_uses_real_observation_when_it_dominates_the_query():
    est = make_adaptive(safety_margin=1.5)
    est.record_observation(n_train=17_000, n_features=50, real_bytes=110_000_000)
    # Query shape is smaller in both dims -> the observation safely bounds it.
    got = est.estimate_bytes(10_000, 0, 50)
    assert got == round(110_000_000 * 1.5)
    # And it's far tighter than the (much more conservative) static formula.
    fallback_est = PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION).estimate_bytes(
        10_000, 0, 50
    )
    assert got < fallback_est


def test_does_not_use_observation_that_does_not_dominate():
    est = make_adaptive()
    # Observation is smaller in n_features than the query -> must not apply.
    est.record_observation(n_train=50_000, n_features=10, real_bytes=50_000_000)
    fallback = PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION)
    got = est.estimate_bytes(10_000, 0, 50)
    assert got == fallback.estimate_bytes(10_000, 0, 50)


def test_picks_tightest_dominating_observation_not_just_any():
    est = make_adaptive(safety_margin=1.0)
    est.record_observation(n_train=100_000, n_features=100, real_bytes=999_000_000)
    est.record_observation(n_train=20_000, n_features=60, real_bytes=120_000_000)
    # Both dominate (10_000, 50), but the second is the tighter (smaller-cells) bound.
    got = est.estimate_bytes(10_000, 0, 50)
    assert got == 120_000_000


def test_predict_time_queries_never_use_observations():
    est = make_adaptive()
    est.record_observation(
        n_train=17_000, n_features=50, real_bytes=1
    )  # absurdly low on purpose
    fallback = PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION)
    # n_test > 0 must always defer to the fallback, even with a dominating
    # observation on record -- predict-time active memory is a different
    # quantity we have no measurement of.
    got = est.estimate_bytes(10_000, 500, 50)
    assert got == fallback.estimate_bytes(10_000, 500, 50)


def test_admit_uses_adaptive_estimate():
    est = make_adaptive(safety_margin=1.0)
    huge_train, huge_features = 900_000, 50
    fallback = PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION)
    assert (
        fallback.admit(huge_train, 0, huge_features) is False
    )  # rejected by the static formula
    # Record a real observation dominating that shape, comfortably under the ceiling.
    est.record_observation(
        n_train=huge_train, n_features=huge_features, real_bytes=200_000_000
    )
    assert est.admit(huge_train, 0, huge_features) is True


def test_observations_are_bounded_fifo():
    est = make_adaptive(max_observations=3)
    for i in range(5):
        est.record_observation(n_train=1000 + i, n_features=10, real_bytes=1_000_000)
    assert len(est._observations) == 3
    # Oldest (n_train=1000, 1001) should have been dropped.
    assert all(o.n_train >= 1002 for o in est._observations)


def test_confidence_reports_observation_count():
    est = make_adaptive()
    assert "0 real operational" in est.confidence()
    est.record_observation(n_train=1000, n_features=10, real_bytes=1_000_000)
    assert "1 real operational" in est.confidence()
