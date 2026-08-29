"""Unit tests for env-driven engine construction (serve/factory.py)."""

import pytest

from tabctx.backends.fake import FakeBackend
from tabctx.serve.factory import (
    BACKEND_ENV_VAR,
    GPU_MEMORY_FRACTION_ENV_VAR,
    ServeSettings,
    build_engine,
    build_estimator,
)


class TestServeSettings:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv(BACKEND_ENV_VAR, raising=False)
        monkeypatch.delenv(GPU_MEMORY_FRACTION_ENV_VAR, raising=False)
        settings = ServeSettings.from_env()
        assert settings.backends == ("tabicl",)
        assert settings.gpu_memory_fraction == 1.0

    def test_fake_backend(self, monkeypatch):
        monkeypatch.setenv(BACKEND_ENV_VAR, "fake")
        assert ServeSettings.from_env().backends == ("fake",)

    def test_backend_whitespace_and_case_tolerated(self, monkeypatch):
        monkeypatch.setenv(BACKEND_ENV_VAR, "  Fake ")
        assert ServeSettings.from_env().backends == ("fake",)

    def test_unknown_backend_rejected(self, monkeypatch):
        monkeypatch.setenv(BACKEND_ENV_VAR, "xgboost")
        with pytest.raises(ValueError, match="xgboost"):
            ServeSettings.from_env()

    def test_tabpfn_is_a_known_backend(self, monkeypatch):
        monkeypatch.setenv(BACKEND_ENV_VAR, "tabpfn")
        assert ServeSettings.from_env().backends == ("tabpfn",)

    @pytest.mark.parametrize("bad", ["0", "-0.5", "1.5", "abc"])
    def test_bad_fraction_rejected(self, monkeypatch, bad):
        monkeypatch.setenv(GPU_MEMORY_FRACTION_ENV_VAR, bad)
        with pytest.raises(ValueError):
            ServeSettings.from_env()

    def test_fraction_parsed(self, monkeypatch):
        monkeypatch.setenv(GPU_MEMORY_FRACTION_ENV_VAR, "0.45")
        assert ServeSettings.from_env().gpu_memory_fraction == 0.45


class TestBuildEstimator:
    def test_fraction_scales_ceiling(self):
        full = build_estimator(ServeSettings(backends=("fake",)))
        half = build_estimator(
            ServeSettings(backends=("fake",), gpu_memory_fraction=0.5)
        )
        assert half.ceiling_bytes() == pytest.approx(
            full.ceiling_bytes() * 0.5, rel=0.01
        )
        assert half.ceiling_bytes() < full.ceiling_bytes()

    def test_fraction_tightens_admission(self):
        # A shape that squeaks past the full-budget gate must be rejected
        # under a small fraction of that budget.
        full = build_estimator(ServeSettings(backends=("fake",)))
        tiny = build_estimator(
            ServeSettings(backends=("fake",), gpu_memory_fraction=0.01)
        )
        n_train, n_features = 5_000, 50
        assert full.admit(n_train, 0, n_features)
        assert not tiny.admit(n_train, 0, n_features)


class TestBuildEngine:
    def test_fake_backend_end_to_end(self):
        built = build_engine(ServeSettings(backends=("fake",)))
        assert isinstance(built.backend, FakeBackend)
        assert "fake" in built.device

        dataset_id = built.engine.fit([[1.0, 2.0], [3.0, 4.0]], ["a", "b"])
        outcome = built.engine.predict(dataset_id, [[5.0, 6.0]])
        assert len(outcome.predictions) == 1

    def test_cache_capacity_matches_estimator_ceiling(self):
        built = build_engine(ServeSettings(backends=("fake",), gpu_memory_fraction=0.5))
        assert built.engine.stats().capacity_bytes == built.estimator.ceiling_bytes()


class TestCalibrationPreload:
    """Builds the TABICL estimator through the factory -- the exact path
    that crashed a live deploy when _preloaded_observations returned a
    raw grid instead of (fit, predict) pairs. Works without tabicl
    installed: the calibration module only needs Observation."""

    def test_tabicl_estimator_builds_with_both_grids(self):
        for mode in ("kv", "repr"):
            est = build_estimator(ServeSettings(backends=("tabicl",), kv_cache=mode))
            assert "preloaded calibration measurement" in est.confidence()
            # A shape inside the measured grid must estimate from the
            # measured peak (well under the formula's number), for both
            # the fit query and the chunking (predict) query.
            fit_est = est.estimate_bytes(50_000, 0, 50)
            assert fit_est < 40_000_000_000
            predict_est = est.estimate_bytes(50_000, 1_000, 50)
            assert predict_est < 10_000_000_000, (
                "predict estimate should come from measured predict peaks, "
                f"got {predict_est} (formula-sized -> 1-row chunking bug)"
            )

    def test_off_mode_and_fake_backend_build_clean(self):
        assert build_estimator(ServeSettings(backends=("tabicl",), kv_cache="off"))
        assert build_estimator(ServeSettings(backends=("fake",)))
