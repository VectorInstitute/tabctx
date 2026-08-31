"""Unit tests for env-driven engine construction (serve/factory.py)."""

import sys

import pytest

from tabctx.backends.fake import FakeBackend
from tabctx.cache.spill import DiskSpillStore
from tabctx.serve.factory import (
    BACKEND_ENV_VAR,
    BATCH_WINDOW_MS_ENV_VAR,
    GPU_MEMORY_FRACTION_ENV_VAR,
    KV_CACHE_ENV_VAR,
    MAX_UPLOAD_BYTES_ENV_VAR,
    SPILL_CAPACITY_ENV_VAR,
    SPILL_DIR_ENV_VAR,
    UPLOAD_TTL_S_ENV_VAR,
    ServeSettings,
    _build_backend,
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

    def test_duplicate_backend_rejected(self, monkeypatch):
        monkeypatch.setenv(BACKEND_ENV_VAR, "fake,fake")
        with pytest.raises(ValueError, match="twice"):
            ServeSettings.from_env()

    @pytest.mark.parametrize("bad", ["repr2", "none", ""])
    def test_bad_kv_cache_mode_rejected(self, monkeypatch, bad):
        monkeypatch.setenv(KV_CACHE_ENV_VAR, bad)
        with pytest.raises(ValueError, match="not a known mode"):
            ServeSettings.from_env()

    def test_kv_cache_mode_parsed(self, monkeypatch):
        monkeypatch.setenv(KV_CACHE_ENV_VAR, " REPR ")
        assert ServeSettings.from_env().kv_cache == "repr"

    def test_bad_batch_window_not_a_float_rejected(self, monkeypatch):
        monkeypatch.setenv(BATCH_WINDOW_MS_ENV_VAR, "abc")
        with pytest.raises(ValueError, match="not a float"):
            ServeSettings.from_env()

    def test_negative_batch_window_rejected(self, monkeypatch):
        monkeypatch.setenv(BATCH_WINDOW_MS_ENV_VAR, "-1")
        with pytest.raises(ValueError, match=">= 0"):
            ServeSettings.from_env()

    def test_batch_window_zero_disables_coalescing_is_valid(self, monkeypatch):
        monkeypatch.setenv(BATCH_WINDOW_MS_ENV_VAR, "0")
        assert ServeSettings.from_env().batch_window_ms == 0

    def test_non_numeric_upload_bytes_or_ttl_rejected(self, monkeypatch):
        monkeypatch.setenv(MAX_UPLOAD_BYTES_ENV_VAR, "not-a-number")
        with pytest.raises(ValueError, match="must be numeric"):
            ServeSettings.from_env()

    @pytest.mark.parametrize(
        "env_var,value",
        [(MAX_UPLOAD_BYTES_ENV_VAR, "0"), (UPLOAD_TTL_S_ENV_VAR, "-1")],
    )
    def test_non_positive_upload_bytes_or_ttl_rejected(
        self, monkeypatch, env_var, value
    ):
        monkeypatch.setenv(env_var, value)
        with pytest.raises(ValueError, match="must be positive"):
            ServeSettings.from_env()

    def test_non_int_spill_capacity_rejected(self, monkeypatch):
        monkeypatch.setenv(SPILL_CAPACITY_ENV_VAR, "not-an-int")
        with pytest.raises(ValueError, match="must be an int"):
            ServeSettings.from_env()

    def test_non_positive_spill_capacity_rejected(self, monkeypatch):
        monkeypatch.setenv(SPILL_CAPACITY_ENV_VAR, "0")
        with pytest.raises(ValueError, match="must be positive"):
            ServeSettings.from_env()

    def test_spill_dir_parsed(self, monkeypatch, tmp_path):
        monkeypatch.setenv(SPILL_DIR_ENV_VAR, str(tmp_path))
        assert ServeSettings.from_env().spill_dir == str(tmp_path)

    def test_spill_dir_unset_is_none(self, monkeypatch):
        monkeypatch.delenv(SPILL_DIR_ENV_VAR, raising=False)
        assert ServeSettings.from_env().spill_dir is None


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

    def test_spill_dir_wires_a_disk_spill_store_into_the_cache(self, tmp_path):
        built = build_engine(ServeSettings(backends=("fake",), spill_dir=str(tmp_path)))
        assert isinstance(built.spill_store, DiskSpillStore)
        assert built.spill_store._dir == tmp_path
        # The SAME store instance backs the engine's own context cache
        # (not just returned alongside it) -- that's the wiring this
        # factory code exists to do.
        assert built.engine._cache._spill is built.spill_store

    def test_no_spill_dir_leaves_spill_store_none(self):
        built = build_engine(ServeSettings(backends=("fake",)))
        assert built.spill_store is None
        assert built.engine._cache._spill is None

    def test_dispatches_non_fake_kinds_to_the_real_backend_path(self, monkeypatch):
        # The real-backend path (_build_real_backend) needs torch/tabicl/
        # tabpfn, which are deliberately not a dev-group dependency (see
        # backends/tabicl.py and backends/tabpfn.py: GPU or licensed
        # weights only, tested locally/on GPU rigs, not in CI) -- and may
        # or may not happen to be installed on any given machine running
        # this suite. Stub it out so this test asserts only what
        # _build_backend itself is responsible for: dispatching non-fake
        # kinds there instead of silently no-op'ing, regardless of what's
        # installed.
        calls = []
        monkeypatch.setattr(
            "tabctx.serve.factory._build_real_backend",
            lambda kind, settings: calls.append((kind, settings)) or (object(), "cpu"),
        )
        settings = ServeSettings(backends=("tabicl",))
        _build_backend("tabicl", settings)
        assert calls == [("tabicl", settings)]


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

    def test_missing_calibration_module_degrades_to_no_preload(self, monkeypatch):
        # Pre-calibration trees (or a build without the generated grid
        # module) must not crash -- just start with no preload and learn
        # from the replica's own fits. Force ImportError on `from
        # tabctx.memory import calibration_tabicl_a100`: both the
        # sys.modules entry AND the parent package's cached attribute
        # must be cleared, or `from X import Y`'s getattr(X, Y)
        # shortcut finds the real module other tests already imported.
        import tabctx.memory as memory_pkg

        monkeypatch.delattr(memory_pkg, "calibration_tabicl_a100", raising=False)
        monkeypatch.setitem(sys.modules, "tabctx.memory.calibration_tabicl_a100", None)
        est = build_estimator(ServeSettings(backends=("tabicl",)))
        assert "0 preloaded calibration measurement" in est.confidence()
