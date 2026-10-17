"""Unit tests for env-driven engine construction (serve/factory.py)."""

import sys

import pytest

from tabctx.backends.fake import FakeBackend
from tabctx.cache.spill import DiskSpillStore
from tabctx.serve.factory import (
    BACKEND_ENV_VAR,
    BATCH_WINDOW_MS_ENV_VAR,
    DEVICES_ENV_VAR,
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


class TestDeviceCapacity:
    def test_capacity_scales_both_ceilings_proportionally(self):
        a100 = build_estimator(ServeSettings(backends=("fake",)))
        l4 = build_estimator(
            ServeSettings(backends=("fake",)), gpu_capacity_bytes=24 * 1024**3
        )
        ratio = (24 * 1024**3) / a100._fallback.gpu_capacity_bytes
        assert l4._fallback.gpu_capacity_bytes == 24 * 1024**3
        assert l4.ceiling_bytes() == pytest.approx(
            a100.ceiling_bytes() * ratio, rel=0.01
        )
        # And the usage-aware admission headroom shrinks with the card.
        assert l4.admission_headroom_bytes(0) < a100.admission_headroom_bytes(0)

    def test_none_capacity_keeps_the_a100_defaults(self):
        from tabctx.memory.estimator import DEFAULT_GPU_CAPACITY_BYTES

        est = build_estimator(
            ServeSettings(backends=("fake",)), gpu_capacity_bytes=None
        )
        assert est._fallback.gpu_capacity_bytes == DEFAULT_GPU_CAPACITY_BYTES

    def test_build_engine_uses_detected_capacity_for_real_backends(self, monkeypatch):
        monkeypatch.setattr(
            "tabctx.serve.factory._build_real_backend",
            lambda kind, settings, device: FakeBackend(name=f"{kind}-model"),
        )
        monkeypatch.setattr(
            "tabctx.serve.factory.detect_gpu_capacity_bytes",
            lambda devices: 24 * 1024**3,
        )
        monkeypatch.setattr("tabctx.serve.factory._visible_cuda_count", lambda: 1)
        built = build_engine(ServeSettings(backends=("tabicl",), devices=("cuda",)))
        assert built.gpu_capacity_bytes == 24 * 1024**3
        assert built.estimator._fallback.gpu_capacity_bytes == 24 * 1024**3

    def test_fake_backend_never_probes_the_device(self, monkeypatch):
        def boom(*_):
            raise AssertionError("fake backend must not touch torch/cuda")

        monkeypatch.setattr("tabctx.serve.factory.detect_gpu_capacity_bytes", boom)
        built = build_engine(ServeSettings(backends=("fake",)))
        assert built.gpu_capacity_bytes is None

    def test_detect_returns_none_without_a_gpu(self, monkeypatch):
        from tabctx.serve.factory import detect_gpu_capacity_bytes

        # Whether or not torch is installed here, no CUDA device is
        # visible on the CI/dev machines this runs on.
        try:
            import torch

            monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        except ImportError:
            pass
        assert detect_gpu_capacity_bytes() is None


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

    def test_default_model_is_the_first_listed_kind_by_model_id(self, monkeypatch):
        # Backends are keyed by model id ("tabicl-v2"), not kind
        # ("tabicl"); the default must still be the first-listed kind's
        # model, resolved through that mapping.
        monkeypatch.setattr(
            "tabctx.serve.factory._build_real_backend",
            lambda kind, settings, device: FakeBackend(name=f"{kind}-model"),
        )
        built = build_engine(
            ServeSettings(backends=("tabpfn", "fake"), devices=("cpu",))
        )
        assert built.default == "tabpfn-model"
        assert built.engine.default_backend == "tabpfn-model"
        assert built.engine.backend_names == ["tabpfn-model", "fake"]

    def test_settings_defaults_match_from_env_defaults(self, monkeypatch):
        for var in (
            BACKEND_ENV_VAR,
            GPU_MEMORY_FRACTION_ENV_VAR,
            KV_CACHE_ENV_VAR,
            BATCH_WINDOW_MS_ENV_VAR,
            MAX_UPLOAD_BYTES_ENV_VAR,
            UPLOAD_TTL_S_ENV_VAR,
            SPILL_DIR_ENV_VAR,
            SPILL_CAPACITY_ENV_VAR,
            DEVICES_ENV_VAR,
            "TABCTX_KV_PRECISION",
        ):
            monkeypatch.delenv(var, raising=False)
        assert ServeSettings.from_env() == ServeSettings()

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
            lambda kind, settings, device: (
                calls.append((kind, settings, device)) or object()
            ),
        )
        settings = ServeSettings(backends=("tabicl",))
        _build_backend("tabicl", settings, "cuda:1")
        assert calls == [("tabicl", settings, "cuda:1")]


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


class TestDevices:
    def test_auto_is_the_default(self, monkeypatch):
        monkeypatch.delenv(DEVICES_ENV_VAR, raising=False)
        assert ServeSettings.from_env().devices == ()
        monkeypatch.setenv(DEVICES_ENV_VAR, " AUTO ")
        assert ServeSettings.from_env().devices == ()

    def test_explicit_list_is_parsed(self, monkeypatch):
        monkeypatch.setenv(DEVICES_ENV_VAR, "cuda:0, cuda:2")
        assert ServeSettings.from_env().devices == ("cuda:0", "cuda:2")

    @pytest.mark.parametrize(
        "raw",
        ["gpu0", "cuda:x", "cuda:0,cuda:0", "tpu", "cuda,cuda:0", "cpu,cuda:0"],
    )
    def test_bad_lists_are_rejected(self, monkeypatch, raw):
        monkeypatch.setenv(DEVICES_ENV_VAR, raw)
        with pytest.raises(ValueError, match=DEVICES_ENV_VAR):
            ServeSettings.from_env()

    @pytest.mark.parametrize(
        ("visible", "expected"),
        [
            (0, ("cpu",)),
            (1, ("cuda",)),
            (4, ("cuda:0", "cuda:1", "cuda:2", "cuda:3")),
        ],
    )
    def test_auto_uses_every_visible_gpu(self, monkeypatch, visible, expected):
        from tabctx.serve.factory import resolve_devices

        monkeypatch.setattr("tabctx.serve.factory._visible_cuda_count", lambda: visible)
        assert resolve_devices(ServeSettings(backends=("tabicl",))) == expected

    def test_explicit_devices_win_and_fake_auto_is_cpu(self, monkeypatch):
        from tabctx.serve.factory import resolve_devices

        monkeypatch.setattr("tabctx.serve.factory._visible_cuda_count", lambda: 4)
        assert resolve_devices(
            ServeSettings(backends=("tabicl",), devices=("cuda:1",))
        ) == ("cuda:1",)
        assert resolve_devices(ServeSettings(backends=("fake",))) == ("cpu",)

    def test_single_device_builds_a_plain_engine(self):
        from tabctx.engine import TabctxEngine

        built = build_engine(ServeSettings(backends=("fake",)))
        assert type(built.engine) is TabctxEngine
        assert built.devices == ("cpu",)

    def test_several_devices_build_one_engine_each(self, monkeypatch, tmp_path):
        from tabctx.multidevice import MultiDeviceEngine

        created = []
        monkeypatch.setattr(
            "tabctx.serve.factory._build_real_backend",
            lambda kind, settings, device: (
                created.append(device) or FakeBackend(name=f"{kind}-model")
            ),
        )
        monkeypatch.setattr(
            "tabctx.serve.factory.detect_gpu_capacity_bytes",
            lambda devices: 48 * 1024**3,
        )
        monkeypatch.setattr("tabctx.serve.factory._visible_cuda_count", lambda: 4)
        built = build_engine(
            ServeSettings(
                backends=("tabicl", "tabpfn"),
                devices=("cuda:0", "cuda:1", "cuda:2"),
                spill_dir=str(tmp_path),
                spill_capacity_bytes=30 * 1024**3,
            )
        )
        assert isinstance(built.engine, MultiDeviceEngine)
        # Every model gets a backend instance on every device.
        assert created == ["cuda:0", "cuda:0", "cuda:1", "cuda:1", "cuda:2", "cuda:2"]
        assert built.engine.backend_names == ["tabicl-model", "tabpfn-model"]
        assert built.default == "tabicl-model"
        # Estimators are shared across devices (identical cards).
        shard_estimators = {
            id(s.estimator_for("tabicl-model")) for s in built.engine._shards
        }
        assert len(shard_estimators) == 1
        # Each device has its own cache, budgeted per device, and spill dir.
        caps = [s.capacity_bytes for s in built.engine.device_stats()]
        assert len(set(caps)) == 1 and caps[0] == built.estimator.ceiling_bytes()
        assert built.spill_store is None and len(built.spill_stores) == 3
        # The spill capacity setting is the replica's total, split evenly.
        assert {s.stats()["spill_capacity_bytes"] for s in built.spill_stores} == {
            10 * 1024**3
        }
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "cuda_0",
            "cuda_1",
            "cuda_2",
        ]
        assert built.device == "cuda:0,cuda:1,cuda:2"


class TestDeviceValidation:
    def test_gpu_index_beyond_what_the_process_sees_is_rejected(self, monkeypatch):
        from tabctx.serve.factory import resolve_devices

        monkeypatch.setattr("tabctx.serve.factory._visible_cuda_count", lambda: 2)
        with pytest.raises(ValueError, match="sees 2 GPU"):
            resolve_devices(
                ServeSettings(backends=("tabicl",), devices=("cuda:0", "cuda:3"))
            )

    def test_explicit_cpu_never_gets_a_gpu_budget(self, monkeypatch):
        from tabctx.serve.factory import detect_gpu_capacity_bytes

        torch = pytest.importorskip("torch")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert detect_gpu_capacity_bytes(("cpu",)) is None

    def test_spill_layout_change_is_reported(self, tmp_path, caplog):
        (tmp_path / "abcd.meta.json").write_text("{}")
        with caplog.at_level("WARNING"):
            build_engine(
                ServeSettings(
                    backends=("fake",),
                    devices=("cuda:0", "cuda:1"),
                    spill_dir=str(tmp_path),
                )
            )
        assert "will not be restored" in caplog.text


class TestKvPrecision:
    def test_default_and_parsing(self, monkeypatch):
        monkeypatch.delenv("TABCTX_KV_PRECISION", raising=False)
        assert ServeSettings.from_env().kv_precision == "int8"
        monkeypatch.setenv("TABCTX_KV_PRECISION", " INT8 ")
        assert ServeSettings.from_env().kv_precision == "int8"
        monkeypatch.setenv("TABCTX_KV_PRECISION", "fp4")
        with pytest.raises(ValueError, match="TABCTX_KV_PRECISION"):
            ServeSettings.from_env()
