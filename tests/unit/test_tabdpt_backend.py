"""TabDPTBackend against stand-ins for torch and tabdpt, so its own logic
runs in the unit environment (neither is installed there). The real model
is tested in tests/integration/test_tabdpt_real.py."""

import contextlib
import sys
import types

import numpy as np
import pytest

from tabctx.client import KNOWN_MODELS
from tabctx.errors import InvalidInputError

LOADS: list[str] = []


class Network:
    use_flash = False


class _Estimator:
    def __init__(self, device=None, compile=True, verbose=True):
        self.device = device
        self.use_flash = device == "cuda"  # TabDPT's own rule
        self._load_model()
        self.calls: list[dict] = []

    def _load_model(self):
        LOADS.append(self.device)
        self.model = Network()
        self.model.use_flash = self.use_flash

    def fit(self, X, y):
        if len(np.unique(y)) < 2 and isinstance(self, TabDPTClassifier):
            raise AssertionError("Number of classes must be greater than 1")
        self.X_train, self.y_train, self.projection = X, y, None

    def _remember(self, X, kwargs):
        if np.isinf(X).any():
            raise ValueError("Input X contains infinity")
        self.X_test = X
        self.calls.append(kwargs)


class TabDPTClassifier(_Estimator):
    def ensemble_predict_proba(self, X, **kwargs):
        self._remember(X, kwargs)
        k = int(self.y_train.max()) + 1
        proba = np.zeros((len(X), k))
        proba[:, int(self.y_train[0])] = 1.0  # the first training row's class
        return proba


class TabDPTRegressor(_Estimator):
    def predict(self, X, output_type="mean", **kwargs):
        self._remember(X, {**kwargs, "output_type": output_type})
        return {"mean": X[:, 0], "logits": None, "borders": None}


def distribution_mean(full):
    return full["mean"]


def distribution_quantiles(full, levels):
    assert all(isinstance(q, float) for q in levels)
    return [full["mean"] + q for q in levels]


@pytest.fixture
def backend_cls(monkeypatch):
    """TabDPTBackend with torch and tabdpt replaced by the stand-ins."""
    LOADS.clear()
    cuda = types.SimpleNamespace(
        init=lambda: None,
        is_available=lambda: False,
        device=lambda d: contextlib.nullcontext(),
        OutOfMemoryError=type("OutOfMemoryError", (RuntimeError,), {}),
    )
    torch = types.SimpleNamespace(cuda=cuda, device=lambda d: d)
    tabdpt = types.ModuleType("tabdpt")
    tabdpt.TabDPTClassifier, tabdpt.TabDPTRegressor = TabDPTClassifier, TabDPTRegressor
    estimator = types.ModuleType("tabdpt.estimator")
    estimator._VERSION = "1_3"
    bars = types.ModuleType("tabdpt.bar_distribution")
    bars.distribution_mean, bars.distribution_quantiles = (
        distribution_mean,
        distribution_quantiles,
    )
    tabdpt.estimator, tabdpt.bar_distribution = estimator, bars
    for name, module in {
        "torch": torch,
        "tabdpt": tabdpt,
        "tabdpt.estimator": estimator,
        "tabdpt.bar_distribution": bars,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    from tabctx.backends.tabdpt import TabDPTBackend

    return TabDPTBackend


X = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 1.0], [4.0, 1.0]])


def test_model_id_follows_the_checkpoint(backend_cls):
    backend = backend_cls(device="cpu")
    assert backend.name == "tabdpt-1.3" and backend.name in KNOWN_MODELS


def test_one_network_per_device_for_every_fit_and_task(backend_cls):
    backend = backend_cls(device="cpu")
    backend.warmup()
    a = backend.fit(X, [1.0, 2.0, 3.0, 4.0], "regression")
    b = backend.fit(X, ["x", "y", "x", "y"], "classification")
    assert LOADS == ["cpu"]  # loaded once
    assert a.estimator.model is b.estimator.model
    assert a.estimator is not b.estimator


def test_flash_attention_on_any_cuda_device(backend_cls):
    """TabDPT enables it only for the string "cuda"; multi-GPU replicas
    name devices "cuda:N"."""
    context = backend_cls(device="cuda:1").fit(X, [1.0, 2.0, 3.0, 4.0], "regression")
    assert context.estimator.use_flash and context.estimator.model.use_flash
    cpu = backend_cls(device="cpu").fit(X, [1.0, 2.0, 3.0, 4.0], "regression")
    assert not cpu.estimator.model.use_flash


def test_numeric_labels_keep_numeric_order(backend_cls):
    backend = backend_cls(device="cpu")
    context = backend.fit(X, [10, 2, 1, 2], "classification")
    assert context.classes == ["1", "2", "10"]
    out = backend.predict(context, X[:2], return_proba=True)
    assert out.predictions == ["10", "10"] and out.classes == ["1", "2", "10"]


def test_every_predict_samples_a_bounded_context_with_a_fixed_seed(backend_cls):
    backend = backend_cls(device="cpu", context_size=512, n_ensembles=2)
    for y, task in (
        ([1.0, 2.0, 3.0, 4.0], "regression"),
        (["x", "y"] * 2, "classification"),
    ):
        context = backend.fit(X, y, task)
        backend.predict(context, X)
        call = context.estimator.calls[-1]
        assert (call["context_size"], call["n_ensembles"], call["seed"]) == (512, 2, 0)
        assert context.estimator.X_test is None  # not kept on the cached context


def test_the_mean_is_the_same_with_or_without_quantiles(backend_cls):
    backend = backend_cls(device="cpu")
    context = backend.fit(X, [1.0, 2.0, 3.0, 4.0], "regression")
    plain = backend.predict(context, X)
    with_q = backend.predict(context, X, quantiles=(0.9, 0.1))
    assert plain.predictions == with_q.predictions
    assert with_q.quantiles == [[v + 0.9, v + 0.1] for v in X[:, 0]]


def test_bad_input_is_invalid_not_a_server_error(backend_cls):
    backend = backend_cls(device="cpu")
    with pytest.raises(InvalidInputError, match="classes"):
        backend.fit(X, ["x"] * 4, "classification")
    with pytest.raises(InvalidInputError, match="2 training rows"):
        backend.fit(X[:1], [1.0], "regression")
    context = backend.fit(X, [1.0, 2.0, 3.0, 4.0], "regression")
    with pytest.raises(InvalidInputError, match="test rows"):
        backend.predict(context, np.array([[np.inf, 0.0]]))


def test_spill_without_the_network_and_only_for_this_version(backend_cls):
    backend = backend_cls(device="cpu")
    context = backend.fit(X, ["x", "y", "x", "y"], "classification")
    blob = backend.dumps_payload(context)
    assert blob.startswith(b"D1:1_3:")
    assert context.estimator.model is not None  # dumping left it intact
    restored = backend.loads_payload(blob)
    assert restored.estimator.model is context.estimator.model
    assert restored.classes == ["x", "y"]
    with pytest.raises(ValueError, match="version"):
        backend.loads_payload(b"D1:1_2:" + blob[len(b"D1:1_3:") :])


def test_settings_from_the_environment(monkeypatch):
    from tabctx.serve.factory import ServeSettings

    monkeypatch.setenv("TABCTX_TABDPT_CONTEXT_SIZE", "4096")
    monkeypatch.setenv("TABCTX_TABDPT_ENSEMBLES", "1")
    settings = ServeSettings.from_env()
    assert (settings.tabdpt_context_size, settings.tabdpt_ensembles) == (4096, 1)
    monkeypatch.setenv("TABCTX_TABDPT_ENSEMBLES", "0")
    with pytest.raises(ValueError, match="positive"):
        ServeSettings.from_env()


def test_a_context_is_charged_its_table_at_float32(backend_cls):
    from tabctx.backends.tabdpt import _context_bytes

    class Fitted:
        X_train = np.zeros((100, 4))
        projection = None

    assert _context_bytes(Fitted()) == 100 * 4 * 4 + 100 * 4
