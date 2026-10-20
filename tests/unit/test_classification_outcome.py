"""Classification predicts report the same labels, whatever return_proba is (#45)."""

import numpy as np
import pytest

from tabctx.backends._outcome import classification_outcome

PROBA = np.array([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]])
CLASSES = np.array([0.0, 1.0, 2.0])  # a JSON train_y of [0, 1, 2] parses as floats


@pytest.mark.parametrize("return_proba", [False, True])
def test_labels_are_class_strings_either_way(return_proba):
    outcome = classification_outcome(PROBA, CLASSES, return_proba)
    assert outcome.predictions == ["1.0", "0.0"]
    if return_proba:
        assert outcome.classes == ["0.0", "1.0", "2.0"]
        assert outcome.probabilities == PROBA.tolist()
    else:
        assert outcome.classes is None and outcome.probabilities is None


class _Classifier:
    """A fitted estimator whose predict() must not be called: a second
    forward pass, and the source of numeric labels (#45)."""

    classes_ = CLASSES

    def predict_proba(self, X):
        return PROBA[: len(X)]

    def predict(self, X):
        raise AssertionError("labels must come from predict_proba")


def _backend(module: str, cls: str):
    backend_cls = getattr(pytest.importorskip(module), cls)
    return object.__new__(backend_cls)  # _predict needs no weights


@pytest.mark.parametrize(
    ("module", "cls"),
    [
        ("tabctx.backends.tabicl", "TabICLBackend"),
        ("tabctx.backends.tabpfn", "TabPFNBackend"),
    ],
)
def test_backends_return_the_same_labels_with_and_without_probabilities(module, cls):
    backend = _backend(module, cls)
    X = np.zeros((2, 3))
    plain = backend._predict(_Classifier(), X, False)
    with_proba = backend._predict(_Classifier(), X, True)
    assert plain.predictions == with_proba.predictions == ["1.0", "0.0"]
    assert set(plain.predictions) <= set(with_proba.classes)
