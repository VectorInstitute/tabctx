"""Unit tests for the deterministic test-double backend (backends/fake.py)."""

from tabctx.backends.fake import FakeBackend


class TestClassification:
    def test_predicts_training_majority_class(self):
        backend = FakeBackend()
        payload = backend.fit([[1.0], [2.0], [3.0]], ["a", "a", "b"], "classification")
        outcome = backend.predict(payload, [[9.0], [10.0]])
        assert outcome.predictions == ["a", "a"]
        assert outcome.probabilities is None
        assert outcome.classes is None

    def test_return_proba_reports_class_fractions(self):
        backend = FakeBackend()
        payload = backend.fit([[1.0], [2.0], [3.0]], ["a", "a", "b"], "classification")
        outcome = backend.predict(payload, [[9.0]], return_proba=True)
        assert outcome.classes == ["a", "b"]
        assert outcome.probabilities == [[2 / 3, 1 / 3]]


class TestRegression:
    def test_predicts_training_mean(self):
        backend = FakeBackend()
        payload = backend.fit([[1.0], [2.0], [3.0]], [1.0, 2.0, 3.0], "regression")
        outcome = backend.predict(payload, [[0.0], [0.0]])
        assert outcome.predictions == [2.0, 2.0]

    def test_empty_training_labels_mean_is_zero(self):
        backend = FakeBackend()
        payload = backend.fit([], [], "regression")
        assert payload.mean_y == 0.0


class TestHints:
    def test_bytes_hint_ignores_shape(self):
        backend = FakeBackend(bytes_hint=555)
        assert backend.context_bytes_hint(n_train=10, n_features=3) == 555
        assert backend.context_bytes_hint(n_train=999, n_features=1) == 555

    def test_bytes_hint_defaults_to_none(self):
        assert FakeBackend().context_bytes_hint(10, 3) is None

    def test_peak_bytes_hint(self):
        assert FakeBackend(peak_bytes_hint=777).fit_peak_bytes_hint() == 777
        assert FakeBackend().fit_peak_bytes_hint() is None


class TestDelaysAndCallCounts:
    def test_fit_delay_actually_sleeps(self):
        backend = FakeBackend(fit_delay_s=0.01)
        backend.fit([[1.0]], ["a"], "classification")
        assert backend.fit_calls == 1

    def test_predict_delay_actually_sleeps(self):
        backend = FakeBackend(predict_delay_s=0.01)
        payload = backend.fit([[1.0]], ["a"], "classification")
        backend.predict(payload, [[1.0]])
        assert backend.predict_calls == 1

    def test_instance_name_overrides_class_default(self):
        assert FakeBackend(name="tabpfn").name == "tabpfn"
        assert FakeBackend().name == "fake"
