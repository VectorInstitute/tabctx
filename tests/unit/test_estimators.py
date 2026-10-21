"""The scikit-learn estimators (tabctx/estimators.py), against a fake
client that serves fits and predicts in memory the way a deployment
does: it reads the files the estimators send and encodes them with the
server's own schema code."""

import copy
import pickle

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("pyarrow")
pd = pytest.importorskip("pandas")

from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score

from tabctx import TabctxClassifier, TabctxRegressor
from tabctx.client import PredictResult, TabctxClient
from tabctx.errors import DatasetNotFoundError
from tabctx.schema import TableSchema
from tabctx.serve.files import read_table, read_training_table


class FakeDeployment:
    """Predicts a row's class as round(its first encoded feature) % k (so
    results depend on how the rows were encoded); regression returns the
    first encoded feature."""

    def __init__(self, tmp_path, classes_order=None):
        self.tmp_path = tmp_path
        self.contexts: dict[str, tuple] = {}
        self.fits: list[dict] = []
        self.predict_calls: list[int] = []
        self.classes_order = classes_order

    def _path(self, data):
        p = self.tmp_path / "f"
        p.write_bytes(data)
        return p

    def fit_file(self, file, target=None, dataset_id=None, model=None, task=None):
        features, y = read_training_table(self._path(file), target, task)
        schema, _ = TableSchema.learn(features.columns, features.names)
        self.fits.append({"dataset_id": dataset_id, "model": model, "task": task})
        classes = sorted(set(y), key=int) if task == "classification" else None
        self.contexts[dataset_id] = (task, schema, classes)
        return dataset_id

    def predict_file(self, dataset_id, file, return_proba=False, model=None):
        self.predict_calls.append(len(file))
        if dataset_id not in self.contexts:
            raise DatasetNotFoundError(
                f"no cached context for dataset_id={dataset_id!r}"
            )
        task, schema, classes = self.contexts[dataset_id]
        table = read_table(self._path(file), text_columns=schema.text_names)
        X = schema.encode(schema.select(table.names, table.columns))
        if task == "regression":
            return PredictResult(X[:, 0].tolist(), None, None, 1.0, None)
        k = len(classes)
        preds = [classes[int(round(v)) % k] if v == v else classes[0] for v in X[:, 0]]
        order = self.classes_order or classes
        proba = [[1.0 if c == p else 0.0 for c in order] for p in preds]
        return PredictResult(preds, proba if return_proba else None, order, 1.0, None)


@pytest.fixture
def deployment(tmp_path):
    return FakeDeployment(tmp_path)


class TestLabels:
    def test_labels_of_any_type_come_back_as_given(self, deployment):
        X = np.array([[0.0], [1.0], [2.0], [0.0]])
        for y in (
            ["cat", "dog", "emu", "cat"],
            [True, False, True, True],
            [10, 20, 30, 10],
        ):
            clf = TabctxClassifier(client=deployment).fit(X, y)
            assert list(clf.classes_) == sorted(set(y))
            k = len(clf.classes_)
            assert clf.predict(X).tolist() == [
                clf.classes_[int(v) % k] for v in X[:, 0]
            ]
        assert deployment.fits[-1] == {
            "dataset_id": clf.dataset_id_,
            "model": "tabpfn-3.5",
            "task": "classification",
        }

    def test_predict_proba_columns_follow_classes_(self, tmp_path):
        # The server may list classes in any order ("10" before "2").
        deployment = FakeDeployment(tmp_path, classes_order=["2", "0", "1"])
        clf = TabctxClassifier(client=deployment).fit(
            [[0.0], [1.0], [2.0]], ["a", "b", "c"]
        )
        assert clf.predict_proba([[0.0], [2.0]]).tolist() == [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

    @pytest.mark.parametrize(
        "y",
        [
            [0, 1, np.nan, 1],
            np.linspace(0, 1, 4),
            [1, "a", 1, "a"],
            ["a", None, "b", "a"],
        ],
    )
    def test_bad_class_labels_are_rejected_here(self, deployment, y):
        with pytest.raises((ValueError, TypeError)):
            TabctxClassifier(client=deployment).fit([[0.0]] * 4, y)
        assert deployment.fits == []

    @pytest.mark.parametrize(
        "y",
        [
            ["a", None, "b", "a"],
            pd.array(["a", pd.NA, "b", "a"], dtype="string"),
            pd.Categorical(["a", None, "b", "a"]),
        ],
    )
    def test_missing_labels_are_a_value_error(self, deployment, y):
        with pytest.raises(ValueError, match="missing labels"):
            TabctxClassifier(client=deployment).fit([[0.0]] * 4, y)

    def test_column_vector_labels_are_raveled(self, deployment):
        with pytest.warns(Warning):
            clf = TabctxClassifier(client=deployment).fit(
                [[0.0], [1.0]], [["a"], ["b"]]
            )
        assert list(clf.classes_) == ["a", "b"]

    def test_regression_labels_must_be_finite(self, deployment):
        with pytest.raises(ValueError, match="missing or infinite"):
            TabctxRegressor(client=deployment).fit([[0.0], [1.0]], [1.0, np.nan])


class TestColumns:
    def test_dataframes_with_text_and_missing_values(self, deployment):
        train = pd.DataFrame({"n": [0.0, 1.0, np.nan], "plan": ["gold", None, "basic"]})
        clf = TabctxClassifier(client=deployment).fit(train, ["x", "y", "x"])
        assert list(clf.feature_names_in_) == ["n", "plan"]
        assert clf.n_features_in_ == 2
        test = pd.DataFrame({"plan": ["gold", "basic"], "n": [1.0, 0.0]})  # any order
        assert clf.predict(test).tolist() == ["y", "x"]

    def test_training_text_stays_text_when_test_values_look_numeric(self, deployment):
        # "c" was text at fit; 0.1 and 2.5 at predict must match "0.1" and "2.5".
        train = pd.DataFrame({"c": ["a", 0.1, "b", 2.5]})
        reg = TabctxRegressor(client=deployment).fit(train, [0.0, 1.0, 2.0, 3.0])
        codes = {"0.1": 0.0, "2.5": 1.0, "a": 2.0, "b": 3.0}
        got = reg.predict(pd.DataFrame({"c": [0.1, 2.5]}))
        assert got.tolist() == [codes["0.1"], codes["2.5"]]
        with pytest.warns(UserWarning, match="by position"):
            assert reg.predict(np.array([[2.5]], dtype=object)).tolist() == [
                codes["2.5"]
            ]

    def test_dataframes_and_arrays_mix_by_position(self, deployment):
        df = pd.DataFrame({"a": [0.0, 1.0], "b": [5.0, 6.0]})
        reg = TabctxRegressor(client=deployment).fit(df, [0.0, 1.0])
        with pytest.warns(UserWarning, match="by position"):
            assert reg.predict(df.to_numpy()[:1]).tolist() == [0.0]
        reg.fit(df.to_numpy(), [0.0, 1.0])
        assert not hasattr(reg, "feature_names_in_")
        with pytest.warns(UserWarning, match="by position"):
            assert reg.predict(df.iloc[1:]).tolist() == [1.0]

    def test_feature_count_is_checked(self, deployment):
        reg = TabctxRegressor(client=deployment).fit([[0.0, 1.0]], [0.0])
        with pytest.raises(
            ValueError, match="X has 3 features, but TabctxRegressor is expecting 2"
        ):
            reg.predict([[0.0, 1.0, 2.0]])

    def test_non_string_column_names_set_no_feature_names(self, deployment):
        reg = TabctxRegressor(client=deployment).fit(pd.DataFrame([[1.0, 2.0]]), [0.0])
        assert not hasattr(reg, "feature_names_in_")


class TestRequests:
    def test_same_data_same_id_and_model_or_labels_change_it(self, deployment):
        X, y = [[0.0], [1.0]], ["a", "b"]
        a = TabctxClassifier(client=deployment).fit(X, y).dataset_id_
        assert TabctxClassifier(client=deployment).fit(X, y).dataset_id_ == a
        assert (
            TabctxClassifier(model="tabicl-v2", client=deployment).fit(X, y).dataset_id_
            != a
        )
        assert TabctxClassifier(client=deployment).fit(X, ["b", "a"]).dataset_id_ != a

    def test_a_lost_context_is_fit_again_once(self, deployment):
        clf = TabctxClassifier(client=deployment).fit([[0.0], [1.0]], ["a", "b"])
        deployment.contexts.clear()  # evicted, or the replica restarted
        assert clf.predict([[1.0]]).tolist() == ["b"]
        assert len(deployment.fits) == 2

    def test_other_404s_are_not_refit(self, deployment):
        clf = TabctxClassifier(client=deployment).fit([[0.0], [1.0]], ["a", "b"])

        def unknown_model(*args, **kwargs):
            raise DatasetNotFoundError("Model 'tabpfn-3.5' not found")

        deployment.predict_file = unknown_model
        with pytest.raises(DatasetNotFoundError, match="Model"):
            clf.predict([[1.0]])
        assert len(deployment.fits) == 1

    def test_the_model_used_is_the_one_fit(self, deployment):
        clf = TabctxClassifier(client=deployment).fit([[0.0], [1.0]], ["a", "b"])
        clf.set_params(model="tabicl-v2")  # without fitting again
        deployment.contexts.clear()
        clf.predict([[0.0]])
        assert deployment.fits[-1]["model"] == "tabpfn-3.5"

    def test_large_predicts_are_split_under_the_request_limit(self, deployment):
        reg = TabctxRegressor(client=deployment, max_request_bytes=40_000).fit(
            [[0.0, 0.0]], [0.0]
        )
        X = np.random.default_rng(0).normal(size=(5000, 2))
        got = reg.predict(X)
        assert len(deployment.predict_calls) > 1
        assert max(deployment.predict_calls) <= 40_000
        assert np.allclose(got, X[:, 0].astype(np.float32))

    def test_uneven_rows_are_split_until_each_fits(self, deployment):
        rng = np.random.default_rng(0)
        text = [None] * 1800 + [
            "".join(rng.choice(list("abcdef"), 2000)) for _ in range(200)
        ]
        X = pd.DataFrame({"t": text})
        reg = TabctxRegressor(client=deployment, max_request_bytes=100_000).fit(
            X.iloc[1798:1802], [0.0, 1.0, 2.0, 3.0]
        )
        assert len(reg.predict(X)) == 2000
        assert max(deployment.predict_calls) <= 100_000

    def test_oversized_tables_are_refused_before_sending(self, deployment):
        X = np.random.default_rng(0).normal(size=(2000, 20))
        with pytest.raises(ValueError, match="training table is"):
            TabctxRegressor(client=deployment, max_request_bytes=10_000).fit(X, X[:, 0])
        assert deployment.fits == []
        reg = TabctxRegressor(client=deployment, max_request_bytes=None).fit(X, X[:, 0])
        with pytest.raises(ValueError, match="one test row"):
            reg.set_params(max_request_bytes=10).predict(X[:2])

    def test_unfitted_predict_raises(self, deployment):
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            TabctxClassifier(client=deployment).predict([[0.0]])


def test_works_with_sklearn_model_selection(deployment):
    X = np.arange(40, dtype=float).reshape(20, 2) % 3
    y = np.array(["a", "b", "c", "a"] * 5)
    assert len(cross_val_score(TabctxClassifier(client=deployment), X, y, cv=2)) == 2
    search = GridSearchCV(
        TabctxClassifier(client=deployment),
        {"model": ["tabpfn-3.5", "tabicl-v2"]},
        cv=2,
    )
    assert search.fit(X, y).best_params_["model"] in ("tabpfn-3.5", "tabicl-v2")
    client = TabctxClient("https://gw.example")
    copied = clone(TabctxClassifier(model="tabicl-v2", client=client))
    assert copied.get_params() == {
        "model": "tabicl-v2",
        "client": client,
        "max_request_bytes": 29 * 1024 * 1024,
    }


class TestApiKey:
    def test_sent_never_shown_never_pickled(self, monkeypatch):
        monkeypatch.delenv("TABCTX_API_KEY", raising=False)
        client = TabctxClient("https://gw.example", api_key="secret-key")
        assert client._headers(None)["Authorization"] == "Bearer secret-key"
        clf = TabctxClassifier(client=client)
        assert "secret-key" not in repr(clf) and "secret-key" not in repr(client)
        assert "secret-key" not in clf._repr_html_()
        assert b"secret-key" not in pickle.dumps(clf)
        # Copies keep the key (they share the client).
        assert clone(clf).client._api_key == "secret-key"
        assert copy.copy(client)._api_key == "secret-key"

    def test_an_unpickled_client_reads_the_environment(self, monkeypatch):
        client = TabctxClient("https://gw.example", api_key="secret-key")
        monkeypatch.setenv("TABCTX_API_KEY", "from-env")
        assert pickle.loads(pickle.dumps(client))._api_key == "from-env"

    def test_defaults_come_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("TABCTX_BASE_URL", "https://gw.example/")
        monkeypatch.setenv("TABCTX_API_KEY", "k")
        client = TabctxClient()
        assert client._base_url == "https://gw.example" and client._api_key == "k"
        monkeypatch.delenv("TABCTX_BASE_URL")
        with pytest.raises(ValueError, match="TABCTX_BASE_URL"):
            TabctxClient()
