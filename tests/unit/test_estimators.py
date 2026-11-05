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

from fake_deployment import FakeDeployment
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score

from tabctx import TabctxClassifier, TabctxRegressor
from tabctx.client import TabctxClient
from tabctx.errors import DatasetNotFoundError, InvalidInputError


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
        "ttl_s": None,
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


class TestDefaultModel:
    """Without a model, the estimators use the deployment's default, so a
    deployment that serves only TabICL works with no arguments."""

    TABICL_ONLY = [{"id": "tabicl-v2", "object": "model", "default": True}]

    def test_uses_the_deployments_default_model(self, tmp_path):
        deployment = FakeDeployment(tmp_path, models=self.TABICL_ONLY)
        clf = TabctxClassifier(client=deployment).fit([[0.0], [1.0]], ["a", "b"])
        assert clf.model_ == "tabicl-v2"
        assert clf.model is None  # the parameter stays as given (clone)
        assert deployment.fits[-1]["model"] == "tabicl-v2"
        assert clf.predict([[1.0]]).tolist() == ["b"]

    def test_regressor_too(self, tmp_path):
        deployment = FakeDeployment(tmp_path, models=self.TABICL_ONLY)
        reg = TabctxRegressor(client=deployment).fit([[0.0], [1.0]], [0.5, 2.5])
        assert reg.model_ == "tabicl-v2"

    def test_clone_and_params_keep_the_unset_model(self, tmp_path):
        deployment = FakeDeployment(tmp_path, models=self.TABICL_ONLY)
        clf = TabctxClassifier(client=deployment).fit([[0.0], [1.0]], ["a", "b"])
        assert clf.get_params()["model"] is None
        copy_ = clone(clf)
        assert copy_.model is None and not hasattr(copy_, "model_")

    def test_through_a_gateway_catalog(self, tmp_path):
        # LLMs listed first and nothing marked default: TabPFN, not an LLM.
        catalog = [
            {"id": "gpt-5", "object": "model", "owned_by": "openai"},
            {"id": "tabicl-v2", "object": "model", "owned_by": "on_prem"},
            {"id": "tabpfn-3.5", "object": "model", "owned_by": "on_prem"},
        ]
        deployment = FakeDeployment(tmp_path, models=catalog)
        clf = TabctxClassifier(client=deployment).fit([[0.0], [1.0]], ["a", "b"])
        assert clf.model_ == "tabpfn-3.5"

    def test_an_explicit_model_is_sent_as_given(self, deployment):
        clf = TabctxClassifier(model="tabicl-v2", client=deployment)
        clf.fit([[0.0], [1.0]], ["a", "b"])
        assert clf.model_ == "tabicl-v2"
        assert deployment.fits[-1]["model"] == "tabicl-v2"

    def test_the_context_id_follows_the_resolved_model(self, tmp_path):
        # Fitting with the default and naming the same model explicitly is
        # the same context, so the server's cache is reused.
        deployment = FakeDeployment(tmp_path, models=self.TABICL_ONLY)
        X, y = [[0.0], [1.0]], ["a", "b"]
        implicit = TabctxClassifier(client=deployment).fit(X, y)
        explicit = TabctxClassifier(model="tabicl-v2", client=deployment).fit(X, y)
        assert implicit.dataset_id_ == explicit.dataset_id_

    def test_a_model_the_deployment_lacks_is_refused(self, tmp_path):
        deployment = FakeDeployment(tmp_path, models=self.TABICL_ONLY)
        with pytest.raises(InvalidInputError, match="serves"):
            TabctxClassifier(model="tabpfn-3.5", client=deployment).fit(
                [[0.0], [1.0]], ["a", "b"]
            )


def test_the_client_is_importable_from_the_package():
    import tabctx
    from tabctx import PredictResult, TabctxClient

    assert TabctxClient is tabctx.client.TabctxClient
    assert PredictResult is tabctx.client.PredictResult
    assert {"TabctxClient", "PredictResult"} <= set(tabctx.__all__)


class TestPredictQuantiles:
    def test_one_column_per_level_in_the_order_given(self, deployment):
        X = np.array([[1.0], [2.0], [3.0]])
        reg = TabctxRegressor(client=deployment).fit(X, [1.0, 2.0, 3.0])
        out = reg.predict_quantiles(X, quantiles=[0.9, 0.1])
        assert out.shape == (3, 2)
        np.testing.assert_allclose(out, [[1.9, 1.1], [2.9, 2.1], [3.9, 3.1]])

    def test_the_default_is_the_median_and_a_90_percent_interval(self, deployment):
        X = np.array([[1.0], [2.0]])
        reg = TabctxRegressor(client=deployment).fit(X, [1.0, 2.0])
        np.testing.assert_allclose(reg.predict_quantiles(X)[0], [1.05, 1.5, 1.95])

    def test_bad_levels_are_a_value_error(self, deployment):
        X = np.array([[1.0], [2.0]])
        reg = TabctxRegressor(client=deployment).fit(X, [1.0, 2.0])
        for bad in ([1.2], [], "0.5"):
            with pytest.raises(ValueError):
                reg.predict_quantiles(X, quantiles=bad)

    def test_a_test_table_over_several_requests(self, deployment):
        # Random values, which Parquet can't compress below the limit.
        X = np.random.default_rng(0).normal(size=(2000, 1))
        reg = TabctxRegressor(client=deployment, max_request_bytes=6_000)
        reg.fit(X[:10], np.arange(10.0))
        out = reg.predict_quantiles(X, quantiles=[0.5])
        assert len(deployment.predict_calls) > 1  # split into several files
        # Tables reach the model as float32.
        np.testing.assert_allclose(out[:, 0], X[:, 0] + 0.5, atol=1e-6)

    def test_a_context_lost_on_the_server_is_fit_again(self, deployment):
        X = np.array([[1.0], [2.0]])
        reg = TabctxRegressor(client=deployment).fit(X, [1.0, 2.0])
        deployment.contexts.clear()  # evicted, or the server restarted
        np.testing.assert_allclose(reg.predict_quantiles(X, [0.5])[:, 0], [1.5, 2.5])


def test_the_ttl_goes_with_every_fit(tmp_path):
    deployment = FakeDeployment(tmp_path)
    TabctxRegressor(client=deployment, ttl_s=60).fit(np.ones((3, 1)), [1.0, 2.0, 3.0])
    assert deployment.fits[-1]["ttl_s"] == 60
