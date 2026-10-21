"""The scikit-learn estimators and the client's file methods against a
real local 2-replica deployment (fake backend: predicts the majority
class, or the mean target)."""

import gzip

import numpy as np
import pytest

pytest.importorskip("ray")
pytest.importorskip("sklearn")
pd = pytest.importorskip("pandas")

from sklearn.model_selection import cross_val_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer  # noqa: E402

from tabctx import TabctxClassifier, TabctxRegressor  # noqa: E402
from tabctx.client import TabctxClient  # noqa: E402


def churn(n=40):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "tenure": rng.integers(1, 60, n).astype(float),
            "plan": rng.choice(["basic", "gold", None], n),
            "spend": np.where(rng.random(n) < 0.2, np.nan, rng.normal(50, 10, n)),
        }
    )
    y = np.array(["stay"] * (n - n // 4) + ["leave"] * (n // 4))
    return X, y


def test_classifier_on_a_dataframe_with_text_and_missing(two_replica_service):
    X, y = churn()
    clf = TabctxClassifier(model="fake", client=TabctxClient(two_replica_service))
    clf.fit(X, y)
    assert clf.predict(X.iloc[:5]).tolist() == ["stay"] * 5
    proba = clf.predict_proba(X[["spend", "plan", "tenure"]].iloc[:3])
    assert proba.shape == (3, 2) and list(clf.classes_) == ["leave", "stay"]
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert clf.score(X, y) == 0.75


def test_regressor_and_numpy_input(two_replica_service):
    X = np.random.default_rng(1).normal(size=(30, 4))
    y = np.arange(30, dtype=float)
    reg = TabctxRegressor(model="fake", client=TabctxClient(two_replica_service))
    pred = reg.fit(X, y).predict(X[:4])
    assert pred.shape == (4,) and np.allclose(pred, y.mean())


def test_cross_validation_and_pipelines(two_replica_service):
    X, y = churn(60)
    client = TabctxClient(two_replica_service)
    pipe = make_pipeline(
        FunctionTransformer(lambda df: df.assign(tenure_years=df.tenure / 12)),
        TabctxClassifier(model="fake", client=client),
    )
    scores = cross_val_score(pipe, X, y, cv=3)
    assert len(scores) == 3 and all(0 <= s <= 1 for s in scores)


def test_client_file_methods(two_replica_service, tmp_path):
    client = TabctxClient(two_replica_service)
    train = tmp_path / "train.csv.gz"
    train.write_bytes(gzip.compress(b"a,b,label\n1,x,yes\n2,y,yes\n3,x,no\n"))
    ds = client.fit_file(train, target="label", dataset_id="client-files", model="fake")
    result = client.predict_file(ds, b"b,a\nx,4\n", return_proba=True)
    assert result.predictions == ["yes"] and result.classes == ["no", "yes"]
