"""An in-memory stand-in for a tabctx deployment, for the client-side
tests (estimators, tabctx-mcp): it reads the files the client sends and
encodes them with the server's own schema code."""

from tabctx.client import PredictResult
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
