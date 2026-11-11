# Quickstart

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VectorInstitute/tabctx/blob/main/examples/quickstart.ipynb)

Run a tabctx server on your laptop, fit a real dataset and predict
against it. A CPU is enough; the server uses a GPU if it finds one.
tabctx needs Python 3.12 or newer. Prefer a notebook? The
[quickstart notebook](https://github.com/VectorInstitute/tabctx/blob/main/examples/quickstart.ipynb)
runs this whole page end to end.

## 1. Install and start a server

```bash
pip install "tabctx[standalone,tabicl,client]"
TABCTX_BACKEND=tabicl python -m tabctx.serve        # http://127.0.0.1:8000
```

The first start downloads the TabICLv2 weights (BSD-3, no account
needed). `curl localhost:8000/v1/models` lists what the server serves.

## 2. Fit once, predict many

```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from tabctx import TabctxClassifier, TabctxClient

X, y = fetch_openml("credit-g", version=1, as_frame=True, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = TabctxClassifier(client=TabctxClient("http://127.0.0.1:8000"))
clf.fit(X_train, y_train)          # encodes the table once, on the server
clf.score(X_test, y_test)          # 0.77, no training loop, no tuning
clf.predict_proba(X_test[:5])      # reuses the cached context
```

`fit` sends the table to the server, which encodes it into a cached
*context*. Every later `predict` and `predict_proba` runs against that
context, without encoding the table again. The DataFrame's text columns,
categories and missing values go as they are.

## 3. On a GPU server

The container image includes TabICL, TabPFN and TabDPT; each
model's weights download when the server starts.

```bash
docker run --gpus all -p 8000:8000 -e TABCTX_BACKEND=tabicl ghcr.io/vectorinstitute/tabctx
```

`TABCTX_BACKEND=tabicl,tabdpt` serves two models from one cache; a fit
picks one with `model=`. Both are licensed for commercial use. `tabpfn`
is a third choice, but its weights need you to accept Prior Labs' terms
first (set `TABPFN_TOKEN`) and are licensed for non-commercial use.

## 4. Without Python

Every client is a thin wrapper over two calls:

```bash
curl localhost:8000/v1/tabctx/fit -F dataset_id=churn -F target=churned -F file=@train.csv
curl localhost:8000/v1/tabctx/predict -F dataset_id=churn -F file=@today.csv
```

## Next

<div class="grid cards" markdown>

-   [**Python and scikit-learn**](python.md): estimators, the client,
    errors, and the engine in your own process.
-   [**HTTP API**](http.md): every route, with curl.
-   [**Serving guide**](../serving.md): Ray Serve, several GPUs, tenants,
    uploads and every setting.
-   [**Coding agents**](../coding-agents.md): give Claude Code or Codex
    these models as tools.

</div>
