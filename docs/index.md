---
hide:
  - navigation
  - toc
---

# tabctx { .tx-sr-only }

<div class="tx-hero" markdown>

![tabctx](assets/logo-light.svg#only-light){ .tx-hero-logo }
![tabctx](assets/logo-dark.svg#only-dark){ .tx-hero-logo }

<p class="tx-tagline">The serving engine for tabular foundation models.</p>

Fit a table once, keep it on the GPU, and answer later predictions
typically well under a second, for many tables and many users, on
hardware you control.

[Get started](getting-started/quickstart.md){ .md-button .md-button--primary }
[Why a server](why-tabctx.md){ .md-button }
[GitHub](https://github.com/VectorInstitute/tabctx){ .md-button }

</div>

```bash
pip install "tabctx[standalone,tabicl,client]"
TABCTX_BACKEND=tabicl python -m tabctx.serve
```

```python
from tabctx import TabctxClassifier, TabctxClient

clf = TabctxClassifier(client=TabctxClient("http://127.0.0.1:8000"))
clf.fit(X_train, y_train)      # encodes the table once, on the server
clf.predict_proba(X_test)      # reuses the cached context
```

## What tabctx does

[TabPFN](https://github.com/PriorLabs/TabPFN) and
[TabICL](https://github.com/soda-inria/tabicl) often beat gradient
boosting with no training loop: the training table is the model's input.
Encoding that table is the expensive part, and it is the same work on
every prediction. tabctx encodes each table once and serves predictions
against the result.
[Why a server, need by need →](why-tabctx.md)

<figure markdown>
![Fit time grows with the training table; predict time against the cached context stays near 0.3 s](assets/fit-vs-predict-light.svg#only-light)
![Fit time grows with the training table; predict time against the cached context stays near 0.3 s](assets/fit-vs-predict-dark.svg#only-dark)
</figure>

<div class="grid cards" markdown>

-   :material-database-clock:{ .lg .middle } **Fit once, predict many**

    ---

    A fitted table becomes a *context*, addressed by id. Every later
    predict skips the re-encode.

    [:octicons-arrow-right-24: How it works](how-it-works.md)

-   :material-memory:{ .lg .middle } **Many tables, one GPU**

    ---

    Contexts share a byte-budgeted cache. The coldest is evicted or
    spilled to disk, and a fit that would run the GPU out of memory is
    refused before it runs.

    [:octicons-arrow-right-24: The context cache](how-it-works.md#how-it-works)

-   :material-swap-horizontal:{ .lg .middle } **Three models, one endpoint**

    ---

    `tabicl-v2`, `tabpfn-3.5` and `tabdpt-1.3` behind `GET /v1/models`
    (two licensed for commercial use); pick one per
    request. Adding a model is one backend class and a few lines of
    wiring.

    [:octicons-arrow-right-24: Add a model](contributing/adding-a-model.md)

-   :material-chart-bell-curve:{ .lg .middle } **Prediction intervals**

    ---

    Regression predicts a distribution, so ask for its quantiles, such
    as a 90% interval per row, from the same model call as the mean.

    [:octicons-arrow-right-24: Prediction intervals](getting-started/python.md#prediction-intervals)

-   :material-language-python:{ .lg .middle } **Drop-in scikit-learn**

    ---

    `TabctxClassifier` and `TabctxRegressor` take DataFrames with text,
    dates and missing values, and work in pipelines and
    `cross_val_score`.

    [:octicons-arrow-right-24: Python guide](getting-started/python.md)

-   :material-robot-outline:{ .lg .middle } **Tools for coding agents**

    ---

    `tabctx-mcp` lets Claude Code, Codex and Gemini CLI fit, predict,
    explain and run what-ifs on a CSV.

    [:octicons-arrow-right-24: Coding agents](coding-agents.md)

-   :material-server-network:{ .lg .middle } **Built for production**

    ---

    Multi-GPU placement, multi-replica routing on Ray Serve or one plain
    process, tenant isolation, list and delete, and a container image.

    [:octicons-arrow-right-24: Serving guide](serving.md)

</div>

## Compared with the alternatives

| | Model in your process | Hosted API | **tabctx** |
|---|:---:|:---:|:---:|
| Data stays on your machines | :material-check: | :material-close: | :material-check: |
| Many tables cached at once, per tenant | :material-close: | :material-check: | :material-check: |
| Refuses a fit before it runs out of GPU memory | :material-close: | n/a | :material-check: |
| Models from several labs behind one endpoint | :material-close: | :material-close: | :material-check: |
| Open source, no per-call metering | :material-check: | :material-close: | :material-check: |
| You can list and delete what the server holds | n/a | varies | :material-check: |

## Models

| Model id | Model | Weights license |
|---|---|---|
| `tabicl-v2` | [TabICLv2](https://github.com/soda-inria/tabicl) (Inria) | BSD-3-Clause: commercial use OK |
| `tabpfn-3.5` | [TabPFN-3.5](https://github.com/PriorLabs/TabPFN) (Prior Labs) | Non-commercial; production use needs a [license from Prior Labs](https://priorlabs.ai/pricing) |
| `tabdpt-1.3` | [TabDPT](https://github.com/layer6ai-labs/TabDPT-inference) (Layer 6) | Apache-2.0: commercial use OK |

tabctx itself is Apache 2.0; each model's weights keep their own license.
