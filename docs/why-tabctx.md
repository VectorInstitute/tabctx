# Why a server

TabPFN, TabICL and TabDPT are good models with small, friendly Python
packages: `fit`, then `predict`, in your own process. That is the right
tool for a notebook. This page is about what changes when a model like
this becomes something other people and programs depend on, and what
tabctx does at each step.

tabctx calls a fitted table a *context*: the model's encoding of the
training table, kept in GPU memory and addressed by an id. In code, the
change is one estimator:

```python
from tabctx import TabctxClassifier, TabctxClient

clf = TabctxClassifier(client=TabctxClient("http://my-server:8000"))   # was TabICLClassifier()
clf.fit(X_train, y_train)
clf.predict_proba(X_test)
```

## Many predictions against one table

A fitted table is reused for every later prediction: a scoring job, an
API behind an app, an agent asking what-ifs. Users of these models run
large numbers of predictions against one fit
([TabPFN #1221](https://github.com/PriorLabs/TabPFN/issues/1221)).

**tabctx:** both packages can keep the encoded table inside one
estimator object (TabICL's is off by default; tabctx turns it on).
tabctx keeps it in a server instead, so every process, job and user
predicting against the table shares it and nobody fits it again.
Predicts that arrive together for the same table are batched into one
GPU call. On an A40, a 100-row predict against a 100,000 × 100 table
(TabICLv2) takes about 0.3 s, against 103 s to fit it
([benchmarks](benchmarks.md)).

## Many tables on one GPU

One process with one fitted object holds one table. A service holds
many: one per customer, product or experiment. Their encodings are large:
a TabICL encoding of 40,000 × 250 is about 5 GB
([tabicl #92](https://github.com/soda-inria/tabicl/issues/92); TabICL's
`repr` mode, `TABCTX_KV_CACHE=repr` in tabctx, is much smaller).

**tabctx:** the tables share a memory budget per GPU. When it is full,
the least recently used table is evicted, or written to local disk
(`TABCTX_SPILL_DIR`) and restored on its next predict.
`GET /v1/tabctx/datasets` lists what is held, and `DELETE` removes it.

## Running out of GPU memory

Out-of-memory errors come up as table sizes and versions change
([TabPFN #849](https://github.com/PriorLabs/TabPFN/issues/849)), and
users ask to set a device-memory budget instead of the model probing for
one by catching out-of-memory errors
([tabicl #177](https://github.com/soda-inria/tabicl/issues/177)).

**tabctx:** before a fit runs, tabctx estimates its peak GPU memory from
the table's shape (using measurements of earlier fits) and checks it
against the GPU's budget minus what the tables already held there take.
If it fits only after freeing memory, the least recently used tables are
evicted first; if it can never fit, it is refused with an error (HTTP
413) before any GPU work, and the server keeps serving everyone else.
`TABCTX_GPU_MEMORY_FRACTION` sets the share of each GPU that budget
covers.

## Keeping a fit

Saving a fitted model in one session and loading it in another has come
up for both packages
([TabPFN #412](https://github.com/PriorLabs/TabPFN/issues/412), now
supported; [tabicl #12](https://github.com/soda-inria/tabicl/issues/12)).
On a hosted API, the provider decides how long a fitted table lives
([TabPFN #1385](https://github.com/PriorLabs/TabPFN/issues/1385)).

**tabctx:** a table stays in the server, under an id you choose, until
you delete it or memory pressure evicts it. With a disk tier, an evicted
table is written to local disk and comes back on its next predict, also
after a server restart on the same node; a table still in GPU memory at
a restart is lost. The scikit-learn estimators keep their training table
and fit again when the server no longer has it. Durable, shared storage
of fitted tables is on the
[roadmap](https://github.com/VectorInstitute/tabctx/issues/6).

## Uncertainty

These models predict a distribution, not just a number. TabPFN and
TabICL return its quantiles in-process; tabctx returns them over its API.

**tabctx:** ask any regression predict for `quantiles`, such as a 90%
interval per row, from the same model call as the mean, for every model
it serves ([prediction intervals](getting-started/python.md#prediction-intervals)).

## Licensing

Model weights carry their own licenses. tabctx is Apache 2.0 and serves
two models whose weights are licensed for commercial use, TabICLv2
(BSD-3) and TabDPT (Apache-2.0), next to TabPFN-3.5, whose weights are
non-commercial unless licensed from Prior Labs. See the
[models table](index.md#models).

## When not to use tabctx

- **One notebook, one table, one person:** the model's own package is
  simpler.
- **Fit once, predict once:** a server adds an upload and an HTTP round
  trip, with no fitted table to reuse.
- **You don't want to run a service:** tabctx is a process to deploy and
  keep up. For several tables in one Python program, its
  [engine runs in your own process](getting-started/python.md#the-engine-in-your-own-process).
- **No GPU and large tables:** these models are slow on a CPU, with or
  without a server.
- **You want someone else to run the GPUs, and your data may leave your
  machines:** a hosted API does that.

**Try it:** the [quickstart](getting-started/quickstart.md) runs a server
on a laptop CPU and fits a real dataset.
