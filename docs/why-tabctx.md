# Why a server

TabPFN, TabICL and TabDPT are good models with small, friendly Python
packages: `fit`, then `predict`, in your own process. That is the right
tool for a notebook. This page is about what changes when a model like
this becomes something other people and programs depend on, and what
tabctx does at each step. The links are to threads where users of these
models raised the same needs.

## Many predictions against one table

A fitted table is reused for every later prediction: a scoring job, an
API behind an app, an agent asking what-ifs. Users of in-process models
ask why a single-row predict is slower than expected
([TabPFN #904](https://github.com/PriorLabs/TabPFN/issues/904)), or run
hundreds of thousands of predicts against one fit
([TabPFN #1221](https://github.com/PriorLabs/TabPFN/issues/1221)).

**tabctx:** the encoded table stays on the GPU as a *context*, and each
predict runs only the test rows against it. Concurrent predicts on one
context share a model call. On an A40, a 100-row predict against a
100,000-row table takes about 0.3 s, against 103 s to fit it
([benchmarks](benchmarks.md)).

## Many tables on one GPU

One process with one fitted object holds one table. A service holds
many: one per customer, product or experiment. Their encodings are large
(a TabICL context for 40,000 × 250 is several GB,
[tabicl #92](https://github.com/soda-inria/tabicl/issues/92)).

**tabctx:** contexts share a byte budget per GPU. The least recently used
is evicted, or written to a disk tier and restored on its next predict.
`GET /v1/tabctx/datasets` lists what is held, and `DELETE` removes it.

## Running out of GPU memory

Users ask to cap the memory a model may use instead of finding out by
crashing ([tabicl #177](https://github.com/soda-inria/tabicl/issues/177)),
and have hit out-of-memory errors as table sizes and versions change
([TabPFN #849](https://github.com/PriorLabs/TabPFN/issues/849),
[#850](https://github.com/PriorLabs/TabPFN/issues/850)).

**tabctx:** every fit is priced before it runs, from measured peaks by
table shape, against what the GPU has free right now. If it fits only
once colder contexts step aside, they do; if it can never fit, it is
refused with a 413 before any GPU work, and the server keeps serving
everyone else. `TABCTX_GPU_MEMORY_FRACTION` caps the share of each GPU
the server may use.

## Keeping a fit

Saving a fitted model with its cache, and loading it in another session,
is a recurring question
([TabPFN #412](https://github.com/PriorLabs/TabPFN/issues/412),
[tabicl #12](https://github.com/soda-inria/tabicl/issues/12)). On a
hosted API, the provider decides how long a fitted table lives
([TabPFN #1385](https://github.com/PriorLabs/TabPFN/issues/1385)).

**tabctx:** a context lives in the server, addressed by an id you choose,
until you delete it or memory pressure evicts it; with a disk tier, it
comes back on its next predict, including after a restart on the same
node. The scikit-learn estimators keep their training table and refit
transparently when a context is gone. Durable,
shared storage of contexts is on the
[roadmap](https://github.com/VectorInstitute/tabctx/issues/6).

## Uncertainty

The models predict a distribution, and users want to see it
([TabPFN #120](https://github.com/PriorLabs/TabPFN/issues/120); quantile
support in [AutoGluon #5571](https://github.com/autogluon/autogluon/pull/5571)).

**tabctx:** ask any regression predict for `quantiles`, such as a 90%
interval per row, from the same model call as the mean
([prediction intervals](getting-started/python.md#prediction-intervals)).

## Licensing

Model weights carry their own licenses, and several recent ones are
non-commercial. tabctx is Apache 2.0 and serves two models whose weights
are licensed for commercial use, TabICLv2 (BSD-3) and TabDPT
(Apache-2.0), next to TabPFN for research use. See the
[models table](index.md#models).

## When not to use tabctx

- One notebook, one table, one person: the model's own package is
  simpler.
- No GPU and large tables: these models are slow on a CPU, with or
  without a server.
- You want someone else to run the GPUs: a hosted API does that.
