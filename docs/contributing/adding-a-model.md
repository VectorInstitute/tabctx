# Add a model

tabctx serves any tabular model that can split its work into *encode a
training table* and *predict against the encoding*. A model is one
class, a **backend**. The cache, admission control, placement, tenancy
and HTTP API come for free: none of them import a model, they only use
this interface.

Open requests for new models: Mitra and TabDPT
([#102](https://github.com/VectorInstitute/tabctx/issues/102)).

## The interface

```python
--8<-- "src/tabctx/backends/base.py:15:"
```

The rules that matter:

- **A fresh model per fit.** `fit()` must build a new underlying
  estimator every call; the payload it returns is cached and must not be
  shared with any other context.
- **One pass per predict.** Derive labels from probabilities rather
  than running the model twice for `return_proba=True`.
- **Labels are strings** for classification (use
  `tabctx.backends._outcome.classification_outcome`).
- **`name`** is the model id clients send as `model` and see in
  `GET /v1/models`, e.g. `tabicl-v2`.

## Optional hooks

The engine probes for these with `getattr`, so a first version needs
none of them:

| Hook | What it buys |
|---|---|
| `warmup()` | Load weights at server start, not inside the first tenant's fit. |
| `fit_peak_bytes_hint()` | The fit's real peak memory, which the admission estimator learns from. |
| `dumps_payload()` / `loads_payload()` | Compact spill files, e.g. leaving shared weights out. Default: pickle. |
| `prepare_fit()` / `fit_prepared()` | Run CPU preprocessing while another fit holds the GPU. |
| `may_offload()` / `host_bytes_estimate()` | Host-memory admission for fits that stage data in RAM. |

## Wire it in

1. Add the backend in `src/tabctx/backends/<model>.py`, importing its
   package lazily so tabctx installs without it.
2. Add its kind to `BackendKind` and `BACKEND_KINDS` in
   `src/tabctx/serve/factory.py`, and build it in `_build_real_backend`.
3. Add an extra in `pyproject.toml` (`tabctx[<model>]`) and a row to the
   models table in the README with its **weights license**.

## Memory calibration

Admission refuses a fit before it runs out of memory using a
per-model estimate of peak memory by table shape. Without a calibration
the engine falls back to a conservative formula and then learns from
each fit's measured peak. To calibrate, run
`benchmarks/calibrate_memory.py` on a GPU over a grid of shapes and add
the points under `src/tabctx/memory/`, as TabICL and TabPFN do.

## Tests

- **Unit tests** need no model weights: test your backend's own logic
  with small fakes, and use `tabctx.backends.fake.FakeBackend` to test
  anything engine-side.
- **An integration test** that skips without the package
  (`pytest.importorskip`) and checks that predictions through the server
  match the model's own package on a small table.
- **Accuracy:** run `benchmarks/eval_real_datasets.py` against the model's
  own package and include the numbers in the pull request.
