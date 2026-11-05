# Add a model

tabctx serves any tabular model that can split its work into *encode a
training table* and *predict against the encoding*. A model is one
class, a **backend**, plus a few lines of wiring. The cache, admission
control, placement, tenancy and HTTP API come with it: none of them
import a model, they only use this interface.

Open requests for new models: Mitra and TabDPT
([#102](https://github.com/VectorInstitute/tabctx/issues/102)).

## The interface

```python
--8<-- "src/tabctx/backends/base.py:15:"
```

### What `fit` and `predict` receive

- **`X`**: a 2-D `float32` numpy array. Missing values are `NaN`. Text
  columns arrive already encoded as category codes (0, 1, 2, ... in
  sorted order); a category the training table never had becomes `NaN`
  at predict. The model can treat every column as numeric.
- **`y`**: the labels. For classification, any values (tabctx returns
  them as strings); for regression, numbers.
- **`task`**: `"classification"` or `"regression"`.

### The rules that matter

- **A fresh model per fit.** `fit()` must build a new underlying
  estimator every call; the payload it returns is cached and must not be
  shared with any other context.
- **One pass per predict.** Derive labels from probabilities rather
  than running the model twice for `return_proba=True`
  (`tabctx.backends._outcome.classification_outcome` does it, and returns
  labels as strings).
- **Report the context's size.** `context_bytes_hint()` returns the
  bytes the fitted context occupies, measured after the fit (on a GPU,
  the allocator's growth), or `None` to let the engine estimate it.
- **Raise tabctx's errors.** Bad input raises `InvalidInputError`
  (HTTP 422); a device failure or out-of-memory raises
  `BackendComputeError` (507). Anything else reaches the caller as a
  500. `backends/tabpfn.py` shows both.
- **`name`** is the model id clients send as `model` and see in
  `GET /v1/models`, e.g. `tabicl-v2`.

## Optional hooks

The engine and the server probe for these with `getattr`, so a first
version needs none of them:

| Hook | What it buys |
|---|---|
| `warmup()` | Load weights at server start, not inside the first tenant's fit. |
| `fit_peak_bytes_hint()` | The fit's real peak memory, which the admission estimator learns from. |
| `dumps_payload()` / `loads_payload()` | Compact spill files, e.g. leaving shared weights out. Default: pickle. |
| `prepare_fit()` / `fit_prepared()` | Run CPU preprocessing while another fit holds the GPU. |
| `may_offload()` / `host_bytes_estimate()` | Host-memory admission for fits that stage data in RAM. |

## Wire it in

1. **The backend**, in `src/tabctx/backends/<model>.py`. Import the
   model's package inside the module (not in `tabctx/__init__.py`), so
   tabctx installs without it.
2. **The server**: in `src/tabctx/serve/factory.py`, add its kind to
   `BackendKind` and `BACKEND_KINDS`, and a branch for it in
   `_build_real_backend` (today an `if tabpfn / else tabicl`).
3. **The client**: add its id to `KNOWN_MODELS` in
   `src/tabctx/client.py`, so the client finds it among a gateway's
   models.
4. **Packaging**: an extra in `pyproject.toml` (`tabctx[<model>]`), and a
   row in the README's models table with the **weights license**.

## Memory calibration

Admission refuses a fit before it runs out of memory, using an estimate
of the fit's peak memory by table shape. A new model starts with the
fallback estimate (TabICL's, calibrated on an A100), which is
conservative, and then learns from every fit's measured peak.

A calibrated model gets its own grid of measured peaks:
`benchmarks/calibrate_memory.py` shows how they are measured (it drives
TabICL today; point it at your backend), the grids live in
`src/tabctx/memory/`, and `factory.py` maps each model to them
(`_REQUIREMENT_CALIBRATIONS`, `_PREDICT_CALIBRATIONS`,
`_preloaded_observations`). This can come in a later pull request.

## Tests

- **Unit tests** need no model weights: test your backend's own logic
  with small fakes, and use `tabctx.backends.fake.FakeBackend` for
  anything engine-side.
- **An integration test** that skips without the package
  (`pytest.importorskip`) and checks that predictions through the
  backend match the model's own package on a small table
  (`tests/integration/test_tabicl_kv_real.py` is an example).
- **Accuracy:** run `benchmarks/eval_real_datasets.py` against the model's
  own package and include the numbers in the pull request.
