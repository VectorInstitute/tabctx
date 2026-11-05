# Deploy on Kubernetes

The Helm chart in
[`deploy/helm/tabctx`](https://github.com/VectorInstitute/tabctx/tree/main/deploy/helm/tabctx)
runs one tabctx server on a GPU node, with its disk tier and a model
weights cache on persistent volumes.

```bash
# From a release (0.19 onwards):
helm install tabctx oci://ghcr.io/vectorinstitute/charts/tabctx --set 'models={tabicl,tabdpt}'

# Or from a clone of the repository:
helm install tabctx ./deploy/helm/tabctx --set 'models={tabicl,tabdpt}'
```

Then:

```bash
kubectl port-forward svc/tabctx 8000:8000
curl localhost:8000/v1/models
```

## What it creates

- A **Deployment** of one pod running `python -m tabctx.serve`, with one
  GPU (`gpus: 1`), probes on `/livez` (liveness and startup: the first
  start downloads weights) and `/readyz` (readiness).
- A **Service** on port 8000.
- Two **PersistentVolumeClaims**, kept when the release is uninstalled:
  the disk tier for evicted contexts (`spill.sizeGi`, 100 GiB) and the
  model weights (`weightsCache.sizeGi`, 20 GiB).
- A **Secret** for TabPFN's token, if you give one.

## Common settings

| Value | Default | |
|---|---|---|
| `models` | `[tabicl]` | Models to serve; the first is the default. `tabicl` and `tabdpt` are licensed for commercial use. |
| `gpus` | `1` | GPUs per pod; `0` runs on the CPU. |
| `requireTenant` | `true` | Every request names its tenant. |
| `tabpfnToken` / `existingTabpfnSecret` | | Required with `tabpfn` in `models`. |
| `spill.enabled`, `spill.sizeGi` | `true`, `100` | The disk tier; tabctx fills at most 90% of it. |
| `extraEnv` | `[]` | Any other [setting](reference/configuration.md). |
| `nodeSelector`, `tolerations`, `affinity`, `resources` | | Pod placement and size. |

`values.yaml` documents every value.

## In production

- **Authenticate in front of it.** tabctx trusts the tenant header as
  sent: put the Service behind a gateway that authenticates callers,
  sets `x-tabctx-tenant-id` itself and strips any value the client sent.
- **One replica.** A context lives in one process. Several replicas need
  a router that sends each dataset to the pod that holds it (Ray Serve
  does; see [serving](serving.md#scaling-out-multi-replica-serving)). For
  more capacity, give one pod more GPUs: tabctx places tables across them.
- **Offline clusters.** Point `HF_HOME` and `TABPFN_MODEL_CACHE_DIR` at a
  volume with the weights already in it (`extraEnv`), with
  `HF_HUB_OFFLINE=1`.
