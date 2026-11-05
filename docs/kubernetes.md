# Deploy on Kubernetes

The Helm chart in
[`deploy/helm/tabctx`](https://github.com/VectorInstitute/tabctx/tree/main/deploy/helm/tabctx)
runs one tabctx server on a GPU node, with a volume that keeps model
weights and compiled kernels across restarts.

```bash
# From a clone of the repository:
helm install tabctx ./deploy/helm/tabctx

# From a release, once the chart is published with one:
helm install tabctx oci://ghcr.io/vectorinstitute/charts/tabctx --version <version>
```

Then:

```bash
kubectl port-forward svc/tabctx 8000:8000
curl localhost:8000/v1/models
```

The chart serves TabICLv2 by default. `--set 'models={tabicl,tabdpt}'`
adds TabDPT (both licensed for commercial use) with an image built after
0.18.0, and `tabpfn` needs a [token](#tabpfn).

To try it on a cluster without GPUs (kind, minikube), shrink it:

```bash
helm install tabctx ./deploy/helm/tabctx --set gpus=0 \
  --set resources.requests.cpu=1 --set resources.requests.memory=4Gi \
  --set resources.limits.memory=4Gi --set weightsCache.sizeGi=5
```

## What it creates

- A **Deployment** of one pod running `python -m tabctx.serve`, with the
  GPUs you ask for (`gpus`), and probes on `/livez` (startup and
  liveness) and `/readyz` (readiness).
- A **Service** on port 8000.
- A **PersistentVolumeClaim** for weights and compiled kernels
  (`weightsCache`), kept when the release is uninstalled; and one for the
  disk tier if you turn it on (`spill`).
- A **Secret** for TabPFN's token, if you give the token itself.

The pod runs as the image's non-root user with no added capabilities and
the runtime's default seccomp profile, so it fits a namespace that
enforces the Pod Security "restricted" level.

## Settings

| Value | Default | Sets | |
|---|---|---|---|
| `models` | `[tabicl]` | `TABCTX_BACKEND` | Models to serve; the first is the default. |
| `gpus` | `1` | `nvidia.com/gpu` | GPUs per pod; `0` runs on the CPU. |
| `gpuMemoryFraction` | `1.0` | `TABCTX_GPU_MEMORY_FRACTION` | Share of each GPU tabctx budgets for. |
| `requireTenant` | `true` | `TABCTX_REQUIRE_TENANT` | Every request names its tenant. |
| `spill.enabled`, `spill.sizeGi` | `false`, `100` | `TABCTX_SPILL_DIR` | The disk tier; tabctx fills at most 90% of it. |
| `weightsCache.enabled`, `.sizeGi`, `.existingClaim` | `true`, `20` | `HF_HOME`, ... | Weights and compiled kernels. |
| `existingTabpfnSecret` / `tabpfnToken` | | `TABPFN_TOKEN` | Required with `tabpfn`. |
| `extraEnv`, `extraVolumes`, `extraVolumeMounts` | | | Any other [setting](reference/configuration.md); a name the chart sets is replaced by yours. |
| `nodeSelector`, `tolerations`, `affinity`, `resources` | | | Placement and size. |

`values.yaml` documents every value.

## Sizing

- **GPU memory** bounds how many fitted tables a pod keeps: tabctx
  admits each fit against its budget for the GPU and evicts the least
  recently used tables to make room.
- **Host memory**: requests equal limits by default, because tabctx
  budgets large fits against the limit (85% of it, less 8 GiB). Large
  fits stage data in host memory, by tabctx's own estimates up to about
  5 KB per cell for a TabICL table too large for the GPU (5 GB per million
  cells) and 600 bytes per cell for TabPFN. Size `resources.limits.memory`
  for the largest tables you will fit.
- **Scratch**: uploads and TabICL's staging go to an `emptyDir` at `/tmp`
  (`tmp.sizeLimit`, 50 GiB).

## Exposing it

Put an authenticating gateway in front: tabctx trusts the tenant header
as sent, so the gateway must authenticate each caller, set
`x-tabctx-tenant-id` itself and strip any value the client sent. Whatever
proxies it needs large bodies and long timeouts: tables up to 4 GiB
(`TABCTX_MAX_UPLOAD_BYTES`) and fits that can take minutes. With
ingress-nginx, for example:

```yaml
nginx.ingress.kubernetes.io/proxy-body-size: "4g"
nginx.ingress.kubernetes.io/proxy-read-timeout: "900"
nginx.ingress.kubernetes.io/proxy-send-timeout: "900"
```

## Upgrades and restarts

A fitted table lives in the server's memory. A restart (an upgrade, a
reschedule) drops every table it holds: clients get 404 and fit again,
which the scikit-learn estimators do on their own. Tables already
evicted to the disk tier, when it is on, come back on their next
predict. On shutdown, in-flight requests get
`terminationGracePeriodSeconds` (300) to finish. Each start loads the
weights and compiled kernels from the weights volume; the first start
downloads and compiles them (`startupSeconds`, 1200, allows for it;
raise it for several models).

## Several GPUs and replicas

Give one pod more GPUs (`gpus: 4`): tabctx places each table on one of
them, with one worker process per GPU. Keep `replicaCount: 1`: the
volumes are ReadWriteOnce, and a fitted table lives in one pod, so
several replicas need a router that sends every request for a dataset to
the pod that holds it. The chart refuses `replicaCount > 1` with its
volumes.

## Offline clusters

Pre-fill a volume with the weights (`HF_HOME/hub` for TabICL and TabDPT,
`TABPFN_MODEL_CACHE_DIR` for TabPFN), give it as
`weightsCache.existingClaim`, and add `HF_HUB_OFFLINE=1` to `extraEnv`.
With TabPFN's weights already there, `tabpfnWithoutToken: true` skips the
token check.

## TabPFN

TabPFN's weights need a token from [Prior Labs](https://ux.priorlabs.ai)
after you accept their terms, and are licensed for non-commercial use.
Put the token in a Secret with a `TABPFN_TOKEN` key and name it in
`existingTabpfnSecret` (a token given as `tabpfnToken` is stored in the
release).

## Monitoring

`GET /readyz` reports the cache (`cache_stats`), the GPU memory torch
really holds (`real_gpu_memory`, measured every 5 s; `device_report_age_s`
says how long ago), host memory, and the server's uptime: a
burst of 404s with a reset uptime means a restart, not an eviction.
