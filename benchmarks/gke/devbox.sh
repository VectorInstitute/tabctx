#!/usr/bin/env bash
# GKE GPU devbox for profiling/optimizing tabctx on real GPUs (#31-#36).
#
# One GPU node pool and one long-running pod with the PROD image (same torch/CUDA
# as inference-platform), the staged tabctx weights (offline, no token), tabicl/
# tabpfn at prod's pins, and this checkout of tabctx installed editable. Iterate
# with `sync` (copy the working tree in) and `exec` (run anything there).
#
#   benchmarks/gke/devbox.sh up            # create cluster + pod (MACHINES order)
#   benchmarks/gke/devbox.sh sync          # copy src/ benchmarks/ tests/ into the pod
#   benchmarks/gke/devbox.sh exec -- CMD   # run CMD in the pod (cwd /work/tabctx)
#                                         # (CMD is joined into one shell string: quote
#                                         #  pipes/redirects as one argument, no bash -c)
#   benchmarks/gke/devbox.sh fetch PATH    # copy /work/tabctx/PATH back to ./gke-out/
#   benchmarks/gke/devbox.sh down          # delete the cluster
#
# The cluster carries purpose=gke-mtest + expiry_epoch (MAX_HOURS) so the org's orphan
# sweep deletes it even if `down` never runs.
set -euo pipefail

PROJECT="${PROJECT:-agentic-ai-evaluation-bootcamp}"
CLUSTER="${CLUSTER:-gke-mtest-tabctx-devbox}"
STATE="${STATE:-$HOME/.cache/tabctx-devbox}"   # remembers the zone/machine chosen
MACHINES="${MACHINES:-a2-highgpu-1g g2-standard-12}"  # 1x A100-40GB, else 1x L4
ZONES="${ZONES:-us-central1-a us-central1-b us-central1-c us-central1-f us-east4-a us-east4-c us-east1-b us-east1-c us-east1-d us-west1-a us-west1-b}"
MAX_HOURS="${MAX_HOURS:-12}"
IMAGE="${IMAGE:-rayproject/ray-llm:nightly.261006.670387-extra-py312-cu130}"
WEIGHTS_BUCKET="${WEIGHTS_BUCKET:-gs://inference-platform-gke-model-weights-bootcamp/tabctx}"
NS=default
POD=tabctx-devbox
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
mkdir -p "$STATE"
# A private kubeconfig: get-credentials would otherwise switch the user's (and any
# concurrently running script's) current kubectl context to this cluster.
export KUBECONFIG="$STATE/kubeconfig"

accelerator_for() {
  case "$1" in
    a2-highgpu-1g) echo "type=nvidia-tesla-a100,count=1" ;;
    a2-highgpu-4g) echo "type=nvidia-tesla-a100,count=4" ;;
    g2-standard-12) echo "type=nvidia-l4,count=1" ;;
    g2-standard-48) echo "type=nvidia-l4,count=4" ;;
    *) echo "unknown machine $1" >&2; exit 1 ;;
  esac
}
gpus_for() { accelerator_for "$1" | sed -E 's/.*count=([0-9]+).*/\1/'; }

has_capacity() {  # zone machine
  local probe="${CLUSTER}-probe"
  if gcloud compute instances create "$probe" --project="$PROJECT" --zone="$1" \
       --machine-type="$2" --maintenance-policy=TERMINATE --no-address \
       --image-family=debian-12 --image-project=debian-cloud --boot-disk-size=10GB \
       --labels="purpose=gke-mtest,test=tabctx-devbox" --quiet >/dev/null 2>"$STATE/probe.err"; then
    gcloud compute instances delete "$probe" --project="$PROJECT" --zone="$1" --quiet >/dev/null 2>&1
    return 0
  fi
  echo "    $1/$2: $(grep -oE 'ZONE_RESOURCE_POOL_EXHAUSTED[A-Z_]*|[Qq]uota [^.]*|does not exist[^.]*' "$STATE/probe.err" | head -1)"
  return 1
}

creds() {
  # shellcheck disable=SC1091
  source "$STATE/env"
  gcloud container clusters get-credentials "$CLUSTER" --zone="$ZONE" --project="$PROJECT" >/dev/null 2>&1
  export KUBECONFIG_CONTEXT
  KUBECONFIG_CONTEXT="gke_${PROJECT}_${ZONE}_${CLUSTER}"
}
k() { kubectl --context "$KUBECONFIG_CONTEXT" "$@"; }

up() {
  if [ ! -f "$STATE/env" ]; then
    for m in $MACHINES; do
      for z in $ZONES; do
        if has_capacity "$z" "$m"; then printf 'ZONE=%s\nMACHINE=%s\n' "$z" "$m" > "$STATE/env"; break 2; fi
      done
    done
    [ -f "$STATE/env" ] || { echo "no capacity for any of: $MACHINES" >&2; exit 1; }
  fi
  # shellcheck disable=SC1091
  source "$STATE/env"
  echo "==> $CLUSTER in $ZONE on $MACHINE"
  if ! gcloud container clusters describe "$CLUSTER" --project="$PROJECT" --zone="$ZONE" >/dev/null 2>&1; then
    gcloud container clusters create "$CLUSTER" --project="$PROJECT" --zone="$ZONE" \
      --release-channel=regular --num-nodes=1 --machine-type=e2-standard-4 \
      --labels="purpose=gke-mtest,test=tabctx-devbox,expiry_epoch=$(( $(date +%s) + MAX_HOURS * 3600 ))"
  fi
  creds
  if ! gcloud container node-pools describe gpu --cluster="$CLUSTER" --project="$PROJECT" --zone="$ZONE" >/dev/null 2>&1; then
    gcloud container node-pools create gpu --project="$PROJECT" --zone="$ZONE" --cluster="$CLUSTER" \
      --machine-type="$MACHINE" --accelerator="$(accelerator_for "$MACHINE"),gpu-driver-version=latest" \
      --num-nodes=1 --disk-type=pd-ssd --disk-size=300 --no-enable-autoupgrade --no-enable-autorepair
  fi
  k wait --for=condition=Ready node -l cloud.google.com/gke-nodepool=gpu --timeout=20m
  local gpus; gpus=$(gpus_for "$MACHINE")
  local phase; phase=$(k get pod $POD -o jsonpath='{.status.phase}' 2>/dev/null || true)
  if [ -n "$phase" ] && [ "$phase" != "Running" ] && [ "$phase" != "Pending" ]; then
    echo "==> pod is $phase (e.g. OOMKilled); recreating"
    k delete pod $POD --wait
  fi
  cat <<YAML | k apply -f -
apiVersion: v1
kind: Pod
metadata: {name: $POD, namespace: $NS}
spec:
  nodeSelector: {cloud.google.com/gke-nodepool: gpu}
  restartPolicy: Never
  volumes:
    - {name: weights, emptyDir: {}}
    - {name: work, emptyDir: {}}
    - {name: shm, emptyDir: {medium: Memory, sizeLimit: 16Gi}}
  initContainers:
    - name: stage-weights
      image: google/cloud-sdk:slim
      command: ["gcloud", "storage", "rsync", "-r", "$WEIGHTS_BUCKET", "/model-weights/tabctx"]
      volumeMounts: [{name: weights, mountPath: /model-weights/tabctx}]
  containers:
    - name: box
      image: $IMAGE
      command: ["sleep", "infinity"]
      # A memory limit, as KubeRay sets in prod: host-RAM exhaustion is then
      # a cgroup OOM inside the container, and the limit is readable there.
      resources: {limits: {nvidia.com/gpu: $gpus, memory: ${POD_MEMORY:-64Gi}}}
      env:
        - {name: TABPFN_MODEL_CACHE_DIR, value: /model-weights/tabctx/tabpfn}
        - {name: HF_HUB_CACHE, value: /model-weights/tabctx/hf/hub}
        - {name: HF_HUB_OFFLINE, value: "1"}
        - {name: TABPFN_MODEL_VERSION, value: "v3.5"}
        - {name: TABPFN_DISABLE_TELEMETRY, value: "1"}
      volumeMounts:
        - {name: weights, mountPath: /model-weights/tabctx, readOnly: true}
        - {name: work, mountPath: /work}
        - {name: shm, mountPath: /dev/shm}
YAML
  k wait --for=condition=Ready pod/$POD --timeout=30m
  sync
  k exec $POD -- bash -lc 'pip install -q tabicl==2.2.0 tabpfn==9.1.0 && pip install -q -e "/work/tabctx[serve]" pytest scikit-learn && python -c "import torch,tabicl,tabpfn,tabctx;print(torch.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(0), tabctx.__file__)"'
}

sync() {
  creds
  k exec $POD -- mkdir -p /work/tabctx
  tar -C "$REPO" --exclude='__pycache__' --exclude='.venv' -czf - src benchmarks tests pyproject.toml README.md LICENSE.md \
    | k exec -i $POD -- tar -C /work/tabctx -xzf -
  echo "synced $(git -C "$REPO" rev-parse --short HEAD)$(git -C "$REPO" diff --quiet || echo +dirty)"
}

case "${1:-}" in
  up) up ;;
  sync) sync ;;
  exec) shift; [ "${1:-}" = "--" ] && shift; creds; k exec $POD -- bash -lc "cd /work/tabctx && $*" ;;
  fetch) creds; mkdir -p gke-out; k cp "$NS/$POD:/work/tabctx/$2" "gke-out/$(basename "$2")" ;;
  down)
    # Only the zone, never credentials: a cluster whose creation failed (state
    # ERROR) has none to fetch, and must still be deletable.
    # shellcheck disable=SC1091
    [ -f "$STATE/env" ] && source "$STATE/env"
    ZONE="${ZONE:-$(gcloud container clusters list --project="$PROJECT" \
      --filter="name=$CLUSTER" --format='value(location)')}"
    [ -n "$ZONE" ] || { echo "no cluster $CLUSTER" >&2; exit 1; }
    gcloud container clusters delete "$CLUSTER" --project="$PROJECT" --zone="$ZONE" --quiet --async
    rm -f "$STATE/env" ;;
  *) sed -n 2,16p "$0"; exit 1 ;;
esac
