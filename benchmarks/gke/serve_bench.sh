#!/usr/bin/env bash
# Serve tabctx with Ray Serve inside the devbox pod, configured like prod's
# tabctx app (one replica over all the pod's GPUs, a worker process per GPU,
# prod's env), and run benchmarks/prod_bench.py against it. Run inside the pod:
#   devbox.sh exec -- 'bash benchmarks/gke/serve_bench.sh out/bench.json [prod_bench args...]'
set -euo pipefail
OUT="$1"; shift
export TABCTX_BACKEND="${TABCTX_BACKEND:-tabicl,tabpfn}"
export TABCTX_SPILL_DIR="${TABCTX_SPILL_DIR:-/tmp/tabctx-spill}"
export TABCTX_SPILL_CAPACITY_BYTES="${TABCTX_SPILL_CAPACITY_BYTES:-53687091200}"
export TABPFN_MODEL_VERSION="${TABPFN_MODEL_VERSION:-v3.5}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
mkdir -p out
rm -rf "$TABCTX_SPILL_DIR"
ray stop --force >/dev/null 2>&1 || true
ray start --head --disable-usage-stats --include-dashboard=true --dashboard-host=127.0.0.1 >/dev/null
nohup serve run tabctx.serve.app:app > out/serve.log 2>&1 &
SERVE_PID=$!
trap 'kill $SERVE_PID 2>/dev/null || true; serve shutdown -y >/dev/null 2>&1 || true; ray stop --force >/dev/null 2>&1 || true' EXIT
for _ in $(seq 1 240); do
  curl -sf http://127.0.0.1:8000/readyz > /dev/null && break
  sleep 5
done
curl -sf http://127.0.0.1:8000/readyz > /dev/null || { tail -50 out/serve.log; exit 1; }
echo "serving; starting benchmark"
VP_API_KEY=local python benchmarks/prod_bench.py --base-url http://127.0.0.1:8000/v1 \
  --dashboard http://127.0.0.1:8265 --save "$OUT" "$@"
