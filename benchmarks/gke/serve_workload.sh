#!/usr/bin/env bash
# Serve tabctx with Ray Serve inside the devbox pod (all of its GPUs, one
# replica, as prod's tabctx worker group does) and run the mixed workload
# against it. Run inside the pod (devbox.sh exec -- 'bash benchmarks/gke/serve_workload.sh ...').
#
#   serve_workload.sh OUT.json [USERS] [DURATION_S] [extra env assignments...]
#   e.g. serve_workload.sh out/wl-procs.json 6 300 TABCTX_WORKER_PROCESSES=1
set -euo pipefail
OUT="$1"; USERS="${2:-6}"; DURATION="${3:-300}"; shift $(( $# < 3 ? $# : 3 ))
for kv in "$@"; do export "$kv"; done
export TABCTX_BACKEND="${TABCTX_BACKEND:-tabicl,tabpfn}"
export TABCTX_SPILL_DIR="${TABCTX_SPILL_DIR:-/tmp/tabctx-spill}"
# A num_gpus=0 actor gets CUDA_VISIBLE_DEVICES="" from Ray unless told not to
# (prod sets the same; see inference-platform's tabctx worker group).
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
rm -rf "$TABCTX_SPILL_DIR"
ray stop --force >/dev/null 2>&1 || true
ray start --head --disable-usage-stats >/dev/null
nohup serve run tabctx.serve.app:app > out/serve.log 2>&1 &
SERVE_PID=$!
trap 'kill $SERVE_PID 2>/dev/null || true; serve shutdown -y >/dev/null 2>&1 || true; ray stop --force >/dev/null 2>&1 || true' EXIT
for _ in $(seq 1 180); do
  if curl -sf http://127.0.0.1:8000/readyz > out/readyz-before.json; then break; fi
  sleep 5
done
curl -sf http://127.0.0.1:8000/readyz > /dev/null || { tail -50 out/serve.log; exit 1; }
python -c "import json; d=json.load(open('out/readyz-before.json')); print('ready:', d['device'], 'workers', len(d.get('devices') or []))"
python benchmarks/bench_workload.py --base-url http://127.0.0.1:8000 \
  --users "$USERS" --duration "$DURATION" --save "$OUT"
curl -sf http://127.0.0.1:8000/readyz > "${OUT%.json}-readyz.json" || true
