#!/usr/bin/env bash
# Launch the full BitLinear-transformer variant sweep on NRP (9 GPU Jobs).
# Each Job queues independently; k8s places them across whatever GPUs are free.
# Usage:  ./launch_all.sh            # apply all 9 variant jobs
#         ./launch_all.sh delete     # tear them all down
set -euo pipefail
CTX="nautilus"; NS="cms-ml"
HERE="$(cd "$(dirname "$0")" && pwd)"
JOBS=(tiny small medium rmsnorm sharednorm noposenc trueposenc clspool gelu sffree)

if [ "${1:-apply}" = "delete" ]; then
  for j in "${JOBS[@]}"; do
    kubectl --context "$CTX" -n "$NS" delete job "kai-bnv-$j" --ignore-not-found
  done
  exit 0
fi

for j in "${JOBS[@]}"; do
  kubectl --context "$CTX" -n "$NS" delete job "kai-bnv-$j" --ignore-not-found >/dev/null 2>&1 || true
  kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bnv-$j.yaml"
done
echo "=== launched ${#JOBS[@]} variant jobs ==="
kubectl --context "$CTX" -n "$NS" get jobs -l app=bnjet-variant
