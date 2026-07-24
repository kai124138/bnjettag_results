#!/usr/bin/env bash
# Launch the ROUND-3 BitLinear sweep on NRP (6 GPU Jobs): LR ceiling + confirm-with-seeds + medium x LR.
# Usage:  ./launch_all_r3.sh            # apply all 6 round-3 jobs
#         ./launch_all_r3.sh delete     # tear them all down
set -euo pipefail
CTX="nautilus"; NS="cms-ml"
HERE="$(cd "$(dirname "$0")" && pwd)"
JOBS=(small-lr025 small-lr035 small-lr05-s1 small-lr05-s2 medium-lr05 medium-lr075)

if [ "${1:-apply}" = "delete" ]; then
  for j in "${JOBS[@]}"; do
    kubectl --context "$CTX" -n "$NS" delete job "kai-bn3-$j" --ignore-not-found
  done
  exit 0
fi

for j in "${JOBS[@]}"; do
  kubectl --context "$CTX" -n "$NS" delete job "kai-bn3-$j" --ignore-not-found >/dev/null 2>&1 || true
  kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn3-$j.yaml"
done
echo "=== launched ${#JOBS[@]} round-3 jobs ==="
kubectl --context "$CTX" -n "$NS" get jobs -l app=bnjet-variant-r3
