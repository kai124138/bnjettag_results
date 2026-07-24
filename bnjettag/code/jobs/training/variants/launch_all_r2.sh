#!/usr/bin/env bash
# Launch the ROUND-2 BitLinear sweep on NRP (10 GPU Jobs): LR sweep + combos + seed repeats.
# Usage:  ./launch_all_r2.sh            # apply all 10 round-2 jobs
#         ./launch_all_r2.sh delete     # tear them all down
set -euo pipefail
CTX="nautilus"; NS="cms-ml"
HERE="$(cd "$(dirname "$0")" && pwd)"
JOBS=(small-lr05 small-lr075 small-lr10 small-lr10-warm3 combo-small combo-medium small-s1 small-s2 clspool-s1 clspool-s2)

if [ "${1:-apply}" = "delete" ]; then
  for j in "${JOBS[@]}"; do
    kubectl --context "$CTX" -n "$NS" delete job "kai-bn2-$j" --ignore-not-found
  done
  exit 0
fi

for j in "${JOBS[@]}"; do
  kubectl --context "$CTX" -n "$NS" delete job "kai-bn2-$j" --ignore-not-found >/dev/null 2>&1 || true
  kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn2-$j.yaml"
done
echo "=== launched ${#JOBS[@]} round-2 jobs ==="
kubectl --context "$CTX" -n "$NS" get jobs -l app=bnjet-variant-r2
