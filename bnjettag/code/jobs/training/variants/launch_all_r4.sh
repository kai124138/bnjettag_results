#!/usr/bin/env bash
# Launch the ROUND-4 BitLinear sweep on NRP (4 GPU Jobs): seed-confirm medium@5e-5 + tune large.
# Usage:  ./launch_all_r4.sh            # apply all 4 round-4 jobs
#         ./launch_all_r4.sh delete     # tear them all down
set -euo pipefail
CTX="nautilus"; NS="cms-ml"
HERE="$(cd "$(dirname "$0")" && pwd)"
JOBS=(medium-lr05-s1 medium-lr05-s2 large-lr05 large-lr075)

if [ "${1:-apply}" = "delete" ]; then
  for j in "${JOBS[@]}"; do
    kubectl --context "$CTX" -n "$NS" delete job "kai-bn4-$j" --ignore-not-found
  done
  exit 0
fi

for j in "${JOBS[@]}"; do
  kubectl --context "$CTX" -n "$NS" delete job "kai-bn4-$j" --ignore-not-found >/dev/null 2>&1 || true
  kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn4-$j.yaml"
done
echo "=== launched ${#JOBS[@]} round-4 jobs ==="
kubectl --context "$CTX" -n "$NS" get jobs -l app=bnjet-variant-r4
