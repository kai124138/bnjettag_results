#!/usr/bin/env bash
# round-7 stage-0b: diagnostic wave (FP32-vs-binary @ small, schedule, LR floor)
# Usage:  ./launch_r7_diag.sh            # apply all (cluster scheduler self-limits; survives laptop close)
#         ./launch_r7_diag.sh delete     # tear down
set -euo pipefail
CTX="nautilus"; NS="cms-ml"
HERE="$(cd "$(dirname "$0")" && pwd)"

ALL=(kai-bn7d-small-fp32 kai-bn7d-small-w8a8 kai-bn7d-small-a8-long kai-bn7d-small-a8-lr02)
if [ "${1:-apply}" = "delete" ]; then
  for j in "${ALL[@]}"; do kubectl --context "$CTX" -n "$NS" delete job "$j" --ignore-not-found; done
  exit 0
fi

for j in "${ALL[@]}"; do
  kubectl --context "$CTX" -n "$NS" delete job "$j" --ignore-not-found >/dev/null 2>&1 || true
  kubectl --context "$CTX" -n "$NS" apply -f "$HERE/$j.yaml"
done

echo
echo "=== round-7 stage-0b: diagnostic wave (FP32-vs-binary @ small, schedule, LR floor) launched ==="
kubectl --context "$CTX" -n "$NS" get jobs -l app=bnjet-r7-diag
