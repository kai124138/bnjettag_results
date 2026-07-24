#!/usr/bin/env bash
# round-7 stage-0: peak-LR probe (binary W1A8 @ small)
# Usage:  ./launch_r7_probe.sh            # apply all (cluster scheduler self-limits; survives laptop close)
#         ./launch_r7_probe.sh delete     # tear down
set -euo pipefail
CTX="nautilus"; NS="cms-ml"
HERE="$(cd "$(dirname "$0")" && pwd)"

ALL=(kai-bn7p-lr05 kai-bn7p-lr20 kai-bn7p-lr50 kai-bn7p-lr100)
if [ "${1:-apply}" = "delete" ]; then
  for j in "${ALL[@]}"; do kubectl --context "$CTX" -n "$NS" delete job "$j" --ignore-not-found; done
  exit 0
fi

for j in "${ALL[@]}"; do
  kubectl --context "$CTX" -n "$NS" delete job "$j" --ignore-not-found >/dev/null 2>&1 || true
  kubectl --context "$CTX" -n "$NS" apply -f "$HERE/$j.yaml"
done

echo
echo "=== round-7 stage-0: peak-LR probe (binary W1A8 @ small) launched ==="
kubectl --context "$CTX" -n "$NS" get jobs -l app=bnjet-r7-probe
