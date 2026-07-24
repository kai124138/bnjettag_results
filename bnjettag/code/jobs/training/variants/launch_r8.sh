#!/usr/bin/env bash
# round-8: the input-standardization 2x2 (mechanism test).
# Usage: ./launch_r8.sh   |   ./launch_r8.sh delete
set -euo pipefail
CTX="nautilus"; NS="cms-ml"; HERE="$(cd "$(dirname "$0")" && pwd)"
ALL=(kai-bn8-a8-nostd kai-bn8-a8-std kai-bn8-fp32-nostd kai-bn8-fp32-std)
if [ "${1:-apply}" = "delete" ]; then
  for j in "${ALL[@]}"; do kubectl --context "$CTX" -n "$NS" delete job "$j" --ignore-not-found; done
  exit 0
fi
for j in "${ALL[@]}"; do
  kubectl --context "$CTX" -n "$NS" delete job "$j" --ignore-not-found >/dev/null 2>&1 || true
  kubectl --context "$CTX" -n "$NS" apply -f "$HERE/$j.yaml"
done
echo; echo "=== round-8 standardization 2x2 launched ==="
kubectl --context "$CTX" -n "$NS" get jobs -l app=bnjet-r8
