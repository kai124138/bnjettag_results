#!/usr/bin/env bash
# round-9: re-tune peak LR for the STANDARDIZED binary model.
set -euo pipefail
CTX="nautilus"; NS="cms-ml"; HERE="$(cd "$(dirname "$0")" && pwd)"
ALL=(kai-bn9-a8-std-lr02 kai-bn9-a8-std-lr05 kai-bn9-a8-std-lr10 kai-bn9-a8-std-lr20)
if [ "${1:-apply}" = "delete" ]; then
  for j in "${ALL[@]}"; do kubectl --context "$CTX" -n "$NS" delete job "$j" --ignore-not-found; done
  exit 0
fi
for j in "${ALL[@]}"; do
  kubectl --context "$CTX" -n "$NS" delete job "$j" --ignore-not-found >/dev/null 2>&1 || true
  kubectl --context "$CTX" -n "$NS" apply -f "$HERE/$j.yaml"
done
echo; kubectl --context "$CTX" -n "$NS" get jobs -l app=bnjet-r8
