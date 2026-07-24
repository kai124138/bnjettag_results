#!/usr/bin/env bash
# Snapshot of the variant sweep: job completion + each pod's latest progress line.
set -uo pipefail
CTX="nautilus"; NS="cms-ml"
JOBS=(tiny small medium rmsnorm sharednorm noposenc trueposenc clspool gelu sffree)
echo "=== jobs ($(date)) ==="
kubectl --context "$CTX" -n "$NS" get jobs -l app=bnjet-variant 2>/dev/null
echo "=== pods ==="
kubectl --context "$CTX" -n "$NS" get pods -l app=bnjet-variant -o wide 2>/dev/null
echo "=== latest train line per running variant ==="
for j in "${JOBS[@]}"; do
  pod=$(kubectl --context "$CTX" -n "$NS" get pods -l "bnv=$j" -o jsonpath='{.items[-1:].metadata.name}' 2>/dev/null)
  [ -z "$pod" ] && continue
  line=$(kubectl --context "$CTX" -n "$NS" logs "$pod" --tail=400 2>/dev/null \
           | grep -E "val_auc|Epoch|\[train\]|\[done\]|Traceback|Error" | tail -1)
  printf "%-12s %s\n" "$j" "${line:-<no log yet>}"
done
