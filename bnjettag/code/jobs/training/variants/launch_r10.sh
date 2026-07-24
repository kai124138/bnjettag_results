#!/usr/bin/env bash
# round-10: pairbias — r8 stdnn flagship recipe + opt-in pairwise-invariant attention-logit
# bias (3 seeds). PREREQ: ConfigMap kai-bn10-code must be built from the pair_bias code
# (see make_code_configmap.sh, name kai-bn10-code) — the jobs hard-fail on a stale ConfigMap.
# Usage: ./launch_r10.sh   |   ./launch_r10.sh delete
set -euo pipefail
CTX="nautilus"; NS="cms-ml"; HERE="$(cd "$(dirname "$0")" && pwd)"
ALL=(kai-bn10-pair-s1 kai-bn10-pair-s2 kai-bn10-pair-s3)
if [ "${1:-apply}" = "delete" ]; then
  for j in "${ALL[@]}"; do kubectl --context "$CTX" -n "$NS" delete job "$j" --ignore-not-found; done
  exit 0
fi
for j in "${ALL[@]}"; do
  kubectl --context "$CTX" -n "$NS" delete job "$j" --ignore-not-found >/dev/null 2>&1 || true
  kubectl --context "$CTX" -n "$NS" apply -f "$HERE/$j.yaml"
done
echo; echo "=== round-10 pairbias (3 seeds) launched ==="
kubectl --context "$CTX" -n "$NS" get jobs -l app=bnjet-r10
