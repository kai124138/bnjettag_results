#!/usr/bin/env bash
# Launch the ROUND-5 quantization sweep on NRP, STAGED <=3 concurrent Jobs.
# Usage:  ./launch_r5_staged.sh            # launch wave-by-wave, waiting between waves
#         ./launch_r5_staged.sh delete     # tear everything down
set -euo pipefail
CTX="nautilus"; NS="cms-ml"
HERE="$(cd "$(dirname "$0")" && pwd)"

ALL=(kai-bn5-large-lr05-s1 kai-bn5-large-lr05-s2 kai-bn5-large-lr05-a6-s1 kai-bn5-large-lr05-a6-s2 kai-bn5-large-lr05-a4-s1 kai-bn5-large-lr05-a4-s2 kai-bn5-fp32-lr05 kai-bn5-w8a8-lr05)
if [ "${1:-apply}" = "delete" ]; then
  for j in "${ALL[@]}"; do kubectl --context "$CTX" -n "$NS" delete job "$j" --ignore-not-found; done
  exit 0
fi

wait_for_wave () {   # block until every job named in $@ is Complete or Failed
  for j in "$@"; do
    echo "[wait] $j"
    while true; do
      s=$(kubectl --context "$CTX" -n "$NS" get job "$j" -o jsonpath="{.status.conditions[?(@.status==\"True\")].type}" 2>/dev/null || true)
      case "$s" in *Complete*|*Failed*) echo "[done] $j -> $s"; break;; esac
      sleep 120
    done
  done
}

echo "=== wave 1: large-lr05-s1 large-lr05-s2 fp32-lr05 ==="
kubectl --context "$CTX" -n "$NS" delete job "kai-bn5-large-lr05-s1" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn5-large-lr05-s1.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bn5-large-lr05-s2" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn5-large-lr05-s2.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bn5-fp32-lr05" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn5-fp32-lr05.yaml"
wait_for_wave kai-bn5-large-lr05-s1 kai-bn5-large-lr05-s2 kai-bn5-fp32-lr05

echo "=== wave 2: large-lr05-a6-s1 large-lr05-a6-s2 w8a8-lr05 ==="
kubectl --context "$CTX" -n "$NS" delete job "kai-bn5-large-lr05-a6-s1" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn5-large-lr05-a6-s1.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bn5-large-lr05-a6-s2" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn5-large-lr05-a6-s2.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bn5-w8a8-lr05" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn5-w8a8-lr05.yaml"
wait_for_wave kai-bn5-large-lr05-a6-s1 kai-bn5-large-lr05-a6-s2 kai-bn5-w8a8-lr05

echo "=== wave 3: large-lr05-a4-s1 large-lr05-a4-s2 ==="
kubectl --context "$CTX" -n "$NS" delete job "kai-bn5-large-lr05-a4-s1" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn5-large-lr05-a4-s1.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bn5-large-lr05-a4-s2" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn5-large-lr05-a4-s2.yaml"
wait_for_wave kai-bn5-large-lr05-a4-s1 kai-bn5-large-lr05-a4-s2

echo "=== round-5 complete ==="
kubectl --context "$CTX" -n "$NS" get jobs -l app=bnjet-variant-r5
