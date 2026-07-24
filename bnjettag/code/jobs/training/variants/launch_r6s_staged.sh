#!/usr/bin/env bash
# Launch round-6-small on NRP, STAGED <=3 concurrent Jobs.
# Usage:  ./launch_r6s_staged.sh            # wave-by-wave
#         ./launch_r6s_staged.sh delete     # tear down
set -euo pipefail
CTX="nautilus"; NS="cms-ml"
HERE="$(cd "$(dirname "$0")" && pwd)"

ALL=(kai-bn6s-a8-s1 kai-bn6s-a6-s1 kai-bn6s-a4-s1 kai-bn6s-fp32-s1 kai-bn6s-w8a8-s1)
if [ "${1:-apply}" = "delete" ]; then
  for j in "${ALL[@]}"; do kubectl --context "$CTX" -n "$NS" delete job "$j" --ignore-not-found; done
  exit 0
fi

wait_for_wave () {
  for j in "$@"; do
    echo "[wait] $j"
    while true; do
      s=$(kubectl --context "$CTX" -n "$NS" get job "$j" -o jsonpath="{.status.conditions[?(@.status==\"True\")].type}" 2>/dev/null || true)
      case "$s" in *Complete*|*Failed*) echo "[done] $j -> $s"; break;; esac
      sleep 120
    done
  done
}

echo "=== wave 1: small-a8-s1 small-fp32-s1 small-w8a8-s1 ==="
kubectl --context "$CTX" -n "$NS" delete job "kai-bn6s-a8-s1" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn6s-a8-s1.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bn6s-fp32-s1" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn6s-fp32-s1.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bn6s-w8a8-s1" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn6s-w8a8-s1.yaml"
wait_for_wave kai-bn6s-a8-s1 kai-bn6s-fp32-s1 kai-bn6s-w8a8-s1

echo "=== wave 2: small-a6-s1 small-a4-s1 ==="
kubectl --context "$CTX" -n "$NS" delete job "kai-bn6s-a6-s1" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn6s-a6-s1.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bn6s-a4-s1" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn6s-a4-s1.yaml"
wait_for_wave kai-bn6s-a6-s1 kai-bn6s-a4-s1

echo "=== round-6-small complete ==="
kubectl --context "$CTX" -n "$NS" get jobs -l app=bnjet-r6-small
