#!/usr/bin/env bash
# Launch the ROUND-7 QAT-stack matrix on NRP, STAGED (one full precision sweep per wave).
# Usage:  ./launch_r7_staged.sh            # wave-by-wave (needs laptop alive)
#         ./launch_r7_staged.sh delete     # tear everything down
# PVC-FREE. PRE-REQS (in order):
#   1. ./make_code_configmap.sh    # code source ConfigMap kai-bnf-code (now packs r7 configs + preflight_r7.sh)
#   2. preflight_r7.sh -> PREFLIGHT_ALL_PASS   (CPU pod with the ConfigMap mounted)
set -euo pipefail
CTX="nautilus"; NS="cms-ml"
HERE="$(cd "$(dirname "$0")" && pwd)"

ALL=(kai-bn7f-small-fp32-s1 kai-bn7f-small-w8a8-s1 kai-bn7f-small-w1a8-s1 kai-bn7f-small-w1a6-s1 kai-bn7f-small-w1a4-s1 kai-bn7f-small-fp32-s2 kai-bn7f-small-w8a8-s2 kai-bn7f-small-w1a8-s2 kai-bn7f-small-w1a6-s2 kai-bn7f-small-w1a4-s2 kai-bn7f-small-fp32-s3 kai-bn7f-small-w8a8-s3 kai-bn7f-small-w1a8-s3 kai-bn7f-small-w1a6-s3 kai-bn7f-small-w1a4-s3 kai-bn7f-tiny-fp32-s1 kai-bn7f-tiny-w8a8-s1 kai-bn7f-tiny-w1a8-s1 kai-bn7f-tiny-w1a6-s1 kai-bn7f-tiny-w1a4-s1 kai-bn7f-tiny-fp32-s2 kai-bn7f-tiny-w8a8-s2 kai-bn7f-tiny-w1a8-s2 kai-bn7f-tiny-w1a6-s2 kai-bn7f-tiny-w1a4-s2 kai-bn7f-tiny-fp32-s3 kai-bn7f-tiny-w8a8-s3 kai-bn7f-tiny-w1a8-s3 kai-bn7f-tiny-w1a6-s3 kai-bn7f-tiny-w1a4-s3)
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
      sleep 60
    done
  done
}

echo "=== wave 1: small-fp32-s1 small-w8a8-s1 small-w1a8-s1 small-w1a6-s1 small-w1a4-s1 ==="
kubectl --context "$CTX" -n "$NS" delete job "kai-bn7f-small-fp32-s1" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn7f-small-fp32-s1.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bn7f-small-w8a8-s1" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn7f-small-w8a8-s1.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bn7f-small-w1a8-s1" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn7f-small-w1a8-s1.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bn7f-small-w1a6-s1" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn7f-small-w1a6-s1.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bn7f-small-w1a4-s1" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn7f-small-w1a4-s1.yaml"
wait_for_wave kai-bn7f-small-fp32-s1 kai-bn7f-small-w8a8-s1 kai-bn7f-small-w1a8-s1 kai-bn7f-small-w1a6-s1 kai-bn7f-small-w1a4-s1

echo "=== wave 2: small-fp32-s2 small-w8a8-s2 small-w1a8-s2 small-w1a6-s2 small-w1a4-s2 ==="
kubectl --context "$CTX" -n "$NS" delete job "kai-bn7f-small-fp32-s2" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn7f-small-fp32-s2.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bn7f-small-w8a8-s2" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn7f-small-w8a8-s2.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bn7f-small-w1a8-s2" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn7f-small-w1a8-s2.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bn7f-small-w1a6-s2" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn7f-small-w1a6-s2.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bn7f-small-w1a4-s2" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn7f-small-w1a4-s2.yaml"
wait_for_wave kai-bn7f-small-fp32-s2 kai-bn7f-small-w8a8-s2 kai-bn7f-small-w1a8-s2 kai-bn7f-small-w1a6-s2 kai-bn7f-small-w1a4-s2

echo "=== wave 3: small-fp32-s3 small-w8a8-s3 small-w1a8-s3 small-w1a6-s3 small-w1a4-s3 ==="
kubectl --context "$CTX" -n "$NS" delete job "kai-bn7f-small-fp32-s3" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn7f-small-fp32-s3.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bn7f-small-w8a8-s3" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn7f-small-w8a8-s3.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bn7f-small-w1a8-s3" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn7f-small-w1a8-s3.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bn7f-small-w1a6-s3" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn7f-small-w1a6-s3.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bn7f-small-w1a4-s3" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn7f-small-w1a4-s3.yaml"
wait_for_wave kai-bn7f-small-fp32-s3 kai-bn7f-small-w8a8-s3 kai-bn7f-small-w1a8-s3 kai-bn7f-small-w1a6-s3 kai-bn7f-small-w1a4-s3

echo "=== wave 4: tiny-fp32-s1 tiny-w8a8-s1 tiny-w1a8-s1 tiny-w1a6-s1 tiny-w1a4-s1 ==="
kubectl --context "$CTX" -n "$NS" delete job "kai-bn7f-tiny-fp32-s1" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn7f-tiny-fp32-s1.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bn7f-tiny-w8a8-s1" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn7f-tiny-w8a8-s1.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bn7f-tiny-w1a8-s1" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn7f-tiny-w1a8-s1.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bn7f-tiny-w1a6-s1" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn7f-tiny-w1a6-s1.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bn7f-tiny-w1a4-s1" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn7f-tiny-w1a4-s1.yaml"
wait_for_wave kai-bn7f-tiny-fp32-s1 kai-bn7f-tiny-w8a8-s1 kai-bn7f-tiny-w1a8-s1 kai-bn7f-tiny-w1a6-s1 kai-bn7f-tiny-w1a4-s1

echo "=== wave 5: tiny-fp32-s2 tiny-w8a8-s2 tiny-w1a8-s2 tiny-w1a6-s2 tiny-w1a4-s2 ==="
kubectl --context "$CTX" -n "$NS" delete job "kai-bn7f-tiny-fp32-s2" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn7f-tiny-fp32-s2.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bn7f-tiny-w8a8-s2" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn7f-tiny-w8a8-s2.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bn7f-tiny-w1a8-s2" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn7f-tiny-w1a8-s2.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bn7f-tiny-w1a6-s2" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn7f-tiny-w1a6-s2.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bn7f-tiny-w1a4-s2" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn7f-tiny-w1a4-s2.yaml"
wait_for_wave kai-bn7f-tiny-fp32-s2 kai-bn7f-tiny-w8a8-s2 kai-bn7f-tiny-w1a8-s2 kai-bn7f-tiny-w1a6-s2 kai-bn7f-tiny-w1a4-s2

echo "=== wave 6: tiny-fp32-s3 tiny-w8a8-s3 tiny-w1a8-s3 tiny-w1a6-s3 tiny-w1a4-s3 ==="
kubectl --context "$CTX" -n "$NS" delete job "kai-bn7f-tiny-fp32-s3" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn7f-tiny-fp32-s3.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bn7f-tiny-w8a8-s3" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn7f-tiny-w8a8-s3.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bn7f-tiny-w1a8-s3" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn7f-tiny-w1a8-s3.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bn7f-tiny-w1a6-s3" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn7f-tiny-w1a6-s3.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bn7f-tiny-w1a4-s3" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bn7f-tiny-w1a4-s3.yaml"
wait_for_wave kai-bn7f-tiny-fp32-s3 kai-bn7f-tiny-w8a8-s3 kai-bn7f-tiny-w1a8-s3 kai-bn7f-tiny-w1a6-s3 kai-bn7f-tiny-w1a4-s3

echo "=== ROUND-7 matrix complete ==="
kubectl --context "$CTX" -n "$NS" get jobs -l app=bnjet-r7f
