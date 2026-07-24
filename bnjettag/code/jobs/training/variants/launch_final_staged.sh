#!/usr/bin/env bash
# Launch the FINAL campaign on NRP, STAGED <=3 concurrent Jobs.
# Usage:  ./launch_final_staged.sh            # wave-by-wave (needs laptop alive)
#         ./launch_final_staged.sh delete     # tear everything down
# PVC-FREE (cephfs-PVC + GPU node is blocked). PRE-REQS (in order):
#   1. ./make_code_configmap.sh      # THE training code source (ConfigMap kai-bnf-code)
#   2. preflight_final.sh -> PREFLIGHT_ALL_PASS   (in a pod with the ConfigMap mounted)
# (sync_code_to_pvc_final.sh is NOT needed for training — it is only for the PVC ROC job.)
set -euo pipefail
CTX="nautilus"; NS="cms-ml"
HERE="$(cd "$(dirname "$0")" && pwd)"

ALL=(kai-bnf-w1a8-s1 kai-bnf-w1a6-s1 kai-bnf-w1a4-s1 kai-bnf-fp32-s1 kai-bnf-w8a8-s1 kai-bnf-w1a8-s2 kai-bnf-w1a6-s2 kai-bnf-w1a4-s2 kai-bnf-fp32-s2 kai-bnf-w8a8-s2 kai-bnf-w1a8-s3 kai-bnf-w1a6-s3 kai-bnf-w1a4-s3 kai-bnf-fp32-s3 kai-bnf-w8a8-s3)
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

echo "=== wave 1: w1a8-s1 w1a6-s1 w1a4-s1 ==="
kubectl --context "$CTX" -n "$NS" delete job "kai-bnf-w1a8-s1" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bnf-w1a8-s1.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bnf-w1a6-s1" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bnf-w1a6-s1.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bnf-w1a4-s1" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bnf-w1a4-s1.yaml"
wait_for_wave kai-bnf-w1a8-s1 kai-bnf-w1a6-s1 kai-bnf-w1a4-s1

echo "=== wave 2: fp32-s1 w8a8-s1 w1a8-s2 ==="
kubectl --context "$CTX" -n "$NS" delete job "kai-bnf-fp32-s1" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bnf-fp32-s1.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bnf-w8a8-s1" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bnf-w8a8-s1.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bnf-w1a8-s2" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bnf-w1a8-s2.yaml"
wait_for_wave kai-bnf-fp32-s1 kai-bnf-w8a8-s1 kai-bnf-w1a8-s2

echo "=== wave 3: w1a6-s2 w1a4-s2 fp32-s2 ==="
kubectl --context "$CTX" -n "$NS" delete job "kai-bnf-w1a6-s2" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bnf-w1a6-s2.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bnf-w1a4-s2" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bnf-w1a4-s2.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bnf-fp32-s2" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bnf-fp32-s2.yaml"
wait_for_wave kai-bnf-w1a6-s2 kai-bnf-w1a4-s2 kai-bnf-fp32-s2

echo "=== wave 4: w8a8-s2 w1a8-s3 w1a6-s3 ==="
kubectl --context "$CTX" -n "$NS" delete job "kai-bnf-w8a8-s2" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bnf-w8a8-s2.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bnf-w1a8-s3" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bnf-w1a8-s3.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bnf-w1a6-s3" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bnf-w1a6-s3.yaml"
wait_for_wave kai-bnf-w8a8-s2 kai-bnf-w1a8-s3 kai-bnf-w1a6-s3

echo "=== wave 5: w1a4-s3 fp32-s3 w8a8-s3 ==="
kubectl --context "$CTX" -n "$NS" delete job "kai-bnf-w1a4-s3" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bnf-w1a4-s3.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bnf-fp32-s3" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bnf-fp32-s3.yaml"
kubectl --context "$CTX" -n "$NS" delete job "kai-bnf-w8a8-s3" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f "$HERE/kai-bnf-w8a8-s3.yaml"
wait_for_wave kai-bnf-w1a4-s3 kai-bnf-fp32-s3 kai-bnf-w8a8-s3

echo "=== FINAL campaign complete ==="
kubectl --context "$CTX" -n "$NS" get jobs -l app=bnjet-final
