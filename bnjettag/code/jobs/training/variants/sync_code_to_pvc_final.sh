#!/usr/bin/env bash
# Sync code/hgq2 -> PVC /data/BNJetTag-hgq2 (what actually trains). md5-verified BOTH ends via a
# throwaway util pod. Stale PVC code has bitten us before — always run this before launch.
set -euo pipefail
CTX="nautilus"; NS="cms-ml"
POD="kai-bnf-sync"
# code/hgq2 is three levels up from this script (variants/ -> training/ -> jobs/ -> code/hgq2)
SRC="$(cd "$(dirname "$0")/../../../hgq2" && pwd)"
TAR="/tmp/bnjettag-hgq2.tar.gz"

echo "[sync] packing $SRC"
tar --exclude='__pycache__' --exclude='*.pyc' --exclude='*.tar.gz' --exclude='*.keras' \
    --exclude='.DS_Store' -czf "$TAR" -C "$(dirname "$SRC")" "$(basename "$SRC")"
LOCAL_MD5=$(md5sum "$TAR" 2>/dev/null | awk '{print $1}' || md5 -q "$TAR")
echo "[sync] local  md5=$LOCAL_MD5  ($(wc -c < "$TAR") bytes)"

cleanup() { kubectl --context "$CTX" -n "$NS" delete pod "$POD" --ignore-not-found >/dev/null 2>&1 || true; }
trap cleanup EXIT
kubectl --context "$CTX" -n "$NS" delete pod "$POD" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f - <<'YAML'
apiVersion: v1
kind: Pod
metadata:
  name: kai-bnf-sync
  labels: {app: kai-util}
spec:
  restartPolicy: Never
  containers:
  - name: util
    image: alpine:3.19
    command: ["sh", "-c", "sleep 3600"]
    resources: {limits: {memory: 512Mi, cpu: "500m"}, requests: {memory: 256Mi, cpu: "200m"}}
    volumeMounts: [{name: data, mountPath: /data}]
  volumes:
  - name: data
    persistentVolumeClaim: {claimName: kai-data}
YAML
echo "[sync] waiting for $POD ..."
kubectl --context "$CTX" -n "$NS" wait --for=condition=Ready "pod/$POD" --timeout=180s

kubectl --context "$CTX" -n "$NS" cp "$TAR" "$POD:/data/bnjettag-hgq2.tar.gz"
REMOTE_MD5=$(kubectl --context "$CTX" -n "$NS" exec "$POD" -- md5sum /data/bnjettag-hgq2.tar.gz | awk '{print $1}')
echo "[sync] remote md5=$REMOTE_MD5"
if [ "$LOCAL_MD5" != "$REMOTE_MD5" ]; then echo "[fatal] tarball md5 MISMATCH (transfer corrupt)"; exit 1; fi
echo "[sync] tarball md5 MATCH -> extracting to /data/BNJetTag-hgq2"
kubectl --context "$CTX" -n "$NS" exec "$POD" -- sh -c '
  rm -rf /data/BNJetTag-hgq2 && mkdir -p /data/BNJetTag-hgq2 &&
  tar -xzf /data/bnjettag-hgq2.tar.gz -C /data/BNJetTag-hgq2 --strip-components=1 &&
  echo "[remote] extracted:" && ls /data/BNJetTag-hgq2 | tr "\n" " " && echo'

# extraction integrity: md5 a few key files on both ends
for f in run_stage.py bnhgq2/qat.py bnhgq2/train.py bnhgq2/config.py; do
  L=$(md5sum "$SRC/$f" 2>/dev/null | awk '{print $1}' || md5 -q "$SRC/$f")
  R=$(kubectl --context "$CTX" -n "$NS" exec "$POD" -- md5sum "/data/BNJetTag-hgq2/$f" | awk '{print $1}')
  if [ "$L" = "$R" ]; then echo "[sync] OK  $f"; else echo "[fatal] MISMATCH $f (local $L != pvc $R)"; exit 1; fi
done
kubectl --context "$CTX" -n "$NS" exec "$POD" -- rm -f /data/bnjettag-hgq2.tar.gz
echo "[sync] DONE — /data/BNJetTag-hgq2 is byte-verified against local code/hgq2"
