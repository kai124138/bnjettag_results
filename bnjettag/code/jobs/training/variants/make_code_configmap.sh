#!/usr/bin/env bash
# Pack code/hgq2 into ConfigMap kai-bnf-code (key hgq2.tar.gz) — the PVC-FREE code source that
# training pods untar to /work/code. THE TRAINING PREREQ (replaces sync_code_to_pvc_final.sh,
# which is only for the PVC-based ROC job). Idempotent (--dry-run=client | apply). Prints the
# tar md5 — pods echo the same md5 for provenance. code/hgq2 packs to ~80 KB (< the 1 MB
# ConfigMap limit); same excludes as the sync script.
set -euo pipefail
CTX="nautilus"; NS="cms-ml"
SRC="$(cd "$(dirname "$0")/../../../hgq2" && pwd)"
TAR="/tmp/bnjettag-hgq2-cm.tar.gz"
tar --exclude='__pycache__' --exclude='*.pyc' --exclude='*.tar.gz' --exclude='*.keras' \
    --exclude='.DS_Store' -czf "$TAR" -C "$(dirname "$SRC")" "$(basename "$SRC")"
MD5=$(md5sum "$TAR" 2>/dev/null | awk '{print $1}' || md5 -q "$TAR")
SZ=$(wc -c < "$TAR")
echo "[cm] packed $SRC -> hgq2.tar.gz  md5=$MD5  bytes=$SZ"
[ "$SZ" -lt 1048576 ] || { echo "[fatal] tar $SZ bytes exceeds the 1 MB ConfigMap limit — trim excludes"; exit 1; }
kubectl --context "$CTX" -n "$NS" create configmap kai-bnf-code \
  --from-file=hgq2.tar.gz="$TAR" --dry-run=client -o yaml \
  | kubectl --context "$CTX" -n "$NS" apply -f -
echo "[cm] applied ConfigMap kai-bnf-code (key hgq2.tar.gz, md5 $MD5)"
echo "[cm] training pods echo this md5 as '[code] configmap tar md5: $MD5' for provenance"
