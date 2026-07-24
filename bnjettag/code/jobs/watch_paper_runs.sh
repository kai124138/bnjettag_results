#!/usr/bin/env bash
# Watch the paper-recipe BitNet runs until all reach a terminal state.
# Reports each job's best (max) val_auc (EarlyStopping monitors val_auc/max).
# bash 3.2-compatible: state tracked via marker files, no associative arrays.
set -uo pipefail
CTX="nautilus"; NS="cms-ml"
JOBS="kai-bn-paper-bin-noclip kai-bn-paper-tern-clip kai-bn-paper-bin-lr15 kai-bn-paper-bin-lr3"
SD="$(mktemp -d /tmp/bnwatch.XXXXXX)"   # one marker file per finished job
best_auc() {  # $1=pod
  kubectl --context "$CTX" -n "$NS" logs "$1" 2>/dev/null \
    | grep -oE 'val_auc: [0-9.]+' | awk '{print $2}' | sort -gr | head -1
}
job_phase() { kubectl --context "$CTX" -n "$NS" get job "$1" -o jsonpath='{.status.succeeded}/{.status.failed}/{.status.active}' 2>/dev/null; }
pod_for()   { kubectl --context "$CTX" -n "$NS" get pods -l job-name="$1" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null; }
iter=0
while :; do
  iter=$((iter+1))
  alldone=1
  echo "===== poll #$iter $(date -u +%H:%M:%S)Z ====="
  for j in $JOBS; do
    ph=$(job_phase "$j")            # succeeded/failed/active  e.g. "1//", "/1/", "//1", "//"
    succ=${ph%%/*}; rest=${ph#*/}; fail=${rest%%/*}
    pod=$(pod_for "$j")
    auc=$(best_auc "$pod"); auc=${auc:-NA}
    if [ "$succ" = "1" ]; then
      if [ ! -f "$SD/$j" ]; then echo "$auc" > "$SD/$j"; echo ">>> $j COMPLETE  best_val_auc=$auc"; fi
    elif [ "$fail" = "1" ]; then
      if [ ! -f "$SD/$j" ]; then echo "FAILED($auc)" > "$SD/$j"; echo ">>> $j FAILED   best_val_auc=$auc"; fi
    else
      alldone=0
      echo "    $j running  best_val_auc_so_far=$auc"
    fi
  done
  [ $alldone -eq 1 ] && break
  sleep 240
done
echo "===== ALL TERMINAL $(date -u +%H:%M:%S)Z ====="
for j in $JOBS; do echo "FINAL  $j  best_val_auc=$(cat "$SD/$j" 2>/dev/null || echo NA)"; done
echo "===== watcher end ====="
