#!/usr/bin/env bash
# Watches the activation-precision sweep jobs (A6, A4). Polls each job's pod log for the
# best val_auc seen so far and the job completion status. Exits when BOTH jobs are
# Complete or Failed. bash-3.2 compatible (macOS): no associative arrays.
set -u
CTX="--context nautilus -n cms-ml"
JOBS="kai-bn-paper-bin-sffree-a6 kai-bn-paper-bin-sffree-a4"
MAX_POLLS=60     # ~4h cap at 240s/poll
SLEEP=240

best_auc() {  # $1 = job name -> prints max val_auc in that pod's log, or NA
  local job="$1" pod log
  pod=$(kubectl $CTX get pods -l job-name="$job" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  [ -z "$pod" ] && { echo "NA"; return; }
  log=$(kubectl $CTX logs "$pod" --tail=400 2>/dev/null)
  [ -z "$log" ] && { echo "NA"; return; }
  echo "$log" | grep -oE 'val_auc: [0-9.]+' | awk '{print $2}' \
    | sort -gr | head -1 | awk '{ if ($1=="") print "NA"; else print $1 }'
}

job_done() {  # $1 = job name -> prints COMPLETE / FAILED / running
  local job="$1" s f
  s=$(kubectl $CTX get job "$job" -o jsonpath='{.status.succeeded}' 2>/dev/null)
  f=$(kubectl $CTX get job "$job" -o jsonpath='{.status.failed}' 2>/dev/null)
  if [ "${s:-0}" = "1" ]; then echo "COMPLETE";
  elif [ "${f:-0}" != "" ] && [ "${f:-0}" != "0" ]; then echo "FAILED";
  else echo "running"; fi
}

i=0
while [ $i -lt $MAX_POLLS ]; do
  i=$((i+1))
  ts=$(date -u +%H:%M:%SZ)
  ndone=0
  line="[poll $i $ts]"
  for j in $JOBS; do
    st=$(job_done "$j")
    au=$(best_auc "$j")
    short=$(echo "$j" | sed 's/kai-bn-paper-bin-sffree-//')
    line="$line  $short=$st(auc=$au)"
    [ "$st" != "running" ] && ndone=$((ndone+1))
  done
  echo "$line"
  if [ $ndone -ge 2 ]; then
    echo ">>> SWEEP DONE"
    for j in $JOBS; do
      short=$(echo "$j" | sed 's/kai-bn-paper-bin-sffree-//')
      echo ">>> $short final: status=$(job_done "$j") best_val_auc=$(best_auc "$j")"
    done
    exit 0
  fi
  sleep $SLEEP
done
echo ">>> WATCHER TIMEOUT after $MAX_POLLS polls"
