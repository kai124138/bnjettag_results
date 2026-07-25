#!/bin/bash
# Experiment 1 — sequential, guarded (shared box; see infrastructure/mulder-setup.md)
set -u
source /data/software/xilinx/Vitis/2023.2/settings64.sh
cd "$(dirname "$0")"
guard() {
  while true; do
    nv=$(pgrep -c vitis_hls || true); free_gb=$(free -g | awk '/Mem:/{print $7}')
    [ "${nv:-0}" -le 2 ] && [ "${free_gb:-0}" -ge 40 ] && break
    echo "guard: vitis=$nv free=${free_gb}G — waiting 60s"; sleep 60
  done
}
for arm in p a a2 b c; do
  guard
  echo "=== arm $arm: $(date) ==="
  (cd "$arm" && vitis_hls -f build.tcl > vitis_hls.log 2>&1)
  echo "=== arm $arm done rc=$? $(date) ==="
done
echo "ALL_ARMS_DONE"
