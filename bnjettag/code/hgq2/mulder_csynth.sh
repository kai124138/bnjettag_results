#!/usr/bin/env bash
# Run Vitis HLS csynth on hls4ml-1.3.0-generated projects (shipped as tarballs).
#
# Usage (on mulder):  ./mulder_csynth.sh <tarball> [<tarball> ...]
# Each tarball extracts to a project dir containing build_prj.tcl.
# Output: <project>/csynth_report.json next to the extracted project.
#
# Toolchain per code/hls/RUN_CSYNTH_ON_VITIS.md (2026-07-02 correction): the FULL
# Vitis settings64.sh must be sourced (provides vitis-run AND vitis_hls).
set -uo pipefail

source /data/software/xilinx/Vitis/2023.2/settings64.sh
command -v vitis_hls >/dev/null || { echo "FATAL: vitis_hls not on PATH"; exit 1; }

HERE="$(cd "$(dirname "$0")" && pwd)"

for tb in "$@"; do
  name=$(basename "$tb" .tar.gz)
  wd="$HERE/$name"
  echo "=== $name  $(date) ==="
  rm -rf "$wd"; mkdir -p "$wd"
  tar -xzf "$tb" -C "$wd" --strip-components=1
  cd "$wd"
  # hls4ml build tcl: run synthesis only (no csim/cosim/export)
  ( time vitis_hls -f build_prj.tcl "reset=1 csim=0 synth=1 cosim=0 validation=0 export=0 vsynth=0" ) \
      > csynth_stdout.log 2>&1
  rc=$?
  xml=$(find . -path "*/syn/report/csynth.xml" | head -1)
  if [ $rc -ne 0 ] || [ -z "$xml" ]; then
    echo "FAIL rc=$rc (no csynth.xml); tail of log:"
    tail -25 csynth_stdout.log
    cd "$HERE"; continue
  fi
  python3 "$HERE/parse_csynth.py" "$xml" > csynth_report.json
  echo "--- $name report ---"
  cat csynth_report.json
  cd "$HERE"
done
echo "=== all done $(date) ==="
