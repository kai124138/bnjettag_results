#!/usr/bin/env bash
# ROUND-7 whole-model hls4ml conversion — the abstract-closing artifact (experiment-log 2026-07-14).
#
# Drives convert_final.py (fetch -> load -> export -> GATE1 -> convert -> GATE2 C-sim -> EBOPs ->
# mulder tarball) on the r7 small binary model r7-small-w1a8-s2 (D32/H4/L2/FFN64, 19,201 params,
# ~190k MACs).  This is the "one call, whole model" round: every weighted layer is <= ~2.1k MACs,
# ~20x under the ~40k Latency-einsum LTO wall, so the NATIVE emission (QEinsumDense/QEinsum/QSoftmax
# as trained, Latency, io_parallel) converts the WHOLE model in a single project (constraints_map.md).
#
# Modes (env WIDEACC, default 1 = the shipped v2 emission):
#   WIDEACC=1 (default) -> --widen-accum: widen EVERY weighted-layer accumulator to frac>=16, flipping
#     the binary ±1 MAC from DSP48 to LUT (measured DSP-0 form, probes_final_v3). Store rf<N>-wideacc/.
#     Expected post-fix DSP (mulder is the arbiter): ~12.8k @ RF=1 (all structural act×act attention),
#     ~1.6k @ RF=8, ~800 @ RF=16.
#   WIDEACC=0 -> the raw BitExact-minimal accums (v1 emission, parks DSP on Wq/Wk/Wv/Wo+fc2). Store rf<N>/.
#
# Emitted RF variants (env RFS, default "1 8 16"), all io_parallel / VU13P / 2.5 ns / Latency strategy:
#   RF=1 fully parallel (latency reference); RF=8 and RF=16 (L1-throughput candidates: at 2.5 ns the
#   40 MHz collision rate allows II<=10, so RF 8-16 brackets the deployable envelope).
# hls4ml 1.3.0 EinsumDense is Latency-strategy-only, so RF acts as the einsum multiplier_limit.
#
# NOTHING here runs csynth (mulder-only) or touches the cluster.
#
# Usage (from code/hgq2/):
#   export WANDB_API_KEY=$(cat ../../wandb-api-key.txt)   # never echoed / committed
#   ./run_convert_r7.sh                                   # WIDEACC=1, RF=1,8,16 -> rf<N>-wideacc/
#   WIDEACC=0 RFS="1 8" ./run_convert_r7.sh               # raw-accum baseline -> rf<N>/
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="${PY:-$ROOT/.venv-hgq2/bin/python}"

CONFIG="$HERE/configs/r7-small-w1a8.json"
VARIANT="w1a8"
SEED="2"
WANDB_RUN="r7-small-w1a8-s2"
WANDB_SUBDIR="small-w1a8-s2"
CACHE="$ROOT/bnjettag/r7/models"
STORE="$ROOT/bnjettag/r7/results/convert/${VARIANT}-s${SEED}"
RFS="${RFS:-1 8 16}"
WIDEACC="${WIDEACC:-1}"
CKPT="$CACHE/$WANDB_SUBDIR/model_best.keras"

SUFFIX=""; WFLAG=()
if [ "$WIDEACC" = 1 ]; then SUFFIX="-wideacc"; WFLAG=(--widen-accum); fi

if [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$CKPT" ]; then
  echo "NOTE: WANDB_API_KEY is not set and no cached checkpoint at $CKPT."
  echo "      export WANDB_API_KEY=\$(cat ../../wandb-api-key.txt)  (never echo it)."
fi

echo "=== r7 whole-model convert  config=$CONFIG  run=$WANDB_RUN  RFs=[$RFS]  WIDEACC=$WIDEACC ==="
rc_all=0
first=1
for rf in $RFS; do
  args=(--variant "$VARIANT" --seed "$SEED" --config "$CONFIG"
        --wandb-run "$WANDB_RUN" --wandb-subdir "$WANDB_SUBDIR"
        --rf "$rf" --run-dir "$STORE/rf${rf}${SUFFIX}" --cache-dir "$CACHE"
        --no-exact-char "${WFLAG[@]}")
  # First RF fetches from W&B into the cache; later RFs reuse the cached checkpoint.
  if [ "$first" = 0 ] && [ -f "$CKPT" ]; then args+=(--checkpoint "$CKPT"); fi
  echo ""
  echo ">>> convert_final RF=$rf  -> $STORE/rf${rf}${SUFFIX}"
  "$PY" "$HERE/convert_final.py" "${args[@]}" || rc_all=$?
  first=0
done
echo ""
echo "=== r7 convert done (rc=$rc_all).  Artifacts: $STORE/rf*${SUFFIX}/ ==="
echo "    mulder: scp bnjettag/r7/results/convert/${VARIANT}-s${SEED}/rf<N>${SUFFIX}/hls_prj_rf<N>.tar.gz \\"
echo "            mulder:~/csynth/ && ssh mulder 'cd ~/csynth && ./mulder_csynth.sh hls_prj_rf<N>.tar.gz'"
exit $rc_all
