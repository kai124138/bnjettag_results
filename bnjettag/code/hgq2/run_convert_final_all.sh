#!/usr/bin/env bash
# FINAL-campaign hls4ml conversion sweep (decisions.md 2026-07-07).
#
# Runs convert_final.py (fetch -> load -> export -> GATE1 -> convert -> GATE2 C-sim ->
# EBOPs -> mulder tarball) for every BINARY variant x seed.  fp32 / w8a8 are SKIPPED:
# there is no binary DSP-0 story for them.  (W8A8 can still get NATIVE EBOPs for the
# cost-frontier table; see the note + the --w8a8-ebops hook below.)
#
# Usage (from code/hgq2/):
#   export WANDB_API_KEY=$(cat ../../wandb-api-key.txt)   # never echoed / committed
#   ./run_convert_final_all.sh                            # w1a{8,6,4} x seeds 1 2 3
#   ./run_convert_final_all.sh --seeds "1" --variants "w1a8"          # subset
#   ./run_convert_final_all.sh --wandb-run final-smoke --variants w1a8 --seeds 1 --tag smoke
#   NOTHING here runs csynth (mulder-only) — it prepares the per-model tarballs.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-$HERE/../../../.venv-hgq2/bin/python}"

VARIANTS="w1a8 w1a6 w1a4"
SEEDS="1 2 3"
WANDB_RUN=""            # default per-variant final-<variant>-s<seed>
TAG=""
EXTRA=()               # forwarded to convert_final.py (e.g. --rf, --strategy, --beta-mode)

while [ $# -gt 0 ]; do
  case "$1" in
    --variants) VARIANTS="$2"; shift 2;;
    --seeds)    SEEDS="$2"; shift 2;;
    --wandb-run) WANDB_RUN="$2"; shift 2;;
    --tag)      TAG="$2"; shift 2;;
    *)          EXTRA+=("$1"); shift;;
  esac
done

if [ -z "${WANDB_API_KEY:-}" ] && [[ ! " ${EXTRA[*]} " == *"--checkpoint"* ]]; then
  echo "NOTE: WANDB_API_KEY is not set — set it (export WANDB_API_KEY=\$(cat "
  echo "      ../../wandb-api-key.txt)) or pass --checkpoint for a local model_best.keras."
fi

echo "=== FINAL convert sweep  variants=[$VARIANTS]  seeds=[$SEEDS] ==="
echo "    SKIPPED: fp32, w8a8 (no binary DSP-0 story; w8a8 EBOPs-only is a separate hook)."
rc_all=0
for v in $VARIANTS; do
  case "$v" in
    fp32|w8a8)
      echo "--- SKIP $v: no binary story (fp32=float baseline; w8a8=8-bit weights)."
      echo "    For the cost frontier, compute native HGQ2 EBOPs for w8a8 separately"
      echo "    (run_stage.py-style ebops on a w8a8 export) — not a DSP-0 convert.";;
    w1a8|w1a6|w1a4)
      for s in $SEEDS; do
        args=(--variant "$v" --seed "$s")
        [ -n "$WANDB_RUN" ] && args+=(--wandb-run "$WANDB_RUN")
        [ -n "$TAG" ] && args+=(--tag "$TAG")
        echo ""
        echo ">>> convert_final $v s$s  ${args[*]} ${EXTRA[*]}"
        "$PY" "$HERE/convert_final.py" "${args[@]}" "${EXTRA[@]}" || rc_all=$?
      done;;
    *) echo "--- SKIP unknown variant $v";;
  esac
done
echo ""
echo "=== sweep done (rc=$rc_all). Per-model artifacts under results/final/runs/<hash>/. ==="
echo "    Next (mulder): scp each hls_prj_rf*.tar.gz to mulder and run ./mulder_csynth.sh."
exit $rc_all
