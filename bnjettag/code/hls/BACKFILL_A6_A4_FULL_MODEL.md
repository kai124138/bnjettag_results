# Backfill the A6/A4 full-model csynth on `mulder` — runbook

**What's pending:** `results/hls_resource_table.md` says *"A6/A4 sweep running on `mulder`
as of 2026-06-26 … backfilled when complete"* — the results were never collected. Only the
A8 full-model composition exists locally (`results/csynth/full_model_shape_*_a8_rf256.json`
+ `full_model_total_a8_rf256.json`).

**Expected outcome (sanity anchors):** DSP is precision-independent, so the A6/A4 composed
totals must come back at **1,049 DSP (100 % LayerNorm)** with modestly smaller LUT/FF than
A8. If DSP ≠ 1,049 or any binary QDense shape reports DSP > 0, stop and investigate — that
would break the structural claim.

## 0. Check whether the 2026-06-26 run already finished

The sweep may have completed after the last fetch. Before re-running anything:

```bash
ssh mulder.t2.ucsd.edu
ls -la ~/csynth/full_csynth_out/ 2>/dev/null | grep -E "a[64]_rf256"
find ~ -name "full_model_total_a[64]_rf256.json" 2>/dev/null
```

If the JSONs exist → skip to **step 4 (fetch)**.

## 1. Environment (verified 2026-06-24 procedure)

```bash
PY=/home/users/kayamaguchi/micromamba/envs/bnjet/bin/python   # hls4ml 1.4.0 + qkeras + tf 2.11
source /data/software/xilinx/Vitis_HLS/2023.2/settings64.sh
command -v vitis_hls    # must print a path
```

## 2. Run the A6/A4 full-model csynth

`full_model_csynth.py` loads the trained checkpoint, transfers weights through the BitNet
binarizer, and synthesizes each of the 5 distinct shapes once per precision (resources are
weight-value-independent, so the lr15 checkpoint is the correct/consistent input — same as
the A8 row).

```bash
export CUDA_VISIBLE_DEVICES=-1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python   # REQUIRED (protobuf clash)
export HLS_MODE=csynth
export HLS_RF=256                                      # folded point; RF=1 intractable
export HLS_ABITS=6,4                                   # A8 already done — only backfill
export HLS_OUT=~/csynth/full_csynth_out
export BN_CKPT=/data/outputs/qk-paper-binary-lr15/bitnet/noNorm_train_bitnetJetTagModel.h5
cd ~/csynth   # wherever code/hls/full_model_csynth.py was scp'd
nohup $PY full_model_csynth.py > backfill_a6a4.log 2>&1 &
```

Budget: ~50 min/precision/shape at RF=256 (~8.5 GB peak for the 256→1024 shape); the two
precisions × 5 shapes can take several hours — run under `nohup`/`tmux` and check
`backfill_a6a4.log`.

## 3. What it produces

```
$HLS_OUT/full_model_shape_<shape>_a6_rf256.json   (×5 shapes)
$HLS_OUT/full_model_shape_<shape>_a4_rf256.json   (×5 shapes)
$HLS_OUT/full_model_total_a6_rf256.json
$HLS_OUT/full_model_total_a4_rf256.json
```

## 4. Fetch + integrate (back on the laptop)

```bash
scp "mulder.t2.ucsd.edu:~/csynth/full_csynth_out/full_model_*_a[64]_rf256.json" \
    bnjettag/results/csynth/
```

Then, locally (this is verification work — numbers only from the JSONs):

1. Fill the A6/A4 full-model rows in `results/hls_resource_table.md` and delete the
   *"backfilled when complete"* placeholder note.
2. Verify DSP = 1,049 at both precisions and per-shape DSP = 0 for every QDense.
3. Log the verification in the internal experiment log.

## Troubleshooting

Same as `RUN_CSYNTH_ON_VITIS.md`: missing `vitis_hls` → re-source `settings64.sh`;
protobuf/TF import death → the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` export;
empty report → parse `*_prj/myproject_prj/solution1/syn/report/*_csynth.rpt` manually.
