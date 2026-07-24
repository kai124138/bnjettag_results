#!/usr/bin/env bash
# Round-5 preflight: dataset check + CPU --sanity build of every round-5 variant.
set -uo pipefail
CODE=/data/BNJetTag
fail=0
echo "=== BitNet round-5 preflight $(date) ==="

# ---- HLS4ML LHC Jet (150p) dataset check (migration 2026-07-01) ----
echo "--- data check: /data/hls4ml_lhc_jet ---"
if python - <<'PYEOF'
import glob
import h5py
for d in ("/data/hls4ml_lhc_jet/train/train", "/data/hls4ml_lhc_jet/val/val"):
    fs = sorted(glob.glob(d + "/*.h5"))
    assert fs, f"no .h5 files in {d}"
    with h5py.File(fs[0], "r") as hf:
        for k in ("jetConstituentList", "jets",
                  "jetFeatureNames", "particleFeatureNames"):
            assert k in hf, f"{fs[0]} missing key {k}"
        names = [n.decode() if isinstance(n, bytes) else str(n)
                 for n in hf["jetFeatureNames"][:]]
        for l in ("j_g", "j_q", "j_w", "j_z", "j_t"):
            assert l in names, f"label {l} not in jetFeatureNames"
        print(f"  OK {d}: {len(fs)} files, "
              f"jetConstituentList {hf['jetConstituentList'].shape}")
PYEOF
then echo "  PASS data"; else echo "  FAIL data"; fail=1; fi

echo "--- r5-large-lr05-s1 (BN_VARIANT=bitnet,BN_ACT_BITS=8,BN_SEED=1) ---"
if env CUDA_VISIBLE_DEVICES=-1 BN_D_MODEL=256 BN_N_HEADS=8 BN_N_LAYERS=8 BN_FFN_DIM=1024 BN_N_PART=10 BN_VARIANT=bitnet BN_ACT_BITS=8 BN_SEED=1 python -u "$CODE/qkerasModel.py" --sanity > /tmp/pf5_large-lr05-s1.log 2>&1 && grep -q "Total trainable parameters" /tmp/pf5_large-lr05-s1.log && ! grep -qE "Traceback|Exception" /tmp/pf5_large-lr05-s1.log; then p=$(grep "Total trainable parameters" /tmp/pf5_large-lr05-s1.log | tail -1); echo "  PASS large-lr05-s1  ($p)"; else echo "  FAIL large-lr05-s1"; tail -30 /tmp/pf5_large-lr05-s1.log; fail=1; fi
echo "--- r5-large-lr05-s2 (BN_VARIANT=bitnet,BN_ACT_BITS=8,BN_SEED=2) ---"
if env CUDA_VISIBLE_DEVICES=-1 BN_D_MODEL=256 BN_N_HEADS=8 BN_N_LAYERS=8 BN_FFN_DIM=1024 BN_N_PART=10 BN_VARIANT=bitnet BN_ACT_BITS=8 BN_SEED=2 python -u "$CODE/qkerasModel.py" --sanity > /tmp/pf5_large-lr05-s2.log 2>&1 && grep -q "Total trainable parameters" /tmp/pf5_large-lr05-s2.log && ! grep -qE "Traceback|Exception" /tmp/pf5_large-lr05-s2.log; then p=$(grep "Total trainable parameters" /tmp/pf5_large-lr05-s2.log | tail -1); echo "  PASS large-lr05-s2  ($p)"; else echo "  FAIL large-lr05-s2"; tail -30 /tmp/pf5_large-lr05-s2.log; fail=1; fi
echo "--- r5-large-lr05-a6-s1 (BN_VARIANT=bitnet,BN_ACT_BITS=6,BN_SEED=1) ---"
if env CUDA_VISIBLE_DEVICES=-1 BN_D_MODEL=256 BN_N_HEADS=8 BN_N_LAYERS=8 BN_FFN_DIM=1024 BN_N_PART=10 BN_VARIANT=bitnet BN_ACT_BITS=6 BN_SEED=1 python -u "$CODE/qkerasModel.py" --sanity > /tmp/pf5_large-lr05-a6-s1.log 2>&1 && grep -q "Total trainable parameters" /tmp/pf5_large-lr05-a6-s1.log && ! grep -qE "Traceback|Exception" /tmp/pf5_large-lr05-a6-s1.log; then p=$(grep "Total trainable parameters" /tmp/pf5_large-lr05-a6-s1.log | tail -1); echo "  PASS large-lr05-a6-s1  ($p)"; else echo "  FAIL large-lr05-a6-s1"; tail -30 /tmp/pf5_large-lr05-a6-s1.log; fail=1; fi
echo "--- r5-large-lr05-a6-s2 (BN_VARIANT=bitnet,BN_ACT_BITS=6,BN_SEED=2) ---"
if env CUDA_VISIBLE_DEVICES=-1 BN_D_MODEL=256 BN_N_HEADS=8 BN_N_LAYERS=8 BN_FFN_DIM=1024 BN_N_PART=10 BN_VARIANT=bitnet BN_ACT_BITS=6 BN_SEED=2 python -u "$CODE/qkerasModel.py" --sanity > /tmp/pf5_large-lr05-a6-s2.log 2>&1 && grep -q "Total trainable parameters" /tmp/pf5_large-lr05-a6-s2.log && ! grep -qE "Traceback|Exception" /tmp/pf5_large-lr05-a6-s2.log; then p=$(grep "Total trainable parameters" /tmp/pf5_large-lr05-a6-s2.log | tail -1); echo "  PASS large-lr05-a6-s2  ($p)"; else echo "  FAIL large-lr05-a6-s2"; tail -30 /tmp/pf5_large-lr05-a6-s2.log; fail=1; fi
echo "--- r5-large-lr05-a4-s1 (BN_VARIANT=bitnet,BN_ACT_BITS=4,BN_SEED=1) ---"
if env CUDA_VISIBLE_DEVICES=-1 BN_D_MODEL=256 BN_N_HEADS=8 BN_N_LAYERS=8 BN_FFN_DIM=1024 BN_N_PART=10 BN_VARIANT=bitnet BN_ACT_BITS=4 BN_SEED=1 python -u "$CODE/qkerasModel.py" --sanity > /tmp/pf5_large-lr05-a4-s1.log 2>&1 && grep -q "Total trainable parameters" /tmp/pf5_large-lr05-a4-s1.log && ! grep -qE "Traceback|Exception" /tmp/pf5_large-lr05-a4-s1.log; then p=$(grep "Total trainable parameters" /tmp/pf5_large-lr05-a4-s1.log | tail -1); echo "  PASS large-lr05-a4-s1  ($p)"; else echo "  FAIL large-lr05-a4-s1"; tail -30 /tmp/pf5_large-lr05-a4-s1.log; fail=1; fi
echo "--- r5-large-lr05-a4-s2 (BN_VARIANT=bitnet,BN_ACT_BITS=4,BN_SEED=2) ---"
if env CUDA_VISIBLE_DEVICES=-1 BN_D_MODEL=256 BN_N_HEADS=8 BN_N_LAYERS=8 BN_FFN_DIM=1024 BN_N_PART=10 BN_VARIANT=bitnet BN_ACT_BITS=4 BN_SEED=2 python -u "$CODE/qkerasModel.py" --sanity > /tmp/pf5_large-lr05-a4-s2.log 2>&1 && grep -q "Total trainable parameters" /tmp/pf5_large-lr05-a4-s2.log && ! grep -qE "Traceback|Exception" /tmp/pf5_large-lr05-a4-s2.log; then p=$(grep "Total trainable parameters" /tmp/pf5_large-lr05-a4-s2.log | tail -1); echo "  PASS large-lr05-a4-s2  ($p)"; else echo "  FAIL large-lr05-a4-s2"; tail -30 /tmp/pf5_large-lr05-a4-s2.log; fail=1; fi
echo "--- r5-fp32-lr05 (BN_VARIANT=vanilla,BN_ACT_BITS=8,BN_SEED=1) ---"
if env CUDA_VISIBLE_DEVICES=-1 BN_D_MODEL=256 BN_N_HEADS=8 BN_N_LAYERS=8 BN_FFN_DIM=1024 BN_N_PART=10 BN_VARIANT=vanilla BN_ACT_BITS=8 BN_SEED=1 python -u "$CODE/qkerasModel.py" --sanity > /tmp/pf5_fp32-lr05.log 2>&1 && grep -q "Total trainable parameters" /tmp/pf5_fp32-lr05.log && ! grep -qE "Traceback|Exception" /tmp/pf5_fp32-lr05.log; then p=$(grep "Total trainable parameters" /tmp/pf5_fp32-lr05.log | tail -1); echo "  PASS fp32-lr05  ($p)"; else echo "  FAIL fp32-lr05"; tail -30 /tmp/pf5_fp32-lr05.log; fail=1; fi
echo "--- r5-w8a8-lr05 (BN_VARIANT=w8a8,BN_ACT_BITS=8,BN_SEED=1) ---"
if env CUDA_VISIBLE_DEVICES=-1 BN_D_MODEL=256 BN_N_HEADS=8 BN_N_LAYERS=8 BN_FFN_DIM=1024 BN_N_PART=10 BN_VARIANT=w8a8 BN_ACT_BITS=8 BN_SEED=1 python -u "$CODE/qkerasModel.py" --sanity > /tmp/pf5_w8a8-lr05.log 2>&1 && grep -q "Total trainable parameters" /tmp/pf5_w8a8-lr05.log && ! grep -qE "Traceback|Exception" /tmp/pf5_w8a8-lr05.log; then p=$(grep "Total trainable parameters" /tmp/pf5_w8a8-lr05.log | tail -1); echo "  PASS w8a8-lr05  ($p)"; else echo "  FAIL w8a8-lr05"; tail -30 /tmp/pf5_w8a8-lr05.log; fail=1; fi
if [ "$fail" = 0 ]; then echo "PREFLIGHT_ALL_PASS"; else echo "PREFLIGHT_HAD_FAILURES"; fi
