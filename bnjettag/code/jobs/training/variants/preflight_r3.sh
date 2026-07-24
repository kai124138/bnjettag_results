#!/usr/bin/env bash
# Round-3 preflight: CPU --sanity build of every round-3 variant.
set -uo pipefail
CODE=/data/BNJetTag
fail=0
echo "=== BitNet round-3 preflight $(date) ==="
echo "--- r3-small-lr025 (recipe[BN_LR=0.000025] arch[default-arch]) ---"
if env CUDA_VISIBLE_DEVICES=-1 BN_D_MODEL=64 BN_N_HEADS=8 BN_N_LAYERS=3 BN_FFN_DIM=256 BN_LR=0.000025 python -u "$CODE/qkerasModel.py" --sanity > /tmp/pf3_small-lr025.log 2>&1 && grep -q "Total trainable parameters" /tmp/pf3_small-lr025.log && ! grep -qE "Traceback|Exception" /tmp/pf3_small-lr025.log; then p=$(grep "Total trainable parameters" /tmp/pf3_small-lr025.log | tail -1); echo "  PASS small-lr025  ($p)"; else echo "  FAIL small-lr025"; tail -30 /tmp/pf3_small-lr025.log; fail=1; fi
echo "--- r3-small-lr035 (recipe[BN_LR=0.000035] arch[default-arch]) ---"
if env CUDA_VISIBLE_DEVICES=-1 BN_D_MODEL=64 BN_N_HEADS=8 BN_N_LAYERS=3 BN_FFN_DIM=256 BN_LR=0.000035 python -u "$CODE/qkerasModel.py" --sanity > /tmp/pf3_small-lr035.log 2>&1 && grep -q "Total trainable parameters" /tmp/pf3_small-lr035.log && ! grep -qE "Traceback|Exception" /tmp/pf3_small-lr035.log; then p=$(grep "Total trainable parameters" /tmp/pf3_small-lr035.log | tail -1); echo "  PASS small-lr035  ($p)"; else echo "  FAIL small-lr035"; tail -30 /tmp/pf3_small-lr035.log; fail=1; fi
echo "--- r3-small-lr05-s1 (recipe[BN_LR=0.00005,BN_SEED=1] arch[default-arch]) ---"
if env CUDA_VISIBLE_DEVICES=-1 BN_D_MODEL=64 BN_N_HEADS=8 BN_N_LAYERS=3 BN_FFN_DIM=256 BN_LR=0.00005 BN_SEED=1 python -u "$CODE/qkerasModel.py" --sanity > /tmp/pf3_small-lr05-s1.log 2>&1 && grep -q "Total trainable parameters" /tmp/pf3_small-lr05-s1.log && ! grep -qE "Traceback|Exception" /tmp/pf3_small-lr05-s1.log; then p=$(grep "Total trainable parameters" /tmp/pf3_small-lr05-s1.log | tail -1); echo "  PASS small-lr05-s1  ($p)"; else echo "  FAIL small-lr05-s1"; tail -30 /tmp/pf3_small-lr05-s1.log; fail=1; fi
echo "--- r3-small-lr05-s2 (recipe[BN_LR=0.00005,BN_SEED=2] arch[default-arch]) ---"
if env CUDA_VISIBLE_DEVICES=-1 BN_D_MODEL=64 BN_N_HEADS=8 BN_N_LAYERS=3 BN_FFN_DIM=256 BN_LR=0.00005 BN_SEED=2 python -u "$CODE/qkerasModel.py" --sanity > /tmp/pf3_small-lr05-s2.log 2>&1 && grep -q "Total trainable parameters" /tmp/pf3_small-lr05-s2.log && ! grep -qE "Traceback|Exception" /tmp/pf3_small-lr05-s2.log; then p=$(grep "Total trainable parameters" /tmp/pf3_small-lr05-s2.log | tail -1); echo "  PASS small-lr05-s2  ($p)"; else echo "  FAIL small-lr05-s2"; tail -30 /tmp/pf3_small-lr05-s2.log; fail=1; fi
echo "--- r3-medium-lr05 (recipe[BN_LR=0.00005] arch[default-arch]) ---"
if env CUDA_VISIBLE_DEVICES=-1 BN_D_MODEL=128 BN_N_HEADS=8 BN_N_LAYERS=4 BN_FFN_DIM=512 BN_LR=0.00005 python -u "$CODE/qkerasModel.py" --sanity > /tmp/pf3_medium-lr05.log 2>&1 && grep -q "Total trainable parameters" /tmp/pf3_medium-lr05.log && ! grep -qE "Traceback|Exception" /tmp/pf3_medium-lr05.log; then p=$(grep "Total trainable parameters" /tmp/pf3_medium-lr05.log | tail -1); echo "  PASS medium-lr05  ($p)"; else echo "  FAIL medium-lr05"; tail -30 /tmp/pf3_medium-lr05.log; fail=1; fi
echo "--- r3-medium-lr075 (recipe[BN_LR=0.000075] arch[default-arch]) ---"
if env CUDA_VISIBLE_DEVICES=-1 BN_D_MODEL=128 BN_N_HEADS=8 BN_N_LAYERS=4 BN_FFN_DIM=512 BN_LR=0.000075 python -u "$CODE/qkerasModel.py" --sanity > /tmp/pf3_medium-lr075.log 2>&1 && grep -q "Total trainable parameters" /tmp/pf3_medium-lr075.log && ! grep -qE "Traceback|Exception" /tmp/pf3_medium-lr075.log; then p=$(grep "Total trainable parameters" /tmp/pf3_medium-lr075.log | tail -1); echo "  PASS medium-lr075  ($p)"; else echo "  FAIL medium-lr075"; tail -30 /tmp/pf3_medium-lr075.log; fail=1; fi
if [ "$fail" = 0 ]; then echo "PREFLIGHT_ALL_PASS"; else echo "PREFLIGHT_HAD_FAILURES"; fi
