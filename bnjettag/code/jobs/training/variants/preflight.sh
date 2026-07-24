#!/usr/bin/env bash
# Preflight: construct + forward + train_on_batch each variant via --sanity.
# Runs on CPU inside the TF image; no data files, no GPU needed.
set -uo pipefail
CODE=/data/BNJetTag
fail=0
echo "=== BitNet variant preflight $(date) ==="
echo "--- var-size-tiny-D32L2H4F128 (defaults(layernorm/per_linear/learned/gap/relu,softmax)) ---"
if env CUDA_VISIBLE_DEVICES=-1 BN_D_MODEL=32 BN_N_HEADS=4 BN_N_LAYERS=2 BN_FFN_DIM=128 python -u "$CODE/qkerasModel.py" --sanity > /tmp/pf_tiny.log 2>&1 && grep -q "Total trainable parameters" /tmp/pf_tiny.log && ! grep -qE "Traceback|Exception" /tmp/pf_tiny.log; then p=$(grep "Total trainable parameters" /tmp/pf_tiny.log | tail -1); echo "  PASS tiny  ($p)"; else echo "  FAIL tiny"; tail -30 /tmp/pf_tiny.log; fail=1; fi
echo "--- var-size-small-D64L3H8F256 (defaults(layernorm/per_linear/learned/gap/relu,softmax)) ---"
if env CUDA_VISIBLE_DEVICES=-1 BN_D_MODEL=64 BN_N_HEADS=8 BN_N_LAYERS=3 BN_FFN_DIM=256 python -u "$CODE/qkerasModel.py" --sanity > /tmp/pf_small.log 2>&1 && grep -q "Total trainable parameters" /tmp/pf_small.log && ! grep -qE "Traceback|Exception" /tmp/pf_small.log; then p=$(grep "Total trainable parameters" /tmp/pf_small.log | tail -1); echo "  PASS small  ($p)"; else echo "  FAIL small"; tail -30 /tmp/pf_small.log; fail=1; fi
echo "--- var-size-medium-D128L4H8F512 (defaults(layernorm/per_linear/learned/gap/relu,softmax)) ---"
if env CUDA_VISIBLE_DEVICES=-1 BN_D_MODEL=128 BN_N_HEADS=8 BN_N_LAYERS=4 BN_FFN_DIM=512 python -u "$CODE/qkerasModel.py" --sanity > /tmp/pf_medium.log 2>&1 && grep -q "Total trainable parameters" /tmp/pf_medium.log && ! grep -qE "Traceback|Exception" /tmp/pf_medium.log; then p=$(grep "Total trainable parameters" /tmp/pf_medium.log | tail -1); echo "  PASS medium  ($p)"; else echo "  FAIL medium"; tail -30 /tmp/pf_medium.log; fail=1; fi
echo "--- var-abl-rmsnorm (BN_NORM_TYPE=rmsnorm) ---"
if env CUDA_VISIBLE_DEVICES=-1 BN_D_MODEL=64 BN_N_HEADS=8 BN_N_LAYERS=3 BN_FFN_DIM=256 BN_NORM_TYPE=rmsnorm python -u "$CODE/qkerasModel.py" --sanity > /tmp/pf_rmsnorm.log 2>&1 && grep -q "Total trainable parameters" /tmp/pf_rmsnorm.log && ! grep -qE "Traceback|Exception" /tmp/pf_rmsnorm.log; then p=$(grep "Total trainable parameters" /tmp/pf_rmsnorm.log | tail -1); echo "  PASS rmsnorm  ($p)"; else echo "  FAIL rmsnorm"; tail -30 /tmp/pf_rmsnorm.log; fail=1; fi
echo "--- var-abl-sharednorm (BN_NORM_PLACEMENT=shared_prenorm) ---"
if env CUDA_VISIBLE_DEVICES=-1 BN_D_MODEL=64 BN_N_HEADS=8 BN_N_LAYERS=3 BN_FFN_DIM=256 BN_NORM_PLACEMENT=shared_prenorm python -u "$CODE/qkerasModel.py" --sanity > /tmp/pf_sharednorm.log 2>&1 && grep -q "Total trainable parameters" /tmp/pf_sharednorm.log && ! grep -qE "Traceback|Exception" /tmp/pf_sharednorm.log; then p=$(grep "Total trainable parameters" /tmp/pf_sharednorm.log | tail -1); echo "  PASS sharednorm  ($p)"; else echo "  FAIL sharednorm"; tail -30 /tmp/pf_sharednorm.log; fail=1; fi
echo "--- var-abl-noposenc (BN_POS_ENC=none) ---"
if env CUDA_VISIBLE_DEVICES=-1 BN_D_MODEL=64 BN_N_HEADS=8 BN_N_LAYERS=3 BN_FFN_DIM=256 BN_POS_ENC=none python -u "$CODE/qkerasModel.py" --sanity > /tmp/pf_noposenc.log 2>&1 && grep -q "Total trainable parameters" /tmp/pf_noposenc.log && ! grep -qE "Traceback|Exception" /tmp/pf_noposenc.log; then p=$(grep "Total trainable parameters" /tmp/pf_noposenc.log | tail -1); echo "  PASS noposenc  ($p)"; else echo "  FAIL noposenc"; tail -30 /tmp/pf_noposenc.log; fail=1; fi
echo "--- var-abl-trueposenc (BN_POS_ENC=learned_real) ---"
if env CUDA_VISIBLE_DEVICES=-1 BN_D_MODEL=64 BN_N_HEADS=8 BN_N_LAYERS=3 BN_FFN_DIM=256 BN_POS_ENC=learned_real python -u "$CODE/qkerasModel.py" --sanity > /tmp/pf_trueposenc.log 2>&1 && grep -q "Total trainable parameters" /tmp/pf_trueposenc.log && ! grep -qE "Traceback|Exception" /tmp/pf_trueposenc.log; then p=$(grep "Total trainable parameters" /tmp/pf_trueposenc.log | tail -1); echo "  PASS trueposenc  ($p)"; else echo "  FAIL trueposenc"; tail -30 /tmp/pf_trueposenc.log; fail=1; fi
echo "--- var-abl-clspool (BN_POOL=cls) ---"
if env CUDA_VISIBLE_DEVICES=-1 BN_D_MODEL=64 BN_N_HEADS=8 BN_N_LAYERS=3 BN_FFN_DIM=256 BN_POOL=cls python -u "$CODE/qkerasModel.py" --sanity > /tmp/pf_clspool.log 2>&1 && grep -q "Total trainable parameters" /tmp/pf_clspool.log && ! grep -qE "Traceback|Exception" /tmp/pf_clspool.log; then p=$(grep "Total trainable parameters" /tmp/pf_clspool.log | tail -1); echo "  PASS clspool  ($p)"; else echo "  FAIL clspool"; tail -30 /tmp/pf_clspool.log; fail=1; fi
echo "--- var-abl-gelu (BN_FFN_ACT=gelu) ---"
if env CUDA_VISIBLE_DEVICES=-1 BN_D_MODEL=64 BN_N_HEADS=8 BN_N_LAYERS=3 BN_FFN_DIM=256 BN_FFN_ACT=gelu python -u "$CODE/qkerasModel.py" --sanity > /tmp/pf_gelu.log 2>&1 && grep -q "Total trainable parameters" /tmp/pf_gelu.log && ! grep -qE "Traceback|Exception" /tmp/pf_gelu.log; then p=$(grep "Total trainable parameters" /tmp/pf_gelu.log | tail -1); echo "  PASS gelu  ($p)"; else echo "  FAIL gelu"; tail -30 /tmp/pf_gelu.log; fail=1; fi
echo "--- var-abl-sffree (BN_SOFTMAX_FREE=1) ---"
if env CUDA_VISIBLE_DEVICES=-1 BN_D_MODEL=64 BN_N_HEADS=8 BN_N_LAYERS=3 BN_FFN_DIM=256 BN_SOFTMAX_FREE=1 python -u "$CODE/qkerasModel.py" --sanity > /tmp/pf_sffree.log 2>&1 && grep -q "Total trainable parameters" /tmp/pf_sffree.log && ! grep -qE "Traceback|Exception" /tmp/pf_sffree.log; then p=$(grep "Total trainable parameters" /tmp/pf_sffree.log | tail -1); echo "  PASS sffree  ($p)"; else echo "  FAIL sffree"; tail -30 /tmp/pf_sffree.log; fail=1; fi
if [ "$fail" = 0 ]; then echo "PREFLIGHT_ALL_PASS"; else echo "PREFLIGHT_HAD_FAILURES"; fi
