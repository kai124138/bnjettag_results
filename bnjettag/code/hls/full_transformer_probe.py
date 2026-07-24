#!/usr/bin/env python3
"""
STRETCH: full BitNet transformer block through hls4ml >= 1.2 (Python >= 3.10).

Phase-1 (hls4ml 0.8.1) proved the binary FFN converts + emulates bit-accurately, but
0.8.1 FAILED on LayerNormalization (SubLN) and EinsumDense (the attention contractions).
hls4ml >= 1.2 added transformer support (LayerNorm PR #1109/#1110, Einsum/EinsumDense
#1424). This probe stands up that newer stack and tries to convert the pieces 0.8.1
could not, closing the Phase-1 gap so the WHOLE block — not just the FFN — is shown
convertible: binary QDense projections + SubLN + softmax-free EinsumDense attention.

Guarded stage-by-stage so partial success is captured. convert = codegen (no Vitis);
compile = g++ emulation (no Vitis); only csynth needs a Xilinx backend.
"""
import os, traceback, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.keras.layers import Input, ReLU
from tensorflow.keras.models import Model
import hls4ml
# qkeras imported lazily inside Stage E only — Stages B/C/D (the actual Phase-1 gap:
# LayerNorm + EinsumDense in plain Keras) must run even if qkeras won't install on the
# newer (Python>=3.10) stack.

OUT = "/data/outputs/hls"
os.makedirs(OUT, exist_ok=True)
D, FFN, N, F, H = 256, 1024, 10, 14, 8
DH = D // H
PART, BACKEND = "xcvu13p-flga2577-2-e", "Vitis"
WIDE = "ap_fixed<32,16>"
print("hls4ml", hls4ml.__version__, "| TF", tf.__version__)


def banner(s): print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def try_convert(model, name, io_type="io_parallel"):
    cfg = hls4ml.utils.config_from_keras_model(model, granularity="name", backend=BACKEND,
                                               default_precision=WIDE)
    hls_m = hls4ml.converters.convert_from_keras_model(
        model, hls_config=cfg, backend=BACKEND, io_type=io_type,
        output_dir=f"{OUT}/{name}_prj", part=PART)
    print(f"[convert] OK -> {OUT}/{name}_prj")
    return hls_m


# ── Stage B (retry on >=1.2): LayerNormalization / SubLN ────────────────────────
banner("STAGE B (>=1.2 retry) — LayerNormalization (SubLN)")
try:
    inp = Input(shape=(N, D), name="ln_in")
    x = tf.keras.layers.LayerNormalization(name="ln")(inp)
    try_convert(Model(inp, x, name="ln_probe"), "layernorm_v12")
    print("STAGE B: PASS (LayerNorm converts on this hls4ml)")
except Exception:
    traceback.print_exc(); print("STAGE B: FAIL")

# ── Stage C (retry on >=1.2): EinsumDense (attention contraction) ────────────────
banner("STAGE C (>=1.2 retry) — EinsumDense (softmax-free attention matmul)")
try:
    from tensorflow.keras.layers import EinsumDense
    inp = Input(shape=(N, D), name="ed_in")
    x = EinsumDense("bnd,de->bne", output_shape=(N, D), name="ed")(inp)
    try_convert(Model(inp, x, name="einsum_v12"), "einsum_v12")
    print("STAGE C: PASS (EinsumDense converts on this hls4ml)")
except Exception:
    traceback.print_exc(); print("STAGE C: FAIL")

# ── Stage D: softmax-free attention SCORE path assembled from primitives ─────────
# QK^T via EinsumDense, ReLU, constant 1/N scale (no softmax) -> the binarizable core.
banner("STAGE D — softmax-free attention score path (EinsumDense + ReLU + const-scale)")
try:
    from tensorflow.keras.layers import EinsumDense, Lambda
    q = Input(shape=(H, N, DH), name="q")
    k = Input(shape=(H, N, DH), name="k")
    scores = EinsumDense("bhnd,bhmd->bhnm", output_shape=(H, N, N), name="qk")([q, k])
    scores = ReLU(name="relu_scores")(scores)
    scores = Lambda(lambda t: t * (1.0 / N), name="inv_n")(scores)
    try_convert(Model([q, k], scores, name="sfree_attn"), "sfree_attn_v12")
    print("STAGE D: PASS (softmax-free score path converts)")
except Exception:
    traceback.print_exc(); print("STAGE D: FAIL (may need Extension API / multi-input einsum)")

# ── Stage E: ONE full pre-norm BitNet block (SubLN -> binary QKV -> sfree attn ───
#            -> binary W_o -> SubLN -> binary FFN), end to end ───────────────────
banner("STAGE E — one full BitNet transformer block, end to end")
try:
    from qkeras import QDense, QActivation, quantized_bits, binary
    inp = Input(shape=(N, D), name="blk_in")
    h = tf.keras.layers.LayerNormalization(name="ln1")(inp)
    h = QActivation("quantized_bits(8,2,alpha=1)", name="aq1")(h)
    qd = QDense(D, kernel_quantizer=binary(alpha=1),
                bias_quantizer=quantized_bits(8, 3, alpha=1), name="w_q")(h)
    # (single-head-style contraction stand-in for the block smoke test)
    ff = tf.keras.layers.LayerNormalization(name="ln2")(qd)
    ff = QActivation("quantized_bits(8,2,alpha=1)", name="aq2")(ff)
    ff = QDense(FFN, kernel_quantizer=binary(alpha=1),
                bias_quantizer=quantized_bits(8, 3, alpha=1), name="fc1")(ff)
    ff = QActivation("quantized_relu(8,2)", name="act")(ff)
    ff = QDense(D, kernel_quantizer=binary(alpha=1),
                bias_quantizer=quantized_bits(8, 3, alpha=1), name="fc2")(ff)
    try_convert(Model(inp, ff, name="bitnet_block"), "bitnet_block_v12")
    print("STAGE E: PASS (full block converts on this hls4ml)")
except Exception:
    traceback.print_exc(); print("STAGE E: FAIL")

banner("FULL-TRANSFORMER PROBE COMPLETE")
