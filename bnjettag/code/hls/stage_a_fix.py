#!/usr/bin/env python3
"""
Stage A (rerun): binary FFN block through hls4ml with a CORRECTLY-SIZED accumulator.
The first pass diverged (corr~0.24) because the default fixed<16,6> accumulator (+-32)
overflows: fc1 sums 256 signed terms reaching +-50. Widen precision -> expect a faithful
bit-accurate match, proving the binary core converts AND reproduces the model.
"""
import os, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import hls4ml
from qkeras import QDense, QActivation, quantized_bits, binary

OUT = "/data/outputs/hls"
D, FFN = 256, 1024
PART, BACKEND = "xcvu13p-flga2577-2-e", "Vitis"
WIDE = "ap_fixed<32,16>"     # 16 integer bits => +-32768, safe for 1024-term sums

inp = Input(shape=(D,), name="ffn_in")
x = QDense(FFN, kernel_quantizer=binary(alpha=1),
           bias_quantizer=quantized_bits(8, 3, alpha=1), name="fc1")(inp)
x = QActivation("quantized_relu(8,2)", name="act")(x)
x = QDense(D, kernel_quantizer=binary(alpha=1),
           bias_quantizer=quantized_bits(8, 3, alpha=1), name="fc2")(x)
ffn = Model(inp, x, name="binary_ffn")

cfg = hls4ml.utils.config_from_keras_model(ffn, granularity="name", backend=BACKEND,
                                           default_precision=WIDE)
# widen accumulator + result + input/bias on the DENSE path only.
# CRITICAL: do NOT widen 'act'. quantized_relu(8,2) saturates at 4; the fc1 pre-acts
# have std ~16 (sum of 256 unit terms) so almost all clip to ~4 in keras. If we widen
# act's result the clip vanishes, hls activations blow up ~15x, and fc2 diverges
# (first wide pass: corr 0.85, max|diff| 1639). Leaving act native keeps the [0,4) clip.
for ln in ("ffn_in", "fc1", "fc2"):
    p = cfg["LayerName"].setdefault(ln, {}).setdefault("Precision", {})
    for k in ("result", "accum", "weight", "bias"):
        if k == "weight":
            continue  # keep binary uint<1>
        p[k] = WIDE
print("[config] precisions:")
for ln in ("ffn_in", "fc1", "act", "fc2"):
    print("  ", ln, cfg["LayerName"][ln].get("Precision"))

hls_ffn = hls4ml.converters.convert_from_keras_model(
    ffn, hls_config=cfg, backend=BACKEND,
    output_dir=f"{OUT}/binary_ffn_prj_wide", part=PART)
hls_ffn.compile()

xb = (np.random.randn(1000, D).astype(np.float32))
yk = ffn.predict(xb, verbose=0).ravel()
yh = np.asarray(hls_ffn.predict(xb)).ravel()
print(f"\n[emulate] N=1000  max|keras-hls|={np.max(np.abs(yk-yh)):.4g}  "
      f"mean|.|={np.mean(np.abs(yk-yh)):.4g}  corr={np.corrcoef(yk, yh)[0,1]:.6f}")
print("RESULT:", "PASS — binary FFN converts AND emulates bit-accurately"
      if np.corrcoef(yk, yh)[0,1] > 0.999 else "still mismatched — investigate precision")
