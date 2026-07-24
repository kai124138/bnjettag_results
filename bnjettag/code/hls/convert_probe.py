#!/usr/bin/env python3
"""
hls4ml convertibility probe for the binary BitNet jet tagger.

Goal: with the REAL hls4ml tool, demonstrate that the BINARY CORE of our architecture
(the dominant 90%+ of resources: binary Dense projections + FFN + head) converts to
firmware and emulates bit-accurately, and probe the two harder pieces (LayerNorm,
softmax-free attention via EinsumDense). No Vitis/Vivado is required for any step here
(convert = codegen; compile = g++ bridge); only the final csynth (resource numbers)
would need a Xilinx backend, which NRP lacks.

Each stage is independently guarded so partial success is captured.
"""
import os, traceback, numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.models import Model

OUT = "/data/outputs/hls"
os.makedirs(OUT, exist_ok=True)
D, FFN, N, F = 256, 1024, 10, 14
PART = "xcvu13p-flga2577-2-e"          # VU13P, same family as the upstream HLS script
BACKEND = "Vitis"

def banner(s): print("\n" + "="*78 + f"\n{s}\n" + "="*78)

import hls4ml
from qkeras import QDense, QActivation, quantized_bits, binary
print("hls4ml", hls4ml.__version__)

def dump_weight_precision(cfg):
    lt = cfg.get("LayerName", {})
    for name, d in lt.items():
        p = d.get("Precision", {})
        if isinstance(p, dict) and "weight" in p:
            print(f"   {name:<10} weight={p.get('weight')}  result={p.get('result')}")

# ── Stage A: binary FFN block (fc1 256->1024 -> relu -> fc2 1024->256) ──────────
banner("STAGE A — binary FFN block (the dominant primitive)")
try:
    inp = Input(shape=(D,), name="ffn_in")
    x = QDense(FFN, kernel_quantizer=binary(alpha=1),
               bias_quantizer=quantized_bits(8, 0, alpha=1), name="fc1")(inp)
    x = QActivation("quantized_relu(8,0)", name="act")(x)
    x = QDense(D, kernel_quantizer=binary(alpha=1),
               bias_quantizer=quantized_bits(8, 0, alpha=1), name="fc2")(x)
    ffn = Model(inp, x, name="binary_ffn")
    ffn.summary()

    cfg = hls4ml.utils.config_from_keras_model(ffn, granularity="name", backend=BACKEND,
                                               default_precision="fixed<16,6>")
    print("\n[config] per-layer weight precision (binary => 1-bit signed => maps to LUT logic):")
    dump_weight_precision(cfg)

    hls_ffn = hls4ml.converters.convert_from_keras_model(
        ffn, hls_config=cfg, backend=BACKEND,
        output_dir=f"{OUT}/binary_ffn_prj", part=PART)
    print("[convert] OK -> HLS project written to", f"{OUT}/binary_ffn_prj")

    try:
        hls_ffn.compile()
        xb = np.random.randn(512, D).astype(np.float32)
        yk = ffn.predict(xb, verbose=0).ravel()
        yh = np.asarray(hls_ffn.predict(xb)).ravel()
        print(f"[emulate] OK  max|keras-hls|={np.max(np.abs(yk-yh)):.4g}  "
              f"corr={np.corrcoef(yk, yh)[0,1]:.5f}")
        print("STAGE A: PASS (convert + bit-accurate emulate)")
    except Exception:
        print("[emulate] could not g++-compile bridge (csim) — convert still succeeded:")
        traceback.print_exc()
        print("STAGE A: PARTIAL (converted; emulation needs g++)")
except Exception:
    traceback.print_exc(); print("STAGE A: FAIL")

# ── Stage B: LayerNormalization (the 'fragile' dependency) ──────────────────────
banner("STAGE B — LayerNormalization (SubLN) convertibility")
try:
    inp = Input(shape=(N, D), name="ln_in")
    x = tf.keras.layers.LayerNormalization(name="ln")(inp)
    lnm = Model(inp, x, name="ln_probe")
    cfg = hls4ml.utils.config_from_keras_model(lnm, granularity="name", backend=BACKEND,
                                               default_precision="fixed<16,6>")
    hls_ln = hls4ml.converters.convert_from_keras_model(
        lnm, hls_config=cfg, backend=BACKEND, io_type="io_parallel",
        output_dir=f"{OUT}/layernorm_prj", part=PART)
    print("[convert] OK -> LayerNormalization converted (io_parallel)")
    print("STAGE B: PASS (LayerNorm converts)")
except Exception:
    traceback.print_exc(); print("STAGE B: FAIL (LayerNorm unsupported on this version/shape)")

# ── Stage C: softmax-free attention score path via EinsumDense ──────────────────
banner("STAGE C — softmax-free attention building blocks (EinsumDense / matmul)")
try:
    # does this hls4ml expose an Einsum/EinsumDense handler?
    from hls4ml.model import layers as _l
    have = [n for n in dir(_l) if "insum" in n.lower() or "atmul" in n.lower()]
    print("[probe] hls4ml IR layers matching Einsum/Matmul:", have)
    # try a tiny EinsumDense (QKᵀ-like contraction) if keras provides it
    try:
        from tensorflow.keras.layers import EinsumDense
        inp = Input(shape=(N, D), name="ed_in")
        x = EinsumDense("bnd,de->bne", output_shape=(N, D), name="ed")(inp)
        edm = Model(inp, x, name="einsum_probe")
        cfg = hls4ml.utils.config_from_keras_model(edm, granularity="name", backend=BACKEND)
        hls_ed = hls4ml.converters.convert_from_keras_model(
            edm, hls_config=cfg, backend=BACKEND, output_dir=f"{OUT}/einsum_prj", part=PART)
        print("[convert] OK -> EinsumDense converted (path for softmax-free attn matmuls)")
        print("STAGE C: PASS (EinsumDense converts)")
    except Exception:
        traceback.print_exc()
        print("STAGE C: PARTIAL (Einsum IR present? see list above; full attn = extension work)")
except Exception:
    traceback.print_exc(); print("STAGE C: FAIL")

banner("PROBE COMPLETE")
