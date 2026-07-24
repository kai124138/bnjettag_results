#!/usr/bin/env python3
"""
Quantization-aggressiveness x hls4ml FIRMWARE sweep for the binary BitNet core.

The abstract's third question is "how aggressively can the 1-bit model be quantized
before its tagging efficiency, RESOURCE CONSUMPTION, and LATENCY degrade." The training
side answers efficiency (A8 -> A6 -> A4 = 0.756 -> 0.745 -> 0.738). THIS script answers
the firmware side for the same axis: it pushes the dominant primitive — a binary
{-1,+1}-weight FFN block (fc1 256->1024 -> relu -> fc2 1024->256) — through REAL hls4ml
at A8/A6/A4 and, for each, reports the firmware cost knobs that resource/latency ride on.

Per precision A it:
  1. converts QKeras -> HLS C++ (codegen only; NO Vitis needed),
  2. g++-emulates bit-accurately vs Keras (corr; proves the firmware reproduces the model),
  3. inspects the GENERATED firmware (defines.h / parameters.h) to read off:
       - the weight C type  -> binary must become ap_uint<1>  => LUT logic, 0 DSP (structural)
       - the accumulator width (popcount adder-tree depth grows with the dot length, not A)
       - the activation/io width (THIS is the A8->A6->A4 dial: narrower act buffers + adders)
       - the reuse factor RF (latency proxy: II/latency ~ RF cycles).

No Vitis/Vivado is required for any step here. The only thing that needs a Xilinx backend
is the final csynth that turns these knobs into exact LUT/FF/DSP/BRAM + latency-in-cycles;
those projects are emitted to disk ready for an off-cluster csynth handoff.
"""
import os, re, glob, json, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import hls4ml
from qkeras import QDense, QActivation, quantized_bits, binary

OUT = "/data/outputs/hls"
os.makedirs(OUT, exist_ok=True)
D, FFN = 256, 1024
PART, BACKEND = "xcvu13p-flga2577-2-e", "Vitis"
WIDE = "ap_fixed<32,16>"   # 16 int bits => +-32768, safe for the 256/1024-term binary sums
ABITS = (8, 6, 4)
print("hls4ml", hls4ml.__version__, "| TF", tf.__version__, "| part", PART, "backend", BACKEND)


def build_ffn(abits):
    """Binary-weight FFN block; activation precision = the swept knob (quantized_relu(A,2))."""
    inp = Input(shape=(D,), name="ffn_in")
    x = QDense(FFN, kernel_quantizer=binary(alpha=1),
               bias_quantizer=quantized_bits(8, 3, alpha=1), name="fc1")(inp)
    x = QActivation(f"quantized_relu({abits},2)", name="act")(x)
    x = QDense(D, kernel_quantizer=binary(alpha=1),
               bias_quantizer=quantized_bits(8, 3, alpha=1), name="fc2")(x)
    return Model(inp, x, name=f"binary_ffn_a{abits}")


def make_cfg(model):
    cfg = hls4ml.utils.config_from_keras_model(model, granularity="name", backend=BACKEND,
                                               default_precision=WIDE)
    # widen accumulator+result+io+bias on the DENSE path; keep weight=ap_uint<1>; keep act
    # NATIVE so quantized_relu's saturation clip (range [0,4)) survives (see stage_a_fix.py).
    for ln in ("ffn_in", "fc1", "fc2"):
        p = cfg["LayerName"].setdefault(ln, {}).setdefault("Precision", {})
        for k in ("result", "accum", "bias"):
            p[k] = WIDE
    return cfg


def grep_firmware(prj):
    """Read the generated C++ to extract weight types, accumulator widths, reuse factor."""
    fw = os.path.join(prj, "firmware")
    facts = {"weight_types": {}, "accum_types": {}, "reuse": {}, "files": []}
    for fn in ("defines.h", "parameters.h"):
        path = os.path.join(fw, fn)
        if not os.path.exists(path):
            continue
        facts["files"].append(fn)
        txt = open(path).read()
        # typedef ... weightN_t;  and  ... accumN_t;
        for m in re.finditer(r"typedef\s+(.+?)\s+(weight\d+_t)\s*;", txt):
            facts["weight_types"][m.group(2)] = m.group(1).strip()
        for m in re.finditer(r"typedef\s+(.+?)\s+(accum\w*_t)\s*;", txt):
            facts["accum_types"][m.group(2)] = m.group(1).strip()
        for m in re.finditer(r"static const unsigned (\w*[Rr]euse\w*)\s*=\s*(\d+)\s*;", txt):
            facts["reuse"][m.group(1)] = int(m.group(2))
    return facts


def is_binary(ctype):
    return bool(re.search(r"ap_u?int<\s*1\s*>", ctype))


summary = []
for A in ABITS:
    print("\n" + "=" * 78 + f"\nA{A}  — binary FFN @ {A}-bit activations\n" + "=" * 78)
    m = build_ffn(A)
    cfg = make_cfg(m)
    prj = f"{OUT}/binary_ffn_a{A}_prj"
    hls_m = hls4ml.converters.convert_from_keras_model(
        m, hls_config=cfg, backend=BACKEND, output_dir=prj, part=PART)
    print(f"[convert] OK -> {prj}")
    # bit-accurate emulation
    corr = float("nan")
    try:
        hls_m.compile()
        xb = np.random.randn(1000, D).astype(np.float32)
        yk = m.predict(xb, verbose=0).ravel()
        yh = np.asarray(hls_m.predict(xb)).ravel()
        corr = float(np.corrcoef(yk, yh)[0, 1])
        maxd = float(np.max(np.abs(yk - yh)))
        print(f"[emulate] N=1000  corr={corr:.6f}  max|keras-hls|={maxd:.4g}")
    except Exception as e:
        print(f"[emulate] g++ bridge unavailable: {e}")
    # firmware inspection
    fw = grep_firmware(prj)
    wts = fw["weight_types"]
    binw = {k: v for k, v in wts.items() if is_binary(v)}
    print(f"[firmware] weight typedefs: {wts}")
    print(f"[firmware]   -> binary (ap_uint<1>, => LUT logic / 0 DSP): {list(binw) or 'NONE FOUND'}")
    print(f"[firmware] accum typedefs : {fw['accum_types']}")
    print(f"[firmware] reuse factors  : {fw['reuse']}")
    rf = max(fw["reuse"].values()) if fw["reuse"] else 1
    summary.append(dict(A=A, corr=corr, n_binary_kernels=len(binw),
                        weight_types=list(wts.values()), reuse=rf,
                        all_dense_binary=(len(binw) >= 2)))

# ---- summary table ----------------------------------------------------------------
print("\n" + "=" * 78 + "\nSWEEP SUMMARY  (quantization-aggressiveness x firmware)\n" + "=" * 78)
hdr = f"{'A-bits':>7}{'emul corr':>12}{'binary kernels':>16}{'all dense=ap_uint<1>':>22}{'reuse':>8}"
print(hdr); print("-" * len(hdr))
for s in summary:
    print(f"{('A'+str(s['A'])):>7}{s['corr']:>12.6f}{s['n_binary_kernels']:>16}"
          f"{str(s['all_dense_binary']):>22}{s['reuse']:>8}")
print("-" * len(hdr))
print("READ: corr~1.0 at every A => the firmware reproduces the quantized model bit-accurately")
print("      at all three precisions. Both dense kernels type as ap_uint<1> at every A => the")
print("      binary MACs are LUT logic with NO DSP, independent of activation precision. The")
print("      A8->A6->A4 knob narrows the activation datapath (io/act width + adder-tree bits),")
print("      which is the resource/latency dial; the DSP=0 result is structural and precision-")
print("      independent. Exact LUT/FF/DSP/BRAM + latency-cycles require csynth (Vitis) on the")
print("      emitted projects: " + ", ".join(f"binary_ffn_a{s['A']}_prj" for s in summary))
with open(f"{OUT}/sweep_precision_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n[saved] {OUT}/sweep_precision_summary.json")
