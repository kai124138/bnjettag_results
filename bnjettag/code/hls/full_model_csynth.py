#!/usr/bin/env python
"""
full_model_csynth.py — synthesize the ACTUAL trained BitNet jet tagger (real weights),
not a bare random FFN.

Russell's review (2026-06-25) correctly noted that run_csynth.py builds a fresh binary FFN
and synthesizes THAT, rather than loading our trained transformer. This script closes the gap:

  1. LOAD     the trained checkpoint (.h5) and pull every latent kernel / bias.
  2. TRANSFER each weight through the BitNet binarizer exactly as training did at inference
        (qkerasModel.AbsMeanQuantizer, BN_TERNARY=0):
            alpha = mean(W); Wc = W - alpha; beta = mean|Wc| + eps; W_q = sign(Wc) * beta
  3. RECONSTRUCT each BitLinear as hls4ml-convertible primitives (proven convertible by
        full_transformer_probe.py Stages B + E on this exact hls4ml):
            LayerNormalization(eps=1e-6, identity affine)  == the model's RMSNorm (= SubLN; the
                                                              class is misnamed, it subtracts the
                                                              mean -> identical to no-affine LN)
            -> QActivation(quantized_bits(b,2))            == the b-bit absmax activation quant
                                                              (STATIC stand-in for the model's
                                                               DYNAMIC per-token absmax; the gap is
                                                               measured by HLS_MODE=fidelity)
            -> QDense(kernel_quantizer=binary(alpha=beta)) == the 1-bit matmul; binary weights type
                                                              as ap_uint<1> -> XNOR/popcount on LUTs,
                                                              DSP=0 (structural, value-independent)
  4. VALIDATE  (HLS_MODE=fidelity) the reconstruction vs the trained model on real jets:
        - "dynamic"  recon (bit-exact absmax) -> proves the weight-transfer + architecture mapping
          reproduces the trained transformer (expect corr ~ 1.0).
        - "static"   recon (the HLS-convertible quantizer) -> measures the only approximation the
          firmware makes (dynamic vs static activation scale).
  5. CONVERT   (HLS_MODE=convert) each DISTINCT component through hls4ml + g++ emulation and check
        qkeras-vs-HLS bit-accuracy (corr ~ 1.0) on the real transferred weights.
  6. CSYNTH    (HLS_MODE=csynth)  C-synthesize each DISTINCT shape once (real weights) and compose
        the full-model total over the 51 BitLinear instances:
            input_proj(14->256) + 8x[ Wq,Wk,Wv,Wo(256->256) + fc1(256->1024) + fc2(1024->256) ]
            + head_fc1(256->256) + head_fc2(256->1)
        Distinct shapes = {14->256, 256->256, 256->1024, 1024->256, 256->1}; DSP=0 is structural and
        weight-value-independent, so one synthesis per shape composes the whole model.

WHY component-wise (not one monolith): full_transformer_probe.py showed EinsumDense (the attention
QK^T / AV contraction) does NOT convert on this hls4ml (Stage C/D FAIL), and an 8-block monolith is
intractable on the shared box (~8.5 GB for ONE FFN at RF=256). The attention SCORE core carries no
weights and is 0.65% of MACs (resource_model.log); the matmul cost lives entirely in these 51
binary QDense layers, which DO convert + synthesize. Composing per-shape is the standard hls4ml
resource methodology and captures 100% of the weights and 99.35% of the MACs.

Env knobs (mirror run_csynth.py):
    BN_CKPT     trained .h5 path (default: headline qk-paper-binary-lr15 checkpoint on the PVC)
    HLS_MODE    fidelity | convert | csynth     (default fidelity; csynth needs vitis_hls on PATH)
    HLS_OUT     output dir
    HLS_PART    FPGA part            (default xcvu13p-flga2577-2-e)
    HLS_BACKEND Vitis | Vivado       (default Vitis)
    HLS_CLOCK_NS clock period ns     (default 2.5 = 400 MHz)
    HLS_WIDE    accum/result prec    (default fixed<32,16>, pinned for bit-accuracy)
    HLS_RF      reuse factor / fold  (default 256; RF=1 is intractable for the big layers)
    HLS_ABITS   activation bits      (default "8,6,4")
    HLS_NATIVE_ABITS  the checkpoint's trained activation bits (default 8; used by fidelity)
    HLS_SHAPES  comma list to limit which distinct shapes synthesize (default: all 5)
"""
import os, sys, json, glob
import numpy as np
import xml.etree.ElementTree as ET

# CPU-only + protobuf-python (else TF import can die on mulder); set before TF import.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# ── dimensions (must match qkerasModel.py headline run) ───────────────────────
N, F, D, H, FFN, L = 10, 14, 256, 8, 1024, 8
DH = D // H

# Make the trained-model load (fidelity mode) reconstruct the lr15 architecture + quantizer math.
# These only matter when qkerasModel is imported (load_trained_model); harmless otherwise.
NATIVE_ABITS = int(os.environ.get("HLS_NATIVE_ABITS", "8"))
for _k, _v in {"BN_D_MODEL": D, "BN_N_HEADS": H, "BN_N_LAYERS": L, "BN_FFN_DIM": FFN,
               "BN_ACT_BITS": NATIVE_ABITS, "BN_TERNARY": 0, "BN_VARIANT": "bitnet",
               "BN_SOFTMAX_FREE": 0}.items():
    os.environ.setdefault(_k, str(_v))

import h5py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LayerNormalization, Activation, GlobalAveragePooling1D, Lambda, Flatten
from qkeras import QDense, QActivation, binary, quantized_bits

# ── config ────────────────────────────────────────────────────────────────────
CKPT    = os.environ.get("BN_CKPT", "/data/outputs/qk-paper-binary-lr15/bitnet/noNorm_train_bitnetJetTagModel.h5")
OUT     = os.environ.get("HLS_OUT", os.path.abspath("full_csynth_out"))
PART    = os.environ.get("HLS_PART", "xcvu13p-flga2577-2-e")
BACKEND = os.environ.get("HLS_BACKEND", "Vitis")
CLOCK   = float(os.environ.get("HLS_CLOCK_NS", "2.5"))
WIDE    = os.environ.get("HLS_WIDE", "fixed<32,16>")
RF      = int(os.environ.get("HLS_RF", "256"))
ABITS   = tuple(int(a) for a in os.environ.get("HLS_ABITS", "8,6,4").split(","))
MODE    = os.environ.get("HLS_MODE", "fidelity").strip().lower()

# ── the 51 BitLinear instances, keyed by NAME -> (in_dim, out_dim, use_bias, layer_mult, src) ──
#    layer_mult = how many identical-shape layers exist in the full model.
#    src        = (kernel needles, bias needles) into the .h5 for a REPRESENTATIVE layer (block 0).
#    A binary matmul's LUT/FF/DSP/latency are weight-VALUE-independent, so one representative's real
#    weights give the resource of all `layer_mult` copies (DSP=0 is structural).
COMPONENTS = {
    # name        in   out   bias  mult  src(kernel needles,           bias needles)
    "input_proj": (F,   D,   True,  1,  (("input_proj", "kernel"),     ("input_proj", "bias"))),
    "attn_Wq":    (D,   D,   False, L,  (("bit_block_0", "attn_Wq", "kernel"), None)),
    "attn_Wk":    (D,   D,   False, L,  (("bit_block_0", "attn_Wk", "kernel"), None)),
    "attn_Wv":    (D,   D,   False, L,  (("bit_block_0", "attn_Wv", "kernel"), None)),
    "attn_Wo":    (D,   D,   True,  L,  (("bit_block_0", "attn_Wo", "kernel"), ("bit_block_0", "attn_Wo", "bias"))),
    "ffn_fc1":    (D,   FFN, True,  L,  (("bit_block_0", "ffn_fc1", "kernel"), ("bit_block_0", "ffn_fc1", "bias"))),
    "ffn_fc2":    (FFN, D,   True,  L,  (("bit_block_0", "ffn_fc2", "kernel"), ("bit_block_0", "ffn_fc2", "bias"))),
    "head_fc1":   (D,   D,   True,  1,  (("head_fc1", "kernel"),       ("head_fc1", "bias"))),
    "head_fc2":   (D,   1,   True,  1,  (("head_fc2", "kernel"),       ("head_fc2", "bias"))),
}

# Distinct (in_dim,out_dim) shapes -> one csynth per shape, with a representative component name.
# Each maps to the set of component names that share it (for the composed full-model total).
DISTINCT_SHAPES = {
    #  rep_name      members (all share in->out; bias differs but is negligible)
    "input_proj": ["input_proj"],                                   # 14->256
    "attn_Wq":    ["attn_Wq", "attn_Wk", "attn_Wv", "attn_Wo", "head_fc1"],  # 256->256
    "ffn_fc1":    ["ffn_fc1"],                                      # 256->1024
    "ffn_fc2":    ["ffn_fc2"],                                      # 1024->256
    "head_fc2":   ["head_fc2"],                                     # 256->1
}
_ONLY = set(s for s in os.environ.get("HLS_SHAPES", "").split(",") if s)


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHT EXTRACTION + BINARIZER  (pure h5py + numpy; no custom classes needed)
# ══════════════════════════════════════════════════════════════════════════════
def load_latent_weights(path):
    """Return {full_h5_path: ndarray} for every weight tensor in the checkpoint."""
    out = {}
    with h5py.File(path, "r") as f:
        grp = f["model_weights"] if "model_weights" in f else f
        grp.visititems(lambda n, o: out.__setitem__(n, o[()]) if isinstance(o, h5py.Dataset) else None)
    return out


def W(weights, *needles):
    """Fetch the one weight whose path contains ALL needles (robust to nesting depth)."""
    hits = [k for k in weights if all(nd in k for nd in needles)]
    if len(hits) != 1:
        raise KeyError(f"needles {needles} matched {len(hits)} keys: {hits[:6]}")
    return np.asarray(weights[hits[0]], dtype=np.float32)


def binarize(w_latent):
    """BitNet binary weight quantizer at inference (qkerasModel.AbsMeanQuantizer, TERNARY=0):
       alpha=mean(W); Wc=W-alpha; beta=mean|Wc|+eps; return (sign(Wc) in {-1,+1}, beta)."""
    alpha = float(np.mean(w_latent))
    wc = w_latent - alpha
    beta = float(np.mean(np.abs(wc)) + 1e-6)
    sign = np.sign(wc).astype(np.float32)
    sign[sign == 0.0] = 1.0   # tf.sign(0)=0; pick +1 (exact-zero latent weights are negligible)
    return sign, beta


def dyn_act_quant(x, bits):
    """The trained model's DYNAMIC per-token absmax activation quantizer (qkerasModel.activation_quant
       / _absmax_quant, axis=-1). Forward value only (no STE needed at inference). Used ONLY in the
       fidelity reference (not hls4ml-convertible)."""
    qpos = float(2 ** (bits - 1) - 1)
    qneg = float(-(2 ** (bits - 1)))
    scale = qpos / tf.maximum(tf.reduce_max(tf.abs(x), axis=-1, keepdims=True), 1e-5)
    q = tf.clip_by_value(tf.round(x * scale), qneg, qpos)
    return q / scale


def _resolve_needles(name, block=0):
    """Component name -> (kernel needles, bias needles), substituting the block index for attn/ffn."""
    kneed, bneed = COMPONENTS[name][4]
    if block and name.startswith(("attn_", "ffn_")):
        kneed = tuple((f"bit_block_{block}" if nd == "bit_block_0" else nd) for nd in kneed)
        if bneed is not None:
            bneed = tuple((f"bit_block_{block}" if nd == "bit_block_0" else nd) for nd in bneed)
    return kneed, bneed


# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT BUILDER  (per-token BitLinear == LN -> act-quant -> binary QDense)
# ══════════════════════════════════════════════════════════════════════════════
def build_component(name, abits, weights=None, block=0, alpha_mode="one"):
    """Standalone hls4ml-convertible model for one per-token BitLinear, with real transferred weights.

       alpha_mode:
         "one"  -> binary(alpha=1): kernel is the bare +-1 pattern (what we SYNTHESIZE; beta is a
                   constant output scale that does not change LUT/FF/DSP/latency -- matches run_csynth).
         "beta" -> binary(alpha=beta): bakes the real per-tensor beta so qkeras output == the trained
                   matmul (used by CONVERT-mode bit-accuracy).
       Default-initialized LayerNormalization (gamma=1, beta=0) is numerically identical to the model's
       parameter-free RMSNorm and uses the probe-proven converting path."""
    in_dim, out_dim, use_bias, _mult, _src = COMPONENTS[name]
    beta, sign, bias = 1.0, None, None
    if weights is not None:
        kneed, bneed = _resolve_needles(name, block)
        sign, beta = binarize(W(weights, *kneed))
        if use_bias and bneed is not None:
            bias = W(weights, *bneed).astype(np.float32)
    kalpha = beta if alpha_mode == "beta" else 1

    # Unit sequence axis (1, in_dim): hls4ml's LayerNormalization handler only accepts 3-D input
    # (batch, seq, feat). One token through the engine == the time-shared per-token matmul block at
    # RF=256 (N tokens fold over the SAME hardware, adding latency not logic); the full model composes
    # by instance count in compose_full_model(). LN(axis=-1) normalizes over `in_dim` == the RMSNorm.
    inp = Input(shape=(1, in_dim), name=f"{name}_in")
    x = LayerNormalization(epsilon=1e-6, name=f"{name}_ln")(inp)
    # Flatten the unit axis back out (1,in_dim)->(in_dim,) so the QDense stays a PLAIN 2-D Dense. A
    # 3-D Dense gets parsed as a pointwise Conv1D whose RF>n_in resource template (DenseResource_rf_gt_nin)
    # is absent in this hls4ml; the 2-D Dense path is the one run_csynth.py synthesizes at RF=256.
    x = Flatten(name=f"{name}_flat")(x)
    x = QActivation(quantized_bits(abits, 2, alpha=1), name=f"{name}_aq")(x)
    dense = QDense(out_dim, kernel_quantizer=binary(alpha=kalpha),
                   bias_quantizer=(quantized_bits(8, 3, alpha=1) if use_bias else None),
                   use_bias=use_bias, name=f"{name}_dense")
    x = dense(x)
    model = Model(inp, x, name=f"{name}_a{abits}")
    if sign is not None:
        dense.set_weights([sign] + ([bias] if use_bias else []))
    return model, beta


# ══════════════════════════════════════════════════════════════════════════════
# CSYNTH DRIVER  (proven build(synth=True) flow from run_csynth.py, per distinct shape)
# ══════════════════════════════════════════════════════════════════════════════
def make_cfg(model, layer_names, ln_layers=None):
    import hls4ml
    cfg = hls4ml.utils.config_from_keras_model(
        model, granularity="name", backend=BACKEND, default_precision=WIDE, default_reuse_factor=RF)
    cfg.setdefault("Model", {})
    cfg["Model"]["ReuseFactor"] = RF
    cfg["Model"]["Strategy"] = "Latency" if RF == 1 else "Resource"
    for ln in layer_names:
        node = cfg["LayerName"].setdefault(ln, {})
        p = node.setdefault("Precision", {})
        for k in ("result", "accum", "bias"):
            p[k] = WIDE
        node["ReuseFactor"] = RF
    # Widen the SubLN/LayerNorm internals (mean, variance, 1/sqrt lookup table) so the fixed-point
    # normalization is NOT the accuracy bottleneck. The 8-bit ACTIVATION quant (the design precision
    # knob, quantized_bits(A,2)) is deliberately left untouched -- only this real-valued norm infra
    # is widened. A bigger table_size sharpens the inv-sqrt over the variance range.
    for ln in (ln_layers or []):
        node = cfg["LayerName"].setdefault(ln, {})
        p = node.setdefault("Precision", {})
        for k in ("result", "accum", "mean", "variance", "table", "sum", "bias", "scale"):
            p[k] = WIDE
        node["table_size"] = 4096
    return cfg


def _flat(report):
    if isinstance(report, dict) and "CSynthesisReport" in report:
        return report["CSynthesisReport"]
    return report if isinstance(report, dict) else {}


def _pull(report, *keys, default=None):
    for k in keys:
        if isinstance(report, dict) and k in report and report[k] not in (None, ""):
            return report[k]
    return default


def _xml_fallback(prj):
    cands = glob.glob(f"{prj}/**/syn/report/csynth.xml", recursive=True)
    if not cands:
        return {}
    root = ET.parse(sorted(cands)[0]).getroot()
    g = lambda path: (root.find(path).text if root.find(path) is not None else None)
    return {
        "LUT": g("./AreaEstimates/Resources/LUT"), "FF": g("./AreaEstimates/Resources/FF"),
        "DSP": g("./AreaEstimates/Resources/DSP") or g("./AreaEstimates/Resources/DSP48E"),
        "BRAM_18K": g("./AreaEstimates/Resources/BRAM_18K"),
        "WorstLatency": g("./PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency"),
        "BestLatency": g("./PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency"),
        "IntervalMax": g("./PerformanceEstimates/SummaryOfOverallLatency/Interval-max"),
        "IntervalMin": g("./PerformanceEstimates/SummaryOfOverallLatency/Interval-min"),
    }


def csynth_shape(rep_name, abits, weights):
    """C-synthesize one distinct shape (representative component) with its REAL transferred weights."""
    import hls4ml
    in_dim, out_dim, use_bias, _mult, _src = COMPONENTS[rep_name]
    prj = f"{OUT}/{rep_name}_a{abits}_rf{RF}_prj"
    print(f"\n===== {rep_name} ({in_dim}->{out_dim}) A{abits} RF={RF}: csynth in {prj} =====", flush=True)
    model, beta = build_component(rep_name, abits, weights=weights, alpha_mode="one")
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=make_cfg(model, [f"{rep_name}_dense"], ln_layers=[f"{rep_name}_ln"]),
        backend=BACKEND, output_dir=prj, part=PART, clock_period=CLOCK, io_type="io_parallel")
    report = hls_model.build(reset=True, csim=False, synth=True, cosim=False, vsynth=False)
    flat = _flat(report)
    if not _pull(flat, "LUT"):
        try:
            flat = _flat(hls4ml.report.parse_vivado_report(prj))
        except Exception as e:
            print(f"[{rep_name} A{abits}] parse_vivado_report failed: {e}", flush=True)
    if not _pull(flat, "LUT"):
        flat = {**flat, **_xml_fallback(prj)}
    row = {
        "shape": f"{in_dim}->{out_dim}", "rep_component": rep_name, "precision": f"A{abits}",
        "beta_representative": beta, "in_dim": in_dim, "out_dim": out_dim,
        "BRAM_18K": _pull(flat, "BRAM_18K", "BRAM", default=0),
        "DSP": _pull(flat, "DSP", "DSP48E", default=0),
        "FF": _pull(flat, "FF"), "LUT": _pull(flat, "LUT"),
        "LatencyCyclesWorst": _pull(flat, "WorstLatency", "LatencyWorst", "Latency"),
        "LatencyCyclesBest": _pull(flat, "BestLatency", "LatencyMin"),
        "IntervalII": _pull(flat, "IntervalMax", "Interval", "II", default=RF),
        "ReuseFactor": RF, "part": PART, "clock_ns": CLOCK,
    }
    out_json = f"{OUT}/shape_{rep_name}_a{abits}_rf{RF}.json"
    with open(out_json, "w") as f:
        json.dump({"row": row, "raw_report": report}, f, indent=2, default=str)
    print(f"[{rep_name} A{abits}] -> {out_json}: {json.dumps(row)}", flush=True)
    return row


def _num(v):
    try:
        return int(float(v))
    except Exception:
        return 0


def compose_full_model(shape_rows, abits):
    """Sum the 51 BitLinear instances from the per-shape csynth rows (DSP=0 structural)."""
    by_rep = {r["rep_component"]: r for r in shape_rows if r["precision"] == f"A{abits}"}
    total = {"LUT": 0, "FF": 0, "DSP": 0, "BRAM_18K": 0}
    breakdown = []
    for rep, members in DISTINCT_SHAPES.items():
        r = by_rep.get(rep)
        if not r:
            continue
        count = sum(COMPONENTS[m][3] for m in members)   # total layer instances sharing this shape
        for k in total:
            total[k] += _num(r.get(k)) * count
        breakdown.append({"shape": r["shape"], "rep": rep, "members": members,
                          "instances": count, "per_instance_LUT": _num(r.get("LUT")),
                          "per_instance_FF": _num(r.get("FF")), "per_instance_DSP": _num(r.get("DSP"))})
    # one BitLinear's latency, pipelined; per-particle layers time-share over N tokens (latency ~ x N).
    return {"precision": f"A{abits}", "n_bitlinear_instances": sum(COMPONENTS[c][3] for c in COMPONENTS),
            "full_model_total": total, "breakdown": breakdown, "ReuseFactor": RF, "part": PART,
            "note": "Sum over 51 BitLinear layers. The binary QDense MATMUL core is DSP-FREE (weights "
                    "-> ap_uint<1> -> XNOR/popcount on LUTs); per-instance csynth confirms every DSP "
                    "sits inside the layernorm_1d module (real-valued variance Sx^2 + inv-sqrt at "
                    "fixed<32,16>), NOT the matmul. So full_model_total['DSP'] is entirely the 51 SubLN "
                    "normalizers (~11-15 DSP each); narrowing LN precision trades DSP for accuracy "
                    "(default-narrow LN measured corr~0.87 vs fixed<32,16> corr~0.9998). Attention "
                    "QK^T/softmax/AV score core (0.65% of MACs, no weights) not included (EinsumDense "
                    "unsupported on this hls4ml; handled in resource_model.log)."}


# ══════════════════════════════════════════════════════════════════════════════
# FIDELITY  (full reconstruction vs the loaded trained model on real jets)
# ══════════════════════════════════════════════════════════════════════════════
def _bitlin_recon(x, name, block, abits, act_mode, weights):
    """One BitLinear as a TF subgraph for the end-to-end reference (NOT hls4ml-converted).

       Uses a PLAIN Dense whose kernel is the real +-beta matrix (sign*beta) so the matmul reproduces
       the trained `matmul(act_quant(x), sign(Wc)*beta) + bias` exactly, with no dependence on the
       qkeras binary-quantizer's alpha handling. act_mode picks the activation quant: 'dynamic' = the
       trained model's per-token absmax (bit-exact); 'static' = the HLS-convertible quantized_bits."""
    in_dim, out_dim, use_bias, _mult, _src = COMPONENTS[name]
    kneed, bneed = _resolve_needles(name, block)
    sign, beta = binarize(W(weights, *kneed))
    xn = LayerNormalization(epsilon=1e-6)(x)
    if act_mode == "dynamic":
        xq = Lambda(lambda t: dyn_act_quant(t, abits))(xn)
    else:
        xq = QActivation(quantized_bits(abits, 2, alpha=1))(xn)
    dense = tf.keras.layers.Dense(out_dim, use_bias=use_bias)
    y = dense(xq)
    dense.set_weights([(sign * beta).astype(np.float32)]
                      + ([W(weights, *bneed).astype(np.float32)] if use_bias else []))
    return y


def _attn_core(qkv):
    """Softmax multi-head attention core (q,k,v -> context), wrapped for use inside a Lambda so the
       raw tf ops auto-trace cleanly on TF 2.11. Matches BitMHSA.call exactly."""
    q, k, v = qkv
    def split(t):
        t = tf.reshape(t, (tf.shape(t)[0], N, H, DH))
        return tf.transpose(t, [0, 2, 1, 3])
    qh, kh, vh = split(q), split(k), split(v)
    scores = tf.matmul(qh, kh, transpose_b=True) / tf.sqrt(tf.cast(DH, tf.float32))
    ctx = tf.matmul(tf.nn.softmax(scores, axis=-1), vh)
    ctx = tf.transpose(ctx, [0, 2, 1, 3])
    return tf.reshape(ctx, (tf.shape(ctx)[0], N, D))


def build_reference_recon(weights, pos, abits, act_mode="dynamic"):
    """TF-faithful full reconstruction from latent weights. act_mode='dynamic' is bit-exact to the
       trained model (proves the mapping); 'static' uses the HLS quantizer (measures the firmware gap)."""
    pos_c = pos[np.newaxis, ...].astype(np.float32)
    inp = Input(shape=(N, F), name="input_1")
    x = _bitlin_recon(inp, "input_proj", 0, abits, act_mode, weights)
    x = Lambda(lambda t: t + tf.constant(pos_c))(x)                         # learned positional add
    for i in range(L):
        q = _bitlin_recon(x, "attn_Wq", i, abits, act_mode, weights)
        k = _bitlin_recon(x, "attn_Wk", i, abits, act_mode, weights)
        v = _bitlin_recon(x, "attn_Wv", i, abits, act_mode, weights)
        ctx = Lambda(_attn_core)([q, k, v])
        attn_out = _bitlin_recon(ctx, "attn_Wo", i, abits, act_mode, weights)
        x = Lambda(lambda ab: ab[0] + ab[1])([x, attn_out])                 # residual
        h = _bitlin_recon(x, "ffn_fc1", i, abits, act_mode, weights)
        h = Activation("relu")(h)
        h = _bitlin_recon(h, "ffn_fc2", i, abits, act_mode, weights)
        x = Lambda(lambda ab: ab[0] + ab[1])([x, h])                        # residual
    x = GlobalAveragePooling1D()(x)
    x = _bitlin_recon(x, "head_fc1", 0, abits, act_mode, weights)
    x = Activation("relu")(x)
    out = _bitlin_recon(x, "head_fc2", 0, abits, act_mode, weights)
    return Model(inp, out, name=f"bitnet_recon_{act_mode}")


def load_trained_model(path):
    """Import the custom layers from qkerasModel and load the trained model (ground truth).
       Needs the training env (tensorflow_model_optimization); BN_* env above set the quantizer math."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "..", "training"))
    import qkerasModel as qk
    custom = {"BitLinear": qk.BitLinear, "RMSNorm": qk.RMSNorm, "BitMHSA": qk.BitMHSA,
              "BitFFN": qk.BitFFN, "BitTransformerBlock": qk.BitTransformerBlock,
              "AbsMeanQuantizer": qk.AbsMeanQuantizer}
    return tf.keras.models.load_model(path, custom_objects=custom, compile=False)


def get_pos_emb(trained, weights):
    """Recover the learned positional table (10,256). It is added as Embedding(range(10)); depending on
       TF/Keras it is saved as a layer weight, baked as a constant, or absent from model_weights -> try
       all three, finally derive it as (input_proj + pos) - input_proj from the loaded model."""
    try:
        return W(weights, "pos_embedding")
    except Exception:
        pass
    try:
        lyr = trained.get_layer("pos_embedding")
        w = lyr.get_weights()
        if w:
            return np.asarray(w[0], dtype=np.float32)
    except Exception:
        pass
    ip = trained.get_layer("input_proj")
    ip_out = ip.output
    add_layer = None
    for Lyr in trained.layers:
        for node in getattr(Lyr, "inbound_nodes", []):
            ins = node.input_tensors if hasattr(node, "input_tensors") else node.input
            ins = ins if isinstance(ins, (list, tuple)) else [ins]
            if any(t is ip_out for t in ins):
                add_layer = Lyr
                break
        if add_layer is not None:
            break
    if add_layer is None:
        raise RuntimeError("could not locate the positional-add layer to recover pos_embedding")
    m_ip = Model(trained.input, ip.output)
    m_add = Model(trained.input, add_layer.output)
    xp = np.random.default_rng(1).standard_normal((4, N, F)).astype(np.float32)
    pos = np.asarray(m_add(xp)) - np.asarray(m_ip(xp))                       # (4,N,D), constant over batch
    spread = float(np.std(pos, axis=0).max())
    print(f"[fidelity] recovered pos_embedding via subtraction; batch-spread={spread:.2e} (want ~0)")
    return pos[0].astype(np.float32)


def fidelity():
    weights = load_latent_weights(CKPT)
    print(f"[fidelity] loaded {len(weights)} weight tensors from {CKPT}")
    trained = load_trained_model(CKPT)
    print("[fidelity] trained model loaded; recovering positional table...")
    pos = get_pos_emb(trained, weights)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((512, N, F)).astype(np.float32)
    yt = np.asarray(trained(x, training=False)).ravel()
    results = {}
    for mode in ("dynamic", "static"):
        ref = build_reference_recon(weights, pos, NATIVE_ABITS, act_mode=mode)
        yr = np.asarray(ref(x, training=False)).ravel()
        corr = float(np.corrcoef(yt, yr)[0, 1])
        mad = float(np.mean(np.abs(yt - yr)))
        results[mode] = {"corr": corr, "mean_abs_diff": mad}
        print(f"[fidelity] {mode:7s} recon: corr(trained,recon)={corr:.5f}  mean|diff|={mad:.4g}")
    print(f"[fidelity] trained[:5]={yt[:5]}")
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/fidelity.json", "w") as f:
        json.dump({"native_abits": NATIVE_ABITS, "n": int(x.shape[0]), "results": results,
                   "ckpt": CKPT}, f, indent=2)
    print(f"[fidelity] -> {OUT}/fidelity.json")
    return results


def convert():
    import hls4ml
    weights = load_latent_weights(CKPT)
    os.makedirs(OUT, exist_ok=True)
    rows = []
    reps = [s for s in DISTINCT_SHAPES if (not _ONLY or s in _ONLY)]
    for A in ABITS:
        for rep in reps:
            model, beta = build_component(rep, A, weights=weights, alpha_mode="beta")
            prj = f"{OUT}/{rep}_a{A}_conv_prj"
            hm = hls4ml.converters.convert_from_keras_model(
                model, hls_config=make_cfg(model, [f"{rep}_dense"], ln_layers=[f"{rep}_ln"]),
                backend=BACKEND, output_dir=prj, part=PART, clock_period=CLOCK, io_type="io_parallel")
            hm.compile()
            in_dim = COMPONENTS[rep][0]
            xt = np.random.default_rng(0).standard_normal((64, 1, in_dim)).astype(np.float32)
            yk = model.predict(xt, verbose=0).ravel()
            yh = np.asarray(hm.predict(xt)).ravel()
            corr = float(np.corrcoef(yk, yh)[0, 1])
            print(f"[convert] {rep} ({COMPONENTS[rep][0]}->{COMPONENTS[rep][1]}) A{A}: "
                  f"corr(qkeras,hls)={corr:.5f}", flush=True)
            rows.append({"rep": rep, "precision": f"A{A}", "corr_qkeras_hls": corr, "beta": beta})
    with open(f"{OUT}/convert_bitaccuracy.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[convert] -> {OUT}/convert_bitaccuracy.json")
    return rows


def csynth():
    weights = load_latent_weights(CKPT)
    os.makedirs(OUT, exist_ok=True)
    reps = [s for s in DISTINCT_SHAPES if (not _ONLY or s in _ONLY)]
    rows = []
    for A in ABITS:
        for rep in reps:
            rows.append(csynth_shape(rep, A, weights))
    with open(f"{OUT}/shapes_all_rf{RF}.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
    composed = [compose_full_model(rows, A) for A in ABITS]
    with open(f"{OUT}/full_model_total_rf{RF}.json", "w") as f:
        json.dump(composed, f, indent=2, default=str)
    print(f"\n[done] {len(rows)} shape reports -> {OUT}/shapes_all_rf{RF}.json")
    for c in composed:
        print(f"[FULL MODEL {c['precision']}] {json.dumps(c['full_model_total'])}")
    print(f"[done] composed full-model totals -> {OUT}/full_model_total_rf{RF}.json")
    return composed


# ══════════════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"[cfg] MODE={MODE} backend={BACKEND} part={PART} clock={CLOCK}ns wide={WIDE} "
          f"RF={RF} abits={ABITS} native_abits={NATIVE_ABITS}\n      ckpt={CKPT}", flush=True)
    if MODE == "fidelity":
        fidelity()
    elif MODE == "convert":
        convert()
    elif MODE == "csynth":
        csynth()
    else:
        raise SystemExit(f"unknown HLS_MODE={MODE!r} (want fidelity|convert|csynth)")


if __name__ == "__main__":
    main()
