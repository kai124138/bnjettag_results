"""monolith-v2 — HYBRID-strategy whole-model emission (full-synthesis Route A/B).

The FINAL flagship graph (build.py) is EinsumDense-dominated: 51 QEinsumDense carry
6.4M ±1 weights, and hls4ml 1.3.0 makes EinsumDense **Latency-only**, so at Latency all
those weights inline as compiler constants → the 27M-IR monolith OOM'd on mulder, and the
big binary einsum-denses (≥~40k MACs) clang-LTO non-converge (constraints_map 2026-07-09).

This module re-expresses every WEIGHTED layer as the v2/v3-proven per-shape recipe, but
COMPOSED inside one graph instead of as isolated probes:

  * a per-token linear projection = a keras QDense applied to the (T, in) token tensor,
    which hls4ml parses as **PointwiseConv1D** (measured 2026-07-10). At Strategy=Resource
    the ±1 kernel goes to a weight ROM (`load_weights_from_txt`) — bounded LTO, the whole
    reason Route A can compile. Head split/merge that the einsum equations did implicitly
    (`btd,dhe->bthe`) becomes explicit keras Reshape, which hls4ml fuses away in io_parallel
    (0 layers, bit-exact — measured). head_fc1/head_fc2 (post-GAP, 2-D) stay genuine QDense.
  * pure ±1 datapath + β via a frozen CSD-2 QBatchNormalization affine (the Resource
    1-bit-ROM trap avoidance), wide accumulator (frac≥16 → the ±1 MAC LUT-maps to 0 DSP),
    ReLU-SAT on fc1/head_fc1 — identical to probes_final_v2/v3.
  * input_proj's positional-encoding (T,D) additive — which bias_axes='td' held in the
    einsum but a PointwiseConv1D's per-channel bias cannot — is reinstated by a
    Reshape (T,D)→(T·D,) → axis=-1 QBatchNormalization with (T·D,) params (γ=CSD-2(β),
    β=(bias_d+PE) flat) → Reshape back. Per-element affine, bit-exact (measured).

  * the WEIGHTLESS attention einsums (QKᵀ, attn·V — 25,600 MACs each, act×act, no inlined
    weights so under the LTO wall) and QSoftmax stay on the native einsum path at **Latency**
    via per-layer strategy overrides (convert(layer_strategies=...)). SubLN unchanged.

Everything else (SubLN, QEinsum, QSoftmax, Add, GAP) is byte-for-byte the build.py graph, so
the attention-path quantization grids are identical; only the 51 projections are reshaped.
The attention quantizer-config helpers below MIRROR build.py's closures exactly (copied, not
imported — build.py's are nested — and guarded by the C-sim gate against drift).
"""
from __future__ import annotations

import os

import numpy as np


# --------------------------------------------------------------------------- #
# quantizer configs — mirror build.py exactly (see module docstring)          #
# --------------------------------------------------------------------------- #
def _binary_kq():
    from hgq.quantizer import QuantizerConfig
    return QuantizerConfig("kbi", "weight", k0=1, b0=1, i0=1, round_mode="RND",
                           overflow_mode="SAT_SYM", trainable=False, heterogeneous_axis=(),
                           bc=None, ic=None, br=None, ir=None)


def _act_q(act_bits, i0):
    from hgq.quantizer import QuantizerConfig
    f0 = act_bits - 1 - int(i0)
    return QuantizerConfig("kif", "datalane", k0=1, i0=int(i0), f0=f0,
                           round_mode="RND_CONV", overflow_mode="SAT", trainable=False,
                           ic=None, fc=None, ir=None, fr=None)


def _wq_wide():
    from hgq.quantizer import QuantizerConfig
    return QuantizerConfig("kif", "weight", k0=1, i0=2, f0=16, round_mode="RND",
                           overflow_mode="SAT", trainable=False, heterogeneous_axis=(),
                           ic=None, fc=None, ir=None, fr=None)


def _bq_wide():
    from hgq.quantizer import QuantizerConfig
    return QuantizerConfig("kif", "bias", k0=1, i0=8, f0=16, round_mode="RND_CONV",
                           overflow_mode="SAT", trainable=False, heterogeneous_axis=(),
                           ic=None, fc=None, ir=None, fr=None)


def _dl(k0, i0_, f0_):
    from hgq.quantizer import QuantizerConfig
    return QuantizerConfig("kif", "datalane", k0=k0, i0=int(i0_), f0=int(f0_),
                           round_mode="RND_CONV", overflow_mode="SAT", trainable=False,
                           heterogeneous_axis=(), homogeneous_axis=None,
                           ic=None, fc=None, ir=None, fr=None)


def _stream_iq(cfg, calib, name):
    margin = 1 + int(os.environ.get("BNHGQ2_STREAM_MARGIN", "0"))
    smax = float(calib.get(f"__stream_max_{name}", 512.0))
    i_ = int(np.ceil(np.log2(max(smax, 1.0) + 1e-9))) + margin
    return _dl(1, i_, cfg["quant"]["act_bits"] - 1)


def _attn_grid():
    return _dl(0, 1, 22)


def _softmax_confs(cfg, calib, blk):
    from hgq.quantizer import QuantizerConfig
    extra = int(os.environ.get("BNHGQ2_SM_EXTRA", "0"))
    smax = float(calib.get(f"__scores_max_{blk}", 4096.0))
    i_exp = int(np.ceil(np.log2(max(smax, 1.0) + 1e-9))) + 1
    f_exp = 10 + extra - i_exp
    mk = lambda k0, i0_, f0_: QuantizerConfig(
        "kif", "datalane", k0=k0, i0=i0_, f0=f0_, round_mode="RND_CONV",
        overflow_mode="SAT", trainable=False, heterogeneous_axis=(),
        homogeneous_axis=None, ic=None, fc=None, ir=None, fr=None)
    mk_t = lambda i0_, f0_: QuantizerConfig(
        "kif", "table", k0=0, i0=i0_, f0=f0_, round_mode="RND_CONV",
        overflow_mode="SAT", trainable=False, ic=None, fc=None, ir=None, fr=None)
    return {
        "exp_iq_conf": mk(0, i_exp, f_exp),
        "inv_iq_conf": mk(0, 4, 8 + extra),
        "exp_oq_conf": mk_t(1, 11 + extra),
        "inv_oq_conf": mk_t(1, 11 + extra),
    }


# --------------------------------------------------------------------------- #
# weighted-layer emission helpers                                             #
# --------------------------------------------------------------------------- #
def _i0(calib, name):
    return int(np.max(calib[name]))


def _pointwise(cfg, calib, name, out_dim, x, reg, use_bias=False):
    """A per-token QDense(out_dim) on a (T, in) tensor → PointwiseConv1D. Pure ±1
    kernel. Registered under `name` for weight porting."""
    from hgq.layers import QDense
    act_bits = cfg["quant"]["act_bits"]
    d = QDense(out_dim, use_bias=use_bias, iq_conf=_act_q(act_bits, _i0(calib, name)),
               kq_conf=_binary_kq(), bq_conf=(_bq_wide() if use_bias else None), name=name)
    reg["dense"][name] = d
    return d(x)


def _dense2d(cfg, calib, name, out_dim, x, reg, use_bias=False):
    """A genuine 2-D QDense (post-GAP head) → nnet::dense."""
    from hgq.layers import QDense
    act_bits = cfg["quant"]["act_bits"]
    d = QDense(out_dim, use_bias=use_bias, iq_conf=_act_q(act_bits, _i0(calib, name)),
               kq_conf=_binary_kq(), bq_conf=(_bq_wide() if use_bias else None), name=name)
    reg["dense"][name] = d
    return d(x)


def _affine(cfg, calib, name, x, reg):
    """β̃·z + b frozen QBatchNormalization (ε=0) on the last axis. iq mirrors build.py's
    affine iq = stream_iq(name)."""
    from hgq.layers import QBatchNormalization
    a = QBatchNormalization(axis=-1, epsilon=0.0, center=True, scale=True,
                            kq_conf=_wq_wide(), bq_conf=_bq_wide(),
                            iq_conf=_stream_iq(cfg, calib, name), name=f"{name}_affine")
    reg["affine"][name] = a
    return a(x)


# --------------------------------------------------------------------------- #
# block emission (shared by monolith-v2 and probe_block)                      #
# --------------------------------------------------------------------------- #
def emit_block(h, cfg, binz, calib, li, reg):
    """Append encoder block `li` to residual-stream tensor h (T,D) and return the new
    residual stream (T,D). Registers weighted layers in reg for porting. The einsum/
    softmax layers are named exactly as build.py so the Latency-override set matches."""
    import keras
    from hgq.layers import QSoftmax
    from hgq.layers.ops.einsum import QEinsum
    from .subln import PSubLN

    A = cfg["arch"]
    T, D, H = A["n_part"], A["d_model"], A["n_heads"]
    E = D // H
    blk = f"bit_block_{li}"

    # ---- attention ----
    ln = PSubLN(name=f"ln_{blk}_attn_qkv")(h)
    q = _pointwise(cfg, calib, f"{blk}_attn_Wq", D, ln, reg)
    k = _pointwise(cfg, calib, f"{blk}_attn_Wk", D, ln, reg)
    v = _pointwise(cfg, calib, f"{blk}_attn_Wv", D, ln, reg)
    q = keras.layers.Reshape((T, H, E), name=f"{blk}_attn_Wq_split")(q)
    k = keras.layers.Reshape((T, H, E), name=f"{blk}_attn_Wk_split")(k)
    v = keras.layers.Reshape((T, H, E), name=f"{blk}_attn_Wv_split")(v)
    scores = QEinsum("bthe,bshe->bhts",
                     iq_confs=[_stream_iq(cfg, calib, f"{blk}_attn_Wq"),
                               _stream_iq(cfg, calib, f"{blk}_attn_Wk")],
                     name=f"{blk}_attn_scores")([q, k])
    sscale = (binz[f"{blk}_attn_Wq"]["beta"] * binz[f"{blk}_attn_Wk"]["beta"]
              / float(np.sqrt(E)))
    attn = QSoftmax(axis=-1, stable=True, input_scaler=float(sscale),
                    name=f"{blk}_attn_softmax", **_softmax_confs(cfg, calib, blk))(scores)
    ctx = QEinsum("bhts,bshe->bthe",
                  iq_confs=[_attn_grid(), _stream_iq(cfg, calib, f"{blk}_attn_Wv")],
                  name=f"{blk}_attn_ctx")([attn, v])
    ctx = keras.layers.Reshape((T, D), name=f"{blk}_attn_ctx_merge")(ctx)   # (T,H,E)->(T,D)
    ctx = PSubLN(name=f"ln_{blk}_attn_Wo")(ctx)                              # norm over D
    wo = _pointwise(cfg, calib, f"{blk}_attn_Wo", D, ctx, reg)
    wo = _affine(cfg, calib, f"{blk}_attn_Wo", wo, reg)
    h = keras.layers.Add(name=f"{blk}_add_attn")([h, wo])

    # ---- FFN ----
    ln = PSubLN(name=f"ln_{blk}_ffn_fc1")(h)
    f1 = _pointwise(cfg, calib, f"{blk}_ffn_fc1", A["ffn_dim"], ln, reg, use_bias=True)
    f1 = keras.layers.ReLU(name=f"{blk}_ffn_act")(f1)
    ln = PSubLN(name=f"ln_{blk}_ffn_fc2")(f1)
    f2 = _pointwise(cfg, calib, f"{blk}_ffn_fc2", D, ln, reg)
    f2 = _affine(cfg, calib, f"{blk}_ffn_fc2", f2, reg)
    h = keras.layers.Add(name=f"{blk}_add_ffn")([h, f2])
    return h


def emit_input_proj(x_in, cfg, binz, calib, pe, reg):
    """SubLN(raw feats) → pointwise ±1 (no bias) → flatten (T,D)→(T·D,) → axis=-1
    QBatchNormalization with (T·D,) params (γ=CSD-2(β), β=(bias_td+PE) flat) → unflatten.
    Reinstates y=β̃·M[t,d]+bias_td[t,d]+PE[t,d]. The flagship's input_proj bias is
    per-position (einsum bias_axes='td', shape (T,D)) so it CANNOT ride a pointwise conv's
    per-channel bias — it (and PE) go into the per-element (T·D) affine. Bit-exact form of
    the affine; a tiny (~3e-4) seed vs the flagship's dense-bias-TABLE quantization remains
    (the affine quantizes the additive directly, the flagship pre-divides by β̃ then rescales
    — see the LEDGER note). Per-shape fidelity is unaffected (block-iso 0.9999987)."""
    import keras
    from hgq.layers import QBatchNormalization
    from .subln import PSubLN
    A = cfg["arch"]
    T, D = A["n_part"], A["d_model"]

    h = PSubLN(name="ln_input_proj")(x_in)
    h = _pointwise(cfg, calib, "input_proj", D, h, reg)                 # ±1, no bias
    h = keras.layers.Reshape((T * D,), name="input_proj_flat")(h)
    aff = QBatchNormalization(axis=-1, epsilon=0.0, center=True, scale=True,
                              kq_conf=_wq_wide(), bq_conf=_bq_wide(),
                              iq_conf=_stream_iq(cfg, calib, "input_proj"),
                              name="input_proj_affine")
    reg["affine_flat"]["input_proj"] = aff
    h = aff(h)
    h = keras.layers.Reshape((T, D), name="input_proj_unflat")(h)
    return h


def _new_reg():
    return {"dense": {}, "affine": {}, "affine_flat": {}}


def build_monolith_v2(cfg, binz, calib, pe, enable_ebops=True):
    """The full flagship model, hybrid emission. Returns (model, reg). Port with
    port_monolith_v2(model, reg, cfg, binz, pe)."""
    import keras
    from hgq.config import LayerConfigScope
    from hgq.layers import QGlobalAveragePooling1D
    from .subln import PSubLN
    A = cfg["arch"]
    T, F, D, L, C = A["n_part"], A["n_feat"], A["d_model"], A["n_layers"], A["n_classes"]
    reg = _new_reg()

    with LayerConfigScope(enable_ebops=enable_ebops, beta0=0.0):
        x_in = keras.Input((T, F), name="input_1")
        h = emit_input_proj(x_in, cfg, binz, calib, pe, reg)
        for li in range(L):
            h = emit_block(h, cfg, binz, calib, li, reg)
        # ---- head ----
        h = QGlobalAveragePooling1D(enable_iq=False, name="gap")(h)
        h = PSubLN(name="ln_head_fc1")(h)
        h = _dense2d(cfg, calib, "head_fc1", D, h, reg, use_bias=True)
        h = keras.layers.ReLU(name="head_act")(h)
        h = PSubLN(name="ln_head_fc2")(h)
        out = _dense2d(cfg, calib, "head_fc2", C, h, reg, use_bias=False)
        out = _affine(cfg, calib, "head_fc2", out, reg)
        model = keras.Model(x_in, out, name=f"{cfg['name']}-monolith-v2")
    port_monolith_v2(model, reg, cfg, binz, pe)
    return model, reg


def build_block_v2(cfg, binz, calib, li, enable_ebops=True):
    """One encoder block (li) as a standalone model: Input(T,D) → block → (T,D).
    Returns (model, reg).

    A lossless identity affine (γ=1, β=0, wide grid) conditions the input so the block's
    residual `Add` enters from a quantizer-bearing layer, not directly from Input — hls4ml
    1.3.0's BitExact pass raises 'unexpected input layer chain' on Input→Merge (in the
    monolith the residual always enters from an internal BatchNorm, so this only bites the
    standalone block). Near-lossless (2^-16 grid) → the block C-sim vs the flagship stands."""
    import keras
    from hgq.config import LayerConfigScope
    from hgq.layers import QBatchNormalization
    A = cfg["arch"]
    T, D = A["n_part"], A["d_model"]
    reg = _new_reg()
    with LayerConfigScope(enable_ebops=enable_ebops, beta0=0.0):
        x_in = keras.Input((T, D), name="block_in")
        cond = QBatchNormalization(axis=-1, epsilon=0.0, center=True, scale=True,
                                   kq_conf=_wq_wide(), bq_conf=_bq_wide(),
                                   iq_conf=_dl(1, 12, 16), name="block_in_cond")
        h = cond(x_in)
        h = emit_block(h, cfg, binz, calib, li, reg)
        model = keras.Model(x_in, h, name=f"{cfg['name']}-block{li}-v2")
    # identity conditioner: y = 1·x + 0
    from .port import _assign_bn
    _assign_bn(cond, np.ones(D, np.float32), np.zeros(D, np.float32))
    port_monolith_v2(model, reg, cfg, binz, pe=None)
    return model, reg


# --------------------------------------------------------------------------- #
# porting                                                                     #
# --------------------------------------------------------------------------- #
def port_monolith_v2(model, reg, cfg, binz, pe):
    """Assign the ±1 sign kernels (reshaped einsum→2-D), bias folds, and CSD-2 affines
    into the built monolith/block. Same fold bookkeeping as port.py / probes_final_v2:
      score/ln folds  → pure ±1, no bias, no affine (β dropped/score-folded)
      bias fold       → pure ±1 + bias=b/β  (fc1, head_fc1)  → ReLU downstream
      explicit fold   → pure ±1, no bias  + affine (γ=CSD-2(β), β=layer bias)
      input_proj      → pure ±1, no bias + (T·D) affine (γ=CSD-2(β), β=(bias_d+PE) flat)
    """
    from .gold import csd2_snap
    from .port import _assign, _assign_bn
    A = cfg["arch"]
    T, D = A["n_part"], A["d_model"]

    # ---- dense/pointwise kernels ----
    for name, layer in reg["dense"].items():
        e = binz[name]
        q = e["q"].astype(np.float32)
        # in/out split is einsum-equation-dependent (probes_final_v2 SHAPES comment):
        # Q/K/V contract the FIRST axis (kernel (D,H,E) 'btd,dhe->bthe' → in=D, out=H·E);
        # everything else contracts all-but-last (kernel (…,out) → in=prod(shape[:-1])).
        if name.endswith(("_attn_Wq", "_attn_Wk", "_attn_Wv")):
            in_dim = int(e["shape"][0])
            out_dim = int(np.prod(e["shape"][1:]))
        else:
            out_dim = int(e["shape"][-1])
            in_dim = int(np.prod(e["shape"])) // out_dim
        q2d = q.reshape(in_dim, out_dim)                      # row-major (h·E+e ordering)
        if e["fold"] == "bias_fold":                          # fc1, head_fc1
            bias = (e["bias"].astype(np.float32) / np.float32(e["beta"]))
            _assign(layer, q2d, bias)
        else:                                                 # input_proj/score/ln/explicit
            _assign(layer, q2d, None)                          # (input_proj bias_td rides affine)

    # ---- last-axis affines (Wo, fc2, head_fc2) ----
    for name, layer in reg["affine"].items():
        e = binz[name]
        width = int(e["shape"][-1])
        gamma = np.full(width, csd2_snap(e["beta"]), dtype=np.float32)
        beta = (np.zeros(width, np.float32) if e["bias"] is None
                else e["bias"].astype(np.float32))
        _assign_bn(layer, gamma, beta)

    # ---- input_proj (T·D) affine: γ=CSD-2(β) const, β=(bias_td + PE) flattened ----
    for name, layer in reg["affine_flat"].items():
        e = binz[name]
        gamma = np.full(T * D, csd2_snap(e["beta"]), dtype=np.float32)
        bias_td = (np.zeros((T, D), np.float32) if e["bias"] is None
                   else e["bias"].astype(np.float32).reshape(T, D))   # per-position (T,D)
        add = bias_td + (pe.astype(np.float32) if pe is not None
                         else np.zeros((T, D), np.float32))            # (T,D)
        _assign_bn(layer, gamma, add.reshape(-1).astype(np.float32))
    return model


# --------------------------------------------------------------------------- #
# hybrid-strategy plumbing                                                    #
# --------------------------------------------------------------------------- #
def latency_layer_names(L):
    """The weightless attention layers that MUST stay Latency (einsums assert it;
    softmax follows the flagship's all-Latency attention path)."""
    names = []
    for li in range(L):
        blk = f"bit_block_{li}"
        names += [f"{blk}_attn_scores", f"{blk}_attn_ctx", f"{blk}_attn_softmax"]
    return names


def _kernel_in_dim(name, shape):
    """Input (contracted) dim of a weighted layer's einsum-native kernel (mirrors the
    porting split): Q/K/V contract the first axis; everything else all-but-last."""
    if name.endswith(("_attn_Wq", "_attn_Wk", "_attn_Wv")):
        return int(shape[0])
    return int(np.prod(shape)) // int(shape[-1])


def per_layer_configs(cfg, binz, rf, latency_names, present=None):
    """LayerName overrides for convert(layer_configs=…): Strategy=Latency on the weightless
    attention layers, and a ReuseFactor cap on every PointwiseConv1D whose in_channels < rf
    (nnet_conv1d_resource asserts RF ≤ in_channels). head_fc1/head_fc2 are post-GAP 2-D Dense
    (nnet::dense_resource handles RF>n_in) so they are NOT capped. `present`: optional set of
    layer names actually in the graph (for probe_block, which has no input_proj) — caps are
    only emitted for present layers."""
    cfgs = {n: {"Strategy": "Latency"} for n in latency_names}
    for name, e in binz.items():
        if name.startswith("_") or name in ("head_fc1", "head_fc2"):
            continue
        if present is not None and name not in present:
            continue
        in_dim = _kernel_in_dim(name, e["shape"])
        if in_dim < rf:                       # pointwise conv: RF ≤ in_channels
            cfgs.setdefault(name, {})["ReuseFactor"] = int(in_dim)
    return cfgs


def widen_conv_dense_accum(hm, min_frac=16):
    """v3 widen_dense_accum generalized to PointwiseConv1D/Conv1D (the per-token
    projections) as well as 2-D Dense (the head). Widening the accum + output fraction
    to ≥min_frac flips the ±1 MAC from a DSP48 to LUTs → 0 DSP (measured DSP↔LUT trade,
    constraints_map 2026-07-09). Bit-exact: downstream re-quantizes to its grid."""
    n = 0
    for nd in hm.get_layers():
        if nd.class_name in ("Dense", "PointwiseConv1D", "Conv1D"):
            prts = []
            a = nd.get_attr("accum_t", None)
            if a is not None:
                prts.append(a.precision)
            prts.append(nd.get_output_variable().type.precision)
            for p in prts:
                frac = int(p.width) - int(p.integer)
                if frac < min_frac:
                    p.width = int(p.integer) + min_frac
            n += 1
    return n
