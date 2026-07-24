"""config → HGQ2 model, binary weights PINNED, static activation quantizers.

Design (all API facts verified by execution against hgq2 0.1.9 + hls4ml 1.3.0,
see LEDGER 2026-07-04 "source deep-dive"):

* Weight quantizer  = KBI(k0=1, b0=1, i0=1, RND, SAT_SYM, trainable=False,
  heterogeneous_axis=()) — passes ±1 kernels through bit-identically and reports
  1 bit to EBOPs. β handling is FOLD-AWARE (see build_hgq2_model docstring):
  most β's fold away exactly; the 2L+2 residual contributors carry a CSD-2 β̃
  as the quantizer scaler (2-signed-digit constants: DSP-free, bit-exact).
* Activation quant  = KIF(k0=1, i0, f0, RND, SAT, trainable=False) per site, i0
  from gold-model calibration, f0 = act_bits − 1 − i0.
* Attention         = composition of native hgq layers (QEinsumDense for Q/K/V/O
  with head axes in the einsum equations, QEinsum for the two act×act contractions,
  QSoftmax stable with 1/√d_head folded into its exp LUT) — the exact same IR
  primitives hls4ml's QMultiHeadAttention handler decomposes into. Native
  QMultiHeadAttention itself cannot express this model's SubLN-inside-Wo.
* Norm              = PSubLN custom layer (parameter-free LayerNorm; hls4ml side
  registered by bnhgq2.subln.register_subln — keras LayerNormalization is
  unconvertible in this stack, all three paths verified broken).
* PE + input bias   = folded into input_proj's QEinsumDense bias table
  (bias_axes='td' → a quantized (T,D) additive constant).
* Residual adds     = plain float adds (keras Add) — bit_exact carries exact widths.
* GAP               = QGlobalAveragePooling1D (bit-exact; 1/10 costs a wide
  accumulator — accepted, head-only).

Layer names mirror the QKeras checkpoint names so port.py maps 1:1.
"""
from __future__ import annotations

import numpy as np


def _quant_cfgs(act_bits: int):
    from hgq.quantizer import QuantizerConfig

    def binary_kq(scaler=None):
        kw = {} if scaler is None else {"scaler": float(scaler)}
        return QuantizerConfig(
            "kbi", "weight", k0=1, b0=1, i0=1,
            round_mode="RND", overflow_mode="SAT_SYM",
            trainable=False, heterogeneous_axis=(),
            bc=None, ic=None, br=None, ir=None, **kw,
        )

    def act_q(i0: int):
        # RND_CONV = ties-to-even, matching TF/numpy rounding (gold model + the
        # trained QKeras forward); SAT clips like the gold static quantizer.
        # heterogeneous (per-element) i/f variables — initialized at the scalar
        # i0, refined to per-channel calibrated values by port.assign_per_channel_ibits
        # (per-element widths are native to the hls4ml io_parallel flow).
        f0 = act_bits - 1 - int(i0)
        return QuantizerConfig(
            "kif", "datalane", k0=1, i0=int(i0), f0=f0,
            round_mode="RND_CONV", overflow_mode="SAT",
            trainable=False,
            ic=None, fc=None, ir=None, fr=None,
        )

    return binary_kq, act_q


def build_hgq2_model(cfg: dict, binz: dict, calib: dict, enable_ebops: bool = True):
    """binz: binarize_checkpoint() output (beta + fold class per layer).
    calib: {layer_name: i0 int or per-channel array — arrays are reduced to their
    max for the layer-level i0; per-channel refinement is assigned post-build}.

    β handling (fold-aware, measured in gold experiments E3/E4, LEDGER 2026-07-04):
      score_fold (Wq,Wk)      → pure ±1 weights; exact β_q·β_k/√d_head folded into
                                 QSoftmax's input_scaler (exp-LUT — free, exact)
      ln_killed  (Wv)         → pure ±1; β dropped (next LN is scale-invariant)
      bias_fold  (fc1,head_fc1)→ pure ±1; bias ported as b/β (next LN kills scale)
      explicit   (input_proj, Wo, fc2, head_fc2) → kq scaler = CSD-2(β):
                                 2-signed-digit constants, DSP-free by the Vitis
                                 ≤2-digit rule, bit-exact (finite binary fraction)
    """
    import keras
    from hgq.config import LayerConfigScope
    from hgq.layers import QEinsumDense, QDense, QSoftmax, QGlobalAveragePooling1D
    from hgq.layers.ops.einsum import QEinsum

    from .subln import PSubLN
    from .gold import csd2_snap

    A = cfg["arch"]
    T, F, D = A["n_part"], A["n_feat"], A["d_model"]
    H, L, FFN, C = A["n_heads"], A["n_layers"], A["ffn_dim"], A["n_classes"]
    E = D // H
    binary_kq, act_q = _quant_cfgs(cfg["quant"]["act_bits"])

    def kq(name):
        # v4: ALL denses carry pure ±1 kernels (2-bit runtime operands → no DSP
        # even in Resource-mode ROM; measured 2026-07-04: in-weight ±β̃ inferred
        # 256 real DSPs at RF=256). Explicit-fold sites get their β̃ from a
        # QBatchNormalization affine appended after the dense.
        return binary_kq(None)

    def wq_wide():
        from hgq.quantizer import QuantizerConfig
        return QuantizerConfig("kif", "weight", k0=1, i0=2, f0=16,
                               round_mode="RND", overflow_mode="SAT",
                               trainable=False, heterogeneous_axis=(),
                               ic=None, fc=None, ir=None, fr=None)

    def bq_wide():
        from hgq.quantizer import QuantizerConfig
        return QuantizerConfig("kif", "bias", k0=1, i0=8, f0=16,
                               round_mode="RND_CONV", overflow_mode="SAT",
                               trainable=False, heterogeneous_axis=(),
                               ic=None, fc=None, ir=None, fr=None)

    def affine(name, x):
        """β̃·z + b as a frozen QBatchNormalization (ε=0, var=1 → scale = γ
        exactly = CSD-2 = compile-time shift-add constants, DSP-free)."""
        from hgq.layers import QBatchNormalization
        return QBatchNormalization(
            axis=-1, epsilon=0.0, center=True, scale=True,
            kq_conf=wq_wide(), bq_conf=bq_wide(), iq_conf=stream_iq(name),
            name=f"{name}_affine",
        )(x)

    def i0(name):
        v = calib[name]
        return int(np.max(v))

    def _dl(k0, i0_, f0_):
        from hgq.quantizer import QuantizerConfig
        return QuantizerConfig("kif", "datalane", k0=k0, i0=int(i0_), f0=int(f0_),
                               round_mode="RND_CONV", overflow_mode="SAT",
                               trainable=False, heterogeneous_axis=(),
                               homogeneous_axis=None,
                               ic=None, fc=None, ir=None, fr=None)

    def stream_iq(name):
        """Exact-passthrough grid for an internal integer-grid stream (Q/K/V):
        values are multiples of 2^-(act_bits-1) with |v| ≤ calibrated max —
        the frozen SAT grid contains them exactly, so quantization is identity.
        (Workaround for hgq 0.1.9's broken enable_iq=False on multi-input layers.)"""
        import os
        margin = 1 + int(os.environ.get("BNHGQ2_STREAM_MARGIN", "0"))  # ablation knob
        smax = float(calib.get(f"__stream_max_{name}", 512.0))
        i_ = int(np.ceil(np.log2(max(smax, 1.0) + 1e-9))) + margin  # SAT guards the tail
        act_bits = cfg["quant"]["act_bits"]
        return _dl(1, i_, act_bits - 1)

    def attn_grid():
        """softmax outputs: products of two ≤11-fraction-bit table values —
        f0=22 holds them exactly."""
        return _dl(0, 1, 22)

    def softmax_confs(blk):
        """Frozen SAT table grids for QSoftmax (defaults are WRAP+uncalibrated —
        the silent-garbage trap found in v1). exp input = max−scores ∈ [0, 2·max];
        grid ~10 total bits with i from calibrated |scores| max (f may be negative:
        coarse-but-covering, score GAPS are O(σ) ≫ step). inv input = Σexp ∈ [1, T]."""
        import os
        from hgq.quantizer import QuantizerConfig
        extra = int(os.environ.get("BNHGQ2_SM_EXTRA", "0"))  # ablation knob
        smax = float(calib.get(f"__scores_max_{blk}", 4096.0))
        i_exp = int(np.ceil(np.log2(max(smax, 1.0) + 1e-9))) + 1  # cover 2·max
        f_exp = 10 + extra - i_exp
        mk = lambda k0, i0_, f0_: QuantizerConfig(
            "kif", "datalane", k0=k0, i0=i0_, f0=f0_,
            round_mode="RND_CONV", overflow_mode="SAT",
            trainable=False, heterogeneous_axis=(), homogeneous_axis=None,
            ic=None, fc=None, ir=None, fr=None)
        mk_t = lambda i0_, f0_: QuantizerConfig(
            "kif", "table", k0=0, i0=i0_, f0=f0_,
            round_mode="RND_CONV", overflow_mode="SAT", trainable=False,
            ic=None, fc=None, ir=None, fr=None)
        return {
            "exp_iq_conf": mk(0, i_exp, f_exp),
            "inv_iq_conf": mk(0, 4, 8 + extra),   # Σexp ∈ [1, T]; 4096-entry inv table
            "exp_oq_conf": mk_t(1, 11 + extra),   # exp values ∈ (0, 1]
            "inv_oq_conf": mk_t(1, 11 + extra),   # 1/Σ ∈ (0, 1]
        }

    # Norm placement: 'subln' (default -> PSubLN, existing behavior) | 'none' (identity
    # passthrough: no layer, no params, no DSP -- the norm-ablation science variant). Guarded
    # on arch.norm so a config without norm=='none' emits a byte/hash-identical export graph.
    norm_kind = str(A.get("norm", "subln")).lower()
    if norm_kind not in ("subln", "none"):
        raise ValueError(f"{cfg['name']}: unsupported arch.norm={norm_kind!r} "
                         "(export stack supports 'subln' or 'none')")

    def norm(name, flatten_axes=1):
        if norm_kind == "none":
            return lambda x: x  # identity passthrough: parameter-free, DSP-free
        return PSubLN(flatten_axes=flatten_axes, name=name)

    with LayerConfigScope(enable_ebops=enable_ebops, beta0=0.0):
        x_in = keras.Input((T, F), name="input_1")

        # ---- input projection: LN -> quant -> ±β̃ einsum-dense with (T,D) bias ----
        h = norm("ln_input_proj")(x_in)
        h = QEinsumDense(
            "btf,fd->btd", output_shape=(T, D), bias_axes="td",
            iq_conf=act_q(i0("input_proj")), kq_conf=kq("input_proj"),
            bq_conf=bq_wide(),
            name="input_proj",
        )(h)  # bias table carries (input_proj bias + folded PE) / beta_csd2
        h = affine("input_proj", h)

        for li in range(L):
            blk = f"bit_block_{li}"
            # ---- attention ----
            ln = norm(f"ln_{blk}_attn_qkv")(h)
            q = QEinsumDense("btd,dhe->bthe", output_shape=(T, H, E),
                             iq_conf=act_q(i0(f"{blk}_attn_Wq")),
                             kq_conf=kq(f"{blk}_attn_Wq"),
                             name=f"{blk}_attn_Wq")(ln)
            k = QEinsumDense("btd,dhe->bthe", output_shape=(T, H, E),
                             iq_conf=act_q(i0(f"{blk}_attn_Wk")),
                             kq_conf=kq(f"{blk}_attn_Wk"),
                             name=f"{blk}_attn_Wk")(ln)
            v = QEinsumDense("btd,dhe->bthe", output_shape=(T, H, E),
                             iq_conf=act_q(i0(f"{blk}_attn_Wv")),
                             kq_conf=kq(f"{blk}_attn_Wv"),
                             name=f"{blk}_attn_Wv")(ln)
            # act×act einsums: exact-passthrough input grids (the trained model
            # holds scores/ctx in float; real quantization happens at the softmax
            # table grids and the next layer's iq)
            scores = QEinsum("bthe,bshe->bhts",
                             iq_confs=[stream_iq(f"{blk}_attn_Wq"),
                                       stream_iq(f"{blk}_attn_Wk")],
                             name=f"{blk}_attn_scores")([q, k])
            sscale = (binz[f"{blk}_attn_Wq"]["beta"] * binz[f"{blk}_attn_Wk"]["beta"]
                      / float(np.sqrt(E)))
            attn = QSoftmax(axis=-1, stable=True, input_scaler=float(sscale),
                            name=f"{blk}_attn_softmax", **softmax_confs(blk))(scores)
            ctx = QEinsum("bhts,bshe->bthe",
                          iq_confs=[attn_grid(), stream_iq(f"{blk}_attn_Wv")],
                          name=f"{blk}_attn_ctx")([attn, v])
            ctx = norm(f"ln_{blk}_attn_Wo", flatten_axes=2)(ctx)
            wo = QEinsumDense("bthe,hed->btd", output_shape=(T, D),
                              iq_conf=act_q(i0(f"{blk}_attn_Wo")),
                              kq_conf=kq(f"{blk}_attn_Wo"),
                              name=f"{blk}_attn_Wo")(ctx)
            wo = affine(f"{blk}_attn_Wo", wo)
            h = keras.layers.Add(name=f"{blk}_add_attn")([h, wo])

            # ---- FFN ----
            ln = norm(f"ln_{blk}_ffn_fc1")(h)
            f1 = QEinsumDense("btd,df->btf", output_shape=(T, FFN), bias_axes="f",
                              iq_conf=act_q(i0(f"{blk}_ffn_fc1")),
                              kq_conf=kq(f"{blk}_ffn_fc1"), bq_conf=bq_wide(),
                              name=f"{blk}_ffn_fc1")(ln)
            f1 = keras.layers.ReLU(name=f"{blk}_ffn_act")(f1)
            ln = norm(f"ln_{blk}_ffn_fc2")(f1)
            f2 = QEinsumDense("btf,fd->btd", output_shape=(T, D),
                              iq_conf=act_q(i0(f"{blk}_ffn_fc2")),
                              kq_conf=kq(f"{blk}_ffn_fc2"),
                              name=f"{blk}_ffn_fc2")(ln)
            f2 = affine(f"{blk}_ffn_fc2", f2)
            h = keras.layers.Add(name=f"{blk}_add_ffn")([h, f2])

        # ---- head ----
        h = QGlobalAveragePooling1D(enable_iq=False, name="gap")(h)
        h = norm("ln_head_fc1")(h)
        h = QDense(D, use_bias=True, bq_conf=bq_wide(),
                   iq_conf=act_q(i0("head_fc1")), kq_conf=kq("head_fc1"),
                   name="head_fc1")(h)
        h = keras.layers.ReLU(name="head_act")(h)
        h = norm("ln_head_fc2")(h)
        out = QDense(C, use_bias=False,
                     iq_conf=act_q(i0("head_fc2")), kq_conf=kq("head_fc2"),
                     name="head_fc2")(h)
        out = affine("head_fc2", out)

        model = keras.Model(x_in, out, name=cfg["name"])
    return model
