#!/usr/bin/env python3
"""Build mulder csynth PROBES from the real ported era-2 large checkpoint.

The full D256/L8 model cannot fit a VU13P spatially (~516% LUT composed, prior
art) — per-shape probes are the established methodology (results/hls_resource_table.md
§B′). The HGQ2 path adds two NEW probes prior art could not synthesize:

  subln_<D>        — the SubLN norm alone (range-reduced 1/√ kernel): the norm's
                     true DSP/LUT/latency cost, to compare against the 1,049-DSP
                     LayerNormalization census of the QKeras path.
  attn_core_a<B>   — scores = QK^T (QEinsum) → stable QSoftmax (β_qβ_k/√d in the
                     exp LUT) → ctx = attn·V (QEinsum): the attention core that was
                     EXCLUDED from all previous full-model synthesis (EinsumDense
                     blocker). Real block-0 quantizer calibration + scale.
  bitlinear_<shape>— PSubLN → QDense(binary ±1, CSD-2 β̃) with the real block-0
                     Wo weights at RF=256/Resource: the HGQ2-path cross-check
                     against the QKeras-path per-shape numbers.

Each probe: build → port real weights → local C-sim gate (bit-exact or corr) →
write hls project + tarball for mulder. Usage:
  python probes.py --config configs/era2-large-w1a8.json --probe attn_core
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bnhgq2.config import load_config, cfg_hash, resolve  # noqa: E402
from bnhgq2.extract import extract_checkpoint  # noqa: E402
from bnhgq2.binarize import binarize_checkpoint  # noqa: E402
from bnhgq2.data import load_eval_set, calibration_batch  # noqa: E402
from bnhgq2 import store  # noqa: E402


def _quant(act_bits, i0):
    from hgq.quantizer import QuantizerConfig
    return QuantizerConfig("kif", "datalane", k0=1, i0=int(i0),
                           f0=act_bits - 1 - int(i0),
                           round_mode="RND_CONV", overflow_mode="SAT",
                           trainable=False, ic=None, fc=None, ir=None, fr=None)


def _binary_kq(scaler=None):
    from hgq.quantizer import QuantizerConfig
    kw = {} if scaler is None else {"scaler": float(scaler)}
    return QuantizerConfig("kbi", "weight", k0=1, b0=1, i0=1,
                           round_mode="RND", overflow_mode="SAT_SYM",
                           trainable=False, heterogeneous_axis=(),
                           bc=None, ic=None, br=None, ir=None, **kw)


def probe_subln(cfg, dim=256):
    import keras
    from bnhgq2.subln import PSubLN
    x = keras.Input((dim,), name="inp")
    y = PSubLN(name=f"subln_{dim}")(x)
    m = keras.Model(x, y, name=f"probe_subln_{dim}")
    X = np.random.default_rng(7).normal(0, 3.0, (2048, dim)).astype(np.float32)
    return m, X, {"strategy": "Latency", "rf": 1}


def probe_bitlinear(cfg, binz, calib, layer="bit_block_0_attn_Wo"):
    """v4 factoring: SubLN → quant → QDense(pure ±1, no bias) → QBN(γ=CSD-2(β),
    β=trained bias, ε=0). The in-weight ±β̃ variant measured 256 DSPs at RF=256
    Resource (ROM operands defeat constant folding) — this probe validates the fix."""
    import keras
    from hgq.config import LayerConfigScope
    from hgq.layers import QDense, QBatchNormalization
    from hgq.quantizer import QuantizerConfig
    from bnhgq2.subln import PSubLN
    from bnhgq2.gold import csd2_snap
    from bnhgq2.port import _assign_bn

    e = binz[layer]
    d_in, d_out = e["shape"]
    i0 = int(np.max(calib[layer]))
    wq_wide = QuantizerConfig("kif", "weight", k0=1, i0=2, f0=16,
                              round_mode="RND", overflow_mode="SAT",
                              trainable=False, heterogeneous_axis=(),
                              ic=None, fc=None, ir=None, fr=None)
    bq_wide = QuantizerConfig("kif", "bias", k0=1, i0=8, f0=16,
                              round_mode="RND_CONV", overflow_mode="SAT",
                              trainable=False, heterogeneous_axis=(),
                              ic=None, fc=None, ir=None, fr=None)
    aff_iq = QuantizerConfig("kif", "datalane", k0=1, i0=10,
                             f0=cfg["quant"]["act_bits"] - 1,
                             round_mode="RND_CONV", overflow_mode="SAT",
                             trainable=False, heterogeneous_axis=(),
                             homogeneous_axis=None,
                             ic=None, fc=None, ir=None, fr=None)
    with LayerConfigScope(enable_ebops=True, beta0=0.0):
        x = keras.Input((d_in,), name="inp")
        h = PSubLN(name="subln")(x)
        h = QDense(d_out, use_bias=False,
                   iq_conf=_quant(cfg["quant"]["act_bits"], i0),
                   kq_conf=_binary_kq(None),
                   name=layer)(h)
        h = QBatchNormalization(axis=-1, epsilon=0.0, center=True, scale=True,
                                kq_conf=wq_wide, bq_conf=bq_wide, iq_conf=aff_iq,
                                name=f"{layer}_affine")(h)
        m = keras.Model(x, h, name=f"probe_bitlinear_{d_in}x{d_out}")
    lay = m.get_layer(layer)
    kv = getattr(lay, "_kernel", None)
    if kv is None:
        kv = lay.kernel
    kv.assign(e["q"].astype(np.float32))
    gamma = np.full(d_out, csd2_snap(e["beta"]), dtype=np.float32)
    beta = (e["bias"].astype(np.float32) if e["bias"] is not None
            else np.zeros(d_out, np.float32))
    _assign_bn(m.get_layer(f"{layer}_affine"), gamma, beta)
    X = np.random.default_rng(7).normal(0, 2.0, (2048, d_in)).astype(np.float32)
    # Latency strategy: weights inlined as compile-time constants → Vitis folds
    # ±1 multiplies into add/sub (DSP-free); Resource-mode ROM operands defeated
    # both const folding and the binary product specialization (measured: 256 DSP).
    return m, X, {"strategy": "Latency", "rf": 256}


def probe_attn_core(cfg, binz, calib, blk="bit_block_0"):
    """scores→softmax→ctx with real block-0 scale + calibrated input grids.
    Inputs are the Q,K,V head tensors (B,T,H,E) as produced by the projections."""
    import keras
    from hgq.config import LayerConfigScope
    from hgq.layers import QSoftmax
    from hgq.layers.ops.einsum import QEinsum

    A = cfg["arch"]
    T, H, E = A["n_part"], A["n_heads"], A["d_model"] // A["n_heads"]
    bits = cfg["quant"]["act_bits"]
    # Q/K/V magnitudes: integer sums of ≤D ±1 terms of quantized inputs → i0 ~ 6-9.
    # Use the post-projection scale: |q| ≤ D (pre-β, pure ±1 path) → i0 = ceil(log2 D)+?
    # calibrated empirically by the caller via taps; default generous:
    i_qkv = int(np.ceil(np.log2(A["d_model"]))) + 1
    from hgq.quantizer import QuantizerConfig

    def _dl(k0, i0_, f0_):
        return QuantizerConfig("kif", "datalane", k0=k0, i0=i0_, f0=f0_,
                               round_mode="RND_CONV", overflow_mode="SAT",
                               trainable=False, heterogeneous_axis=(),
                               homogeneous_axis=None,
                               ic=None, fc=None, ir=None, fr=None)

    def _tb(i0_, f0_):
        return QuantizerConfig("kif", "table", k0=0, i0=i0_, f0=f0_,
                               round_mode="RND_CONV", overflow_mode="SAT",
                               trainable=False, ic=None, fc=None, ir=None, fr=None)

    with LayerConfigScope(enable_ebops=True, beta0=0.0):
        qi = keras.Input((T, H, E), name="q_in")
        ki = keras.Input((T, H, E), name="k_in")
        vi = keras.Input((T, H, E), name="v_in")
        iq = lambda: _quant(16, i_qkv)  # wide input grids: the core is the DUT
        scores = QEinsum("bthe,bshe->bhts", iq_confs=[iq(), iq()],
                         name="attn_scores")([qi, ki])
        sscale = (binz[f"{blk}_attn_Wq"]["beta"] * binz[f"{blk}_attn_Wk"]["beta"]
                  / float(np.sqrt(E)))
        # raw |scores| bound for random N(0,12) inputs: ~E * (5σ)^2-ish → 2^14 grid
        attn = QSoftmax(axis=-1, stable=True, input_scaler=float(sscale),
                        exp_iq_conf=_dl(0, 14, -4), inv_iq_conf=_dl(0, 4, 8),
                        exp_oq_conf=_tb(1, 11), inv_oq_conf=_tb(1, 11),
                        name="attn_softmax")(scores)
        ctx = QEinsum("bhts,bshe->bthe", iq_confs=[_dl(0, 1, 22), iq()],
                      name="attn_ctx")([attn, vi])
        m = keras.Model([qi, ki, vi], ctx, name=f"probe_attn_core_a{bits}")
    rng = np.random.default_rng(7)
    # realistic magnitude: ±1-weight sums over 256 quantized-normed features
    Q = rng.normal(0, 12.0, (2048, T, H, E)).astype(np.float32)
    K = rng.normal(0, 12.0, (2048, T, H, E)).astype(np.float32)
    V = rng.normal(0, 12.0, (2048, T, H, E)).astype(np.float32)
    return m, [Q, K, V], {"strategy": "Latency", "rf": 1}


PROBES = {
    "subln": probe_subln,
    "bitlinear": probe_bitlinear,
    "attn_core": probe_attn_core,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--probe", required=True, choices=list(PROBES))
    ap.add_argument("--rf", type=int, default=None)
    ap.add_argument("--layer", default="bit_block_0_attn_Wo",
                    help="checkpoint layer for the bitlinear probe")
    a = ap.parse_args()

    cfg = load_config(a.config)
    h = cfg_hash(cfg)
    ck = extract_checkpoint(resolve(cfg, "checkpoint"))
    binz = binarize_checkpoint(ck["layers"])

    calib_path = os.path.join(store.STORE, "runs", h, "calib.npz")
    calib = {}
    if os.path.exists(calib_path):
        z = np.load(calib_path)
        calib = {k: z[k] for k in z.files}

    if a.probe == "subln":
        model, X, hcfg = probe_subln(cfg)
    elif a.probe == "bitlinear":
        model, X, hcfg = probe_bitlinear(cfg, binz, calib, layer=a.layer)
    else:
        model, X, hcfg = probe_attn_core(cfg, binz, calib)

    from bnhgq2.convert import convert, pack_for_mulder
    from bnhgq2.verify import predict_in_batches
    rf = a.rf or hcfg["rf"]
    tag = a.probe
    if a.probe == "bitlinear" and a.layer != "bit_block_0_attn_Wo":
        tag = f"{a.probe}_{a.layer}"
    out = os.path.join(store.STORE, "runs", h, f"probe_{tag}_rf{rf}")
    if isinstance(X, list):
        keras_ref = np.asarray(model([x[:512] for x in X], training=False))
        csim_X = [x[:512] for x in X]
    else:
        keras_ref = predict_in_batches(model, X[:512])
        csim_X = X[:512]
    hm, report = convert(model, cfg, out, rf=rf, strategy=hcfg["strategy"],
                         csim_X=csim_X, keras_ref=keras_ref)
    tar = pack_for_mulder(out, out + ".tar.gz")
    report["tarball"] = tar
    report["probe"] = tag
    store.write_stage(cfg, h, f"probe_{tag}", report)
    print(json.dumps(report.get("csim", {}), indent=1))
    print(f"[probe] project: {out}\n[probe] tarball: {tar}")


if __name__ == "__main__":
    main()
