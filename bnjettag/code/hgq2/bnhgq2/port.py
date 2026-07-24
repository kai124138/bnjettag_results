"""Port extracted+binarized QKeras weights into the HGQ2 rebuild.

Mapping (checkpoint kernel (in, out) layouts → HGQ2 einsum kernels):
  input_proj  (16,256)   → 'btf,fd->btd'  kernel (16,256); bias table (10,256) =
                            bias[None,:] + PE (the folded positional constant)
  attn_Wq/Wk/Wv (256,256)→ 'btd,dhe->bthe' kernel (256,H,E)  [row-major reshape]
  attn_Wo     (256,256)  → 'bthe,hed->btd' kernel (H,E,256)  [row d_in = h*E+e]
  ffn_fc1     (256,1024) → 'btd,df->btf'   kernel as-is; bias (1024,)
  ffn_fc2     (1024,256) → 'btf,fd->btd'   kernel as-is; bias (256,)
  head_fc1/2             → QDense kernels as-is
Kernels are assigned as the SIGN matrices (±1); the per-tensor β̃ (pow2-snapped)
lives in the weight quantizer's scaler (set at build time from the same binz dict).
"""
from __future__ import annotations

import numpy as np


def pow2_snap(beta: float) -> float:
    return float(2.0 ** np.round(np.log2(beta)))


def betas_pow2(binz: dict) -> dict:
    return {n: pow2_snap(e["beta"]) for n, e in binz.items() if not n.startswith("_")}


def port_weights(model, cfg: dict, binz: dict, pe: np.ndarray):
    A = cfg["arch"]
    H, E = A["n_heads"], A["d_model"] // A["n_heads"]

    def q(name):
        return binz[name]["q"].astype(np.float32)

    def b(name):
        e = binz[name]
        bias = e["bias"].astype(np.float32)
        if e["fold"] == "bias_fold":
            # pure ±1 weights; the next LN kills the missing β scale exactly,
            # provided the bias is rescaled to b/β (see gold.py fold math)
            bias = bias / np.float32(e["beta"])
        return bias

    from .gold import csd2_snap

    n_set = 0
    for layer in model.layers:
        nm = layer.name
        if nm.endswith("_affine"):
            # v4: β̃·z + b as frozen BN (ε=0): γ=CSD-2(β) const, β=trained bias
            # (input_proj's bias+PE live in its dense table → affine β=0),
            # moving stats identity.
            src = nm[:-len("_affine")]
            e = binz[src]
            width = e["shape"][-1]  # output feature dim = last kernel axis (2D
            # QKeras-path kernels: shape[-1]==shape[1]; QAT einsum-native kernels
            # are 3D, e.g. Wo (H,E,D) -> D is the last axis, not shape[1]=E)
            # Norm-free graphs (arch.norm='none', round 8): the ln_killed/bias_fold
            # βs have no LayerNorm to die in — they propagate multiplicatively into
            # this downstream affine (exact by linearity of the ctx einsum and
            # positive-homogeneity of ReLU): γ_Wo carries β_v, γ_fc2 carries β_fc1,
            # γ_head_fc2 carries β_head_fc1. Normed graphs: carry=1, byte-identical.
            carry = 1.0
            if str(cfg["arch"].get("norm", "subln")).lower() == "none":
                if src.endswith("_attn_Wo"):
                    carry = binz[src.replace("_attn_Wo", "_attn_Wv")]["beta"]
                elif src.endswith("_ffn_fc2"):
                    carry = binz[src.replace("_ffn_fc2", "_ffn_fc1")]["beta"]
                elif src == "head_fc2":
                    carry = binz["head_fc1"]["beta"]
            gamma = np.full(width, csd2_snap(e["beta"] * carry), dtype=np.float32)
            beta = (np.zeros(width, np.float32) if src == "input_proj"
                    else e["bias"].astype(np.float32))
            _assign_bn(layer, gamma, beta)
            n_set += 1
            continue
        if nm not in binz or nm.startswith("_"):
            continue  # norm/softmax/einsum/add layers carry no ported weights
        if nm == "input_proj":
            # dense output later scaled by β̃ in the affine → table pre-divided
            bias_table = (b(nm)[None, :] + pe.astype(np.float32)) \
                / np.float32(csd2_snap(binz[nm]["beta"]))
            _assign(layer, q(nm), bias_table)
        elif nm.endswith(("_attn_Wq", "_attn_Wk", "_attn_Wv")):
            _assign(layer, q(nm).reshape(-1, H, E), None)
        elif nm.endswith("_attn_Wo"):
            _assign(layer, q(nm).reshape(H, E, -1), None)  # bias in the affine
        elif nm.endswith("_ffn_fc2") or nm == "head_fc2":
            _assign(layer, q(nm), None)  # bias in the affine
        elif nm.endswith("_ffn_fc1") or nm == "head_fc1":
            _assign(layer, q(nm), b(nm))
        else:
            continue
        n_set += 1
    expected = (3 + 6 * A["n_layers"]) + (2 * A["n_layers"] + 2)  # denses + affines
    if n_set != expected:
        raise RuntimeError(f"ported {n_set} layers, expected {expected}")
    return n_set


def _assign_bn(layer, gamma, beta):
    """Set a frozen QBatchNormalization to y = γ·x + β exactly (ε must be 0).
    Name matching must be exact AND shape-aware: QLayerBase also owns a SCALAR
    variable named 'beta' (the EBOPs loss weight) that must not be touched."""
    want_shape = tuple(gamma.shape)
    found = set()
    for v in layer.variables:
        p = (v.path if hasattr(v, "path") else v.name).split("/")[-1]
        if tuple(v.shape) != want_shape:
            continue
        if p in ("gamma", "bn_gamma"):
            v.assign(gamma); found.add("gamma")
        elif p in ("beta", "bn_beta"):
            v.assign(beta); found.add("beta")
        elif p == "moving_mean":
            v.assign(np.zeros(want_shape, np.float32)); found.add("mean")
        elif p in ("moving_variance", "moving_var"):
            v.assign(np.ones(want_shape, np.float32)); found.add("var")
    missing = {"gamma", "beta", "mean", "var"} - found
    if missing:
        raise RuntimeError(f"{layer.name}: BN variables not found: {missing} "
                           f"(have: {[getattr(v,'path',v.name) for v in layer.variables]})")


def _assign(layer, kernel, bias):
    ker_var = getattr(layer, "_kernel", None)
    if ker_var is None:
        ker_var = layer.kernel
    if tuple(ker_var.shape) != tuple(kernel.shape):
        raise ValueError(f"{layer.name}: kernel shape {tuple(ker_var.shape)} != {kernel.shape}")
    ker_var.assign(kernel)
    if bias is not None:
        bias_var = getattr(layer, "_bias", None)
        if bias_var is None:
            bias_var = layer.bias
        bias_var.assign(bias.reshape(tuple(bias_var.shape)))


def assign_per_channel_ibits(model, calib: dict, act_bits: int):
    """Refinement: push per-channel (last-axis) integer-bit choices into each
    layer's input quantizer (i and f variables). Heterogeneous quantizer vars
    may be per-element shaped (e.g. (T, D)); per-channel values broadcast over
    the leading axes. Returns {layer_name: 'per_channel'|'scalar'} describing
    what was actually applied (verify.py mirrors this in the gold model)."""
    applied = {}
    for layer in model.layers:
        name = layer.name
        if name not in calib:
            continue
        vals = np.asarray(calib[name], dtype=np.float32)
        iq = getattr(layer, "iq", None)
        if iq is None or vals.ndim == 0:
            if iq is not None:
                applied[name] = "scalar"
            continue
        qz = iq.quantizer  # internal FixedPointQuantizerKIF
        try:
            ishape = tuple(qz._i.shape)
        except AttributeError:
            ishape = tuple(qz.i.shape)
        try:
            if len(ishape) >= 1 and ishape[-1] == vals.shape[-1]:
                tiled = np.broadcast_to(vals, ishape).astype(np.float32)
                (qz._i if hasattr(qz, "_i") else qz.i).assign(tiled)
                (qz._f if hasattr(qz, "_f") else qz.f).assign(act_bits - 1 - tiled)
                applied[name] = "per_channel"
            else:
                applied[name] = "scalar"
        except Exception:
            applied[name] = "scalar"
    return applied
