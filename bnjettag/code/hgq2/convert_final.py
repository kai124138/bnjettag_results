#!/usr/bin/env python3
"""FINAL-campaign hls4ml conversion driver (decisions.md 2026-07-07).

Trained QAT checkpoint  ->  hardware-faithful HGQ2 export  ->  C-sim fidelity gate
->  mulder-ready Vitis project.  The 15 real jobs train natively in HGQ2 (qat.py):
the trained model IS (essentially) the hardware model, so the §6' rebuild Δ collapses
to a re-factoring step, not a re-approximation.  This driver performs that factoring
and proves it is faithful.

Chain (per variant/seed):
  1. fetch     model_best.keras + train_meta.json from W&B (bnjettag-final) or local.
  2. load      the QAT model (BitQEinsumDense/BitQDense + PSubLN + QSoftmax/QEinsum).
  3. export    -> the build.py hardware graph: latent kernels re-binarized with the
               SAME absmean math (binarize.py == qat.bitnet_binary_ste, exact), pure
               ±1 datapath + fold-aware β (score fold / ln-kill / bias fold + CSD-2
               affines at the 2L+2 residual sites), STATIC act grids READ BACK from
               the trained QAT quantizers (NOT re-calibrated).
  4. GATE 1    export vs QAT predictions on real jets -> corr >= 0.9999 (numerically
               near-identical; CSD-2 β is the only intended departure, characterized
               against an exact-β build).
  5. convert   through bnhgq2.convert (hls4ml 1.3.0, Vitis, VU13P) + GATE 2 (C-sim):
               hls4ml vs the exported HGQ2 model on real jets -> corr >= 0.997.
  6. EBOPs     native HGQ2 EBOPs for the exported model.
  7. store     results/final/runs/<hash>/<variant>-s<seed>[/tag]/: export_verify.json,
               csim_verify.json, ebops.json, convert.json + the hls project + tarball.

Weight quantizer / act policy / β folding are the PROVEN rebuild path (RESEARCH.md §6',
constraints_map.md).  This driver never modifies qat.py / train.py / build.py semantics.

Usage:
  export WANDB_API_KEY=$(cat bnjettag/wandb-api-key.txt)     # never echoed
  python convert_final.py --variant w1a8 --seed 1            # real run final-w1a8-s1
  python convert_final.py --variant w1a8 --seed 1 --wandb-run final-smoke --tag smoke
  python convert_final.py --checkpoint /path/model_best.keras --variant w1a8 --seed 1
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from bnhgq2.config import load_config, cfg_hash, PROJECT_ROOT  # noqa: E402

ENTITY = "kayamaguchi-uc-san-diego"
PROJECT = "bnjettag-final"
BINARY_VARIANTS = ("w1a8", "w1a6", "w1a4")


# --------------------------------------------------------------------------- #
# (1) fetch                                                                    #
# --------------------------------------------------------------------------- #
def fetch_checkpoint(variant: str, seed: int, wandb_run: str, dest_dir: str,
                     local_checkpoint: str | None = None,
                     subdir: str | None = None) -> tuple[str, str | None]:
    """Return (model_best.keras path, train_meta.json path|None).

    Local path wins; otherwise pull `<subdir>/model_best.keras`
    (+ train_meta.json) from the W&B run `wandb_run` in bnjettag-final.
    `subdir` defaults to `<variant>-s<seed>` (the FINAL-campaign layout); pass it
    explicitly for other layouts (e.g. r7's `small-w1a8-s2`).
    WANDB_API_KEY must be in the environment (never printed here)."""
    if local_checkpoint:
        kp = os.path.abspath(local_checkpoint)
        if not os.path.isfile(kp):
            raise FileNotFoundError(kp)
        mp = os.path.join(os.path.dirname(kp), "train_meta.json")
        return kp, (mp if os.path.isfile(mp) else None)

    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("WANDB_API_KEY not set (export WANDB_API_KEY=$(cat "
                         "bnjettag/wandb-api-key.txt); never echo it)")
    import wandb

    subdir = subdir or f"{variant}-s{seed}"
    api = wandb.Api(timeout=180)
    runs = [r for r in api.runs(f"{ENTITY}/{PROJECT}") if r.name == wandb_run]
    if len(runs) != 1:
        raise SystemExit(f"{wandb_run}: {len(runs)} runs match in {ENTITY}/{PROJECT} "
                         "(need exactly 1)")
    run = runs[0]
    os.makedirs(dest_dir, exist_ok=True)
    got = {}
    for f in run.files():
        if f.name in (f"{subdir}/model_best.keras", f"{subdir}/train_meta.json"):
            f.download(root=dest_dir, replace=True)
            got[os.path.basename(f.name)] = os.path.join(dest_dir, f.name)
            print(f"[fetch] {f.name} ({f.size/1e6:.2f} MB)", flush=True)
    if "model_best.keras" not in got:
        raise SystemExit(f"{wandb_run}: no {subdir}/model_best.keras attached")
    return got["model_best.keras"], got.get("train_meta.json")


# --------------------------------------------------------------------------- #
# (2) load                                                                     #
# --------------------------------------------------------------------------- #
def patch_resource_einsum_check():
    """hls4ml 1.3.0 Vitis `ValidateResourceStrategy` matches QEinsumDense via the
    'Dense' substring of 'EinsumDense', but `get_layer_mult_size` returns n_in=None for
    einsum layers, so its `rf > n_in` QoR-warning check crashes the whole conversion
    (never hit by the rebuild: its probes used Latency or plain QDense).  The pass ONLY
    prints a urem-core QoR note, so we guard it against None n_in.  Idempotent."""
    from hls4ml.backends.vitis.passes.feature_check import ValidateResourceStrategy
    if getattr(ValidateResourceStrategy, "_bnhgq2_guarded", False):
        return
    orig = ValidateResourceStrategy.transform

    def transform(self, model, node):
        try:
            n_in, _ = model.config.backend.get_layer_mult_size(node)
        except Exception:
            return
        if n_in is None:
            return  # EinsumDense: no scalar n_in -> skip the (warning-only) check
        return orig(self, model, node)

    ValidateResourceStrategy.transform = transform
    ValidateResourceStrategy._bnhgq2_guarded = True


def fix_relu_saturation(hm):
    """GATE-2 repair (bit-exact). `keras.layers.ReLU` is parsed by hls4ml 1.3.0 as
    `ParametrizedActivation('thresholdedrelu')` whose BitExact-assigned output precision is
    UNSIGNED with **WRAP** saturation. A negative input then wraps to ~2^i instead of clamping
    to 0 — on the wide binary FFN contraction (fc1 spans ±O(2^10)) this makes every negative
    element (~half of them) explode to ~1024 (C-sim max|Δ| = 1023.999, corr −0.06). Setting the
    output saturation mode to SAT clamps negatives to the unsigned minimum (0), reproducing keras
    ReLU bit-exactly (positives are in-range, frac bits preserved). Scoped to relu-family
    activations only. Returns the count patched. See constraints_map.md 2026-07-07 addendum."""
    from hls4ml.model.types import SaturationMode
    n = 0
    for l in hm.get_layers():
        act = l.attributes.get("activation") if getattr(l, "attributes", None) else None
        if act and "relu" in str(act).lower():
            p = l.get_output_variable().type.precision
            if getattr(p, "saturation_mode", None) != SaturationMode.SAT:
                p.saturation_mode = SaturationMode.SAT
                n += 1
    return n


def widen_weighted_accum(hm, min_frac: int = 16):
    """Widen every WEIGHTED-layer accumulator (and its output precision) so the fractional
    width is >= min_frac, keeping the integer range unchanged. This flips Vitis's ±1-MAC
    mapping from DSP48 to LUT -> 0 DSP on the binary denses (the measured DSP<->LUT trade,
    constraints_map.md 2026-07-09; the fc1 0-DSP point is ap_fixed<27,11> = 16 frac). It is
    the WHOLE-MODEL generalization of probes_final_v3.widen_dense_accum: that helper matched
    only class_name=='Dense', but in the native monolith the binary projections/FFN are
    EinsumDense — both register TypeAttribute('accum'), so get_attr('accum_t') works for both.

    Bit-exactness: the ±1-weight MAC over a b-frac quantized activation is an integer multiple
    of 2^-b (adds create no new fraction), and every weighted layer here has activation frac
    <= act_bits-1 < 16, so a >=16-frac accum represents the exact result; the downstream grid
    re-quantizes to its own precision either way. So widening only makes the accum MORE precise
    -> C-sim unchanged (the gate is the arbiter). The weightless act×act QEinsum score/ctx cores
    (class_name=='Einsum') are deliberately NOT touched: their multiplies are the structural
    attention DSPs, expected and precision-independent. Mutates the existing precision objects
    (they carry definition_cpp; do NOT replace). Returns the count of layers widened."""
    n = 0
    for nd in hm.get_layers():
        if nd.class_name not in ("Dense", "EinsumDense"):
            continue
        precs = []
        at = nd.get_attr("accum_t")
        if at is not None and getattr(at, "precision", None) is not None:
            precs.append(at.precision)
        try:
            precs.append(nd.get_output_variable().type.precision)
        except Exception:
            pass
        widened = False
        for p in precs:
            if p is None or not hasattr(p, "width") or not hasattr(p, "integer"):
                continue
            frac = int(p.width) - int(p.integer)
            if frac < min_frac:
                p.width = int(p.integer) + min_frac  # keep integer range, widen frac
                widened = True
        if widened:
            n += 1
    return n


def audit_weighted_accums(out_dir: str, cfg: dict, min_frac: int = 16):
    """Read the GENERATED firmware/defines.h and report every WEIGHTED-layer accumulator
    typedef (ap_fixed<W,I>), its fractional width, and whether it clears the min_frac wall.
    Evidence for the DSP-0 claim: 'no narrow-frac accumulator remains on any weighted layer'.
    Returns {name: {typedef, width, integer, frac, wide}} + an 'all_wide' bool + the offenders."""
    import re
    from bnhgq2.extract import layer_names
    weighted = set(layer_names(cfg["arch"]["n_layers"]))  # input_proj/Q/K/V/O/fc1/fc2/head_*
    defines = os.path.join(out_dir, "firmware", "defines.h")
    pat = re.compile(r"typedef\s+ap_u?fixed<(\d+),(\d+)[^>]*>\s+([A-Za-z0-9_]+)_accum_t;")
    found = {}
    with open(defines) as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            w, i, name = int(m.group(1)), int(m.group(2)), m.group(3)
            if name in weighted:
                found[name] = {"typedef": line.strip().replace("typedef ", ""),
                               "width": w, "integer": i, "frac": w - i,
                               "wide": (w - i) >= min_frac}
    offenders = sorted(n for n, v in found.items() if not v["wide"])
    return {"min_frac": min_frac, "n_weighted_layers": len(weighted),
            "n_accums_found": len(found), "per_layer": found,
            "narrow_frac_offenders": offenders, "all_wide": len(offenders) == 0}


def load_qat_model(keras_path: str):
    """Fresh-process reload recipe (train.py interface contract)."""
    from bnhgq2.compat import apply_keras_compat
    from bnhgq2.subln import register_subln
    import bnhgq2.qat  # noqa: F401 — registers BitQEinsumDense/BitQDense/AddPositional
    import keras

    apply_keras_compat()
    register_subln()
    return keras.models.load_model(keras_path)


# --------------------------------------------------------------------------- #
# (3) export: QAT model  ->  build.py hardware graph                           #
# --------------------------------------------------------------------------- #
def _to_np(x):
    from keras import ops
    return np.asarray(ops.convert_to_numpy(x))


def _read_kernel(layer):
    kv = getattr(layer, "_kernel", None)
    if kv is None:
        kv = layer.kernel
    return _to_np(kv)


def _read_bias(layer):
    bv = getattr(layer, "_bias", None)
    if bv is None:
        bv = getattr(layer, "bias", None)
    return None if bv is None else _to_np(bv)


def extract_qat_layers(model, names):
    """{name: {'kernel': latent fp32, 'bias': latent|None}} straight off the QAT
    bit-layers — the input to binarize_checkpoint (same absmean math the forward used)."""
    by = {ly.name: ly for ly in model.layers}
    out = {}
    for n in names:
        ly = by[n]
        out[n] = {"kernel": _read_kernel(ly), "bias": _read_bias(ly)}
    return out


def read_qat_act_ibits(model, names):
    """Effective per-tensor integer-bit count of each bit-layer's INPUT quantizer, read
    back from the trained/frozen QAT grid (NOT re-calibrated).

    Uses the canonical `quantizer.kif` — the SAME (k, i, f) hls4ml reads via
    `extract_fixed_quantizer_config` — so the export grid == the converted grid == the QAT
    forward grid, bit-exactly. This is REQUIRED for the trainable-scale KBI act quantizers
    (`qat._trainable_act`, act_calib='trainable', e.g. final-w1a4): there the raw `_i` is a
    CONTINUOUS trained latent (e.g. 1.49) and only `.kif` gives the hard rounded integer
    bits actually used at inference/export. Backward-compatible with the frozen-KIF grids
    (act_calib='frozen'): kif.i == _i there. Verified 51/51 layers on final-w1a4-s1
    (round(raw _i) matched kif.i, all widths == act_bits). The exported static grid is
    SAT<act_bits, 1+i>, f = act_bits-1-i (build.py.act_q reconstructs it from i)."""
    by = {ly.name: ly for ly in model.layers}
    calib = {}
    for n in names:
        _, i, _ = by[n].iq.quantizer.kif  # KBI-safe: hard effective integer bits
        calib[n] = int(round(float(np.max(_to_np(i)))))
    return calib


def qat_stream_ranges(model, cfg, binz, X):
    """Measure the internal stream/score maxima from the QAT model on real jets and
    convert them into the raw (pure-±1) magnitudes the export graph will see.  Only
    the softmax score range needs to be accurate (it sizes the exp table); stream
    grids keep f=act_bits-1 and merely widen their integer range, so a generous i is
    lossless -> those get a 2x safety margin.  Returns extra calib keys for build.py."""
    import keras
    from bnhgq2.gold import csd2_snap

    A = cfg["arch"]
    E = A["d_model"] // A["n_heads"]
    L = A["n_layers"]

    taps = {}
    by = {ly.name: ly for ly in model.layers}
    for li in range(L):
        blk = f"bit_block_{li}"
        taps[f"{blk}_attn_scores"] = by[f"{blk}_attn_scores"].output
        for w in ("Wq", "Wk", "Wv"):
            taps[f"{blk}_attn_{w}"] = by[f"{blk}_attn_{w}"].output
        taps[f"{blk}_attn_Wo"] = by[f"{blk}_attn_Wo"].output
        taps[f"{blk}_ffn_fc2"] = by[f"{blk}_ffn_fc2"].output
    taps["input_proj"] = by["input_proj"].output
    taps["head_fc2"] = by["head_fc2"].output

    probe = keras.Model(model.inputs, taps)
    vals = probe.predict(X, batch_size=512, verbose=0)
    amax = {k: float(np.abs(np.asarray(v)).max()) for k, v in vals.items()}

    calib = {}
    for li in range(L):
        blk = f"bit_block_{li}"
        bq = binz[f"{blk}_attn_Wq"]["beta"]
        bk = binz[f"{blk}_attn_Wk"]["beta"]
        # softmax exp table sees the RAW ±1-contraction score (β_q·β_k folded out
        # into input_scaler) -> accurate range:
        calib[f"__scores_max_{blk}"] = amax[f"{blk}_attn_scores"] / (bq * bk)
        # Q/K/V exact-passthrough stream grids (f fixed) -> generous is lossless:
        for w in ("Wq", "Wk", "Wv"):
            beta = binz[f"{blk}_attn_{w}"]["beta"]
            calib[f"__stream_max_{blk}_attn_{w}"] = 2.0 * amax[f"{blk}_attn_{w}"] / beta

    for name in ("input_proj", "head_fc2", *[f"bit_block_{li}_attn_Wo" for li in range(L)],
                 *[f"bit_block_{li}_ffn_fc2" for li in range(L)]):
        e = binz[name]
        bmax = float(np.abs(e["bias"]).max()) if e["bias"] is not None else 0.0
        calib[f"__stream_max_{name}"] = 2.0 * (amax[name] + bmax) / csd2_snap(e["beta"])
    return calib


def _norm_free_carry_sites(binz):
    """The norm-free (arch.norm='none') export sites whose INPUT stream is 1/β-inflated by
    the β-carry fold (port.py): ctx→Wo, relu(fc1)→fc2, relu(head_fc1)→head_fc2. These are the
    'explicit'-fold sites that RECEIVE a downstream β-carry (γ = β·β_upstream, carry≠1);
    input_proj is 'explicit' too but its input is the raw model input (carry=1), so it is
    EXCLUDED. Returns the ordered layer-name list (2L+1 sites)."""
    return [n for n, e in binz.items()
            if not n.startswith("_") and e.get("fold") == "explicit" and n != "input_proj"]


def _assign_iq_grid(iq, i_val, f_val):
    """Set a single-input datalane quantizer's (i, f) to scalars broadcast over its
    (possibly per-element) grid vars. Mirrors assign_per_channel_ibits/match_attention_to_qat."""
    qz = iq.quantizer
    iv = qz._i if hasattr(qz, "_i") else qz.i
    fv = qz._f if hasattr(qz, "_f") else getattr(qz, "f", None)
    iv.assign(np.full(iv.shape, float(i_val), dtype="float32"))
    if fv is not None:
        fv.assign(np.full(fv.shape, float(f_val), dtype="float32"))


def calibrate_norm_free_grids(export_model, cfg, binz, X, margin=1):
    """Two-pass per-site sizing of the β-carry-inflated internal-stream input grids for
    NORM-FREE exports (arch.norm=='none'). A NO-OP that returns {} for every normed graph, so
    those exports stay byte/hash-identical (guarded on arch.norm).

    Why (round-8 stdnn flagship): without LayerNorm the residual stream grows block-over-block
    and the β-carry fold leaves the ctx→Wo / relu(fc1)→fc2 / relu(head_fc1)→head_fc2 inputs
    scaled by 1/β (β dropped upstream, restored only in the downstream affine). Sizing those
    grids from the QAT sites' i-bits (read_qat_act_ibits) saturates them catastrophically —
    e.g. a ctx of |max|≈54 into a SAT<8,1+2>=±4 grid — the dominant GATE-1 collapse (0.754).
    The residual/normalized dense inputs (input_proj, Q/K/V, fc1, head_fc1) are NOT inflated
    (they reproduce the QAT quantization at act_bits bit-for-bit), so they are left untouched.

    Two passes on the real calibration jets X:
      1. widen every carry-site grid permissively (i=WIDE, f=act_bits-1) so NONE clips — a clip
         anywhere corrupts the downstream residual and mis-measures later sites;
      2. tap the true pre-quant input max at each carry site on this non-clipping graph and set
         i = ceil(log2(max)) + margin with f = act_bits-1 KEPT (the exact-passthrough precedent
         of build.py.stream_iq): total width WIDENS on these internal streams rather than
         stealing fractional resolution.
    Returns {site: {max, i, f, width}} (empty for normed graphs)."""
    if str(cfg["arch"].get("norm", "subln")).lower() != "none":
        return {}
    import keras
    ab = int(cfg["quant"]["act_bits"])
    sites = _norm_free_carry_sites(binz)
    by = {ly.name: ly for ly in export_model.layers}
    WIDE_I = 24  # covers any residual magnitude; measurement is pre-quant, so only the
    for n in sites:  # DOWNSTREAM (non-clipping) effect of these grids matters for pass 1
        _assign_iq_grid(by[n].iq, WIDE_I, ab - 1)
    taps = {n: by[n].input for n in sites}
    outs = keras.Model(export_model.inputs, taps).predict(X, batch_size=512, verbose=0)
    chosen = {}
    for n in sites:
        mx = float(np.abs(np.asarray(outs[n])).max())
        i_ = int(np.ceil(np.log2(max(mx, 1.0) + 1e-9))) + int(margin)
        _assign_iq_grid(by[n].iq, i_, ab - 1)
        chosen[n] = {"max": mx, "i": i_, "f": ab - 1, "width": 1 + i_ + (ab - 1)}
    return chosen


def build_export(cfg, binz, calib, pe, beta_mode="csd2",
                 qat_model=None, attn_grids="build_default", calib_jets=None):
    """Build the hardware-faithful HGQ2 export from pre-computed QAT-sourced pieces.

    Reuses the PROVEN build.py graph (pure ±1 datapath, PSubLN, QEinsum/QSoftmax
    attention, 2L+2 CSD-2 affines) and port.py, sourcing everything from the QAT model:
      * binz  = binarize_checkpoint(QAT latent kernels)   (== qat forward binarization)
      * calib = QAT act i-bits (read back)  + stream/score ranges (measured)
      * pe    = the trained AddPositional table (folds into input_proj's bias)
    beta_mode='csd2'  -> the DSP-free deployable point (build.py default).
    beta_mode='exact' -> β kept exact in the affines (fidelity reference; may cost
                         affine DSPs on mulder — used to characterize the CSD-2 Δ)."""
    from bnhgq2.build import build_hgq2_model
    from bnhgq2.port import port_weights, assign_per_channel_ibits

    model = build_hgq2_model(cfg, binz, calib)
    port_weights(model, cfg, binz, pe)
    assign_per_channel_ibits(model, calib, cfg["quant"]["act_bits"])
    if beta_mode == "exact":
        _apply_exact_beta(model, binz, pe, cfg)
    if attn_grids == "copy_qat":
        if qat_model is None:
            raise ValueError("attn_grids='copy_qat' needs qat_model")
        match_attention_to_qat(model, qat_model, binz, cfg)  # scaler=1/β (keras-only, see fn)
    # Norm-free (arch.norm='none') ONLY: re-size the β-carry-inflated internal-stream input
    # grids from measured export-side ranges (two-pass). NO-OP + returns {} for normed graphs,
    # so every existing normed export stays byte/hash-identical. Runs last so it measures on
    # the fully-ported/affine-set graph. calib_jets=None (default) preserves prior behaviour.
    model._bnhgq2_norm_free_grids = (
        calibrate_norm_free_grids(model, cfg, binz, calib_jets)
        if calib_jets is not None else {})
    return model


def qat_binz_pe(qat_model, cfg):
    """binz (== the QAT forward binarization) + the folded PE table, from the QAT model."""
    from bnhgq2.binarize import binarize_checkpoint
    from bnhgq2.extract import layer_names
    names = layer_names(cfg["arch"]["n_layers"])
    binz = binarize_checkpoint(extract_qat_layers(qat_model, names))
    zeros = binz["_summary"]["total_sign_zeros"]
    if zeros:
        raise SystemExit(f"STOP: {zeros} sign-zeros in binarization — binary "
                         "{-1,+1} claim would be void")
    pe = _to_np(qat_model.get_layer("pos_enc").pos).reshape(
        cfg["arch"]["n_part"], cfg["arch"]["d_model"])
    return binz, pe, names


def match_attention_to_qat(export_model, qat_model, binz, cfg):
    """`attn_grids='copy_qat'`: reproduce the TRAINED attention quantization exactly.

    The β-drop (Wq/Wk/Wv → pure ±1, β folded into the softmax input_scaler) rescales the
    attention streams the export sees, so build.py's independently-sized softmax/stream
    grids do NOT reproduce the QAT model's attention quantization — the dominant gate-1
    residual.  This copies the QAT scores/softmax/ctx grids with **scaler=1/β**: the
    quantizer computes base(x/scaler)*scaler, so base=QAT-grid, scaler=1/β reproduces the
    QAT quantization of β·x exactly under the β-fold. Lifts gate-1 to the ~0.99 ceiling.

    HARDWARE CAVEAT (MEASURED, not assumed): hls4ml's HGQ2 frontend
    `extract_fixed_quantizer_config` (keras_v3/hgq2/_base.py) reads ONLY the internal
    quantizer's k/i/f + SAT/RND — it **ignores `q.scaler`**. Confirmed two ways: (1) source
    read; (2) C-sim of a copy_qat export → GATE-2 collapses (hls4ml computes the base grid
    without the 1/β scale, keras computes with it). A power-of-2-shifted, scaler-FREE copy
    is worse than build_default (gate-1 0.949 — it reproduces QAT's coarse SATURATING grids
    and the non-integer β rounds badly). Net: copy_qat is a genuine keras-side gate-1
    improvement but is NOT hardware-faithful on this hls4ml — `build_default` is the shipped
    export. Exact QAT-attention reproduction in hardware would need per-tensor non-power-of-2
    scale support in the FixedPointQuantizer frontend, or ±β projection weights (DSP cost on
    the Q/K/V projections). Returns the number of quantizers retuned."""
    from keras import ops
    L = cfg["arch"]["n_layers"]

    def set_grid(dst, src):
        # read the source grid via .kif (robust to KIF and trainable-KBI); write to dst's
        # KIF vars (build.py stream/softmax grids are KIF).
        k, i, f = src.quantizer.kif
        for dv, sv in ((dst.quantizer._k, k), (dst.quantizer._i, i), (dst.quantizer._f, f)):
            dv.assign(np.full(dv.shape, float(np.max(np.asarray(ops.convert_to_numpy(sv)))),
                              dtype="float32"))

    n = 0
    for li in range(L):
        blk = f"bit_block_{li}"
        bq = binz[f"{blk}_attn_Wq"]["beta"]
        bk = binz[f"{blk}_attn_Wk"]["beta"]
        bv = binz[f"{blk}_attn_Wv"]["beta"]
        qs, es = qat_model.get_layer(f"{blk}_attn_scores"), export_model.get_layer(f"{blk}_attn_scores")
        set_grid(es.iq[0], qs.iq[0]); es.iq[0].scaler = 1.0 / bq
        set_grid(es.iq[1], qs.iq[1]); es.iq[1].scaler = 1.0 / bk
        qsm, esm = qat_model.get_layer(f"{blk}_attn_softmax"), export_model.get_layer(f"{blk}_attn_softmax")
        set_grid(esm.exp_table.iq, qsm.exp_table.iq); esm.exp_table.iq.scaler = 1.0 / (bq * bk)
        set_grid(esm.exp_table.oq, qsm.exp_table.oq)
        set_grid(esm.inv_table.iq, qsm.inv_table.iq)
        set_grid(esm.inv_table.oq, qsm.inv_table.oq)
        qcx, ecx = qat_model.get_layer(f"{blk}_attn_ctx"), export_model.get_layer(f"{blk}_attn_ctx")
        set_grid(ecx.iq[0], qcx.iq[0])
        set_grid(ecx.iq[1], qcx.iq[1]); ecx.iq[1].scaler = 1.0 / bv
        n += 7
    return n


def _apply_exact_beta(model, binz, pe, cfg):
    """Re-assign the 2L+2 affine gammas (and input_proj's pre-divided bias table) with
    the EXACT per-tensor β instead of the CSD-2 snap — the fidelity-reference build that
    ISOLATES the CSD-2 β Δ. For NORM-FREE graphs (arch.norm='none') the gamma must carry the
    dropped upstream β exactly like port.py's csd2 path (γ_Wo·β_v, γ_fc2·β_fc1,
    γ_head_fc2·β_head_fc1) — WITHOUT that carry the exact-β build collapses (measured
    corr≈0.036), which would make the CSD-2 characterization meaningless. Normed graphs:
    carry=1, byte-identical to before."""
    from bnhgq2.port import _assign_bn, _assign
    norm_free = str(cfg["arch"].get("norm", "subln")).lower() == "none"
    by = {ly.name: ly for ly in model.layers}
    for name, e in binz.items():
        if name.startswith("_") or e["fold"] != "explicit":
            continue
        beta_exact = float(e["beta"])
        width = e["shape"][-1]
        carry = 1.0
        if norm_free:
            if name.endswith("_attn_Wo"):
                carry = float(binz[name.replace("_attn_Wo", "_attn_Wv")]["beta"])
            elif name.endswith("_ffn_fc2"):
                carry = float(binz[name.replace("_ffn_fc2", "_ffn_fc1")]["beta"])
            elif name == "head_fc2":
                carry = float(binz["head_fc1"]["beta"])
        gamma = np.full(width, beta_exact * carry, dtype=np.float32)
        aff_bias = (np.zeros(width, np.float32) if name == "input_proj"
                    else e["bias"].astype(np.float32))
        _assign_bn(by[f"{name}_affine"], gamma, aff_bias)
        if name == "input_proj":
            bias_table = ((e["bias"].astype(np.float32) + pe.astype(np.float32))
                          / np.float32(beta_exact))
            _assign(by["input_proj"], e["q"].astype(np.float32), bias_table)


# --------------------------------------------------------------------------- #
# (4) GATE 1: export vs QAT                                                     #
# --------------------------------------------------------------------------- #
def _softmax(z):
    z = z.astype(np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def predict(model, X, batch=1024):
    out = []
    for i in range(0, len(X), batch):
        out.append(np.asarray(model(X[i:i + batch], training=False)))
    return np.concatenate(out)


def weight_export_exactness(qat_model, binz):
    """max|β·q (export) − bitnet_binary_ste(latent) (QAT)| over all bit-layers.
    Proves the datapath weights are the exact {−β,+β} the QAT forward computes."""
    import bnhgq2.qat as qat
    by = {ly.name: ly for ly in qat_model.layers}
    worst = 0.0
    for name, e in binz.items():
        if name.startswith("_"):
            continue
        qat_eff = _to_np(qat.bitnet_binary_ste(by[name]._kernel))
        exp_eff = (e["q"].astype(np.float64) * e["beta"]).astype(np.float32)
        worst = max(worst, float(np.abs(qat_eff - exp_eff.reshape(qat_eff.shape)).max()))
    return worst


def gate1(qat_model, export_model, X):
    yq_logits = predict(qat_model, X)
    ye_logits = predict(export_model, X).reshape(yq_logits.shape)
    sq, se = _softmax(yq_logits), _softmax(ye_logits)
    corr_scores = float(np.corrcoef(se.ravel(), sq.ravel())[0, 1])
    corr_logits = float(np.corrcoef(ye_logits.ravel(), yq_logits.ravel())[0, 1])
    per_class = [float(np.corrcoef(se[:, c], sq[:, c])[0, 1]) for c in range(sq.shape[1])]
    argmax_agree = float(np.mean(se.argmax(1) == sq.argmax(1)))
    return {
        "n": int(len(X)),
        "corr_scores": corr_scores,
        "corr_logits": corr_logits,
        "corr_per_class": per_class,
        "max_abs_diff_scores": float(np.abs(se - sq).max()),
        "mean_abs_diff_scores": float(np.abs(se - sq).mean()),
        "max_abs_diff_logits": float(np.abs(ye_logits - yq_logits).max()),
        "argmax_agreement": argmax_agree,
    }


# --------------------------------------------------------------------------- #
# driver                                                                       #
# --------------------------------------------------------------------------- #
def _load_jets(data_dir, n_part, n):
    from bnhgq2.data import load_eval_set
    X, y = load_eval_set(data_dir, n_part=n_part)
    idx = np.linspace(0, len(X) - 1, min(n, len(X))).astype(np.int64)
    return X[idx].astype(np.float32), y[idx]


def _store(run_dir, name, payload):
    from bnhgq2.store import _jsonable
    os.makedirs(run_dir, exist_ok=True)
    rec = {"written": time.strftime("%Y-%m-%dT%H:%M:%S"), **_jsonable(payload)}
    p = os.path.join(run_dir, name)
    with open(p, "w") as f:
        json.dump(rec, f, indent=1)
    return p


def run_convert_final(variant, seed, *, wandb_run=None, checkpoint=None, tag=None,
                      rf=None, strategy="Latency", beta_mode="csd2",
                      attn_grids="build_default",
                      n_gate=4096, n_csim=128, n_ebops=4096, data_dir=None,
                      cache_dir=None, store_root=None, characterize_exact=True,
                      config_path=None, wandb_subdir=None, run_dir=None,
                      widen_accum=False, min_frac=16, layer_configs=None,
                      input_std_path=None):
    if variant not in BINARY_VARIANTS:
        raise SystemExit(f"{variant}: convert_final only converts binary variants "
                         f"{BINARY_VARIANTS} (fp32/w8a8 have no binary story; w8a8 = "
                         "EBOPs-only, see run_convert_final_all.sh)")
    # config_path lets a non-FINAL model (e.g. r7-small-w1a8) reuse this exact chain;
    # default keeps the FINAL-campaign path byte-identical.
    cfg_path = config_path or os.path.join(_HERE, "configs", f"final-{variant}.json")
    cfg = load_config(cfg_path)
    h = cfg_hash(cfg)
    A = cfg["arch"]
    rf = rf or cfg["hls"].get("rf", 256)
    leaf = f"{variant}-s{seed}" + (f"-{tag}" if tag else "")
    wandb_run = wandb_run or f"final-{variant}-s{seed}"
    data_dir = data_dir or os.path.join(PROJECT_ROOT, "data", "val")
    cache_dir = cache_dir or os.path.join(PROJECT_ROOT, "bnjettag", "models", "final")
    store_root = store_root or os.path.join(PROJECT_ROOT, "bnjettag", "results", "final")
    # run_dir override lets a caller pin the exact results leaf (e.g. r7 rf1/rf8 dirs);
    # default is the FINAL-campaign <store_root>/runs/<hash>/<leaf> layout.
    run_dir = run_dir or os.path.join(store_root, "runs", h, leaf)
    os.makedirs(run_dir, exist_ok=True)

    print(f"=== convert_final {leaf}  cfg={cfg['name']} [{h}] rf={rf} "
          f"strategy={strategy} beta={beta_mode} attn_grids={attn_grids} ===", flush=True)

    # (1) fetch + (2) load
    kp, mp = fetch_checkpoint(variant, seed, wandb_run,
                              os.path.join(cache_dir, leaf) if tag else cache_dir,
                              local_checkpoint=checkpoint, subdir=wandb_subdir)
    meta = json.load(open(mp)) if mp else {}
    qat_model = load_qat_model(kp)
    print(f"[load] {os.path.basename(kp)}  params={qat_model.count_params():,}", flush=True)

    # real jets
    Xg, yg = _load_jets(data_dir, A["n_part"], n_gate)
    _std = None
    if input_std_path:
        from bnhgq2.data import apply_input_std
        _std = json.load(open(input_std_path))
        Xg = apply_input_std(Xg, _std["mu"], _std["sigma"])
        print(f"[data] input_std applied from {input_std_path} (round-8 contract)", flush=True)
    print(f"[data] {len(Xg)} real jets from {os.path.relpath(data_dir, PROJECT_ROOT)}",
          flush=True)

    # (3) export: binarize once, read act grids, measure stream/score ranges, build.
    binz, pe, names = qat_binz_pe(qat_model, cfg)
    calib = read_qat_act_ibits(qat_model, names)
    calib.update(qat_stream_ranges(qat_model, cfg, binz, Xg[:2048]))
    export_model = build_export(cfg, binz, calib, pe, beta_mode=beta_mode,
                                qat_model=qat_model, attn_grids=attn_grids,
                                calib_jets=Xg[:2048])
    nf_grids = getattr(export_model, "_bnhgq2_norm_free_grids", {})
    if nf_grids:
        print("[grids] norm-free carry-site widths (measured export-side ranges): "
              + ", ".join(f"{n.split('_')[-1]}<{v['width']},{1+v['i']}>@max{v['max']:.1f}"
                          for n, v in nf_grids.items()), flush=True)
    w_exact = weight_export_exactness(qat_model, binz)
    print(f"[export] {export_model.count_params():,} params; "
          f"weight |Δ| vs QAT forward = {w_exact:.3e} (0 => datapath ±β exact)", flush=True)

    # (4) GATE 1
    g1 = gate1(qat_model, export_model, Xg)
    g1["weight_export_max_abs_diff"] = w_exact
    g1["beta_mode"] = beta_mode
    g1_pass = g1["corr_scores"] >= 0.9999
    print(f"[GATE1] export vs QAT ({beta_mode}): corr_scores={g1['corr_scores']:.6f} "
          f"corr_logits={g1['corr_logits']:.6f} argmax={g1['argmax_agreement']:.4f} "
          f"pass={g1_pass}", flush=True)

    export_verify = {"config": cfg["name"], "config_hash": h, "variant": variant,
                     "seed": seed, "tag": tag, "wandb_run": wandb_run, "attn_grids": attn_grids,
                     "checkpoint": os.path.relpath(kp, PROJECT_ROOT) if kp.startswith(PROJECT_ROOT) else kp,
                     "train_meta": meta, "n_bitlayers": len([n for n in binz if not n.startswith('_')]),
                     "act_ibits": {n: calib[n] for n in calib if not n.startswith("__")},
                     "norm_free_grids": nf_grids,
                     "gate1": g1, "gate1_pass": bool(g1_pass), "gate1_threshold": 0.9999}

    # Characterize the fidelity axes when shipping the default export (csd2 + build_default):
    #   * exact-β  — isolates the CSD-2 β Δ (tiny)
    #   * copy_qat — reproduces the TRAINED attention grids via scaler=1/β -> gate-1 ceiling.
    #     hls4ml IGNORES that datalane scaler (proven), so this is a keras-side ceiling; its
    #     gate-2 drop is measured after conversion below (demonstrates non-hardware-faithfulness).
    do_copyqat_demo = False
    if characterize_exact and beta_mode == "csd2" and attn_grids == "build_default":
        exact_model = build_export(cfg, binz, calib, pe, beta_mode="exact",
                                   calib_jets=Xg[:2048])
        g1e = gate1(qat_model, exact_model, Xg)
        export_verify["gate1_exact_beta"] = {
            "corr_scores": g1e["corr_scores"], "corr_logits": g1e["corr_logits"],
            "argmax_agreement": g1e["argmax_agreement"],
            "note": "exact-β affine (DSP cost on mulder unmeasured) — isolates the CSD-2 β Δ",
        }
        print(f"[GATE1] exact-β reference: corr_scores={g1e['corr_scores']:.6f} "
              f"(Δ from csd2 = {g1e['corr_scores'] - g1['corr_scores']:+.2e})", flush=True)
        try:  # optional keras-side characterization — must never block the shipped gates
            cq_model = build_export(cfg, binz, calib, pe, beta_mode="exact",
                                    qat_model=qat_model, attn_grids="copy_qat",
                                    calib_jets=Xg[:2048])
            g1m = gate1(qat_model, cq_model, Xg)
            export_verify["gate1_copy_qat_ceiling"] = {
                "corr_scores": g1m["corr_scores"], "corr_logits": g1m["corr_logits"],
                "argmax_agreement": g1m["argmax_agreement"],
                "note": "attn_grids='copy_qat' (QAT attention grids via scaler=1/β). KERAS-SIDE "
                        "ceiling: hls4ml's FixedPointQuantizer frontend ignores q.scaler (measured "
                        "gate-2 below), so this is not hardware-faithful; build_default ships.",
            }
            print(f"[GATE1] copy_qat ceiling: corr_scores={g1m['corr_scores']:.6f}", flush=True)
            del cq_model
            do_copyqat_demo = True
        except Exception as e:
            export_verify["gate1_copy_qat_ceiling"] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[GATE1] copy_qat characterization skipped ({type(e).__name__}: {e})",
                  flush=True)
        del exact_model

    _store(run_dir, "export_verify.json", export_verify)

    # (6) EBOPs (native HGQ2, exported model)
    from bnhgq2.ebops_calc import compute_ebops
    Xe, _ = _load_jets(data_dir, A["n_part"], n_ebops)
    if _std is not None:
        from bnhgq2.data import apply_input_std
        Xe = apply_input_std(Xe, _std["mu"], _std["sigma"])
    eb = compute_ebops(export_model, Xe)
    eb.update({"config": cfg["name"], "config_hash": h, "variant": variant, "seed": seed,
               "beta_mode": beta_mode, "convention": "HGQ2_native_trace_minmax"})
    _store(run_dir, "ebops.json", eb)
    print(f"[EBOPs] total = {eb['total']:,}", flush=True)

    # (5) convert + GATE 2 (C-sim)
    from bnhgq2.convert import convert, pack_for_mulder
    patch_resource_einsum_check()
    out = os.path.join(run_dir, f"hls_prj_rf{rf}")
    csim_X = Xg[:n_csim]
    keras_ref = predict(export_model, csim_X)
    widen_count = {"n": 0}

    def _post_parse(hm):
        fix_relu_saturation(hm)
        if widen_accum:
            widen_count["n"] = widen_weighted_accum(hm, min_frac=min_frac)

    hm, report = convert(export_model, cfg, out, rf=rf, strategy=strategy,
                         csim_X=csim_X, keras_ref=keras_ref,
                         post_parse=_post_parse, layer_configs=layer_configs)
    accum_audit = audit_weighted_accums(out, cfg, min_frac=min_frac)
    if widen_accum:
        print(f"[widen] widened {widen_count['n']} weighted-layer accums to frac>={min_frac}; "
              f"audit all_wide={accum_audit['all_wide']} "
              f"offenders={accum_audit['narrow_frac_offenders']}", flush=True)
    tar = pack_for_mulder(out, out + ".tar.gz")
    cs = report.get("csim", {})
    g2_pass = cs.get("corr", 0.0) >= 0.997
    print(f"[GATE2] hls4ml C-sim vs export: corr={cs.get('corr')} "
          f"max|Δ|={cs.get('max_abs_diff')} bit_exact={cs.get('bit_exact')} "
          f"pass={g2_pass}", flush=True)

    csim_verify = {"config": cfg["name"], "config_hash": h, "variant": variant,
                   "seed": seed, "tag": tag, "rf": rf, "strategy": strategy,
                   "beta_mode": beta_mode, "attn_grids": attn_grids,
                   "backend": report.get("backend"),
                   "macos_local_csim": platform.system() == "Darwin",
                   "csim": cs, "gate2_pass": bool(g2_pass), "gate2_threshold": 0.997,
                   "relu_sat_fix": "applied (fix_relu_saturation)",
                   "widen_accum": bool(widen_accum), "widen_min_frac": min_frac,
                   "n_layers_widened": widen_count["n"], "accum_audit": accum_audit,
                   "layer_configs": layer_configs}

    # Demonstration (measured, not assumed): convert the copy_qat export and C-sim it.
    # hls4ml drops the datalane scaler -> its output reverts to the build_default grids while
    # the keras ref uses scaler=1/β -> GATE-2 collapses. This is why copy_qat is keras-only.
    if do_copyqat_demo:
        try:
            cq = build_export(cfg, binz, calib, pe, beta_mode=beta_mode,
                              qat_model=qat_model, attn_grids="copy_qat",
                              calib_jets=Xg[:2048])
            cq_out = os.path.join(run_dir, f"hls_prj_copyqat_rf{rf}")
            cq_ref = predict(cq, csim_X)
            _, cq_report = convert(cq, cfg, cq_out, rf=rf, strategy=strategy,
                                   csim_X=csim_X, keras_ref=cq_ref,
                                   post_parse=fix_relu_saturation)
            cqcs = cq_report.get("csim", {})
            csim_verify["gate2_copy_qat_demo"] = {
                "corr": cqcs.get("corr"), "max_abs_diff": cqcs.get("max_abs_diff"),
                "note": "hls4ml vs copy_qat keras export. Low corr = hls4ml ignored the 1/β "
                        "datalane scaler (keras_v3/hgq2/_base.py extract_fixed_quantizer_config "
                        "reads only kif) -> copy_qat cannot be a hardware model on this hls4ml.",
            }
            import shutil
            shutil.rmtree(cq_out, ignore_errors=True)  # demo only, not a mulder artifact
            print(f"[GATE2] copy_qat demo (hls4ml drops scaler): corr={cqcs.get('corr')}",
                  flush=True)
        except Exception as e:
            csim_verify["gate2_copy_qat_demo"] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[GATE2] copy_qat demo skipped ({type(e).__name__}: {e})", flush=True)

    _store(run_dir, "csim_verify.json", csim_verify)

    convert_rec = {"config_hash": h, "variant": variant, "seed": seed, "tag": tag,
                   "output_dir": os.path.relpath(out, PROJECT_ROOT),
                   "tarball": os.path.relpath(tar, PROJECT_ROOT),
                   "rf": rf, "strategy": strategy, "beta_mode": beta_mode,
                   "part": cfg["hls"].get("part"), "clock_ns": cfg["hls"].get("clock_ns"),
                   "mulder_cmd": f"./mulder_csynth.sh {os.path.basename(tar)}"}
    _store(run_dir, "convert.json", convert_rec)

    print(f"\n[done] {leaf}: GATE1({beta_mode})={'PASS' if g1_pass else 'CHECK'} "
          f"GATE2={'PASS' if g2_pass else 'CHECK'}  EBOPs={eb['total']:,}\n"
          f"       results  -> {os.path.relpath(run_dir, PROJECT_ROOT)}\n"
          f"       mulder   -> scp {os.path.relpath(tar, PROJECT_ROOT)} mulder:...; "
          f"./mulder_csynth.sh {os.path.basename(tar)}", flush=True)
    return {"gate1": g1, "gate1_pass": g1_pass, "gate2": cs, "gate2_pass": g2_pass,
            "ebops": eb["total"], "run_dir": run_dir, "tarball": tar}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", required=True, choices=BINARY_VARIANTS)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--wandb-run", default=None, help="W&B run display name "
                    "(default final-<variant>-s<seed>; use final-smoke for the smoke ckpt)")
    ap.add_argument("--checkpoint", default=None, help="local model_best.keras (skips W&B)")
    ap.add_argument("--tag", default=None, help="results leaf suffix, e.g. 'smoke'")
    ap.add_argument("--rf", type=int, default=None)
    ap.add_argument("--strategy", default="Latency",
                    help="hls4ml strategy. MUST be Latency for the full model: hls4ml "
                         "1.3.0 EinsumDense codegen asserts Latency-only (Resource raises); "
                         "einsum folding uses ReuseFactor as multiplier_limit.")
    ap.add_argument("--beta-mode", default="csd2", choices=("csd2", "exact"))
    ap.add_argument("--attn-grids", default="build_default",
                    choices=("build_default", "copy_qat"),
                    help="build_default = the shipped hardware-faithful export (gate-2 ~1.0). "
                         "copy_qat = reproduce the TRAINED attention grids via scaler=1/β for a "
                         "higher gate-1 (keras-side ceiling; hls4ml drops the scaler, so its "
                         "gate-2 collapses — not hardware-faithful, see match_attention_to_qat).")
    ap.add_argument("--n-gate", type=int, default=4096)
    ap.add_argument("--n-csim", type=int, default=128)
    ap.add_argument("--n-ebops", type=int, default=4096)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--no-exact-char", action="store_true",
                    help="skip the exact-β characterization build")
    ap.add_argument("--config", default=None,
                    help="explicit config JSON path (default configs/final-<variant>.json; "
                         "point at e.g. configs/r7-small-w1a8.json for the round-7 model)")
    ap.add_argument("--wandb-subdir", default=None,
                    help="W&B file subdir prefix (default <variant>-s<seed>; e.g. small-w1a8-s2)")
    ap.add_argument("--run-dir", default=None,
                    help="explicit results dir (default <store-root>/runs/<hash>/<leaf>)")
    ap.add_argument("--store-root", default=None, help="results store root override")
    ap.add_argument("--cache-dir", default=None, help="checkpoint cache dir override")
    ap.add_argument("--widen-accum", action="store_true",
                    help="widen EVERY weighted-layer accumulator to frac>=--min-frac (the measured "
                         "DSP-0 form, probes_final_v3): flips the binary ±1 MAC from DSP48 to LUT. "
                         "Bit-exact; C-sim is the arbiter. Weightless act×act einsums are untouched.")
    ap.add_argument("--min-frac", type=int, default=16,
                    help="min fractional width for --widen-accum (default 16 = the fc1 0-DSP point)")
    ap.add_argument("--input-std", default=None,
                    help="input_std.json (round-8: standardize the gate/ebops jets with "
                         "the checkpoint's train-split stats)")
    ap.add_argument("--layer-configs", default=None,
                    help="JSON file {layer_name: {hls_key: value}} merged into hls_config['LayerName'] "
                         "(bnhgq2.convert layer_configs). Per-layer ReuseFactor maps for the LUT-fit "
                         "operating-point study (einsums Latency-only). Default None = single-RF path.")
    a = ap.parse_args()
    lc = json.load(open(a.layer_configs)) if a.layer_configs else None
    run_convert_final(a.variant, a.seed, wandb_run=a.wandb_run, checkpoint=a.checkpoint,
                      tag=a.tag, rf=a.rf, strategy=a.strategy, beta_mode=a.beta_mode,
                      attn_grids=a.attn_grids,
                      n_gate=a.n_gate, n_csim=a.n_csim, n_ebops=a.n_ebops,
                      data_dir=a.data_dir, characterize_exact=not a.no_exact_char,
                      config_path=a.config, wandb_subdir=a.wandb_subdir,
                      run_dir=a.run_dir, store_root=a.store_root, cache_dir=a.cache_dir,
                      input_std_path=a.input_std,
                      widen_accum=a.widen_accum, min_frac=a.min_frac, layer_configs=lc)


if __name__ == "__main__":
    main()
