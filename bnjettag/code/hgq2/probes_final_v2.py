#!/usr/bin/env python3
"""v2 csynth probes: the BIG binary shapes as 2-D QDense at RESOURCE strategy.

Why v2 (measured, constraints_map.md 2026-07-09): the v1 big binary shapes are QEinsumDense at
Latency (EinsumDense is Latency-only in hls4ml 1.3.0), and clang LTO does NOT converge for
≥~40k-MAC einsum-dense (attn_qkv/attn_Wo 65k, ffn_fc1 262k all hit the 3 h mulder timeout with
clang at 100 % CPU). At Latency the folded RF doesn't reduce the inlined-constant weight volume.
The §6′-proven route is **plain 2-D Dense at Resource** (weights → ROM, LTO stays bounded).

For each contraction `...,in->...,out` the token axis flattens (einsum btf,fd->btd ≡ per-token
Dense F→D), so we re-emit as a 2-D QDense probe carrying the flagship's EXACT config: pure ±1
kernel (binz sign matrix), KBI-derived static act grid (calib i via .kif), and — the Resource
1-bit-trap avoidance — β in a FROZEN CSD-2 QBatchNormalization affine, NEVER in-weight ±β (the
in-weight scale at Resource = a guaranteed 256-DSP ROM trap, measured 2026-07-04). fc1/head_fc1
carry their SAT-pinned ReLU. C-sim gate: hls4ml vs the flagship sublayer output on REAL activations
(same bar as v1). Output: distinct-named tarballs + manifest v2 (does NOT overwrite v1). csynth is
mulder-only — this ships tarballs + the mulder command.

Usage:
  export WANDB_API_KEY=$(cat bnjettag/wandb-api-key.txt)
  python probes_final_v2.py --checkpoint bnjettag/models/final/w1a8-s3/model_best.keras
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import convert_final as cf  # noqa: E402
from bnhgq2.config import load_config, cfg_hash, PROJECT_ROOT  # noqa: E402
from bnhgq2.gold import csd2_snap  # noqa: E402

FLAGSHIP_VARIANT, FLAGSHIP_SEED, FLAGSHIP_RUN = "w1a8", 3, "final-w1a8-s3"
FLAGSHIP_CONFIG = f"final-{FLAGSHIP_VARIANT}"       # configs/final-w1a8.json
# Seed a config ships with, when none is given (the flagship's is s3; new configs start at s1).
DEFAULT_SEED = {FLAGSHIP_CONFIG: FLAGSHIP_SEED}

# NOTE: SHAPES below is FLAGSHIP-PINNED (literal D=256 / FFN=1024 / F=16 dims) and drives only
# the per-shape probe drivers (this file's main(), probes_final_v3 — both iterate it directly).
# The WHOLE-MODEL emitters (monolith_v2, probe_block) do not touch it: they take every dim from
# cfg['arch'] via load_model() + bnhgq2.monolith, so they run on any config.
# fold: 'explicit' (±1 dense no-bias -> CSD-2 affine γ=csd2(β), β=layer bias)
#       'explicit_nobias' (input_proj: per-position bias+PE dropped (0-DSP) -> affine β=0)
#       'score' (Q/K/V: ±1 dense, no bias, no affine — β folds into softmax downstream)
#       'bias'  (fc1/head_fc1: ±1 dense with bias=b/β, then SAT-pinned ReLU)
# in_dim/out_dim are EXPLICIT: the einsum contracts different axes per layer (Wq contracts the
# first kernel axis D; Wo contracts the first TWO axes H,E), so kernel.shape alone is ambiguous.
# np.reshape(q, (in_dim, out_dim)) is row-major-correct for both once in/out are right.
SHAPES = [
    dict(name="input_proj", layer="input_proj", subln="ln_input_proj",
         flag_out="input_proj_affine", fold="explicit_nobias", relu=False,
         in_dim=16, out_dim=256,
         desc="F=16 -> D=256 (per-token; SubLN on raw feats; PE+per-pos bias dropped = 0-DSP)"),
    dict(name="attn_qkv", layer="bit_block_0_attn_Wq", subln="ln_bit_block_0_attn_qkv",
         flag_out="bit_block_0_attn_Wq", fold="score", relu=False,
         in_dim=256, out_dim=256,
         desc="D=256 -> (H*E)=256 (Q/K/V shape; β score-folded, no affine)"),
    dict(name="attn_Wo", layer="bit_block_0_attn_Wo", subln="ln_bit_block_0_attn_Wo",
         flag_out="bit_block_0_attn_Wo_affine", fold="explicit", relu=False,
         in_dim=256, out_dim=256,
         desc="(H*E)=256 -> D=256"),
    dict(name="ffn_fc1", layer="bit_block_0_ffn_fc1", subln="ln_bit_block_0_ffn_fc1",
         flag_out="bit_block_0_ffn_act", fold="bias", relu=True,
         in_dim=256, out_dim=1024,
         desc="D=256 -> FFN=1024 + ReLU(SAT) (262k MACs — the v1 LTO wall)"),
    dict(name="ffn_fc2", layer="bit_block_0_ffn_fc2", subln="ln_bit_block_0_ffn_fc2",
         flag_out="bit_block_0_ffn_fc2_affine", fold="explicit", relu=False,
         in_dim=1024, out_dim=256,
         desc="FFN=1024 -> D=256"),
    dict(name="head_fc1", layer="head_fc1", subln="ln_head_fc1",
         flag_out="head_act", fold="bias", relu=True,
         in_dim=256, out_dim=256,
         desc="256 -> 256 + ReLU(SAT)"),
]


def resolve_source(config_name, variant=None, seed=None, wandb_run=None):
    """Where <config_name>'s trained checkpoint lives, by convention:
         variant   = the config name minus its 'final-' prefix   (w1a8, small-w1a8)
         seed      = DEFAULT_SEED[config] if pinned, else 1      (the flagship shipped s3)
         wandb_run = f'{config_name}-s{seed}'                    (final-w1a8-s3)
    W&B keeps the checkpoint at '<variant>-s<seed>/model_best.keras' inside that run (see
    cf.fetch_checkpoint). These conventions REPRODUCE the FLAGSHIP_* constants exactly, so
    'final-w1a8' still resolves to w1a8 / s3 / final-w1a8-s3. Idempotent, and every field is
    overridable — round-7's W&B layout is not frozen yet, so pass --variant/--seed/--wandb-run
    if it differs, or just --checkpoint (which bypasses W&B entirely)."""
    if variant is None:
        variant = (config_name[len("final-"):] if config_name.startswith("final-")
                   else config_name)
    if seed is None:
        seed = DEFAULT_SEED.get(config_name, 1)
    if wandb_run is None:
        wandb_run = f"{config_name}-s{seed}"
    return variant, int(seed), wandb_run


def load_model(config_name=FLAGSHIP_CONFIG, checkpoint=None, variant=None, seed=None,
               wandb_run=None, n_jets=512):
    """Load ANY config in configs/ and return everything the emitters need:
    (cfg, export, qat, binz, calib, pe, X, ckpt_rel).

    The architecture (d_model / n_heads / n_layers / ffn_dim / n_part / n_feat / n_classes)
    is taken ENTIRELY from cfg['arch'] — nothing here is pinned to the flagship's
    D256/H8/L8/FFN1024, and every callee (cf.qat_binz_pe -> extract.layer_names(n_layers),
    cf.qat_stream_ranges, bnhgq2.build/monolith) is already config-driven.
    Checkpoint precedence: explicit `checkpoint` > cfg['checkpoint'] (resolved against
    PROJECT_ROOT) > W&B fetch of resolve_source()'s run."""
    cfg = load_config(os.path.join(_HERE, "configs", f"{config_name}.json"))
    variant, seed, wandb_run = resolve_source(config_name, variant, seed, wandb_run)
    if checkpoint is None and cfg.get("checkpoint"):
        checkpoint = os.path.join(PROJECT_ROOT, cfg["checkpoint"])
    kp, _ = cf.fetch_checkpoint(variant, seed, wandb_run,
                                os.path.join(PROJECT_ROOT, "bnjettag", "models", "final"),
                                local_checkpoint=checkpoint)
    qat = cf.load_qat_model(kp)
    X, _ = cf._load_jets(os.path.join(PROJECT_ROOT, "data", "val"),
                         cfg["arch"]["n_part"], n_jets)
    binz, pe, names = cf.qat_binz_pe(qat, cfg)
    calib = cf.read_qat_act_ibits(qat, names)
    calib.update(cf.qat_stream_ranges(qat, cfg, binz, X))
    exp = cf.build_export(cfg, binz, calib, pe, beta_mode="csd2")
    ckpt_rel = os.path.relpath(kp, PROJECT_ROOT) if kp.startswith(PROJECT_ROOT) else kp
    return cfg, exp, qat, binz, calib, pe, X, ckpt_rel


def load_flagship(checkpoint=None):
    """Back-compat wrapper — exactly load_model(FLAGSHIP_CONFIG). The per-shape probe drivers
    (this file's main(), probes_final_v3) stay flagship-only because SHAPES is flagship-pinned."""
    return load_model(FLAGSHIP_CONFIG, checkpoint)


def _cfgs(act_bits, i0, in_dim):
    from hgq.quantizer import QuantizerConfig
    binary_kq = QuantizerConfig("kbi", "weight", k0=1, b0=1, i0=1, round_mode="RND",
                                overflow_mode="SAT_SYM", trainable=False, heterogeneous_axis=(),
                                bc=None, ic=None, br=None, ir=None)
    act_q = QuantizerConfig("kif", "datalane", k0=1, i0=int(i0), f0=act_bits - 1 - int(i0),
                            round_mode="RND_CONV", overflow_mode="SAT", trainable=False,
                            heterogeneous_axis=(), ic=None, fc=None, ir=None, fr=None)
    wq_wide = QuantizerConfig("kif", "weight", k0=1, i0=2, f0=16, round_mode="RND",
                              overflow_mode="SAT", trainable=False, heterogeneous_axis=(),
                              ic=None, fc=None, ir=None, fr=None)
    bq_wide = QuantizerConfig("kif", "bias", k0=1, i0=8, f0=16, round_mode="RND_CONV",
                              overflow_mode="SAT", trainable=False, heterogeneous_axis=(),
                              ic=None, fc=None, ir=None, fr=None)
    # affine input = the ±1 contraction over `in_dim` — size i to cover it (SAT clips tail)
    i_aff = int(np.ceil(np.log2(max(in_dim, 2)))) + 2
    aff_iq = QuantizerConfig("kif", "datalane", k0=1, i0=i_aff, f0=act_bits - 1,
                             round_mode="RND_CONV", overflow_mode="SAT", trainable=False,
                             heterogeneous_axis=(), homogeneous_axis=None,
                             ic=None, fc=None, ir=None, fr=None)
    return binary_kq, act_q, wq_wide, bq_wide, aff_iq


def build_probe(cfg, binz, calib, spec):
    import keras
    from hgq.config import LayerConfigScope
    from hgq.layers import QDense, QBatchNormalization
    from bnhgq2.subln import PSubLN
    from bnhgq2.port import _assign_bn

    e = binz[spec["layer"]]
    q = e["q"]
    in_dim, out_dim = int(spec["in_dim"]), int(spec["out_dim"])
    assert q.size == in_dim * out_dim, f"{spec['name']}: kernel {q.shape} != {in_dim}x{out_dim}"
    q2d = q.reshape(in_dim, out_dim).astype(np.float32)  # einsum kernel -> 2-D (row-major)
    beta = float(e["beta"])
    act_bits = int(cfg["quant"]["act_bits"])
    i0 = int(np.max(calib[spec["layer"]]))
    binary_kq, act_q, wq_wide, bq_wide, aff_iq = _cfgs(act_bits, i0, in_dim)
    use_bias = spec["fold"] == "bias"

    with LayerConfigScope(enable_ebops=True, beta0=0.0):
        x = keras.Input((in_dim,), name="inp")
        h = PSubLN(flatten_axes=1, name=f"ln_{spec['name']}")(x)
        dense = QDense(out_dim, use_bias=use_bias, iq_conf=act_q, kq_conf=binary_kq,
                       bq_conf=(bq_wide if use_bias else None), name=f"dense_{spec['name']}")
        h = dense(h)
        aff = None
        if spec["fold"] in ("explicit", "explicit_nobias"):
            aff = QBatchNormalization(axis=-1, epsilon=0.0, center=True, scale=True,
                                      kq_conf=wq_wide, bq_conf=bq_wide, iq_conf=aff_iq,
                                      name=f"dense_{spec['name']}_affine")
            h = aff(h)
        if spec["relu"]:
            h = keras.layers.ReLU(name=f"{spec['name']}_act")(h)
        m = keras.Model(x, h, name=f"probe2_dense_{spec['name']}")

    # port weights
    kv = getattr(dense, "_kernel", None)
    if kv is None:
        kv = dense.kernel
    kv.assign(q2d)
    if use_bias:  # bias_fold: b/β (next-LN kills the 1/β scale in the full model)
        bv = getattr(dense, "_bias", None)
        if bv is None:
            bv = dense.bias
        bv.assign((e["bias"].astype(np.float32) / np.float32(beta)).reshape(bv.shape))
    if aff is not None:
        gamma = np.full(out_dim, csd2_snap(beta), dtype=np.float32)
        bbeta = (np.zeros(out_dim, np.float32) if spec["fold"] == "explicit_nobias"
                 else e["bias"].astype(np.float32))
        _assign_bn(m.get_layer(f"dense_{spec['name']}_affine"), gamma, bbeta)
    return m, in_dim, out_dim


def flagship_ref(exp, byn, spec, binz, pe, X, n, in_dim, out_dim):
    """keras_ref = the flagship sublayer output on real activations, reshaped to (N, out_dim);
    csim_X = the flagship SubLN input, reshaped to (N, in_dim) — the per-token flatten (Wo's
    (T,H,E) input groups H*E=256, not the last axis 32). For input_proj the per-position
    (bias+PE) additive is subtracted (the probe measures the 0-DSP-add-free binary core)."""
    import keras
    tin = keras.Model(exp.inputs, byn[spec["subln"]].input)
    tout = keras.Model(exp.inputs, byn[spec["flag_out"]].output)
    xin = np.asarray(tin.predict(X[:n], verbose=0))
    yout = np.asarray(tout.predict(X[:n], verbose=0))
    if spec["fold"] == "explicit_nobias":  # input_proj: strip per-position bias(T,D)+PE(T,D)
        e = binz[spec["layer"]]
        add = e["bias"].astype(np.float32) + pe.astype(np.float32)   # (T, D)
        yout = yout - add[None, :, :]
    xin = xin.reshape(-1, in_dim)
    yout = yout.reshape(-1, out_dim)
    return np.ascontiguousarray(xin.astype(np.float32)), np.ascontiguousarray(yout.astype(np.float32))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--n-csim", type=int, default=64)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()

    cfg, exp, qat, binz, calib, pe, X, ckpt_rel = load_flagship(a.checkpoint)
    byn = {l.name: l for l in exp.layers}
    cf.patch_resource_einsum_check()
    from bnhgq2.convert import convert, pack_for_mulder

    out_root = os.path.join(PROJECT_ROOT, "bnjettag", "results", "final", "probes")
    os.makedirs(out_root, exist_ok=True)
    want = set(a.only.split(",")) if a.only else None
    manifest = {"source": {"flagship": FLAGSHIP_RUN, "config": cfg["name"],
                           "config_hash": cfg_hash(cfg), "checkpoint": ckpt_rel,
                           "note": "v2 big-shape probes: 2-D QDense at RESOURCE/RF256 (einsum "
                                   "btf,fd->btd ≡ per-token Dense). Pure ±1 datapath + frozen "
                                   "CSD-2 affine (Resource 1-bit-trap avoidance). Sliced config "
                                   "(±1 sign matrix, KBI act grid, ReLU-SAT) from build_export("
                                   "final-w1a8-s3)."},
                "hls": {"part": cfg["hls"].get("part"), "clock_ns": cfg["hls"].get("clock_ns"),
                        "backend": "Vitis", "io": "io_parallel", "strategy": "Resource", "rf": 256},
                "written": time.strftime("%Y-%m-%dT%H:%M:%S"), "probes": {}}
    tarballs = []
    for spec in SHAPES:
        if want and spec["name"] not in want:
            continue
        m, in_dim, out_dim = build_probe(cfg, binz, calib, spec)
        xin, yref = flagship_ref(exp, byn, spec, binz, pe, X, a.n_csim, in_dim, out_dim)
        repro = float(np.corrcoef(np.asarray(m(xin, training=False)).ravel(), yref.ravel())[0, 1])
        proj = os.path.join(out_root, f"probe2_dense_{spec['name']}_rf256_resource")
        shutil.rmtree(proj, ignore_errors=True)
        _, report = convert(m, cfg, proj, rf=256, strategy="Resource",
                            csim_X=xin, keras_ref=yref, post_parse=cf.fix_relu_saturation)
        tar = pack_for_mulder(proj, proj + ".tar.gz")
        tarballs.append(os.path.basename(tar))
        cs = report.get("csim", {})
        corr = cs.get("corr")
        ok = bool(corr is not None and corr >= 0.997)
        manifest["probes"][f"dense_{spec['name']}"] = {
            "shape": f"{in_dim}->{out_dim}", "desc": spec["desc"], "fold": spec["fold"],
            "rf": 256, "strategy": "Resource", "relu_sat": spec["relu"],
            "csim_vs_flagship": {"n": cs.get("n"), "corr": corr,
                                 "max_abs_diff": cs.get("max_abs_diff"), "sanity_pass": ok},
            "keras_reproduces_flagship_corr": repro,
            "expected_checks": ("binary 2-D dense: DSP=0 IF hls4ml emits true 1-bit weight types; "
                                "else ~256 DSP (270 with SubLN) = the HGQ2-path 1-bit-emission gap "
                                "(constraints_map 2026-07-09) — mulder is the arbiter. Trap "
                                "avoided: pure ±1 datapath + frozen CSD-2 affine (0 DSP). "
                                "SubLN = expected DSP consumer. ReLU (fc1/head_fc1) = 0 DSP, SAT."),
            "tarball": os.path.relpath(tar, PROJECT_ROOT),
        }
        shutil.rmtree(proj, ignore_errors=True)
        print(f"[probe2] dense_{spec['name']:12s} {in_dim:>4d}->{out_dim:<4d} Resource/RF256  "
              f"csim={corr}  repro={repro:.6f}  sanity={'PASS' if ok else 'CHECK'}", flush=True)

    manifest["mulder_command"] = "./mulder_csynth.sh " + " ".join(sorted(tarballs))
    mpath = os.path.join(out_root, "probes_manifest_v2.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[done] {len(manifest['probes'])} v2 probes -> {os.path.relpath(out_root, PROJECT_ROOT)}")
    print(f"       manifest: {os.path.relpath(mpath, PROJECT_ROOT)}")
    print(f"       mulder:   {manifest['mulder_command']}")


if __name__ == "__main__":
    main()
