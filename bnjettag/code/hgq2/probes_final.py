#!/usr/bin/env python3
"""Per-shape csynth PROBE generator from the FINAL flagship (decisions.md 2026-07-07).

The whole-model monolith cannot be C-synthesized: even folded RF=256 at D256/L8, Vitis 2023.2
OOM-killed during Unroll/Inline at 27M instructions (~24 h on mulder's 125 GB box). So the
deployable route is the §6' per-shape probe methodology (probes.py precedent) — but here each
probe is SLICED FROM THE FINAL FLAGSHIP EXPORT MODEL (build_export on final-w1a8-s3), so it
carries the final stack's EXACT configuration: KBI-derived static act grids (via quantizer.kif),
the ReLU unsigned-WRAP→SAT pins (fix_relu_saturation), CSD-2/frozen-affine β handling, and the
range-reduced SubLN. No re-porting, no drift — the probe IS a sub-graph of the shipped model.

Each probe -> one self-contained hls4ml project (Vitis, VU13P, 2.5 ns) + tarball for mulder, with
a local C-sim sanity gate (corr vs the sliced keras sub-model on REAL tapped activations) before
it ships. This script DOES NOT run csynth (mulder-only); it emits tarballs + a manifest + the
exact mulder command list under results/final/probes/.

Usage:
  export WANDB_API_KEY=$(cat bnjettag/wandb-api-key.txt)     # never echoed
  python probes_final.py                                     # fetch flagship, emit all probes
  python probes_final.py --checkpoint /path/model_best.keras # local flagship
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

FLAGSHIP_VARIANT = "w1a8"
FLAGSHIP_SEED = 3
FLAGSHIP_RUN = "final-w1a8-s3"

# blk-0 is representative of every block (identical shapes/config across L). Probe = one distinct
# critical-path shape. kind: 'bitlinear' (SubLN->act-quant->±1 einsum/dense->affine/ReLU) |
# 'subln' | 'attn_core'. `inp` names a layer whose .input is the probe input (single-input); for
# attn_core, `inp_out` names layers whose .output are the (multi) probe inputs.
PROBES = [
    dict(name="bitlinear_input_proj", kind="bitlinear", rf=256,
         inp="ln_input_proj", out="input_proj_affine",
         shape="btf,fd->btd  F=16 -> D=256 (per-token; SubLN on RAW features, var~1e6)",
         checks="binary einsum-dense = 0 DSP (±1 inlined, Latency); input_proj_affine (CSD-2 β) "
                "= 0 DSP; SubLN(F=16 on raw features) = the DSP consumer here."),
    dict(name="bitlinear_attn_qkv", kind="bitlinear", rf=256,
         inp="ln_bit_block_0_attn_qkv", out="bit_block_0_attn_Wq",
         shape="btd,dhe->bthe  D=256 -> (H=8,E=32)=256  (Q/K/V share this shape; β score-folded, no affine)",
         checks="binary einsum-dense = 0 DSP; NO affine (β folds into the softmax input_scaler); "
                "SubLN(256) = DSP consumer."),
    dict(name="bitlinear_attn_Wo", kind="bitlinear", rf=256,
         inp="ln_bit_block_0_attn_Wo", out="bit_block_0_attn_Wo_affine",
         shape="bthe,hed->btd  (H,E)=256 -> D=256  (SubLN flatten_axes=2 over H*E)",
         checks="binary einsum-dense = 0 DSP; Wo_affine (CSD-2 β) = 0 DSP; SubLN(256) = DSP consumer."),
    dict(name="bitlinear_ffn_fc1", kind="bitlinear", rf=256,
         inp="ln_bit_block_0_ffn_fc1", out="bit_block_0_ffn_act",
         shape="btd,df->btf  D=256 -> FFN=1024  + ReLU (SAT-pinned)",
         checks="binary einsum-dense = 0 DSP; ReLU = 0 DSP (must be ap_ufixed<.,.,AP_SAT>, not WRAP); "
                "SubLN(256) = DSP consumer. (The widest fan-out; ~262k MACs.)"),
    dict(name="bitlinear_ffn_fc2", kind="bitlinear", rf=256,
         inp="ln_bit_block_0_ffn_fc2", out="bit_block_0_ffn_fc2_affine",
         shape="btf,fd->btd  FFN=1024 -> D=256",
         checks="binary einsum-dense = 0 DSP; fc2_affine (CSD-2 β) = 0 DSP; SubLN(1024) = DSP "
                "consumer (widest normalize)."),
    dict(name="bitlinear_head_fc1", kind="bitlinear", rf=256,
         inp="ln_head_fc1", out="head_act",
         shape="256 -> 256 (2-D QDense) + ReLU (SAT-pinned)",
         checks="binary dense = 0 DSP; ReLU = 0 DSP (SAT); SubLN(256, 2-D) = DSP consumer."),
    dict(name="bitlinear_head_fc2", kind="bitlinear", rf=256,
         inp="ln_head_fc2", out="head_fc2_affine",
         shape="256 -> 5 (2-D QDense, logits)",
         checks="binary dense = 0 DSP; head_fc2_affine (CSD-2 β) = 0 DSP; SubLN(256, 2-D) = DSP consumer."),
    dict(name="subln_256", kind="subln", rf=256,
         inp="ln_head_fc1", out="ln_head_fc1",
         shape="SubLN dim=256 (folded, range-reduced 1/sqrt)",
         checks="THE known DSP consumer (old fixed<32,16> LayerNorm census: 1,049 DSP for 51 "
                "instances). Isolated here: variance + inv-sqrt path; DSP > 0 EXPECTED and counted "
                "(folded RF=256 << the RF=1 census of 1,792)."),
    dict(name="attn_core_rf64", kind="attn_core", rf=64,
         inp_out=["bit_block_0_attn_Wq", "bit_block_0_attn_Wk", "bit_block_0_attn_Wv"],
         out="bit_block_0_attn_ctx",
         shape="QK^T (einsum) -> stable QSoftmax (β_qβ_k/sqrt(E) in exp LUT) -> attn·V (einsum); "
               "T=10,H=8,E=32; the §6' folded operating point RF=64",
         checks="act×act einsums -> DSPs EXPECTED and COUNTED (not a binary matmul: no weights to "
                "binarize; §6' rf64 measured 820 DSP total, ~400/einsum, softmax ~10). This is the "
                "0.65%-of-MACs piece weight-binarization structurally cannot touch."),
]


def build_flagship_export(checkpoint):
    cfg = load_config(os.path.join(_HERE, "configs", f"final-{FLAGSHIP_VARIANT}.json"))
    kp, _ = cf.fetch_checkpoint(FLAGSHIP_VARIANT, FLAGSHIP_SEED, FLAGSHIP_RUN,
                               os.path.join(PROJECT_ROOT, "bnjettag", "models", "final"),
                               local_checkpoint=checkpoint)
    qat = cf.load_qat_model(kp)
    A = cfg["arch"]
    X, _ = cf._load_jets(os.path.join(PROJECT_ROOT, "data", "val"), A["n_part"], 512)
    binz, pe, names = cf.qat_binz_pe(qat, cfg)
    calib = cf.read_qat_act_ibits(qat, names)
    calib.update(cf.qat_stream_ranges(qat, cfg, binz, X))
    exp = cf.build_export(cfg, binz, calib, pe, beta_mode="csd2")
    return cfg, exp, X, os.path.relpath(kp, PROJECT_ROOT) if kp.startswith(PROJECT_ROOT) else kp


def slice_probe(exp, byn, spec):
    import keras
    if spec["kind"] == "attn_core":
        ins = [byn[n].output for n in spec["inp_out"]]
        return keras.Model(ins, byn[spec["out"]].output, name=f"probe_{spec['name']}")
    return keras.Model(byn[spec["inp"]].input, byn[spec["out"]].output,
                       name=f"probe_{spec['name']}")


def tap_inputs(exp, byn, spec, X, n=64):
    import keras
    if spec["kind"] == "attn_core":
        tap = keras.Model(exp.inputs, [byn[n_].output for n_ in spec["inp_out"]])
        outs = tap.predict(X[:n], verbose=0)
        return [np.ascontiguousarray(np.asarray(o, np.float32)) for o in outs]
    tap = keras.Model(exp.inputs, byn[spec["inp"]].input)
    return np.ascontiguousarray(np.asarray(tap.predict(X[:n], verbose=0), np.float32))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default=None, help="local flagship model_best.keras (skips W&B)")
    ap.add_argument("--n-csim", type=int, default=64)
    ap.add_argument("--only", default=None, help="comma-list of probe names to (re)build")
    a = ap.parse_args()

    cfg, exp, X, ckpt_rel = build_flagship_export(a.checkpoint)
    byn = {l.name: l for l in exp.layers}
    cf.patch_resource_einsum_check()
    from bnhgq2.convert import convert, pack_for_mulder

    out_root = os.path.join(PROJECT_ROOT, "bnjettag", "results", "final", "probes")
    os.makedirs(out_root, exist_ok=True)
    h = cfg_hash(cfg)
    want = set(a.only.split(",")) if a.only else None

    manifest = {"source": {"flagship": FLAGSHIP_RUN, "config": cfg["name"], "config_hash": h,
                           "checkpoint": ckpt_rel,
                           "note": "probes SLICED from build_export(final-w1a8-s3); carry the final "
                                   "stack's exact act grids (KBI via .kif), ReLU-SAT pins, CSD-2 β."},
                "hls": {"part": cfg["hls"].get("part"), "clock_ns": cfg["hls"].get("clock_ns"),
                        "backend": "Vitis", "io": "io_parallel", "strategy": "Latency"},
                "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "probes": {}}
    tarballs = []
    for spec in PROBES:
        if want and spec["name"] not in want:
            continue
        name = spec["name"]
        sub = slice_probe(exp, byn, spec)
        xin = tap_inputs(exp, byn, spec, X, a.n_csim)
        keras_ref = np.asarray(sub(xin if isinstance(xin, list) else xin, training=False))
        proj = os.path.join(out_root, f"probe_{name}_rf{spec['rf']}")
        shutil.rmtree(proj, ignore_errors=True)
        _, report = convert(sub, cfg, proj, rf=spec["rf"], strategy="Latency",
                            csim_X=xin, keras_ref=keras_ref, post_parse=cf.fix_relu_saturation)
        tar = pack_for_mulder(proj, proj + ".tar.gz")
        tarballs.append(os.path.basename(tar))
        cs = report.get("csim", {})
        corr = cs.get("corr")
        ok = bool(corr is not None and corr >= 0.997)
        manifest["probes"][name] = {
            "kind": spec["kind"], "rf": spec["rf"], "strategy": "Latency",
            "shape": spec["shape"], "layers": [l.name for l in sub.layers],
            "csim": {"n": cs.get("n"), "corr": corr, "max_abs_diff": cs.get("max_abs_diff"),
                     "sanity_pass": ok},
            "expected_checks": spec["checks"],
            "tarball": os.path.relpath(tar, PROJECT_ROOT),
        }
        # keep only the tarball (regenerable project dir is large)
        shutil.rmtree(proj, ignore_errors=True)
        print(f"[probe] {name:24s} rf{spec['rf']:<3d} csim corr={corr}  sanity={'PASS' if ok else 'CHECK'}",
              flush=True)

    manifest["mulder_command"] = "./mulder_csynth.sh " + " ".join(sorted(tarballs))
    mpath = os.path.join(out_root, "probes_manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[done] {len(manifest['probes'])} probes -> {os.path.relpath(out_root, PROJECT_ROOT)}")
    print(f"       manifest: {os.path.relpath(mpath, PROJECT_ROOT)}")
    print(f"       mulder:   cd <probes dir on mulder>; {manifest['mulder_command']}")


if __name__ == "__main__":
    main()
