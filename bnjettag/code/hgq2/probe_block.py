#!/usr/bin/env python3
"""probe_block — one full flagship ENCODER BLOCK as a single hls4ml project (Route B).

The deployable fallback to the monolith: synthesize block-0 end-to-end
(SubLN → QKV → attention core → Wo → residual → SubLN → fc1 → ReLU(SAT) → fc2 → residual)
under the SAME hybrid rules as monolith-v2 (weighted layers = PointwiseConv1D at Resource,
weight ROMs, wide accumulator, CSD-2 affine; the weightless QKᵀ/attn·V einsums + QSoftmax at
Latency; SubLN unchanged). ~1/8 of the monolith; 8 instances + input/head compose the whole
model at the RTL/IP level (the standard way large FPGA designs are built).

C-sim gate: hls4ml(block) vs the FLAGSHIP's block-0 on REAL tapped activations
(input = the export's input_proj_affine output; reference = the export's bit_block_0_add_ffn
output) — the same activation-tapping the per-shape probes use. ≥ 0.997.

Deliverable: tarball + manifest + LEDGER note. csynth is mulder-only.

Usage:
  python probe_block.py --checkpoint bnjettag/models/final/w1a8-s3/model_best.keras
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
import probes_final_v2 as pv2  # noqa: E402
from bnhgq2.config import cfg_hash, PROJECT_ROOT  # noqa: E402
from bnhgq2.monolith import (build_block_v2, latency_layer_names,  # noqa: E402
                             per_layer_configs, widen_conv_dense_accum)


def run(config=pv2.FLAGSHIP_CONFIG, checkpoint=None, li=0, rf=None, n_csim=64):
    variant, seed, wandb_run = pv2.resolve_source(config)
    cfg, exp, qat, binz, calib, pe, X, ckpt_rel = pv2.load_model(config, checkpoint)
    h = cfg_hash(cfg)
    L = cfg["arch"]["n_layers"]
    rf = rf or cfg["hls"].get("rf", 256)
    blk = f"bit_block_{li}"
    store = os.path.join(PROJECT_ROOT, "bnjettag", "results", "final", "probes")
    os.makedirs(store, exist_ok=True)
    print(f"=== probe_block {blk}  cfg={cfg['name']} [{h}] rf={rf} ===", flush=True)

    # (tap) the flagship's REAL block input/output on real jets
    import keras
    ebn = {l.name: l for l in exp.layers}
    in_src = "input_proj_affine" if li == 0 else f"bit_block_{li-1}_add_ffn"
    tin = keras.Model(exp.inputs, ebn[in_src].output)
    tout = keras.Model(exp.inputs, ebn[f"{blk}_add_ffn"].output)
    xin = np.ascontiguousarray(np.asarray(tin.predict(X[:n_csim], verbose=0)).astype(np.float32))
    yflag = np.ascontiguousarray(np.asarray(tout.predict(X[:n_csim], verbose=0)).astype(np.float32))

    # (build) block-li standalone, hybrid emission
    block, reg = build_block_v2(cfg, binz, calib, li)
    ykeras = np.asarray(block(xin, training=False)).reshape(yflag.shape)
    repro = float(np.corrcoef(ykeras.ravel(), yflag.ravel())[0, 1])
    print(f"[build] block params={block.count_params():,}  "
          f"keras(block) vs flagship block-{li} corr={repro:.7f} "
          f"max|Δ|={np.abs(ykeras - yflag).max():.3e}", flush=True)

    # (convert) hybrid strategy
    cf.patch_resource_einsum_check()
    from bnhgq2.convert import convert, pack_for_mulder
    lat = [n for n in latency_layer_names(li + 1) if n.startswith(blk)]   # this block only
    present = set(reg["dense"]) | set(lat)
    lcfg = per_layer_configs(cfg, binz, rf, lat, present=present)

    def post_parse(hm):
        post_parse._n = (cf.fix_relu_saturation(hm), widen_conv_dense_accum(hm))
    post_parse._n = (0, 0)

    proj = os.path.join(store, f"probe_block{li}_rf{rf}_hybrid")
    shutil.rmtree(proj, ignore_errors=True)
    # gate the C-sim against the flagship block output (the true deployable reference)
    hm, report = convert(block, cfg, proj, rf=rf, strategy="Resource",
                         csim_X=xin, keras_ref=yflag,
                         post_parse=post_parse, layer_configs=lcfg)
    nre, nwi = post_parse._n
    cs = report.get("csim", {})
    g_pass = cs.get("corr", 0.0) >= 0.997
    print(f"[GATE C-sim] hls4ml(block) vs FLAGSHIP block-{li}: corr={cs.get('corr')} "
          f"max|Δ|={cs.get('max_abs_diff')} relu_sat={nre} wide_accum={nwi} pass={g_pass}",
          flush=True)

    # (inspect)
    fw = os.path.join(proj, "firmware")
    mcpp = open(os.path.join(fw, "myproject.cpp")).read()
    inspect = {
        "nnet_pointwise_conv_calls": mcpp.count("nnet::pointwise_conv_1d"),
        "nnet_einsum_dense_calls": mcpp.count("nnet::einsum_dense"),   # MUST be 0
        "nnet_einsum_calls": mcpp.count("nnet::einsum<"),
        "nnet_softmax_calls": mcpp.count("nnet::softmax"),
        "nnet_subln_calls": mcpp.count("nnet::subln"),
        "weight_roms_load_from_txt": mcpp.count("load_weights_from_txt"),
    }
    print(f"[inspect] pointwise={inspect['nnet_pointwise_conv_calls']} "
          f"einsum_DENSE={inspect['nnet_einsum_dense_calls']} (must=0) "
          f"einsum(act×act)={inspect['nnet_einsum_calls']} subln={inspect['nnet_subln_calls']} "
          f"weight_ROMs={inspect['weight_roms_load_from_txt']}", flush=True)

    tar = pack_for_mulder(proj, proj + ".tar.gz")
    shutil.rmtree(proj, ignore_errors=True)

    manifest = {"written": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "probe": f"block{li}", "config": cfg["name"], "config_hash": h,
                "wandb_run": wandb_run, "variant": variant, "seed": seed,
                "checkpoint": ckpt_rel, "rf": rf,
                "part": cfg["hls"].get("part"), "clock_ns": cfg["hls"].get("clock_ns"),
                "strategy": "Resource(weighted)+Latency(einsum/softmax)",
                "contents": ("SubLN→QKV(pointwise)→QKᵀ→QSoftmax→attn·V→SubLN→Wo(pointwise)→"
                             "residual→SubLN→fc1(pointwise)+ReLU(SAT)→SubLN→fc2(pointwise)→residual"),
                "keras_vs_flagship_block_corr": repro,
                "csim_vs_flagship_block": {"n": cs.get("n"), "corr": cs.get("corr"),
                                           "max_abs_diff": cs.get("max_abs_diff"),
                                           "bit_exact": cs.get("bit_exact"),
                                           "threshold": 0.997, "pass": bool(g_pass)},
                "post_parse": {"relu_sat_patched": nre, "wide_accum_widened": nwi},
                "project_inspection": inspect,
                "compose_note": (f"{L} block instances + input_proj + GAP/head compose the whole "
                                 "model at the RTL/IP level; per-block csynth on mulder gives a "
                                 "measured (not estimated) whole-model resource/latency table."),
                "tarball": os.path.relpath(tar, PROJECT_ROOT),
                "mulder_cmd": f"./mulder_csynth.sh {os.path.basename(tar)}"}
    mpath = os.path.join(store, f"probe_block{li}_manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=1)

    print(f"\n[done] probe_block{li}: C-sim={'PASS' if g_pass else 'CHECK'} "
          f"einsum_DENSE_remaining={inspect['nnet_einsum_dense_calls']}\n"
          f"       manifest -> {os.path.relpath(mpath, PROJECT_ROOT)}\n"
          f"       tarball  -> {os.path.relpath(tar, PROJECT_ROOT)}\n"
          f"       mulder   -> scp {os.path.relpath(tar, PROJECT_ROOT)} mulder:~/ ; "
          f"./mulder_csynth.sh {os.path.basename(tar)}", flush=True)
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=pv2.FLAGSHIP_CONFIG,
                    help="config name in configs/ WITHOUT .json (default: flagship final-w1a8)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--rf", type=int, default=None, help="default: cfg['hls']['rf']")
    ap.add_argument("--n-csim", type=int, default=64)
    a = ap.parse_args()
    run(config=a.config, checkpoint=a.checkpoint, li=a.block, rf=a.rf, n_csim=a.n_csim)


if __name__ == "__main__":
    main()
