#!/usr/bin/env python3
"""Pipeline CLI: python run_stage.py <stage|all> --config configs/X.json [options]

Stages:
  train     — FINAL campaign: native HGQ2 QAT from scratch (needs a config with a
              "train" block). CLI: train --config configs/final-<variant>.json
              --seed <n> [--smoke].  Writes model_best.keras + train_meta.json.
  extract   — checkpoint → weights + PE + binarization summary (STOP if sign-zeros)
  calibrate — gold-model MSE-per-channel activation calibration → calib.npz
  build     — HGQ2 model, port weights, per-channel i-bits; save summary
  verify    — Gate 2 (HGQ2 ↔ gold) on a subset, then Gate 1 (HGQ2 ↔ trained npz)
              on the FULL val split; writes scores npz for ROC plots
  ebops     — native HGQ2 EBOPs (trace_minmax)
  convert   — hls4ml Vitis project + local C-sim bit-exactness (Gate 3) + tarball
  all       — everything above in order

Every stage writes results/hgq2/runs/<hash>/<stage>.json via bnhgq2.store.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bnhgq2.config import load_config, cfg_hash, resolve, PROJECT_ROOT  # noqa: E402
from bnhgq2 import store  # noqa: E402
from bnhgq2.compat import apply_keras_compat  # noqa: E402

apply_keras_compat()  # keras-3.15 EinsumDense.full_output_shape (EBOPs needs it)


def _run_dir(h):
    d = os.path.join(store.STORE, "runs", h)
    os.makedirs(d, exist_ok=True)
    return d


def stage_train(cfg, h, ctx, seed=1, smoke=False):
    from bnhgq2.train import train
    if "train" not in cfg:
        raise SystemExit(f"{cfg['name']}: not a training config (no 'train' block)")
    out_dir = ctx.get("out_dir") or os.path.join(
        os.environ.get("BNHGQ2_OUT_ROOT", os.path.join(PROJECT_ROOT, "bnjettag", "models", "final")),
        f"{_variant(cfg)}-s{seed}")
    meta = train(cfg, seed=seed, out_dir=out_dir, smoke=smoke, cfg_hash=h)
    # model_best.keras + train_meta.json (PVC + W&B) are the source of truth; the store
    # JSON is a local-analysis convenience and must never fail the training run.
    try:
        store.write_stage(cfg, h, "train", {**meta, "seed": int(seed), "out_dir": out_dir})
    except Exception as e:
        print(f"[train] store.write_stage skipped ({e})", flush=True)


def _variant(cfg):
    w = cfg["quant"]["weight"]
    return "fp32" if w == "none" else "w8a8" if w == "int8_absmax" else f"w1a{cfg['quant']['act_bits']}"


def stage_extract(cfg, h, ctx):
    from bnhgq2.extract import extract_checkpoint, layer_names
    from bnhgq2.binarize import binarize_checkpoint
    from bnhgq2.gold import csd2_snap

    ck = extract_checkpoint(resolve(cfg, "checkpoint"))
    binz = binarize_checkpoint(ck["layers"])
    names = layer_names(cfg["arch"]["n_layers"])
    missing = [n for n in names if n not in ck["layers"]]
    zeros = binz["_summary"]["total_sign_zeros"]
    if missing or zeros:
        raise SystemExit(f"STOP: missing layers {missing} / sign-zeros {zeros} — "
                         "binary {-1,+1} claim would be void")
    payload = {
        "checkpoint": cfg["checkpoint"],
        "n_bitlinear": len(names),
        "sign_zeros": zeros,
        "pe_shape": list(ck["pos_encoding"].shape),
        "betas": {n: binz[n]["beta"] for n in names},
        "beta_csd2": {n: csd2_snap(binz[n]["beta"]) for n in names
                      if binz[n]["fold"] == "explicit"},
        "folds": {n: binz[n]["fold"] for n in names},
    }
    store.write_stage(cfg, h, "extract", payload)
    ctx.update(ck=ck, binz=binz)
    print(f"[extract] {len(names)} BitLinears, 0 sign-zeros, PE {ck['pos_encoding'].shape}")


def _load_data(cfg, ctx):
    if "X" not in ctx:
        from bnhgq2.data import load_eval_set
        ctx["X"], ctx["y"] = load_eval_set(resolve(cfg, "eval_data"),
                                           n_part=cfg["arch"]["n_part"])
    return ctx["X"], ctx["y"]


def stage_calibrate(cfg, h, ctx):
    from bnhgq2.data import calibration_batch
    from bnhgq2.gold import GoldModel

    X, _ = _load_data(cfg, ctx)
    Xc = calibration_batch(X, cfg["quant"].get("calib_n", 8192))
    gm = GoldModel(cfg, ctx["binz"], ctx["ck"]["pos_encoding"],
                   mode="static", beta_mode="fold_csd2")
    calib = gm.calibrate(Xc, policy="mse_per_channel")
    # per-block score ranges for the softmax exp-table grid (v2 fix: QSoftmax
    # defaults are WRAP/uncalibrated); taps run with the final calib active
    gm.calib = calib
    gm.forward(Xc[:2048], taps=True)
    for li in range(cfg["arch"]["n_layers"]):
        blk = f"bit_block_{li}"
        # gold taps hold SCALED scores; the exp-table grid sees raw QK^T
        raw_max = float(np.abs(gm._taps[f"{blk}_scores"]).max() / gm.score_scale(blk))
        calib[f"__scores_max_{blk}"] = raw_max
        # Q/K/V stream ranges for the exact-passthrough einsum input grids
        for w in ("Wq", "Wk", "Wv"):
            calib[f"__stream_max_{blk}_attn_{w}"] = \
                float(np.abs(gm._taps[f"{blk}_attn_{w}"]).max())
    # explicit-site pre-affine ranges: z = (y − b)/β̃ bounded by (|y|max+|b|max)/β̃
    from bnhgq2.gold import csd2_snap
    for name, e in ctx["binz"].items():
        if name.startswith("_") or e["fold"] != "explicit":
            continue
        bmax = float(np.abs(e["bias"]).max()) if e["bias"] is not None else 0.0
        ymax = float(np.abs(gm._taps[name]).max())
        calib[f"__stream_max_{name}"] = (ymax + bmax) / csd2_snap(e["beta"])
    np.savez(os.path.join(_run_dir(h), "calib.npz"),
             **{k: np.asarray(v) for k, v in calib.items()})
    ctx["calib"] = calib
    isummary = {k: [int(np.min(v)), int(np.max(v))] for k, v in calib.items()}
    store.write_stage(cfg, h, "calibrate", {
        "policy": "mse_per_channel", "n_calib": int(len(Xc)),
        "i_range_per_site": isummary,
    })
    print(f"[calibrate] {len(calib)} sites, per-channel MSE policy")


def _ensure_calib(cfg, h, ctx):
    if "calib" not in ctx:
        p = os.path.join(_run_dir(h), "calib.npz")
        if os.path.exists(p):
            z = np.load(p)
            ctx["calib"] = {k: z[k] for k in z.files}
        else:
            stage_calibrate(cfg, h, ctx)
    return ctx["calib"]


def stage_build(cfg, h, ctx):
    from bnhgq2.build import build_hgq2_model
    from bnhgq2.port import port_weights, assign_per_channel_ibits

    calib = _ensure_calib(cfg, h, ctx)
    model = build_hgq2_model(cfg, ctx["binz"], calib)
    n = port_weights(model, cfg, ctx["binz"], ctx["ck"]["pos_encoding"])
    applied = assign_per_channel_ibits(model, calib, cfg["quant"]["act_bits"])
    ctx["model"] = model
    ctx["applied"] = applied
    n_pc = sum(1 for v in applied.values() if v == "per_channel")
    store.write_stage(cfg, h, "build", {
        "n_ported": n, "params": int(model.count_params()),
        "iq_granularity": applied,
        "n_per_channel": n_pc,
    })
    print(f"[build] ported {n} layers, {n_pc}/{len(applied)} sites per-channel i-bits")


def stage_verify(cfg, h, ctx, subset=20000):
    from bnhgq2.verify import gate1_vs_trained, gate2_vs_gold

    X, y = _load_data(cfg, ctx)
    calib = ctx["calib"]
    # mirror what the model actually applies
    applied = ctx.get("applied", {})
    calib_mirror = {k: (v if applied.get(k) == "per_channel" else int(np.max(v)))
                    for k, v in calib.items()}
    g2 = gate2_vs_gold(ctx["model"], cfg, ctx["binz"], ctx["ck"]["pos_encoding"],
                       calib_mirror, X[:4096])
    print(f"[verify] gate2 HGQ2↔gold: corr={g2['corr_logits']:.6f} "
          f"max|Δ|={g2['max_abs_diff']:.4f} pass={g2['pass']}")
    g1, scores = gate1_vs_trained(ctx["model"], X, y, resolve(cfg, "reference_npz"))
    print(f"[verify] gate1 HGQ2↔trained (n={g1['n']}): corr={g1['corr_scores']:.6f} "
          f"AUC {g1['auc_macro']:.5f} vs ref {g1['auc_macro_ref']:.5f} "
          f"(Δ {g1['d_auc']:+.5f})")
    np.savez_compressed(os.path.join(_run_dir(h), "scores.npz"),
                        y=y[:g1["n"]], score=scores,
                        meta=str({"config": cfg["name"], "hash": h,
                                  "auc_macro": g1["auc_macro"]}))
    store.write_stage(cfg, h, "verify", {"gate2_vs_gold": g2, "gate1_vs_trained": g1})


def stage_ebops(cfg, h, ctx):
    from bnhgq2.data import calibration_batch
    from bnhgq2.ebops_calc import compute_ebops

    X, _ = _load_data(cfg, ctx)
    Xc = calibration_batch(X, 4096)
    res = compute_ebops(ctx["model"], Xc)
    store.write_stage(cfg, h, "ebops", res)
    print(f"[ebops] total = {res['total']:,}")


def stage_convert(cfg, h, ctx, rf=None):
    from bnhgq2.convert import convert, pack_for_mulder
    from bnhgq2.verify import predict_in_batches

    X, _ = _load_data(cfg, ctx)
    out = os.path.join(_run_dir(h), f"hls_prj_rf{rf or cfg['hls'].get('rf', 1)}")
    csim_X = X[:512]
    keras_ref = predict_in_batches(ctx["model"], csim_X)
    hm, report = convert(ctx["model"], cfg, out, rf=rf, csim_X=csim_X,
                         keras_ref=keras_ref)
    tar = pack_for_mulder(out, out + ".tar.gz")
    report["tarball"] = os.path.relpath(tar, PROJECT_ROOT)
    store.write_stage(cfg, h, "convert", report)
    cs = report.get("csim", {})
    print(f"[convert] {out}\n  csim: bit_exact={cs.get('bit_exact')} "
          f"max|Δ|={cs.get('max_abs_diff')} corr={cs.get('corr')}")


STAGES = {
    "train": stage_train,
    "extract": stage_extract,
    "calibrate": stage_calibrate,
    "build": stage_build,
    "verify": stage_verify,
    "ebops": stage_ebops,
    "convert": stage_convert,
}
ORDER = ["extract", "calibrate", "build", "verify", "ebops", "convert"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES) + ["all"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--rf", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-convert", action="store_true")
    # Optional explicit checkpoint/output directory for the `train` stage. Absent (None)
    # reproduces the historical behavior exactly (BNHGQ2_OUT_ROOT/<variant>-s<seed>); the
    # round-7 matrix passes it so checkpoints land under <size>-<variant>-s<seed>/ (the
    # per-size axis needs a name the <variant>-only default cannot express). Additive and
    # reversible — the final campaign never passes it.
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    h = cfg_hash(cfg)
    print(f"=== {cfg['name']} [{h}] ===")
    ctx = {}
    if a.out_dir:
        ctx["out_dir"] = a.out_dir

    # TRAIN is a standalone from-scratch stage (no extract/calibrate/build preamble)
    if a.stage == "train":
        stage_train(cfg, h, ctx, seed=a.seed, smoke=a.smoke)
        return

    stages = ORDER if a.stage == "all" else [a.stage]
    if a.stage != "all":
        # earlier stages are cheap and idempotent; rebuild the in-memory ctx
        for s in ORDER[:ORDER.index(a.stage)]:
            if s in ("extract", "calibrate", "build"):
                STAGES[s](cfg, h, ctx)
    for s in stages:
        if s == "convert" and a.skip_convert:
            continue
        if s == "convert":
            STAGES[s](cfg, h, ctx, rf=a.rf)
        else:
            STAGES[s](cfg, h, ctx)


if __name__ == "__main__":
    main()
