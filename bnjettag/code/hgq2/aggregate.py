#!/usr/bin/env python3
"""Stage (g): aggregate the results store into the tradeoff table + ROC figure.

Real rows only. Every number carries its provenance:
  [measured/HGQ2]   — produced by this pipeline (store JSONs)
  [measured/QKeras] — pre-existing verified artifacts (roc-results/r5, results/csynth)
  [blocked]         — stage not run (e.g. csynth awaiting mulder) — printed as such

Outputs: results/hgq2/tradeoff_table.md, results/hgq2/roc_hgq2_overlay.png
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bnhgq2.config import load_config, cfg_hash, resolve, PROJECT_ROOT  # noqa: E402
from bnhgq2 import store  # noqa: E402

CONFIGS = ["era2-large-w1a8", "era2-large-w1a6", "era2-large-w1a4"]
HERE = os.path.dirname(os.path.abspath(__file__))

# fixed categorical assignment (dataviz-validated 2026-07-04); FP32 = neutral reference
COLORS = {"W1A8": "#2a78d6", "W1A6": "#1baf7a", "W1A4": "#eda100", "W8A8": "#4a3aa7"}
REF_COLOR = "#555555"
CLASS_NICE = ["g", "q", "W", "Z", "t"]


def _stage(h, name):
    r = store.read_stage(h, name)
    return r["payload"] if r else None


def collect():
    rows = []
    for cname in CONFIGS:
        cfg = load_config(os.path.join(HERE, "configs", f"{cname}.json"))
        h = cfg_hash(cfg)
        rows.append({
            "config": cname, "hash": h, "cfg": cfg,
            "act_bits": cfg["quant"]["act_bits"],
            "verify": _stage(h, "verify"),
            "ebops": _stage(h, "ebops"),
            "convert": _stage(h, "convert"),
            "probes": {p: _stage(h, f"probe_{p}") for p in ("subln", "bitlinear", "attn_core")},
            "csynth": _collect_csynth(h),
        })
    return rows


def _collect_csynth(h):
    """csynth_report.json files brought back from mulder into the run dir."""
    out = {}
    d = os.path.join(store.STORE, "runs", h)
    if not os.path.isdir(d):
        return out
    for root, _, files in os.walk(d):
        for f in files:
            if f == "csynth_report.json":
                key = os.path.basename(root)
                with open(os.path.join(root, f)) as fh:
                    out[key] = json.load(fh)
    return out


def make_table(rows):
    L = []
    L.append("# HGQ2 rebuild — efficiency vs resource vs latency (era-2 large, D256/L8)\n")
    L.append("All AUCs are **era-2 ROC-test macro-OvR** (held-out val split, n=260,000).")
    L.append("`ref` = the trained QKeras model's verified scores (roc-results/r5, verified 2026-07-03).")
    L.append("HGQ2 rebuild = binary {−1,+1} pinned, static per-channel act quant, fold-aware CSD-2 β̃.\n")
    L.append("| model | AUC ref [measured/QKeras] | AUC HGQ2 rebuild | Δ | corr(scores) | EBOPs (HGQ2-native) | hls4ml C-sim bit-exact |")
    L.append("|---|---|---|---|---|---|---|")
    for r in rows:
        v, e = r["verify"], r["ebops"]
        name = f"W1A{r['act_bits']}"
        if v:
            g1 = v["gate1_vs_trained"]
            csim = (r["convert"] or {}).get("csim", {})
            be = csim.get("bit_exact", "—") if csim else "[blocked: convert not run]"
            L.append(f"| {name} | {g1['auc_macro_ref']:.4f} | {g1['auc_macro']:.4f} | "
                     f"{g1['d_auc']:+.4f} | {g1['corr_scores']:.4f} | "
                     f"{e['total']:,} | {be} |" if e else
                     f"| {name} | {g1['auc_macro_ref']:.4f} | {g1['auc_macro']:.4f} | "
                     f"{g1['d_auc']:+.4f} | {g1['corr_scores']:.4f} | [blocked] | — |")
        else:
            L.append(f"| {name} | — | [blocked: verify not run] | — | — | — | — |")
    L.append("")
    # baselines from the verified npz (never recomputed here — quoted with source)
    L.append("Baselines (same table, source `roc-results/r5/roc_auc.md`, verified 2026-07-03): "
             "FP32 0.8765 · W8A8 0.8642 [measured/QKeras, single-run lr05].\n")
    # csynth section
    L.append("## Real synthesis (Vitis HLS 2023.2, VU13P, mulder)\n")
    any_syn = False
    L.append("| probe | precision | LUT | FF | DSP | BRAM_18K | latency (cycles) | II | est. clock |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        for key, c in sorted(r["csynth"].items()):
            any_syn = True
            L.append(f"| {key} | A{r['act_bits']} | {c['LUT']:,} | {c['FF']:,} | "
                     f"**{c['DSP']}** | {c['BRAM_18K']} | "
                     f"{c['LatencyBest']}–{c['LatencyWorst']} | "
                     f"{c['IntervalMin']} | {c.get('estimated_clock_ns','—')} ns |")
    if not any_syn:
        L.append("| *(none yet — projects packed, csynth pending on mulder)* | | | | | | | | |")
    L.append("")
    L.append("**Per-instance DSP split (from the per-function csynth reports — the numbers "
             "that carry the thesis):** probe_bitlinear_head_fc2_rf32 (Latency, pure ±1 + "
             "CSD-2 affine): binary dense **0 DSP** · affine **0 DSP** · SubLN 112 DSP = "
             "all of the probe's DSPs. probe_bitlinear_rf256/v2 (Resource): dense 256 DSP — "
             "the documented Resource-ROM trap, kept as the negative result; SubLN folded "
             "= 14 DSP there. See constraints_map.md.\n")
    L.append("Prior verified per-shape csynth (QKeras path, results/hls_resource_table.md §B′, "
             "RF=256): matmul cores 0 DSP at A8/A6/A4; all 1,049 model DSPs in the "
             "old fixed<32,16> LayerNorm; composed whole-model latency upper bound "
             "23,409 cycles ≈ 58.5 µs @ 400 MHz (attention score core excluded there — "
             "the HGQ2 attn_core probe closes exactly that gap when its csynth lands).\n")
    return "\n".join(L)


def make_roc_figure(rows, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    # reference curves from stored npz (FP32 + per-variant trained refs)
    fp32 = np.load(os.path.join(PROJECT_ROOT,
                   "bnjettag/roc-results/r5/FP32-lr05.npz"))
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.12, wspace=0.08)

    curves = []  # (label, color, ls, y, score)
    curves.append(("FP32 vanilla (trained ref)", REF_COLOR, "--",
                   fp32["y"], fp32["score"]))
    for r in rows:
        if not r["verify"]:
            continue
        sc = np.load(os.path.join(store.STORE, "runs", r["hash"], "scores.npz"))
        name = f"W1A{r['act_bits']}"
        curves.append((f"{name} HGQ2 rebuild", COLORS[name], "-", sc["y"], sc["score"]))
        ref = np.load(resolve(r["cfg"], "reference_npz"))
        curves.append((f"{name} trained (QKeras)", COLORS[name], ":",
                       ref["y"], ref["score"]))

    for c in range(5):
        ax = axes.flat[c]
        for label, color, ls, y, score in curves:
            fpr, tpr, _ = roc_curve(y[:, c], score[:, c])
            ax.plot(tpr, np.clip(fpr, 1e-5, 1), color=color, ls=ls, lw=1.6,
                    label=label if c == 0 else None)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1)
        ax.set_xlim(0, 1)
        ax.text(0.05, 0.92, f"{CLASS_NICE[c]} vs rest", transform=ax.transAxes,
                fontsize=12, fontweight="bold")
        ax.grid(True, which="both", alpha=0.18, lw=0.5)
        if c >= 2:
            ax.set_xlabel("Signal efficiency (TPR)")
        if c % 3 == 0:
            ax.set_ylabel("Mistag rate (FPR)")
    axes.flat[5].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[5].legend(handles, labels, loc="center", fontsize=10, frameon=False)
    fig.suptitle("BitNet jet tagger — HGQ2 rebuild vs trained model, era-2 val (n=260k), "
                 "HEP convention: linear TPR / log FPR", fontsize=13)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[aggregate] wrote {out_png}")


def main():
    rows = collect()
    os.makedirs(store.STORE, exist_ok=True)
    table = make_table(rows)
    tpath = os.path.join(store.STORE, "tradeoff_table.md")
    with open(tpath, "w") as f:
        f.write(table)
    print(f"[aggregate] wrote {tpath}")
    if any(r["verify"] for r in rows):
        make_roc_figure(rows, os.path.join(store.STORE, "roc_hgq2_overlay.png"))


if __name__ == "__main__":
    main()
