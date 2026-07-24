#!/usr/bin/env python3
"""Generate the publication *results* figures for the BitNet 1-bit jet tagger run.

These are the RESULTS plots (efficiency + firmware), distinct from the four
data/architecture diagnostics already in results/plots/. Three PNGs are emitted:

  results_auc_by_variant.png    - tagging efficiency (val AUC) across precisions:
                                  vanilla FP32 / W8A8 / 1-bit BitNet / ternary / softmax-free,
                                  with the cost of going 1-bit annotated.
  results_quant_axis.png        - the abstract's two axes in one figure: val AUC (top) and
                                  measured FPGA resource (bottom) vs activation bits A8->A6->A4.
  results_csynth_resources.png  - measured Vitis HLS C-synthesis utilisation (% of VU13P) for
                                  the binary FFN block; DSP = 0 is the headline.

Data provenance (single sources of truth, no invented numbers):
  - val AUCs            : results/RESULTS.md  §1a (central comparison), §1b (quant axis),
                          §1c (appendix). Hardcoded below WITH the section cited.
  - FPGA resources      : read live at run time from
                          results/csynth/csynth_report_a{8,6,4}_rf256.json
                          (real Vitis HLS 2023.2 C-synthesis on `mulder`, RF=256,
                          xcvu13p-flga2577-2-e, 400 MHz target, 2026-06-24).
  - VU13P denominators  : LUT 1,728,000 / FF 3,456,000 / DSP 12,288 / BRAM_18K 5,376.

Run (uses the self-contained venv created for plotting):
  .venv-plots/bin/python code/plots/make_results_plots.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless PNG
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # run root (bnjettag/)
CSYNTH_DIR = ROOT / "results" / "csynth"
OUT_DIR = ROOT / "results" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# VU13P (xcvu13p-flga2577-2-e) available resources -- the synthesis target part.
VU13P = {"LUT": 1_728_000, "FF": 3_456_000, "DSP": 12_288, "BRAM_18K": 5_376}

# ----------------------------------------------------------------------------
# Authoritative val AUCs  (source: results/RESULTS.md)
# ----------------------------------------------------------------------------
# Each tuple: (label, val_AUC, category)   category drives colour/legend.
AUC_VARIANTS = [
    ("Vanilla\nFP32",        0.7703, "baseline"),   # RESULTS.md sec1a - full-precision ref
    ("W8A8\n(int8)",         0.7719, "baseline"),   # RESULTS.md sec1a - 8-bit ~lossless
    ("1-bit BitNet\n(W1A8)", 0.7530, "deployed"),   # RESULTS.md sec1a - THE deployed model
    ("Ternary\n{-1,0,+1}",   0.7685, "appendix"),   # RESULTS.md sec1c - appendix
    ("Softmax-free\n(W1A8)", 0.7562, "appendix"),   # RESULTS.md sec1c - appendix
]
FP32_REF = 0.7703  # vanilla full-precision reference line

# Quantization axis, canonical softmax (RESULTS.md sec1b): (activation_bits, val_AUC)
QUANT = [(8, 0.7524), (6, 0.7507), (4, 0.7437)]

# House style ----------------------------------------------------------------
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)
C_BASE = "#5b6770"      # baselines (FP32 / 8-bit)
C_DEPLOY = "#c0392b"    # the deployed 1-bit model (thesis target)
C_APPENDIX = "#8e7cc3"  # appendix variants
C_LUT = "#2c7fb8"
C_FF = "#41b6c4"
C_BRAM = "#fdae61"


# ----------------------------------------------------------------------------
# Load the measured csynth reports
# ----------------------------------------------------------------------------
def load_csynth() -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for ab in (8, 6, 4):
        p = CSYNTH_DIR / f"csynth_report_a{ab}_rf256.json"
        d = json.loads(p.read_text())["row"]
        rows[ab] = {
            "LUT": int(d["LUT"]),
            "FF": int(d["FF"]),
            "BRAM_18K": int(d["BRAM_18K"]),
            "DSP": int(d["DSP"]),
            "lat_cyc": int(d["LatencyCyclesWorst"]),
            "clock_ns": float(d["clock_ns"]),
            "II": int(d["IntervalII"]),
        }
    return rows


def pct(val: int, key: str) -> float:
    return 100.0 * val / VU13P[key]


# ----------------------------------------------------------------------------
# Figure 1 - val AUC across precisions
# ----------------------------------------------------------------------------
def fig_auc_by_variant(path: Path) -> None:
    labels = [v[0] for v in AUC_VARIANTS]
    aucs = [v[1] for v in AUC_VARIANTS]
    cats = [v[2] for v in AUC_VARIANTS]
    cmap = {"baseline": C_BASE, "deployed": C_DEPLOY, "appendix": C_APPENDIX}
    colors = [cmap[c] for c in cats]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    x = np.arange(len(labels))
    ax.bar(x, aucs, width=0.66, color=colors, edgecolor="black", linewidth=0.6, zorder=3)

    # full-precision reference line
    ax.axhline(FP32_REF, ls="--", lw=1.0, color=C_BASE, zorder=2)
    ax.text(len(labels) - 0.55, FP32_REF + 0.0004, "vanilla FP32 ref",
            ha="right", va="bottom", fontsize=8, color=C_BASE)

    # exact value on every bar
    for xi, a in zip(x, aucs):
        ax.text(xi, a + 0.0005, f"{a:.4f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    # cost-of-1-bit gap (FP32 ref -> binary), drawn at the binary bar (x=2)
    ax.annotate("", xy=(2.0, FP32_REF), xytext=(2.0, 0.7530),
                arrowprops=dict(arrowstyle="<->", color=C_DEPLOY, lw=1.4))
    ax.text(2.13, (FP32_REF + 0.7530) / 2, "-1.7 pt", color=C_DEPLOY,
            fontsize=9.5, va="center", fontweight="bold")

    ax.set_ylim(0.735, 0.776)
    ax.set_ylabel("Validation AUC")
    ax.set_title("Tagging efficiency across weight / activation precision\n"
                 "BitNet jet tagger (D256 / H8 / L8 / FFN1024)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.grid(axis="y", ls=":", alpha=0.5, zorder=0)
    ax.text(2.0, 0.7748,
            "y-axis truncated; exact AUC printed on each bar. Source: RESULTS.md sec1.",
            ha="center", va="top", fontsize=7, color="#888", style="italic")

    legend = [
        Patch(facecolor=C_BASE, edgecolor="black", label="baseline (FP32 / 8-bit)"),
        Patch(facecolor=C_DEPLOY, edgecolor="black", label="deployed 1-bit (thesis)"),
        Patch(facecolor=C_APPENDIX, edgecolor="black", label="appendix variant"),
    ]
    ax.legend(handles=legend, fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 2 - quantization axis: AUC (top) + FPGA resource (bottom) vs A-bits
# ----------------------------------------------------------------------------
def fig_quant_axis(path: Path, cs: dict[int, dict]) -> None:
    bits = [q[0] for q in QUANT]
    aucs = [q[1] for q in QUANT]
    lut = [pct(cs[b]["LUT"], "LUT") for b in bits]
    ff = [pct(cs[b]["FF"], "FF") for b in bits]
    xpos = np.arange(len(bits))

    fig, (axT, axB) = plt.subplots(
        2, 1, figsize=(6.8, 5.8), sharex=True,
        gridspec_kw=dict(height_ratios=[1.0, 1.15], hspace=0.10),
    )

    # --- top: AUC ---
    axT.plot(xpos, aucs, "-o", color=C_DEPLOY, lw=2.2, ms=9, zorder=3)
    for xi, a in zip(xpos, aucs):
        axT.text(xi, a + 0.0007, f"{a:.4f}", ha="center", fontsize=9, fontweight="bold")
    axT.set_ylim(0.7395, 0.7555)
    axT.set_ylabel("Validation AUC")
    axT.grid(axis="y", ls=":", alpha=0.5)
    axT.set_title("Quantization axis - efficiency vs FPGA resources\n"
                  "1-bit weights, canonical softmax; activations swept A8 -> A4")
    axT.text(0.02, 0.08, "gentle, monotonic: ~0.9 pt over 8 -> 4 bits (no cliff)",
             transform=axT.transAxes, fontsize=8, color="#666", style="italic")

    # --- bottom: resources ---
    bram = [pct(cs[b]["BRAM_18K"], "BRAM_18K") for b in bits]
    axB.plot(xpos, lut, "-s", color=C_LUT, lw=2.2, ms=8, label="LUT")
    axB.plot(xpos, ff, "-^", color="#1a9850", lw=2.2, ms=8, label="FF")
    axB.plot(xpos, bram, "-D", color="#d6800a", lw=2.0, ms=6, label="BRAM")
    for xi, v in zip(xpos, lut):
        axB.text(xi, v + 0.8, f"{v:.1f}%", ha="center", fontsize=8.5, color=C_LUT, fontweight="bold")
    for xi, v in zip(xpos, ff):
        axB.text(xi, v + 0.9, f"{v:.1f}%", ha="center", fontsize=8.5, color="#1a9850", fontweight="bold")
    axB.text(xpos[0] - 0.05, bram[0] + 1.1, f"BRAM {bram[0]:.2f}%", ha="left",
             fontsize=8, color="#d6800a", fontweight="bold")
    axB.axhline(0.0, color=C_DEPLOY, lw=1.6, ls="--")
    axB.text(1.0, -1.7, "DSP = 0%  (binary -> no multipliers)",
             ha="center", fontsize=9, color=C_DEPLOY, fontweight="bold")
    axB.set_ylim(-3.2, 29)
    axB.set_ylabel("% of VU13P")
    axB.set_xticks(xpos)
    axB.set_xticklabels([f"A{b}\n({b}-bit)" for b in bits])
    axB.set_xlabel("activation precision")
    axB.grid(axis="y", ls=":", alpha=0.5)
    axB.legend(fontsize=8.5, loc="center right", framealpha=0.92)
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 3 - measured csynth utilisation (% VU13P), DSP = 0 headline
# ----------------------------------------------------------------------------
def fig_csynth_resources(path: Path, cs: dict[int, dict]) -> None:
    res_keys = ["LUT", "FF", "BRAM_18K", "DSP"]
    res_lbls = ["LUT", "FF", "BRAM", "DSP"]
    bits = [8, 6, 4]
    bar_c = {8: "#2c7fb8", 6: "#41b6c4", 4: "#a1dab4"}

    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    x = np.arange(len(res_keys))
    w = 0.25
    for i, b in enumerate(bits):
        vals = [pct(cs[b][k], k) for k in res_keys]
        xb = x + (i - 1) * w
        ax.bar(xb, vals, width=w, color=bar_c[b], edgecolor="black", linewidth=0.5,
               label=f"A{b}", zorder=3)
        for xi, v in zip(xb, vals):
            txt = "0" if v == 0 else (f"{v:.2f}" if v < 2 else f"{v:.1f}")
            ax.text(xi, v + 0.3, txt, ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(res_lbls)
    ax.set_ylabel("% of VU13P utilised")
    ax.set_ylim(0, 29)
    ax.set_title("Measured Vitis HLS C-synthesis - binary FFN block\n"
                 "RF=256, xcvu13p @ 400 MHz target  (2026-06-24, `mulder`)")
    ax.grid(axis="y", ls=":", alpha=0.5, zorder=0)
    ax.legend(title="activation", fontsize=8, ncol=3, loc="upper right", framealpha=0.92)

    # DSP = 0 callout (DSP is the rightmost group, x=3, all zero)
    ax.annotate("DSP = 0\nthe structural\nbinary win", xy=(3.0, 0.3), xytext=(2.25, 11.5),
                ha="center", fontsize=9.5, color=C_DEPLOY, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_DEPLOY, lw=1.5))

    # latency note box
    lat_us = cs[8]["lat_cyc"] * cs[8]["clock_ns"] / 1000.0
    ax.text(0.27, 0.60,
            f"latency {cs[8]['lat_cyc']} cycles  ~ {lat_us:.2f} us @ 400 MHz\nII = {cs[8]['II']} cycles",
            transform=ax.transAxes, fontsize=8.5, color="#222", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", fc="#f3f3f3", ec="#bbb"))
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main() -> None:
    cs = load_csynth()
    print("loaded csynth:", {b: {"LUT%": round(pct(cs[b]["LUT"], "LUT"), 2),
                                 "DSP": cs[b]["DSP"]} for b in (8, 6, 4)})
    f1 = OUT_DIR / "results_auc_by_variant.png"
    f2 = OUT_DIR / "results_quant_axis.png"
    f3 = OUT_DIR / "results_csynth_resources.png"
    fig_auc_by_variant(f1)
    fig_quant_axis(f2, cs)
    fig_csynth_resources(f3, cs)
    for f in (f1, f2, f3):
        print("wrote", f.relative_to(ROOT), f"({f.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
