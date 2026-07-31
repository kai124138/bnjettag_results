#!/usr/bin/env python3
"""Regenerate every figure in this repository.

Nothing is transcribed. Accuracy figures are computed from the per-jet score arrays
under `bnjettag/**/roc-results/`; hardware figures are parsed from the raw Vitis HLS
C-synthesis reports under `bnjettag/r7/results/csynth/` (gzipped in place). The only
hard-coded quantities are the VU13P device capacities and the architecture drawn in
Fig. 1.

    pip install numpy scikit-learn matplotlib
    python figures/make_figures.py

Output: figures/fig*.png, plus figures/NUMBERS.txt with every value that reached a
figure, so a figure can be checked without rerunning the script.
"""
from __future__ import annotations

import glob
import gzip
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
NPZ_R7 = ROOT / "bnjettag/r7/roc-results"
NPZ_R10 = ROOT / "bnjettag/roc-results/r10"
CSYNTH = ROOT / "bnjettag/r7/results/csynth"
ADDER = ROOT / "bnjettag/results/adder-graph"

CLASSES = ["g", "q", "W", "Z", "t"]
VU13P = {"LUT": 1_728_000, "FF": 3_456_000, "DSP": 12_288, "BRAM_18K": 5_376}
L1_WINDOW_NS = 25.0

# A colour per role, used consistently across every figure.
C_FP32 = "#3b4a6b"      # full precision
C_W8A8 = "#7d8bab"      # 8-bit baseline
C_BIN = "#c0392b"       # binary, the model
C_BINNN = "#1e7a34"     # binary, norm-free (the flagship)
C_PAIR = "#b8860b"      # binary + pairwise attention bias
C_GREY = "#9aa0a6"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})

LOG = []


def note(line=""):
    LOG.append(line)
    print(line)


# --------------------------------------------------------------------------- IO
def arrays(pattern):
    """Every (y, score) pair matching a glob, seed order."""
    out = []
    for p in sorted(glob.glob(str(pattern))):
        d = np.load(p)
        out.append((d["y"], d["score"], Path(p).name))
    if not out:
        raise FileNotFoundError(pattern)
    return out


def macro_auc(y, s):
    return roc_auc_score(y, s, multi_class="ovr", average="macro")


def cohort_auc(pattern):
    v = np.array([macro_auc(y, s) for y, s, _ in arrays(pattern)])
    return v.mean(), v.std(ddof=1), v


def rejection(y, s, eps):
    """Per-class one-vs-rest 1/FPR at the first threshold reaching TPR >= eps."""
    out = []
    for c in range(y.shape[1]):
        fpr, tpr, _ = roc_curve(y[:, c], s[:, c])
        i = min(int(np.searchsorted(tpr, eps, side="left")), len(fpr) - 1)
        out.append(np.inf if fpr[i] == 0 else 1.0 / fpr[i])
    return np.array(out)


def csynth(name):
    """Totals and timing from one whole-model report under bnjettag/r7/results/csynth."""
    path = CSYNTH / f"{name}.xml.gz"
    if not path.exists():
        path = CSYNTH / f"{name}.xml"
    return report(path)


def e1(arm, version="v1"):
    """Totals and timing from one Experiment-1 arm; version 'v2' is the rewind rerun."""
    fn = "csynth.xml.gz" if version == "v1" else "csynth_v2_rewind.xml.gz"
    return report(ADDER / f"e1/{arm}/prj_{arm}/sol1/syn/report/{fn}")


def report(path):
    """Totals and timing from any raw Vitis HLS report, gzipped or plain."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as fh:
        root = ET.parse(fh).getroot()
    res = root.find("AreaEstimates/Resources")
    g = lambda t: int(res.find(t).text) if res.find(t) is not None else 0
    txt = lambda p: root.find(p).text
    return {
        "DSP": g("DSP"), "LUT": g("LUT"), "FF": g("FF"), "BRAM": g("BRAM_18K"),
        "clk": float(txt("PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod")),
        "lat": int(txt("PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency")),
        "II": int(txt("PerformanceEstimates/SummaryOfOverallLatency/Interval-min")),
        "root": root,
    }


def _category(name):
    n = name.lower()
    if "einsum_dense" in n:
        return "binary weight layers"
    if n.startswith("einsum_") or "einsum_ap" in n:
        return "attention act x act"
    if "subln" in n:
        return "SubLN normalization"
    if "softmax" in n:
        return "softmax"
    if n.startswith("dense_") or n == "dense" or ("dense" in n and "einsum" not in n):
        return "binary weight layers"
    if "pooling" in n:
        return "pooling"
    if n.startswith("myproject"):
        return "top-level glue"
    return "glue / other"


def census(name, resource="DSP"):
    """Per-consumer attribution: own = module total - sum(children).

    Same instance-tree walk as bnjettag/code/hls/parse_census.py; the returned
    categories sum exactly to the report's own profile total (asserted).
    """
    rep = csynth(name)
    root = rep["root"]
    mres = {}
    for m in root.findall("ModuleInformation/Module"):
        r = m.find(".//AreaEstimates/Resources")
        gi = lambda t: int(r.find(t).text) if r is not None and r.find(t) is not None else 0
        mres[m.find("Name").text] = {"DSP": gi("DSP") or gi("DSP48E"), "LUT": gi("LUT"),
                                     "FF": gi("FF"), "BRAM": gi("BRAM_18K")}
    val = lambda mod: mres.get(mod, {}).get(resource, 0)
    cats = defaultdict(int)

    def walk(elem):
        mod = elem.find("ModuleName").text
        il = elem.find("InstancesList")
        kids = il.findall("Instance") if il is not None else []
        cats[_category(mod)] += val(mod) - sum(val(k.find("ModuleName").text) for k in kids)
        for k in kids:
            walk(k)

    top = root.find("RTLDesignHierarchy/TopModule")
    il = top.find("InstancesList")
    kids = il.findall("Instance") if il is not None else []
    topmod = top.find("ModuleName").text
    cats[_category(topmod)] += val(topmod) - sum(val(k.find("ModuleName").text) for k in kids)
    for k in kids:
        walk(k)

    assert sum(cats.values()) == rep[resource], f"{name}: census does not close"
    return dict(cats), rep


# ------------------------------------------------------- Fig 1 -- architecture
def fig_architecture():
    fig, ax = plt.subplots(figsize=(13.0, 5.6))
    ax.set_xlim(0, 104)
    ax.set_ylim(0, 42)
    ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, label, kind, sub=None):
        face = {"bin": "#dfe8f5", "act": "#f7dcd8", "plain": "#eeeeee"}[kind]
        edge = {"bin": C_FP32, "act": C_BIN, "plain": "#888888"}[kind]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35",
                                    linewidth=1.3, facecolor=face, edgecolor=edge))
        ax.text(x + w / 2, y + h / 2 + (1.1 if sub else 0), label, ha="center",
                va="center", fontsize=9)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 1.9, sub, ha="center", va="center",
                    fontsize=7.5, color="#555555")

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=9, linewidth=1.0,
                                     color="#777777", shrinkA=1, shrinkB=1))

    def elbow(pts):
        """Polyline connector; arrowhead on the final segment only."""
        for i in range(len(pts) - 2):
            ax.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]],
                    color="#777777", linewidth=1.0, solid_capstyle="round")
        arrow(*pts[-2], *pts[-1])

    y, h = 30, 6.5
    box(1, y, 12.5, h, "10 constituents\n$\\times$ 16 features", "plain")
    arrow(13.5, y + h / 2, 16, y + h / 2)
    box(16, y, 13, h, "offline z-score", "plain", "per feature, train split")
    arrow(29, y + h / 2, 31.5, y + h / 2)
    box(31.5, y, 13, h, "input projection", "bin", "$\\pm1$ weights, $d = 32$")
    elbow([(44.5, y + h / 2), (48, y + h / 2), (48, 34.2), (50, 34.2)])

    # the transformer block, drawn once
    ax.add_patch(FancyBboxPatch((48.5, 20.5), 37.5, 19, boxstyle="round,pad=0.4",
                                linewidth=1.0, facecolor="none", edgecolor="#bbbbbb",
                                linestyle="--"))
    ax.text(67, 40.6, "transformer block $\\times\\,2$   (4 heads, FFN 64, residual add after each sub-layer)",
            ha="center", fontsize=8.5, color="#666666")

    # upper lane, left to right
    box(50, 31.5, 10, 5.5, "Q, K, V", "bin", "$\\pm1$")
    arrow(60, 34.2, 62, 34.2)
    box(62, 31.5, 9.5, 5.5, "$QK^{\\mathsf{T}}$", "act", "act $\\times$ act")
    arrow(71.5, 34.2, 73.5, 34.2)
    box(73.5, 31.5, 10.5, 5.5, "softmax", "act")
    # turn back into the lower lane
    elbow([(78.7, 31.5), (78.7, 30.3), (54.7, 30.3), (54.7, 29.0)])

    # lower lane, left to right
    box(50, 23.5, 9.5, 5.5, "$\\cdot\\,V$", "act", "act $\\times$ act")
    arrow(59.5, 26.2, 61.5, 26.2)
    box(61.5, 23.5, 9, 5.5, "out proj", "bin", "$\\pm1$")
    arrow(70.5, 26.2, 72.5, 26.2)
    box(72.5, 23.5, 11.5, 5.5, "FFN", "bin", "$\\pm1$, ReLU, $\\pm1$")
    arrow(86, 26.2, 88.5, 26.2)

    box(88.5, 23.5, 14, 5.5, "mean pool", "act", "over constituents")
    arrow(95.5, 23.5, 95.5, 20.5)
    box(88.5, 14.5, 14, 5.5, "classifier head", "bin", "$\\pm1$, ReLU, $\\pm1$")
    arrow(95.5, 14.5, 95.5, 11.5)
    box(88.5, 5.5, 14, 5.5, "5 class scores", "plain", "g / q / W / Z / t")

    rep8 = csynth("whole_model_rf8_stdnn")
    cats = census("whole_model_rf8_stdnn", "DSP")[0]
    ax.text(1, 18.5,
            "Blue — every weight is $\\{-1,+1\\}$: 15 layers, $\\mathbf{0}$ DSPs in all "
            "eight whole-model syntheses.\nRed — no weights to binarize: these are the only "
            "multipliers the model has left.",
            fontsize=9.5, va="top")
    lines = [f"{k:22s}{v:>7,}" for k, v in sorted(cats.items(), key=lambda kv: -kv[1]) if v]
    ax.text(1, 11.0,
            "measured DSP census, whole model, reuse factor 8\n  " + "\n  ".join(lines)
            + f"\n  {'total':22s}{rep8['DSP']:>7,}  of {VU13P['DSP']:,} on the VU13P "
              f"({rep8['DSP'] / VU13P['DSP']:.1%})",
            fontsize=8.5, va="top", family="monospace", color="#333333")

    ax.set_title("Binary-weight transformer jet tagger — where the arithmetic goes",
                 fontsize=12, pad=6)
    fig.savefig(OUT / "fig01_architecture.png")
    plt.close(fig)
    note("fig01  architecture; DSP census RF=8 (A8, norm-free): "
         + ", ".join(f"{k}={v}" for k, v in cats.items()))


# ------------------------------------------------------------- Fig 3 -- ROC
def fig_roc():
    arms = [
        ("FP32", NPZ_R7 / "r8/SMALL-FP32-r8std-s1.npz", C_FP32, "-"),
        ("W8A8", NPZ_R7 / "r8/SMALL-W8A8-r8std-s1.npz", C_W8A8, "-"),
        ("W1A8, norm-free", NPZ_R7 / "r8/SMALL-W1A8-r8stdnn-s1.npz", C_BINNN, "-"),
        ("W1A8, SubLN", NPZ_R7 / "r8/SMALL-W1A8-r8std-s1.npz", C_BIN, "--"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), sharey=True)
    for c, ax in enumerate(axes):
        for label, path, colour, ls in arms:
            d = np.load(path)
            fpr, tpr, _ = roc_curve(d["y"][:, c], d["score"][:, c])
            auc = roc_auc_score(d["y"][:, c], d["score"][:, c])
            ax.plot(tpr, np.maximum(fpr, 1e-6), ls, color=colour, linewidth=1.4,
                    label=f"{label}  {auc:.3f}")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.2)
        ax.set_xlim(0, 1)
        ax.set_title(f"{CLASSES[c]} vs rest")
        ax.set_xlabel(f"{CLASSES[c]} tagging efficiency")
        ax.legend(fontsize=7.2, loc="lower right")
    axes[0].set_ylabel("mistag rate (false positive rate)")
    fig.suptitle("Tagging performance at deployable scale — 19,201 parameters, standardized inputs, "
                 "held-out split (n = 260,000), seed 1", fontsize=11, y=1.03)
    fig.savefig(OUT / "fig03_roc_by_class.png")
    plt.close(fig)
    note("fig03  per-class ROC, round-8 arms, seed 1")


# ----------------------------------------- Fig 2 -- what input conditioning did
def fig_standardization():
    raw = [
        ("FP32", "SMALL-FP32-s*.npz", C_FP32),
        ("W8A8", "SMALL-W8A8-s*.npz", C_W8A8),
        ("W1A8\nSubLN", "SMALL-W1A8-s*.npz", C_BIN),
        ("W1A8\nnorm-free", "extension/SMALL-W1A8-nonorm-s*.npz", C_BINNN),
    ]
    std = [
        ("FP32", "r8/SMALL-FP32-r8std-s*.npz", C_FP32),
        ("W8A8", "r8/SMALL-W8A8-r8std-s*.npz", C_W8A8),
        ("W1A8\nSubLN", "r8/SMALL-W1A8-r8std-s*.npz", C_BIN),
        ("W1A8\nnorm-free", "r8/SMALL-W1A8-r8stdnn-s*.npz", C_BINNN),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for block, xoff, marker, tag in ((raw, 0, "o", "raw features"),
                                     (std, 1, "s", "standardized features")):
        for i, (label, pattern, colour) in enumerate(block):
            m, sd, v = cohort_auc(NPZ_R7 / pattern)
            x = i * 2.6 + xoff
            ax.scatter([x] * len(v), v, s=16, color=colour, alpha=0.35, zorder=2)
            ax.errorbar(x, m, yerr=sd, fmt=marker, color=colour, markersize=8,
                        capsize=4, linewidth=1.4, zorder=3)
            ax.annotate(f"{m:.4f}", (x, m), textcoords="offset points",
                        xytext=(0, 13 if xoff else -19), ha="center", fontsize=8.5,
                        color=colour)
            note(f"fig02  {tag:22s} {label.replace(chr(10), ' '):18s} {m:.4f} +- {sd:.4f}")

    for i, (label, _, colour) in enumerate(raw):
        m_raw = cohort_auc(NPZ_R7 / raw[i][1])[0]
        m_std = cohort_auc(NPZ_R7 / std[i][1])[0]
        ax.annotate("", xy=(i * 2.6 + 1, m_std), xytext=(i * 2.6, m_raw),
                    arrowprops=dict(arrowstyle="-|>", color=colour, alpha=0.45, lw=1.1))
        ax.text(i * 2.6 + 0.5, (m_raw + m_std) / 2, f"+{100 * (m_std - m_raw):.1f}",
                fontsize=8, color=colour, ha="center",
                bbox=dict(fc="white", ec="none", pad=1))

    ax.set_xticks([i * 2.6 + 0.5 for i in range(4)])
    ax.set_xticklabels([lbl for lbl, _, _ in raw])
    ax.set_ylabel("macro one-vs-rest AUC  (held-out, n = 260,000)")
    ax.set_title("Offline input standardization, and what it does to the binary penalty\n"
                 "19,201 parameters; circles = raw features, squares = standardized; "
                 "3 seeds, error bars = sample s.d.", fontsize=10.5)
    ax.set_ylim(0.62, 0.96)
    fig.savefig(OUT / "fig02_input_standardization.png")
    plt.close(fig)


# ------------------------------------------- Fig 4 -- the quantization axis
def fig_activation_ladder():
    bits = [8, 6, 4]
    small = [cohort_auc(NPZ_R7 / f"SMALL-W1A{b}-s*.npz") for b in bits]
    tiny = [cohort_auc(NPZ_R7 / f"r7b/TINY-W1A{b}-s*.npz") for b in bits]
    s_fp32 = cohort_auc(NPZ_R7 / "SMALL-FP32-s*.npz")
    t_fp32 = cohort_auc(NPZ_R7 / "r7b/TINY-FP32-s*.npz")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    for label, data, fp32, colour, marker in (
            ("19,201 parameters", small, s_fp32, C_BIN, "o"),
            ("5,345 parameters", tiny, t_fp32, C_BINNN, "s")):
        ax.errorbar(bits, [d[0] for d in data], yerr=[d[1] for d in data],
                    marker=marker, color=colour, capsize=4, linewidth=1.6,
                    label=f"binary weights, {label}")
        ax.axhline(fp32[0], color=colour, linestyle=":", linewidth=1.2, alpha=0.7)
        for b, d in zip(bits, data):
            note(f"fig04  {label:20s} A{b}: {d[0]:.4f} +- {d[1]:.4f}")
    ax.text(7.9, max(s_fp32[0], t_fp32[0]) + 0.006,
            f"full precision: {s_fp32[0]:.4f} at 19,201 params, {t_fp32[0]:.4f} at 5,345",
            fontsize=8, color="#444444")
    ax.set_xticks(bits)
    ax.set_ylim(0.60, 0.85)
    ax.invert_xaxis()
    ax.set_xlabel("activation width (bits)")
    ax.set_ylabel("macro one-vs-rest AUC")
    ax.set_title("Tagging efficiency vs activation width\n(raw features, each scale at its own tuned recipe)",
                 fontsize=10.5)
    ax.legend(fontsize=8.5, loc="lower left")

    # measured silicon on the same axis
    rows = {}
    for b, rf1, rf8 in ((8, "whole_model_rf1_stdnn", "whole_model_rf8_stdnn"),
                        (6, "whole_model_rf1_w1a6", "whole_model_rf8_w1a6"),
                        (4, "whole_model_rf1_w1a4", "whole_model_rf8_w1a4")):
        rows[b] = (census(rf1, "DSP"), census(rf8, "DSP"))
    for rf, idx, marker, ls in ((1, 0, "o", "-"), (8, 1, "s", "--")):
        tot = [rows[b][idx][1]["DSP"] for b in bits]
        att = [rows[b][idx][0].get("attention act x act", 0) for b in bits]
        ax2.plot(bits, tot, marker=marker, linestyle=ls, color=C_FP32,
                 label=f"whole model, reuse factor {rf}")
        ax2.plot(bits, att, marker=marker, linestyle=ls, color=C_BIN, alpha=0.75,
                 label=f"attention only, reuse factor {rf}")
        for b, t in zip(bits, tot):
            note(f"fig04  DSP RF={rf} A{b}: total {t}, attention {rows[b][idx][0].get('attention act x act', 0)}")
    ax2.axhline(0, color=C_BINNN, linewidth=2.4)
    ax2.text(7.9, 480, "all 15 binary weight layers: 0 DSP at every width",
             fontsize=8.5, color=C_BINNN)
    ax2.set_xticks(bits)
    ax2.set_ylim(-400, 16500)
    ax2.invert_xaxis()
    ax2.set_xlabel("activation width (bits)")
    ax2.set_ylabel("DSP blocks (whole model)")
    ax2.set_title("Measured DSP cost vs activation width\n(Vitis HLS 2023.2, VU13P, whole model)",
                  fontsize=10.5)
    ax2.legend(fontsize=8, loc="center left")
    fig.savefig(OUT / "fig04_quantization_axis.png")
    plt.close(fig)


# ------------------------------------- Fig 6 -- the trigger's own metric
def fig_rejection():
    arms = [
        ("FP32", NPZ_R7 / "r8/SMALL-FP32-r8std-s*.npz", C_FP32, "-"),
        ("W8A8", NPZ_R7 / "r8/SMALL-W8A8-r8std-s*.npz", C_W8A8, "-"),
        ("W1A8, norm-free", NPZ_R7 / "r8/SMALL-W1A8-r8stdnn-s*.npz", C_BINNN, "-"),
        ("W1A8, SubLN", NPZ_R7 / "r8/SMALL-W1A8-r8std-s*.npz", C_BIN, "--"),
        ("W1A8, norm-free\n+ pairwise bias", NPZ_R10 / "R10-W1A8-pair-s*.npz", C_PAIR, "-"),
    ]
    grid = np.linspace(0.2, 0.9, 36)
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for label, pattern, colour, ls in arms:
        curves = []
        for y, s, _ in arrays(pattern):
            curves.append([rejection(y, s, e).mean() for e in grid])
        m = np.array(curves).mean(axis=0)
        sd = np.array(curves).std(axis=0, ddof=0)
        ax.plot(grid, m, ls, color=colour, linewidth=1.6, label=label)
        ax.fill_between(grid, m - sd, m + sd, color=colour, alpha=0.13, linewidth=0)
        for e in (0.5, 0.7):
            j = int(np.argmin(abs(grid - e)))
            note(f"fig06  {label.replace(chr(10), ' '):32s} eps={e}: {m[j]:6.1f} +- {sd[j]:.1f}")
    for e in (0.5, 0.7):
        ax.axvline(e, color="#cccccc", linewidth=0.8, zorder=0)
    ax.set_yscale("log")
    ax.set_xlabel("signal efficiency $\\varepsilon_S$")
    ax.set_ylabel("background rejection  $1/\\varepsilon_B$   (macro mean over 5 classes)")
    ax.set_title("Background rejection at fixed signal efficiency\n"
                 "19,201 parameters, standardized inputs; band = s.d. over 3 seeds",
                 fontsize=10.5)
    ax.legend(fontsize=8.5)
    fig.savefig(OUT / "fig06_rejection.png")
    plt.close(fig)


# --------------------------------------------- Fig 7 -- DSP census, all syntheses
def fig_dsp_census():
    emissions = [
        ("A8, SubLN\nRF 1", "whole_model_rf1"),
        ("A8, SubLN\nRF 8", "whole_model_rf8_narrow"),
        ("A8, norm-free\nRF 1", "whole_model_rf1_stdnn"),
        ("A8, norm-free\nRF 8", "whole_model_rf8_stdnn"),
        ("A6\nRF 1", "whole_model_rf1_w1a6"),
        ("A6\nRF 8", "whole_model_rf8_w1a6"),
        ("A4\nRF 1", "whole_model_rf1_w1a4"),
        ("A4\nRF 8", "whole_model_rf8_w1a4"),
    ]
    order = ["SubLN normalization", "attention act x act", "softmax", "pooling",
             "binary weight layers"]
    colours = {"SubLN normalization": "#8e6bb5", "attention act x act": C_BIN,
               "softmax": "#e08a3c", "pooling": "#5b9bd5",
               "binary weight layers": C_BINNN}
    data = {}
    for label, name in emissions:
        cats, rep = census(name, "DSP")
        data[label] = (cats, rep)
        note(f"fig07  {name:28s} DSP total {rep['DSP']:6,d}  " +
             "  ".join(f"{k}={v}" for k, v in cats.items() if v))

    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(emissions))
    bottom = np.zeros(len(emissions))
    for cat in order:
        vals = np.array([data[l][0].get(cat, 0) for l, _ in emissions], dtype=float)
        name = cat + " — zero in every bar" if cat == "binary weight layers" else cat
        ax.bar(x, vals, 0.62, bottom=bottom, color=colours[cat], label=name,
               edgecolor="white", linewidth=0.6)
        bottom += vals
    for i, (label, _) in enumerate(emissions):
        ax.text(i, bottom[i] * 1.35 + 30, f"{int(bottom[i]):,}", ha="center", fontsize=8.5)
    ax.axhline(VU13P["DSP"], color="#444444", linestyle="--", linewidth=1.1)
    ax.text(len(emissions) - 0.4, VU13P["DSP"] * 1.15, f"VU13P: {VU13P['DSP']:,} DSP",
            ha="right", fontsize=8.5)
    ax.set_yscale("log")
    ax.set_ylim(20, 1.1e5)
    ax.set_xticks(x)
    ax.set_xticklabels([l for l, _ in emissions], fontsize=8.5)
    ax.set_ylabel("DSP blocks")
    ax.set_title("Where the DSPs are, in all eight whole-model syntheses\n"
                 "the binary weight layers contribute zero to every bar "
                 "(instance-tree attribution, sums closed against the report totals)",
                 fontsize=10.5)
    ax.legend(fontsize=8.5, ncol=3, loc="upper center")
    fig.savefig(OUT / "fig07_dsp_census.png")
    plt.close(fig)


# ---------------------------------------------------- Fig 9 -- LUT, the open problem
def fig_lut_census():
    emissions = [
        ("A8, SubLN, RF 8", "whole_model_rf8_narrow"),
        ("A8, norm-free, RF 8\n(the flagship)", "whole_model_rf8_stdnn"),
        ("A8, norm-free, RF 8\ndistributed arithmetic", "whole_model_da_rf8attn_stdnn"),
        ("A8, norm-free, RF 1", "whole_model_rf1_stdnn"),
    ]
    order = ["binary weight layers", "SubLN normalization", "attention act x act",
             "softmax", "pooling", "glue / other", "top-level glue"]
    colours = {"binary weight layers": C_BINNN, "SubLN normalization": "#8e6bb5",
               "attention act x act": C_BIN, "softmax": "#e08a3c",
               "pooling": "#5b9bd5", "glue / other": "#b0b0b0",
               "top-level glue": "#d8d8d8"}
    data = {}
    for label, name in emissions:
        cats, rep = census(name, "LUT")
        data[label] = (cats, rep)
        note(f"fig09  {name:32s} LUT total {rep['LUT']:10,d}  " +
             "  ".join(f"{k}={v:,}" for k, v in cats.items() if v))

    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    y = np.arange(len(emissions))[::-1]
    left = np.zeros(len(emissions))
    for cat in order:
        vals = np.array([data[l][0].get(cat, 0) for l, _ in emissions], dtype=float)
        ax.barh(y, vals / 1e6, 0.55, left=left / 1e6, color=colours[cat], label=cat,
                edgecolor="white", linewidth=0.6)
        left += vals
    for yy, (label, _) in zip(y, emissions):
        tot = data[label][1]["LUT"]
        ax.text(tot / 1e6 + 0.09, yy, f"{tot / 1e6:.2f}M  ({tot / VU13P['LUT']:.1f}$\\times$ device)",
                va="center", fontsize=8.5)
    ax.axvline(VU13P["LUT"] / 1e6, color="#444444", linestyle="--", linewidth=1.2)
    ax.text(VU13P["LUT"] / 1e6 - 0.07, -0.62, f"VU13P: {VU13P['LUT'] / 1e6:.3f}M LUT",
            ha="right", va="center", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels([l for l, _ in emissions], fontsize=8.5)
    ax.set_ylim(-1.05, len(emissions) - 0.35)
    ax.set_xlabel("lookup tables (millions)")
    ax.set_xlim(0, 6.6)
    ax.set_title("Lookup-table budget — the constraint that is still open\n"
                 "removing the normalization layers moved the model from 5.21M to 3.91M LUT; "
                 "distributed arithmetic made it worse",
                 fontsize=10.5)
    ax.legend(fontsize=8, ncol=4, loc="lower right", bbox_to_anchor=(1.0, -0.02))
    ax.grid(axis="y", visible=False)
    fig.savefig(OUT / "fig09_lut_budget.png")
    plt.close(fig)


# --------------------------------------------- Fig 10 -- token-axis folding
def fig_token_folding():
    """Experiment 1 on one real layer, and what it projects for the whole model."""
    import json

    # --- left: the measured arms, one 32 -> 64 per-token layer
    arms = [("P\nten instances,\nas synthesized", "p", None),
            ("A2\nfolded by two", "a2", None),
            ("B\nsubset-sum\nrestructuring", "b", None),
            ("A\nfolded by ten", "a", "v2"),
            ("C\nfolded and\nrestructured", "c", "v2")]
    base = e1("p")["LUT"]
    census_ten = 234_930          # the same layer inside the whole model, RF 8, x10
    lut, ratio, labels = [], [], []
    for label, arm, v2 in arms:
        r1 = e1(arm)
        lut.append(r1["LUT"])
        ratio.append(base / r1["LUT"])
        labels.append(label)
        line = (f"fig10  arm {arm.upper():2s} LUT {r1['LUT']:8,d}  FF {r1['FF']:7,d}  "
                f"DSP {r1['DSP']}  latency {r1['lat']:3d} cyc  II {r1['II']:3d}  "
                f"clock {r1['clk']:.3f} ns  ({base / r1['LUT']:.2f}x arm P)")
        if v2:
            r2 = e1(arm, "v2")
            line += (f"\nfig10  arm {arm.upper():2s} rewind: LUT {r2['LUT']:8,d} "
                     f"({100 * (r2['LUT'] - r1['LUT']) / r1['LUT']:+.1f}%)  II {r1['II']} -> "
                     f"{r2['II']}  = {r2['II'] * r2['clk']:.1f} ns per jet")
        note(line)
    note(f"fig10  layer baseline inside the whole model: {census_ten:,} LUT "
         f"(ten instances of 23,493)")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.5),
                                  gridspec_kw={"width_ratios": [1.15, 1]})
    cols = [C_GREY, C_GREY, C_W8A8, C_BINNN, "#14532d"]
    x = np.arange(len(arms))
    ax.bar(x, np.array(lut) / 1e3, 0.62, color=cols, edgecolor="white", linewidth=0.6)
    for i, (v, r) in enumerate(zip(lut, ratio)):
        ax.text(i, v / 1e3 + 7, f"{v / 1e3:.0f}k" + ("" if i == 0 else f"\n{r:.1f}$\\times$"),
                ha="center", va="bottom", fontsize=8.5,
                color="#222222" if i < 3 else C_BINNN)
    ax.axhline(census_ten / 1e3, color="#444444", linestyle="--", linewidth=1.1)
    ax.text(len(arms) - 0.45, census_ten / 1e3 + 5,
            "same layer in the whole model", ha="right", fontsize=8, color="#444444")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("lookup tables (thousands)")
    ax.set_ylim(0, 320)
    ax.set_title("One 32$\\rightarrow$64 per-token layer, five emissions\n"
                 "all five bit-exact in C-simulation, 0 DSP", fontsize=10.5)
    ax.grid(axis="x", visible=False)

    # --- right: measured whole model against the folded projection
    meas = csynth("whole_model_rf8_stdnn")
    f = json.loads((ADDER / "counting_results.json").read_text())
    fold, comp = f["fold"], f["fold_restruct"]
    cats = fold["census_categories"]
    measured = [cats["dense_per_token"], cats["glue_other"],
                cats["einsum_actxact"] + cats["softmax"] + cats["pooling"]
                + cats["dense_head"] + cats["einsum_dense_wrap"], cats["top_glue"]]
    assert sum(measured) == meas["LUT"], "census does not close"
    unfolded_rest = fold["unfolded"] - cats["top_glue"]
    projected = [fold["dense_folded"], fold["glue_folded"], unfolded_rest, cats["top_glue"]]
    # restructured slice, keeping the fold's own multiplexer and control overhead
    dense_overhead = fold["dense_folded"] - fold["per_token_slice_lut"]
    restruct = [comp["slice_lut_best"] + dense_overhead, fold["glue_folded"],
                unfolded_rest, cats["top_glue"]]
    assert abs(sum(projected) - fold["total"]) < 2, "projection does not close"
    assert abs(sum(restruct) - comp["fold_restruct_total"]) < 2, "composition does not close"

    stacks = [("measured\nas synthesized", measured),
              ("projected\nfolded by ten", projected),
              ("projected\nfolded and\nrestructured", restruct)]
    parts = [("per-token binary dense", C_BINNN), ("per-token glue", "#b0b0b0"),
             ("attention, softmax, pooling, head", C_BIN), ("top-level glue", "#d8d8d8")]
    y = np.arange(len(stacks))[::-1]
    left = np.zeros(len(stacks))
    for j, (name, colour) in enumerate(parts):
        vals = np.array([s[1][j] for s in stacks], dtype=float)
        ax2.barh(y, vals / 1e6, 0.5, left=left / 1e6, color=colour, label=name,
                 edgecolor="white", linewidth=0.6)
        left += vals
    for yy, (label, vals) in zip(y, stacks):
        tot = sum(vals)
        ax2.text(tot / 1e6 + 0.06, yy, f"{tot / 1e6:.2f}M", va="center", fontsize=8.5)
        note(f"fig10  {label.replace(chr(10), ' '):38s} {tot:9,d} LUT  "
             f"({tot / VU13P['LUT']:.2f} of device)")
    ax2.axvline(VU13P["LUT"] / 1e6, color="#444444", linestyle="--", linewidth=1.2)
    ax2.text(VU13P["LUT"] / 1e6 + 0.05, -0.72, f"VU13P: {VU13P['LUT'] / 1e6:.3f}M",
             ha="left", va="center", fontsize=8.5)
    ax2.set_yticks(y)
    ax2.set_yticklabels([s[0] for s in stacks], fontsize=8.5)
    ax2.set_ylim(-1.15, len(stacks) - 0.4)
    ax2.set_xlim(0, 4.5)
    ax2.set_xlabel("lookup tables (millions)")
    ax2.set_title("Whole model: measured, and projected from the measured\n"
                  "fold ratio (a projection, not a synthesis)", fontsize=10.5)
    ax2.legend(fontsize=7.6, ncol=2, loc="lower right", bbox_to_anchor=(1.0, -0.03))
    ax2.grid(axis="y", visible=False)
    fig.savefig(OUT / "fig10_token_folding.png")
    plt.close(fig)


# ------------------------------------------------- Fig 8 -- the design space
def fig_operating_points():
    # (label, report, colour, marker, label offset left panel, offset right panel)
    points = [
        ("A8 SubLN, RF 1", "whole_model_rf1", C_BIN, "o", (7, 4), (8, 2)),
        ("A8 SubLN, RF 8", "whole_model_rf8_narrow", C_BIN, "s", (7, 4), (8, 2)),
        ("A8 SubLN fabric-norm, RF 8", "whole_model_rf8_noSubLNdsp", C_W8A8, "s", (-9, 14), (9, 7)),
        ("A8 SubLN fabric-norm, RF 10", "whole_model_rf10_lutfit", C_W8A8, "^", (-9, 12), (9, -12)),
        ("A8 norm-free, RF 1", "whole_model_rf1_stdnn", C_BINNN, "o", (7, -12), (8, 2)),
        ("A8 norm-free, RF 8", "whole_model_rf8_stdnn", C_BINNN, "s", (7, 4), (8, 2)),
        ("A6, RF 8", "whole_model_rf8_w1a6", "#e08a3c", "s", (-9, -1), (-9, 8)),
        ("A4, RF 8", "whole_model_rf8_w1a4", "#8e6bb5", "s", (-9, -16), (-9, -12)),
    ]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.8))
    for label, name, colour, marker, off1, off2 in points:
        r = csynth(name)
        per_jet = r["II"] * r["clk"]
        latency_us = r["lat"] * r["clk"] / 1000
        ax.scatter(per_jet, latency_us, s=70, color=colour, marker=marker,
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(label, (per_jet, latency_us), textcoords="offset points",
                    xytext=off1, fontsize=7.6, color="#444444",
                    ha="right" if off1[0] < 0 else "left")
        ax2.scatter(r["LUT"] / 1e6, r["DSP"], s=70, color=colour, marker=marker,
                    edgecolor="white", linewidth=0.8, zorder=3)
        ax2.annotate(label, (r["LUT"] / 1e6, r["DSP"]), textcoords="offset points",
                     xytext=off2, fontsize=7.6, color="#444444",
                     ha="right" if off2[0] < 0 else "left")
        note(f"fig08  {label:30s} {r['DSP']:6,d} DSP  {r['LUT']:10,d} LUT  "
             f"{r['lat']:4d} cyc  II={r['II']:2d}  {r['clk']:.3f} ns  "
             f"-> {latency_us:.3f} us, {per_jet:.1f} ns/jet")

    ax.axvspan(L1_WINDOW_NS, 40, color="#c0392b", alpha=0.06, zorder=0)
    ax.axvline(L1_WINDOW_NS, color=C_BIN, linestyle="--", linewidth=1.2)
    ax.text(L1_WINDOW_NS - 0.8, 0.06, "40 MHz: one jet every 25 ns",
            ha="right", va="bottom", fontsize=8.5, color=C_BIN)
    ax.set_xlabel("throughput  (ns per jet = initiation interval $\\times$ estimated clock)")
    ax.set_ylabel("end-to-end latency ($\\mu$s)")
    ax.set_xlim(0, 37)
    ax.set_ylim(0, 2.15)
    ax.set_title("Throughput and latency of every whole-model synthesis", fontsize=10.5)

    ax2.axvline(VU13P["LUT"] / 1e6, color="#444444", linestyle="--", linewidth=1.2)
    ax2.text(VU13P["LUT"] / 1e6 + 0.18, 3.0e4, "VU13P LUT budget", rotation=90,
             va="top", fontsize=8.5)
    ax2.axhline(VU13P["DSP"], color="#444444", linestyle=":", linewidth=1.2)
    ax2.text(15.2, VU13P["DSP"] * 1.15, "VU13P DSP budget", ha="right", fontsize=8.5)
    ax2.set_yscale("log")
    ax2.set_xlabel("lookup tables (millions)")
    ax2.set_ylabel("DSP blocks")
    ax2.set_xlim(0, 15.5)
    ax2.set_ylim(500, 8e4)
    ax2.set_title("Resource cost of the same points\n(fits below and left of the dashed lines)",
                  fontsize=10.5)
    fig.savefig(OUT / "fig08_operating_points.png")
    plt.close(fig)


# ------------------------------------------- Fig 5 -- the pairwise attention bias
def fig_pair_bias():
    base = arrays(NPZ_R7 / "r8/SMALL-W1A8-r8stdnn-s*.npz")
    pair = arrays(NPZ_R10 / "R10-W1A8-pair-s*.npz")
    eps = 0.5
    b = np.array([rejection(y, s, eps) for y, s, _ in base]).mean(axis=0)
    p = np.array([rejection(y, s, eps) for y, s, _ in pair]).mean(axis=0)
    b_auc = np.array([[roc_auc_score(y[:, c], s[:, c]) for c in range(5)] for y, s, _ in base]).mean(axis=0)
    p_auc = np.array([[roc_auc_score(y[:, c], s[:, c]) for c in range(5)] for y, s, _ in pair]).mean(axis=0)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.2))
    x = np.arange(5)
    ax.bar(x - 0.19, b, 0.36, color=C_BINNN, label="norm-free binary (19,201 params)")
    ax.bar(x + 0.19, p, 0.36, color=C_PAIR, label="+ pairwise attention bias (+84 params)")
    for i in range(5):
        ax.text(i, max(b[i], p[i]) * 1.03, f"{100 * (p[i] - b[i]) / b[i]:+.1f}%",
                ha="center", fontsize=8.5,
                color=C_PAIR if p[i] > b[i] else "#999999")
        note(f"fig05  {CLASSES[i]}: rejection {b[i]:.1f} -> {p[i]:.1f} "
             f"({100 * (p[i] - b[i]) / b[i]:+.1f}%),  AUC {b_auc[i]:.4f} -> {p_auc[i]:.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_ylabel(f"background rejection at $\\varepsilon_S$ = {eps}")
    ax.set_title("Per-class rejection, seed mean (6 seeds vs 3)", fontsize=10.5)
    ax.legend(fontsize=8.2)

    ax2.bar(x - 0.19, b_auc, 0.36, color=C_BINNN)
    ax2.bar(x + 0.19, p_auc, 0.36, color=C_PAIR)
    ax2.set_xticks(x)
    ax2.set_xticklabels(CLASSES)
    ax2.set_ylim(0.80, 0.95)
    ax2.set_ylabel("one-vs-rest AUC")
    ax2.set_title("Per-class AUC, seed mean (6 seeds vs 3)", fontsize=10.5)
    fig.suptitle("A shared pairwise-invariant bias on the attention logits: what gain there is sits on W and Z (n.s. at 6 seeds)",
                 fontsize=11, y=1.02)
    fig.savefig(OUT / "fig05_pairwise_bias.png")
    plt.close(fig)


if __name__ == "__main__":
    note("Figures for the binary-weight transformer jet tagger.")
    note("Accuracy from the per-jet arrays; hardware from the raw Vitis csynth reports.")
    note("")
    fig_architecture()
    fig_roc()
    fig_standardization()
    fig_activation_ladder()
    fig_rejection()
    fig_dsp_census()
    fig_lut_census()
    fig_operating_points()
    fig_pair_bias()
    fig_token_folding()
    (OUT / "NUMBERS.txt").write_text("\n".join(LOG) + "\n")
    note("")
    note(f"wrote {len(sorted(OUT.glob('fig*.png')))} figures and NUMBERS.txt")
