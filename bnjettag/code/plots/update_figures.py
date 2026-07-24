#!/usr/bin/env python3
"""
Editable source for the June-29 "update" figures used in the progress doc.

These six figures were originally produced by an ad-hoc session script that was
never committed, so the PNGs existed with no editable source. This script
reproduces them from the verified numbers (inline below, sourced from
the internal experiment log) and from the era-1 ROC .npz files, so the
figures are now fully editable and reproducible.

Edit the DATA blocks or the styling, then re-run:

    .venv-hgq2/bin/python bnjettag/code/plots/update_figures.py

Output goes to bnjettag/results/plots/regen/ (non-destructive: it does NOT
overwrite the originals). To promote a regenerated figure, copy it up one level.

All AUC numbers here are ERA-1 (old private 2-class dataset). Do not compare
to era-2 (public 5-class) numbers.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]          # .../bnjettag
OUT = ROOT / "results" / "plots" / "regen"
OUT.mkdir(parents=True, exist_ok=True)
NPZ = ROOT / "roc-results"

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.grid": True,
    "grid.alpha": 0.3, "grid.linestyle": ":", "savefig.dpi": 200,
    "savefig.bbox": "tight", "figure.autolayout": False,
})
BLUE, RED, GREEN, GREY = "#1f77b4", "#c0392b", "#1e7a34", "#7f7f7f"


# --------------------------------------------------------------------------
# 1. sweep_auc_vs_size — validation AUC vs model size, lr15 vs tuned LR
# --------------------------------------------------------------------------
def sweep_auc_vs_size():
    sizes = ["tiny\n27K", "small\n154K", "medium\n808K", "large\n6.37M"]
    x = np.arange(4)
    lr15 = [0.7112, 0.7335, 0.7538, 0.7530]
    tuned = [np.nan, 0.7499, 0.7567, 0.7672]
    tuned_err = [0, 0.0033, 0.0052, 0]          # seed spread where measured

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(x, lr15, "o--", color=BLUE, lw=2, ms=9,
            label="lr15 recipe (peak LR 1.5e-4, un-tuned)")
    ax.errorbar(x, tuned, yerr=tuned_err, fmt="s-", color=RED, lw=2.5, ms=10,
                capsize=4, label="tuned LR (~5e-5)")
    for xi, v in zip(x, lr15):
        ax.annotate(f"{v:.4f}", (xi, v), textcoords="offset points",
                    xytext=(0, -16), ha="center", color=BLUE, fontsize=10)
    for xi, v in zip(x, tuned):
        if not np.isnan(v):
            ax.annotate(f"{v:.4f}", (xi, v), textcoords="offset points",
                        xytext=(0, 10), ha="center", color=RED, fontweight="bold")
    ax.annotate("+0.0142\nby lowering LR", (3, 0.7601), color=RED, fontsize=11,
                fontweight="bold", ha="left", va="center")
    ax.annotate("", xy=(3, 0.7672), xytext=(3, 0.7530),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.5))
    ax.text(0, 0.7135, "tiny: not LR-swept", color=GREY, fontsize=10, style="italic")
    ax.set_xticks(x); ax.set_xticklabels(sizes)
    ax.set_xlabel("model size  (trainable parameters, log scale)")
    ax.set_ylabel("best validation AUC")
    ax.set_title("Once the LR is tuned, bigger wins\n"
                 "validation AUC vs model size — the recipe was the hidden ceiling")
    ax.legend(loc="lower right")
    fig.savefig(OUT / "sweep_auc_vs_size.png"); plt.close(fig)


# --------------------------------------------------------------------------
# 2. sweep_efficiency_frontier — best tuned AUC per size
# --------------------------------------------------------------------------
def sweep_efficiency_frontier():
    x = np.arange(1, 4)
    auc = [0.7499, 0.7567, 0.7672]
    err = [0.0033, 0.0052, 0]
    fig, ax = plt.subplots(figsize=(8, 5.6))
    ax.errorbar(x, auc, yerr=err, fmt="o-", color=RED, lw=2.5, ms=11, capsize=4)
    ax.plot(0, 0.7112, "o", mfc="white", mec=GREY, ms=10)
    ax.text(0.05, 0.7112, "tiny (lr15 only)", color=GREY, fontsize=10,
            va="center", style="italic")
    for xi, v in zip(x, auc):
        ax.annotate(f"{v:.4f}", (xi, v), textcoords="offset points",
                    xytext=(0, 12), ha="center", color=RED, fontweight="bold")
    ax.annotate("best AUC", (3, 0.7660), color=RED, fontweight="bold", ha="right")
    ax.annotate("efficiency pick:\nwithin ~0.010 of large\nat ~8x fewer params",
                xy=(2, 0.7567), xytext=(2.15, 0.7505), color=GREEN, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.set_xticks(range(4))
    ax.set_xticklabels(["tiny\n27K", "small\n154K", "medium\n808K", "large\n6.37M"])
    ax.set_xlabel("trainable parameters  (log scale)")
    ax.set_ylabel("best validation AUC achieved")
    ax.set_title("Efficiency frontier — best validation AUC per model size\n"
                 "binary {-1,+1} weights - A8 activations - tuned LR")
    fig.savefig(OUT / "sweep_efficiency_frontier.png"); plt.close(fig)


# --------------------------------------------------------------------------
# 3. sweep_lr_small — small-model LR sweep
# --------------------------------------------------------------------------
def sweep_lr_small():
    lr = np.array([2.5, 3.5, 5.0, 7.5, 10, 15])
    auc = [0.7500, 0.7537, 0.7499, 0.7480, 0.7344, 0.7335]
    err = [0, 0, 0.0036, 0, 0, 0]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.errorbar(lr, auc, yerr=err, fmt="o-", color=RED, lw=2.5, ms=9, capsize=4)
    for xi, v in zip(lr, auc):
        ax.annotate(f"{v:.4f}", (xi, v), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontweight="bold")
    ax.axhline(0.750, color=GREY, ls=":", lw=1)
    ax.text(2.5, 0.7503, "plateau ~0.750", color=GREY, style="italic", fontsize=10)
    ax.text(3.5, 0.7545, "best 0.7537 @3.5e-5", color=GREY, fontsize=10)
    ax.annotate("lr15: too hot ->\npeaks ep6 then collapses", (15, 0.7360),
                color=RED, ha="right", fontsize=10)
    ax.set_xlabel(r"peak learning rate  ($\times 10^{-5}$)")
    ax.set_ylabel("best validation AUC")
    ax.set_title("Lower peak LR fixes the collapse — small-model LR sweep\n"
                 "(rounds 2-3; 5e-5 point is a 3-seed mean +/- spread)")
    fig.savefig(OUT / "sweep_lr_small.png"); plt.close(fig)


# --------------------------------------------------------------------------
# 4. hls_dsp_breakdown — where every DSP comes from (era-1 full model)
# --------------------------------------------------------------------------
def hls_dsp_breakdown():
    labels = ["binary matmul core (51 BitLinears)", "LN: input_proj (14) x1",
              "LN: head_fc2 (256) x1", "LN: ffn_fc1 (256) x8",
              "LN: ffn_fc2 (1024) x8", "LN: attn proj + head_fc1 (256) x33"]
    vals = [0, 11, 15, 120, 408, 495]
    colors = [RED, "#a9cce3", "#5dade2", "#48c9b0", "#2874a6", "#1b4f72"]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors, edgecolor="white")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    for yi, v in zip(y, vals):
        if v == 0:
            ax.text(6, yi, "0  <- the binary win", color=RED, fontweight="bold", va="center")
        else:
            ax.text(v + 6, yi, str(v), va="center", fontweight="bold")
    ax.set_xlabel("DSP48 blocks")
    ax.set_title("Where every DSP comes from — the binary win made precise\n"
                 "full trained transformer: matmul = 0 DSP, 100% of DSP = LayerNorm")
    ax.text(0.98, 0.06, "total = 1,049 DSP = 8.5% of a VU13P\n(precision-independent: A8=A6=A4)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round", fc="#eeeeee", ec="#bbbbbb"))
    ax.set_xlim(0, 560)
    fig.savefig(OUT / "hls_dsp_breakdown.png"); plt.close(fig)


# --------------------------------------------------------------------------
# 5. roc_overlay_clean — era-1 held-out ROC (from the .npz), HEP log-FPR style
# --------------------------------------------------------------------------
def roc_overlay():
    files = [("FP32-vanilla.npz", "FP32", "black"),
             ("W8A8-baseline.npz", "W8A8", "#7f7f7f"),
             ("A8-binary-softmax.npz", "binary A8", RED),
             ("A6-binary-softmax.npz", "binary A6", "#e67e22"),
             ("A4-binary-softmax.npz", "binary A4", "#8e44ad")]
    fig, ax = plt.subplots(figsize=(7, 6))
    n = None
    for fn, name, c in files:
        p = NPZ / fn
        if not p.exists():
            print(f"  [skip] {fn} not found"); continue
        d = np.load(p); y, s = d["y"], d["score"]; n = len(y)
        fpr, tpr, _ = roc_curve(y, s)
        auc = roc_auc_score(y, s)
        ax.plot(tpr, np.clip(fpr, 1e-4, 1), color=c, lw=1.8,
                label=f"{name} (AUC {auc:.4f})")
    ax.set_yscale("log")
    ax.set_xlabel("tagging efficiency (signal eff, TPR)")
    ax.set_ylabel("mistag rate (QCD, FPR) — log")
    ax.set_title(f"ROC — era-1 held-out test set (n = {n:,})\n"
                 "binary BitNet vs FP32 / W8A8 baselines")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(1e-4, 1)
    fig.savefig(OUT / "roc_overlay_clean.png"); plt.close(fig)


# --------------------------------------------------------------------------
# 6. workflow_code — the system & code map (schematic)
# --------------------------------------------------------------------------
def workflow_code():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 11); ax.set_ylim(0, 6); ax.axis("off")
    C = {"code": "#2e86c1", "trained": "#7d3c98", "hls": "#c0392b", "infra": "#7f7f7f"}

    def box(x, y, w, h, title, sub, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
                                    fc=color, ec="white", alpha=0.92))
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
                color="white", fontsize=9.5, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                color="white", fontsize=7.5)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=12, color="#444", lw=1.2))

    ax.text(5.5, 5.85, "BNJetTag — system & code map",
            ha="center", fontsize=13, fontweight="bold", color="#1f4e79")
    box(4.2, 5.15, 2.6, 0.55, "W&B  bnjettag-bitnet", "per-epoch scalars", C["infra"])
    top = [("1 Training core", "qkerasModel.py"), ("2 Variant engine", "gen_round{2,3,4}_jobs.py"),
           ("3 Run on GPU", "NRP Nautilus k8s Jobs"), ("4 Durable logs", "/data/qk-variants/* +util"),
           ("5 Analyze", "variant_sweep.md + plots")]
    for i, (tt, ss) in enumerate(top):
        box(0.2 + i * 2.16, 4.2, 2.0, 0.7, tt, ss, C["infra"] if i >= 2 else C["code"])
    box(4.3, 3.0, 2.4, 0.7, "Trained model", "lr15_bitnetJetTagModel.h5", C["trained"])
    bottom = [("ROC-test AUC", "roc_auc.md (n=222,912)", C["infra"]),
              ("Offline ROC", "make_roc.py  ->  .npz", C["code"]),
              ("HLS rebuild", "model_csynth.py / run_csynth.py", C["hls"]),
              ("Vitis csynth - mulder", "0 DSP", C["hls"])]
    for i, (tt, ss, cc) in enumerate(bottom):
        box(0.2 + i * 2.75, 1.5, 2.5, 0.7, tt, ss, cc)
    arrow(5.5, 4.2, 5.5, 3.7); arrow(5.5, 3.0, 5.5, 2.2)
    ax.text(5.5, 0.7, "blue = code we wrote / edited   purple = trained artifact   "
            "red = HLS reconstruction   grey = infrastructure / outputs",
            ha="center", fontsize=8, style="italic", color="#555")
    fig.savefig(OUT / "workflow_code.png"); plt.close(fig)


if __name__ == "__main__":
    sweep_auc_vs_size()
    sweep_efficiency_frontier()
    sweep_lr_small()
    hls_dsp_breakdown()
    roc_overlay()
    workflow_code()
    print(f"wrote 6 figures to {OUT}")
