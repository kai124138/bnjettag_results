#!/usr/bin/env python3
"""Background rejection at fixed signal efficiency — the trigger-native metric.

AUC is a threshold-free summary; the trigger operates at a fixed signal
efficiency and cares about how much background survives there.  The two can
disagree badly: BitParT (arXiv:2508.07431) reports flat AUC with 15-17% lost
rejection, and this project's own 19.2k flagship reads "2.6 AUC points" as a
~35% rejection loss.  Every accuracy claim about a binary arm should therefore
quote both.

Convention (fixed 2026-07-24, verification gate; do not change silently):

    rejection(eps_S) = macro-mean over classes of the per-class one-vs-rest
                       1 / FPR at the FIRST threshold where TPR >= eps_S

  * per-class OvR, unweighted macro over the 5 classes -- matches macro_ovr_auc
  * "first TPR >= eps" via the sklearn roc_curve arrays, side='left'
  * across seeds: POPULATION std (ddof=0) for rejection
                  SAMPLE std     (ddof=1) for AUC
    The split is historical but real -- RESEARCH.md's rejection tables are ddof=0
    and its AUC tables are ddof=1.  Both are printed with their convention named.

Inputs are the standard ROC arrays written by roc_final.py: an .npz with `y`
(N x C one-hot) and `score` (N x C).

Usage
-----
    # one arm
    python rejection.py --arm "W1A8-stdnn=r7/roc-results/r8/SMALL-W1A8-r8stdnn-s*.npz"

    # several arms with retention against a named baseline, as a markdown table
    python rejection.py \
      --arm "FP32-std=.../SMALL-FP32-r8std-s*.npz" \
      --arm "W1A8-stdnn=.../SMALL-W1A8-r8stdnn-s*.npz" \
      --baseline FP32-std --md table.md

    # the gate: reproduce every published figure or exit nonzero
    python rejection.py --selftest
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import sys

import numpy as np

EPS_DEFAULT = (0.3, 0.5, 0.7)
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# --------------------------------------------------------------------------- #
# core                                                                          #
# --------------------------------------------------------------------------- #
def per_class_rejection(y_onehot: np.ndarray, scores: np.ndarray,
                        eps: float) -> list[float]:
    """1/FPR at the first threshold where TPR >= eps, per class."""
    from sklearn.metrics import roc_curve

    out = []
    for c in range(y_onehot.shape[1]):
        fpr, tpr, _ = roc_curve(y_onehot[:, c], scores[:, c])
        i = min(int(np.searchsorted(tpr, eps, side="left")), len(fpr) - 1)
        f = float(fpr[i])
        out.append(float("inf") if f <= 0.0 else 1.0 / f)
    return out


def macro_ovr_auc(y_onehot: np.ndarray, scores: np.ndarray) -> tuple[float, list[float]]:
    """Project-canonical macro one-vs-rest AUC (mirrors roc_final.macro_ovr_auc)."""
    from sklearn.metrics import roc_auc_score

    per = [float(roc_auc_score(y_onehot[:, c], scores[:, c]))
           for c in range(y_onehot.shape[1])]
    return float(np.mean(per)), per


def load_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return d["y"], d["score"]


class Arm:
    """One (variant, set of seeds) point: AUC and rejection, aggregated over seeds."""

    def __init__(self, label: str, paths: list[str], eps=EPS_DEFAULT):
        if not paths:
            raise SystemExit(f"{label}: no .npz matched")
        self.label, self.paths, self.eps = label, sorted(paths), tuple(eps)
        self.n_seed = len(self.paths)
        aucs, macro, per = [], {e: [] for e in self.eps}, {e: [] for e in self.eps}
        n_rows = set()
        for p in self.paths:
            y, s = load_npz(p)
            n_rows.add(int(y.shape[0]))
            a, _ = macro_ovr_auc(y, s)
            aucs.append(a)
            for e in self.eps:
                r = per_class_rejection(y, s, e)
                per[e].append(r)
                macro[e].append(float(np.mean(r)))
        if len(n_rows) != 1:
            raise SystemExit(f"{label}: seeds disagree on n ({sorted(n_rows)})")
        self.n = n_rows.pop()
        self.auc = float(np.mean(aucs))
        self.auc_sd = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0   # sample
        self.auc_per_seed = [float(a) for a in aucs]
        self.rej = {e: float(np.mean(macro[e])) for e in self.eps}
        self.rej_sd = {e: float(np.std(macro[e], ddof=0)) for e in self.eps}  # population
        self.rej_per_seed = {e: [float(v) for v in macro[e]] for e in self.eps}
        self.rej_per_class = {e: np.mean(np.array(per[e]), axis=0) for e in self.eps}


# --------------------------------------------------------------------------- #
# reporting                                                                     #
# --------------------------------------------------------------------------- #
def table(arms: list[Arm], baseline: str | None, classes: list[str] | None) -> str:
    eps = arms[0].eps
    base = next((a for a in arms if a.label == baseline), None)
    L = [f"n = {arms[0].n:,} · macro-OvR · rejection = macro-mean 1/FPR at first "
         f"TPR ≥ ε_S · rejection ± population std (ddof=0) · AUC ± sample std (ddof=1)",
         ""]
    head = "| arm | seeds | ROC-test AUC | " + " | ".join(f"ε_S={e}" for e in eps) + " |"
    L += [head, "|" + "---|" * (3 + len(eps))]
    for a in arms:
        cells = []
        for e in eps:
            c = f"{a.rej[e]:.1f} ± {a.rej_sd[e]:.1f}"
            if base is not None and base.rej[e] > 0:
                c += f" ({100 * a.rej[e] / base.rej[e]:.1f}%)"
            cells.append(c)
        L.append(f"| {a.label} | {a.n_seed} | {a.auc:.4f} ± {a.auc_sd:.4f} | "
                 + " | ".join(cells) + " |")
    if base is not None:
        L += ["", f"Percentages are retention of **{base.label}** rejection at the same ε_S."]
    if classes:
        mid = eps[len(eps) // 2]
        L += ["", f"Per-class rejection at ε_S={mid} (retention of {baseline} in brackets):", "",
              "| arm | " + " | ".join(classes) + " |", "|" + "---|" * (1 + len(classes))]
        for a in arms:
            cells = []
            for i in range(len(classes)):
                v = a.rej_per_class[mid][i]
                cells.append(f"{v:.1f}" + (f" [{100 * v / base.rej_per_class[mid][i]:.0f}%]"
                                           if base is not None else ""))
            L.append(f"| {a.label} | " + " | ".join(cells) + " |")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# the gate                                                                      #
# --------------------------------------------------------------------------- #
# Every rejection figure this project has published, with its source. The
# self-test recomputes each from the stored arrays and refuses to pass on drift.
SELFTEST = {
    # RESEARCH.md §2 SMALL (19.2k, round 8, standardized) + COMPILER.md §8 ε=0.3 row
    "r7/roc-results/r8/SMALL-FP32-r8std-s*.npz":     {0.3: 150.1, 0.5: 47.8, 0.7: 16.7},
    "r7/roc-results/r8/SMALL-W8A8-r8std-s*.npz":     {0.3: 148.6, 0.5: 46.8, 0.7: 16.4},
    "r7/roc-results/r8/SMALL-W1A8-r8stdnn-s*.npz":   {0.3: 91.2, 0.5: 31.0, 0.7: 11.5},
    "r7/roc-results/r8/SMALL-W1A8-r8std-s*.npz":     {0.3: 63.3, 0.5: 24.6, 0.7: 9.7},
    # RESEARCH.md §2 FINAL (6.4M, raw)
    "roc-results/final/FP32-s*.npz":                 {0.5: 20.4, 0.7: 8.2},
    "roc-results/final/W8A8-s*.npz":                 {0.5: 21.5, 0.7: 8.2},
    "roc-results/final/W1A8-s*.npz":                 {0.5: 19.3, 0.7: 7.8},
    "roc-results/final/W1A6-s*.npz":                 {0.5: 16.7, 0.7: 6.9},
    "roc-results/final/W1A4-s*.npz":                 {0.5: 9.7, 0.7: 3.8},
    # roc-results/r10/roc_auc_r10.md (6 seeds)
    "roc-results/r10/R10-W1A8-pair-s*.npz":          {0.3: 95.4, 0.5: 32.5, 0.7: 11.9},
    # roc-results/r11/roc_auc_r11_control.md — the norm-free 8-bit control at d32
    "roc-results/r11/R11-W8A8-stdnn-s*.npz":         {0.3: 215.8, 0.5: 60.2, 0.7: 19.4},
}


# Published RETENTION percentages (RESEARCH.md §2): (arm, baseline, eps) -> percent.
# Macro means alone cannot catch a drift in how retention is formed, so these are checked too.
_SM = "r7/roc-results/r8/SMALL-{}-s*.npz"
_FN = "roc-results/final/{}-s*.npz"
SELFTEST_RETENTION = {
    (_SM.format("W8A8-r8std"), _SM.format("FP32-r8std")): {0.5: 97.9, 0.7: 98.4},
    (_SM.format("W1A8-r8stdnn"), _SM.format("FP32-r8std")): {0.5: 64.9, 0.7: 68.8},
    (_SM.format("W1A8-r8std"), _SM.format("FP32-r8std")): {0.5: 51.5, 0.7: 58.0},
    (_FN.format("W8A8"), _FN.format("FP32")): {0.5: 105.6, 0.7: 99.7},
    (_FN.format("W1A8"), _FN.format("FP32")): {0.5: 94.7, 0.7: 95.5},
    (_FN.format("W1A6"), _FN.format("FP32")): {0.5: 81.6, 0.7: 84.6},
    (_FN.format("W1A4"), _FN.format("FP32")): {0.5: 47.8, 0.7: 46.6},
    ("roc-results/r11/R11-W8A8-stdnn-s*.npz", _SM.format("FP32-r8std")):
        {0.3: 143.8, 0.5: 126.1, 0.7: 116.4},
}

# Published PER-CLASS rejection (COMPILER.md §8 / RESEARCH.md §2), order (g,q,W,Z,t).
SELFTEST_PER_CLASS = {
    _SM.format("FP32-r8std"): {0.5: [34.2, 26.2, 34.1, 58.6, 85.7]},
    _SM.format("W1A8-r8stdnn"): {0.5: [23.9, 23.2, 18.2, 22.1, 67.7]},
    "roc-results/r11/R11-W8A8-stdnn-s*.npz": {0.5: [39.7, 28.1, 40.9, 82.0, 110.5]},
}

# Published SPREADS, one per convention, so a ddof drift cannot pass silently.
SELFTEST_STD = [
    (_SM.format("W1A8-r8stdnn"), 0.5, 0, 2.0),    # RESEARCH.md §2: population
    ("roc-results/r10/R10-W1A8-pair-s*.npz", 0.5, 1, 2.2),  # roc_auc_r10.md: sample
]


def _paths(pat):
    return _glob.glob(os.path.join(REPO, pat))


def selftest(tol: float = 0.05) -> int:
    bad = n = 0

    def check(label, got, want, t=tol):
        nonlocal bad, n
        n += 1
        ok = abs(got - want) <= t
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {label}  got {got:8.2f}  published {want:8.2f}")

    for pat, expect in SELFTEST.items():
        if not _paths(pat):
            print(f"SKIP  {pat} (no arrays present)")
            continue
        a = Arm(pat, _paths(pat), eps=tuple(expect))
        for e, want in expect.items():
            check(f"macro   {pat:46s} ε={e}", a.rej[e], want)

    for (pat, base_pat), expect in SELFTEST_RETENTION.items():
        if not (_paths(pat) and _paths(base_pat)):
            print(f"SKIP  retention {pat}")
            continue
        eps = tuple(expect)
        a, b = Arm(pat, _paths(pat), eps=eps), Arm(base_pat, _paths(base_pat), eps=eps)
        for e, want in expect.items():
            check(f"retain  {pat:46s} ε={e}", 100 * a.rej[e] / b.rej[e], want, 0.1)

    for pat, expect in SELFTEST_PER_CLASS.items():
        if not _paths(pat):
            print(f"SKIP  per-class {pat}")
            continue
        a = Arm(pat, _paths(pat), eps=tuple(expect))
        for e, want in expect.items():
            for i, w in enumerate(want):
                check(f"class{i} {pat:46s} ε={e}", float(a.rej_per_class[e][i]), w)

    for pat, e, ddof, want in SELFTEST_STD:
        if not _paths(pat):
            print(f"SKIP  std {pat}")
            continue
        a = Arm(pat, _paths(pat), eps=(e,))
        got = float(np.std(a.rej_per_seed[e], ddof=ddof))
        check(f"std{ddof}    {pat:46s} ε={e}", got, want)

    print(f"\n{'SELFTEST PASS' if not bad else f'SELFTEST FAIL ({bad} mismatches)'} "
          f"({n - bad}/{n} checks)")
    return 1 if bad else 0


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", default=[],
                    help="LABEL=GLOB (repeatable), glob over roc .npz files")
    ap.add_argument("--eps", type=float, nargs="+", default=list(EPS_DEFAULT))
    ap.add_argument("--baseline", default=None, help="arm label to take retention against")
    ap.add_argument("--classes", default="g,q,W,Z,t",
                    help="class names for the per-class table ('' to skip)")
    ap.add_argument("--md", default=None, help="write the markdown table here")
    ap.add_argument("--json", default=None, help="write the full numbers here")
    ap.add_argument("--selftest", action="store_true",
                    help="recompute every published figure and exit nonzero on drift")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())
    if not a.arm:
        ap.error("give at least one --arm LABEL=GLOB (or --selftest)")

    arms = []
    for spec in a.arm:
        if "=" not in spec:
            ap.error(f"--arm needs LABEL=GLOB, got {spec!r}")
        label, pat = spec.split("=", 1)
        paths = _glob.glob(pat if os.path.isabs(pat) else os.path.join(REPO, pat))
        arms.append(Arm(label, paths, eps=a.eps))

    classes = [c for c in a.classes.split(",") if c] or None
    out = table(arms, a.baseline, classes)
    print(out)
    if a.md:
        with open(a.md, "w") as f:
            f.write(out + "\n")
        print(f"\n[wrote] {a.md}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({x.label: {"n": x.n, "n_seed": x.n_seed, "paths": x.paths,
                                 "auc": x.auc, "auc_sd_sample": x.auc_sd,
                                 "auc_per_seed": x.auc_per_seed,
                                 "rejection": {str(e): x.rej[e] for e in x.eps},
                                 "rejection_sd_pop": {str(e): x.rej_sd[e] for e in x.eps},
                                 "rejection_per_seed": {str(e): x.rej_per_seed[e] for e in x.eps},
                                 "rejection_per_class": {str(e): list(map(float, x.rej_per_class[e]))
                                                         for e in x.eps}}
                       for x in arms}, f, indent=2)
        print(f"[wrote] {a.json}")


if __name__ == "__main__":
    main()
