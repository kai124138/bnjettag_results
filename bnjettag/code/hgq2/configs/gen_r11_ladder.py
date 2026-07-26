#!/usr/bin/env python3
"""Round-11 capacity ladder — config generator.

Era-2 has a 332x hole on the capacity axis (19,201 -> 6,375,173 params) and the
standardized-input regime the flagship lives in has exactly one point.  Round 11
fills it: the three r8 arms re-run at four larger d_model, so background rejection
at fixed signal efficiency can be read against parameter count at fixed
preprocessing.

Scaling rule (fixed before any run): hold n_heads = 4 and n_layers = 2, scale
ffn_dim = 2 * d_model -- exactly the r8 `small` proportions (D32/H4/L2/FFN64).
Only d_model moves, so the curve has one independent variable.

Arms are the r8 arms verbatim, changed in nothing but the architecture block:
  w1a8-stdnn  binary_absmean + norm "none"  (the flagship arm, best binary)
  w8a8-std    int8_absmax    + norm "subln"
  fp32-std    none           + norm "subln"

Plus one control at the flagship scale, `r11-d32-w8a8-stdnn`: 8-bit weights with
the norms removed.  Every existing 8-bit arm carries SubLN while the binary
flagship does not, so any W1-vs-W8 resource comparison at d32 is confounded by
normalization.  This control removes it for three runs of cluster time.

Stage 0 is an LR probe: r7b's lesson is that an inherited peak LR invalidated a
whole column (tiny's optimum was 5x above small's).  Probe {1e-5, 2e-5, 5e-5} on
the binary arm at d64 and d128, one seed, then run the matrix at the winner.

Usage (from code/hgq2/configs/):
    python gen_r11_ladder.py            # write the configs
    python gen_r11_ladder.py --check    # print param counts, write nothing
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

SCALES = [48, 64, 96, 128]
LR_GRID = {"1e5": 1e-05, "2e5": 2e-05, "5e5": 5e-05, "1e4": 1e-04}
# Which peak LRs to probe at which scale. The first pass covered {1e-5, 2e-5, 5e-5}
# at d64 and d128; d64's optimum came out at 5e-5, i.e. ON THE EDGE of the grid,
# which is precisely the failure that invalidated round 7b's tiny column (its true
# optimum sat 5x above the inherited value). The grid is therefore extended upward
# and the two unprobed scales are covered directly rather than interpolated -- the
# LR is not monotone in scale in this project (tiny 1e-4, small 2e-5, large 5e-5),
# so interpolating between probed scales would be an assumption, not a measurement.
LR_PROBE = {48: ["2e5", "5e5"],
            64: ["1e5", "2e5", "5e5", "1e4"],
            96: ["2e5", "5e5", "1e4"],
            128: ["1e5", "2e5", "5e5", "1e4"]}

# Peak LR for the LADDER, one per scale, chosen from the stage-0 probe.
#
# Selection rule, fixed before the probe returned: take the LR whose best epoch falls
# INSIDE the 101-epoch budget (an optimum at the budget edge means the point is
# under-trained, which would flatten the capacity curve exactly where the hypothesis
# is being tested), and among those take the highest validation macro AUC.  Where two
# rungs tie, prefer the lower -- decisions.md 2026-07-16 chose the stable rung over the
# nominally-equal hotter one for the same reason.
#
# Stage-0 result (validation macro AUC, best_epoch in brackets, 1 seed):
#   d64   1e-5 0.90771 [85] · 2e-5 0.91915 [78] · 5e-5 0.91984 [29] · 1e-4 0.91988 [14]
#   d96   2e-5 0.92475 [79] · 5e-5 0.92832 [55]
#   d128  1e-5 0.92449 [100] · 2e-5 0.92951 [100] · 5e-5 0.93524 [97]
#   d48   2e-5 0.91219 [89] · 5e-5 0.91154 [24]
# d64's optimum is a PLATEAU from 5e-5 to 1e-4 and d96/d128 both turn back DOWN at 1e-4,
# so 5e-5 is an interior optimum, not a grid edge -- which is what the extended grid was
# added to check.  Note d128 at 1e-5 and 2e-5 both peak at epoch 100: those rungs are
# budget-limited, not converged, and the rule rejects them regardless of their AUC.
# d48 is the one scale where 2e-5 wins, by 0.065 pts on one seed -- inside noise, but the
# rule is applied as written rather than overridden for tidiness, and it also happens to
# interpolate d32's own tuned 2e-5.
LADDER_LR = {48: 2e-05, 64: 5e-05, 96: 5e-05, 128: 5e-05}

ARMS = {
    # name suffix -> (weight quant, act_bits, act_policy/calib keys, norm)
    "w1a8-stdnn": dict(weight="binary_absmean", act_bits=8, norm="none"),
    "w8a8-std": dict(weight="int8_absmax", act_bits=8, norm="subln"),
    "fp32-std": dict(weight="none", act_bits=32, norm="subln"),
    # Added 2026-07-25 after the d32 control: norm removal is worth ~+0.8 AUC pts to 8-bit
    # as well as ~+1.4 to binary, so a ladder whose binary arm is norm-free and whose 8-bit
    # arm is normed compares across the norm axis at EVERY scale -- the exact confound the
    # d32 control was built to remove. This arm makes the ladder's central W1-vs-W8
    # comparison matched.
    "w8a8-stdnn": dict(weight="int8_absmax", act_bits=8, norm="none"),
}


def arch(d_model: int, norm: str) -> dict:
    return {
        "n_part": 10,
        "n_feat": 16,
        "d_model": d_model,
        "n_heads": 4,
        "n_layers": 2,
        "ffn_dim": 2 * d_model,
        "n_classes": 5,
        "pool": "gap",
        "ffn_act": "relu",
        "norm": norm,
        "norm_placement": "per_linear",
        "pos_enc": "learned",
        "softmax_free": False,
        "input_std": True,
    }


def quant(weight: str, act_bits: int) -> dict:
    if weight == "none":
        return {"weight": "none", "act_bits": 32, "act_policy": "none", "calib_n": 8192}
    return {
        "weight": weight,
        "act_bits": act_bits,
        "act_policy": "static_per_tensor_mse_calibrated",
        "calib_n": 8192,
        "act_calib": "trainable",
    }


def train(lr: float) -> dict:
    return {
        "data": "/data/hls4ml_lhc_jet/train/train",
        "validation_split": 0.2,
        "epochs": 101,
        "batch": 256,
        "lr": lr,
        "warmup_epochs": 1,
        "decay_epochs": 100,
        "decay_power": 1.0,
        "beta2": 0.98,
        "weight_decay": 0.01,
        "clip_mode": "value",
        "clipvalue": 1.0,
        "clipnorm": 1.0,
        "es_patience": 15,
        "wandb_project": "bnjettag-final",
    }


HLS = {
    "backend": "Vitis",
    "part": "xcvu13p-flga2577-2-e",
    "clock_ns": 2.5,
    "rf": 256,
    "io": "io_parallel",
}


def cfg(name: str, d_model: int, armkey: str, lr: float = 2e-05) -> dict:
    a = ARMS[armkey]
    return {
        "name": name,
        "era": 2,
        "arch": arch(d_model, a["norm"]),
        "quant": quant(a["weight"], a["act_bits"]),
        "train": train(lr),
        "hls": HLS,
    }


def all_configs() -> dict:
    out = {}
    # stage 0 -- LR probe on the binary arm only
    for d, tags in LR_PROBE.items():
        for tag in tags:
            n = f"r11-lrprobe-d{d}-w1a8-stdnn-lr{tag}"
            out[n] = cfg(n, d, "w1a8-stdnn", lr=LR_GRID[tag])
    # stage 1 -- the ladder, each scale at its own probed peak LR
    for d in SCALES:
        for armkey in ARMS:
            n = f"r11-d{d}-{armkey}"
            out[n] = cfg(n, d, armkey, lr=LADDER_LR[d])
    # Norm-free controls at the flagship scale. Every 8-bit and FP32 arm in the project
    # carries SubLN while the binary flagship does not, so EVERY published W1-vs-W8 and
    # W1-vs-FP32 comparison crosses the norm axis. These two remove that confound.
    # The FP32 one is not optional: without it, "norm-free 8-bit beats full precision"
    # commits the same cross-axis error it was meant to expose (verification gate, 2026-07-25).
    for arm in ("w8a8-std", "fp32-std"):
        n = f"r11-d32-{arm.split('-')[0]}-stdnn"
        c = cfg(n, 32, arm)
        c["arch"]["norm"] = "none"
        c["name"] = n
        out[n] = c
    return out


def count_params(c: dict) -> int:
    """Build the QAT model and report its parameter count (needs .venv-hgq2)."""
    import sys

    sys.path.insert(0, os.path.dirname(HERE))
    from bnhgq2 import qat  # noqa: E402

    # enable_ebops=True matches bnhgq2/train.py:203, i.e. the AS-TRAINED count.
    # Building with False under-reports by 50 (the EBOPs counters) and produced a
    # table mixing conventions -- caught twice by the verification gate.
    m, _ = qat.build_qat_model(c, seed=1, enable_ebops=True)
    return int(m.count_params())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="build each model and print its param count; write nothing")
    args = ap.parse_args()

    cfgs = all_configs()
    if args.check:
        for n, c in sorted(cfgs.items()):
            A = c["arch"]
            print(f"{n:42s} d={A['d_model']:3d} ffn={A['ffn_dim']:3d} "
                  f"norm={A['norm']:5s} w={c['quant']['weight']:15s} "
                  f"lr={c['train']['lr']:g}  params={count_params(c):,}")
        return

    for n, c in sorted(cfgs.items()):
        p = os.path.join(HERE, f"{n}.json")
        with open(p, "w") as f:
            json.dump(c, f, indent=2)
            f.write("\n")
        print(f"wrote {p}")
    print(f"{len(cfgs)} configs")


if __name__ == "__main__":
    main()
