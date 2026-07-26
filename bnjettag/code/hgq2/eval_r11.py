#!/usr/bin/env python3
"""Round-11 evaluation driver: W&B checkpoints -> ROC .npz -> claim table.

`roc_final.py eval-all` cannot be used for round 11: it has no `--input-std`, and
every r11 arm is `input_std: true`, so each checkpoint must be evaluated per-seed
with its own standardization stats (experiment log 2026-07-24).  This script is
that loop, made repeatable instead of retyped.

It also fetches what `convert_final.fetch_checkpoint` does not: that helper pulls
only `model_best.keras` and `train_meta.json`, and the round-8 contract needs
`input_std.json` beside them.

Usage (from code/hgq2/, with WANDB_API_KEY exported and never echoed):
    python eval_r11.py --arms w1a8-stdnn w8a8-std fp32-std --scales 48 64 96 128
    python eval_r11.py --only r11-d32-w8a8-stdnn          # the control
    python eval_r11.py --table                            # rebuild tables only

Outputs: `bnjettag/roc-results/r11/R11-<ARM>-d<D>-s<seed>.npz` and, with --table,
per-scale markdown via `rejection.py`.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.abspath(os.path.join(REPO, ".."))
PY = os.path.join(ROOT, ".venv-hgq2", "bin", "python")
NPZ_DIR = os.path.join(REPO, "roc-results", "r11")
CKPT_DIR = os.path.join(REPO, "models", "r11")
DATA = os.path.join(ROOT, "data", "val")

# variant label roc_final.py expects, per arm
VARIANT = {"w1a8-stdnn": "w1a8", "w8a8-std": "w8a8", "fp32-std": "fp32",
           "w8a8-stdnn": "w8a8", "fp32-stdnn": "fp32"}


def fetch(run: str) -> str | None:
    """Pull model_best.keras + input_std.json for `run`; return the run dir."""
    dest = os.path.join(CKPT_DIR, run)
    kp = os.path.join(dest, run, "model_best.keras")
    isj = os.path.join(dest, run, "input_std.json")
    if os.path.isfile(kp) and os.path.isfile(isj):
        return os.path.join(dest, run)
    if not os.environ.get("WANDB_API_KEY"):
        sys.exit("WANDB_API_KEY not set (export WANDB_API_KEY=$(cat bnjettag/wandb-api-key.txt);"
                 " never echo it)")
    import wandb

    api = wandb.Api(timeout=180)
    hits = [r for r in api.runs("kayamaguchi-uc-san-diego/bnjettag-final") if r.name == run]
    if not hits:
        print(f"[skip] {run}: no W&B run")
        return None
    # Duplicates are expected, not exceptional: `backoffLimit: 2` means a job that is
    # retried (e.g. after the k8s admission failures of 2026-07-25) logs a second run
    # under the same display name. Prefer a finished run that actually carries the
    # checkpoint, newest first — the same rule the r10 job comment states for the ROC
    # fetcher. Requiring exactly one run silently drops real results.
    want = (f"{run}/model_best.keras", f"{run}/input_std.json")

    def rank(r):
        return (r.state == "finished", r.summary.get("best_val_macro_auc") is not None,
                str(r.created_at))

    for r in sorted(hits, key=rank, reverse=True):
        got = set()
        for f in r.files():
            if f.name in want:
                f.download(root=dest, replace=True)
                got.add(f.name)
        if len(got) == 2:
            if len(hits) > 1:
                print(f"[fetch] {run}  ({len(hits)} W&B runs; took state={r.state} {r.id})")
            else:
                print(f"[fetch] {run}")
            return os.path.join(dest, run)
    print(f"[skip] {run}: {len(hits)} W&B run(s), none carrying both {want}")
    return None


def evaluate(run: str, d: str, arm: str, seed: int) -> str | None:
    npz = os.path.join(NPZ_DIR, f"R11-{arm.upper()}-d{d}-s{seed}.npz")
    if os.path.isfile(npz):
        print(f"[have] {os.path.basename(npz)}")
        return npz
    rd = fetch(run)
    if rd is None:
        return None
    os.makedirs(NPZ_DIR, exist_ok=True)
    cmd = [PY, os.path.join(HERE, "roc_final.py"), "eval",
           "--keras", os.path.join(rd, "model_best.keras"),
           "--input-std", os.path.join(rd, "input_std.json"),
           "--variant", VARIANT[arm], "--seed", str(seed), "--size", f"d{d}",
           "--data", DATA, "--npz", npz, "--expect-n", "260000", "--batch", "4096"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("[roc]"):
            print("  " + line)
    if r.returncode != 0:
        print(f"[FAIL] {run}\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
        return None
    return npz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", nargs="+", default=["48", "64", "96", "128"])
    ap.add_argument("--arms", nargs="+", default=["w1a8-stdnn", "w8a8-std", "fp32-std"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--only", default="", help="a single config name, e.g. r11-d32-w8a8-stdnn")
    ap.add_argument("--table", action="store_true", help="(re)build the per-scale tables")
    a = ap.parse_args()

    jobs = ([(a.only.split("-d")[1].split("-")[0], a.only.split("-", 2)[2], s)
             for s in a.seeds] if a.only
            else [(d, arm, s) for d in a.scales for arm in a.arms for s in a.seeds])

    ok = 0
    for d, arm, s in jobs:
        run = f"r11-d{d}-{arm}-s{s}"
        ok += evaluate(run, d, arm, s) is not None
    print(f"\n[eval] {ok}/{len(jobs)} arrays present")

    if a.table:
        for d in a.scales:
            args = [PY, os.path.join(HERE, "rejection.py"), "--baseline", "FP32"]
            for arm in a.arms:
                lbl = {"w1a8-stdnn": "W1A8", "w8a8-std": "W8A8", "fp32-std": "FP32"}[arm]
                args += ["--arm", f"{lbl}={NPZ_DIR}/R11-{arm.upper()}-d{d}-s*.npz"]
            args += ["--md", os.path.join(NPZ_DIR, f"rejection_d{d}.md")]
            print(f"\n=== d{d} ===")
            subprocess.run(args)


if __name__ == "__main__":
    main()
