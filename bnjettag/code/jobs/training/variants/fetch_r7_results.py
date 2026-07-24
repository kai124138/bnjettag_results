#!/usr/bin/env python3
"""Pull the ROUND-7 ROC artifacts from W&B into bnjettag/r7/roc-results/.

The kai-roc-r7-gpu.yaml job uploads its 30 per-model npz + roc_auc.md +
roc_overlay_SMALL.png + roc_overlay_TINY.png to the W&B run `r7-roc-artifacts`
(project bnjettag-final) as the durable copy (PVC-free; the 2026-07-02 ceph incident
taught us not to trust the PVC alone). This fetches them back locally so the verification procedure recomputes AUC from the npz.

Sibling of fetch_final_results.py (unchanged) — same mechanism, round-7 run + landing dir.
Auth: WANDB_API_KEY from env, else ~/.netrc (machine api.wandb.ai).

Usage:
    python fetch_r7_results.py                 # -> bnjettag/r7/roc-results/
    python fetch_r7_results.py --dest /tmp/x   # custom dir
    python fetch_r7_results.py --list          # just list files, do not download
"""
import argparse
import os
import sys

ENTITY = "kayamaguchi-uc-san-diego"
PROJECT = "bnjettag-final"
RUN_NAME = "r7-roc-artifacts"
# repo-relative default landing dir: .../variants/ -> bnjettag is 4 levels up
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DEST = os.path.abspath(
    os.path.join(_HERE, "..", "..", "..", "..", "r7", "roc-results"))
KEEP_EXT = (".npz", ".md", ".png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=RUN_NAME, help="W&B run display name")
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--entity", default=ENTITY)
    ap.add_argument("--dest", default=DEFAULT_DEST, help="local landing dir")
    ap.add_argument("--list", action="store_true", help="list files, do not download")
    a = ap.parse_args()

    try:
        import wandb
    except ImportError:
        sys.exit("wandb not installed in this interpreter; `pip install wandb`")

    api = wandb.Api(timeout=120)
    runs = list(api.runs(f"{a.entity}/{a.project}", filters={"display_name": a.run}))
    if len(runs) != 1:
        sys.exit(f"{a.run}: {len(runs)} W&B runs match in {a.entity}/{a.project} "
                 f"(need exactly 1)")
    run = runs[0]

    files = [f for f in run.files() if f.name.endswith(KEEP_EXT)]
    if not files:
        sys.exit(f"{a.run}: no {KEEP_EXT} files attached")
    files.sort(key=lambda f: f.name)

    if a.list:
        for f in files:
            print(f"{f.size/1e6:8.2f} MB  {f.name}")
        return

    os.makedirs(a.dest, exist_ok=True)
    for f in files:
        f.download(root=a.dest, replace=True)
        print(f"[fetch] {f.name}  ({f.size/1e6:.2f} MB) -> {a.dest}")
    print(f"\n[done] {len(files)} files -> {a.dest}\n"
          f"       recompute AUC from the npz next (the ROC verification procedure).")


if __name__ == "__main__":
    main()
