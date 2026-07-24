#!/usr/bin/env python3
"""
Pull the BitLinear-transformer variant sweep results from W&B and print a
comparison table sorted by best val AUC.

Auth: uses WANDB_API_KEY from env, else ~/.netrc (machine api.wandb.ai).
Reads entity kayamaguchi-uc-san-diego / project bnjettag-bitnet.

Usage:  python fetch_results.py            # all var-* runs + the lr15 large anchor
        python fetch_results.py --csv out.csv
"""
import argparse
import os
import sys

ENTITY = "kayamaguchi-uc-san-diego"
PROJECT = "bnjettag-bitnet"
# include the existing large anchor so the size axis is complete
ANCHOR_SUBSTRINGS = ("paper-binary-D256",)  # lr15 large


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--prefix", default="var-", help="run-name prefix to include")
    args = ap.parse_args()

    try:
        import wandb
    except ImportError:
        sys.exit("wandb not installed in this interpreter; `pip install wandb`")

    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}")

    rows = []
    for r in runs:
        name = r.name or ""
        keep = name.startswith(args.prefix) or any(s in name for s in ANCHOR_SUBSTRINGS)
        if not keep:
            continue
        s = r.summary
        cfg = r.config

        def g(*keys, default=None):
            for k in keys:
                if k in s and s[k] is not None:
                    return s[k]
            for k in keys:
                if k in cfg and cfg[k] is not None:
                    return cfg[k]
            return default

        rows.append(dict(
            name=name,
            state=r.state,
            val_auc=g("val_auc", "best_val_auc", "val_auc_max"),
            val_loss=g("val_loss", "best_val_loss"),
            params=g("n_params", "params", "trainable_params"),
            D=g("d_model"), L=g("n_layers"), H=g("n_heads"), FFN=g("ffn_dim"),
            norm=g("norm_type"), placement=g("norm_placement"),
            posenc=g("pos_enc"), pool=g("pool"), ffn_act=g("ffn_act"),
            softmax_free=g("softmax_free"),
            epochs=g("epoch", "epochs"),
        ))

    def keyf(d):
        v = d["val_auc"]
        return v if isinstance(v, (int, float)) else -1.0

    rows.sort(key=keyf, reverse=True)

    hdr = ["run", "state", "val_auc", "params", "D", "L", "H", "FFN",
           "norm", "place", "posenc", "pool", "ffn_act", "sf"]
    widths = [30, 9, 9, 9, 4, 3, 3, 5, 10, 14, 9, 6, 8, 3]

    def fmt(vals):
        return "  ".join(str(v if v is not None else "-")[:w].ljust(w)
                         for v, w in zip(vals, widths))

    print(fmt(hdr))
    print("-" * (sum(widths) + 2 * len(widths)))
    for d in rows:
        va = d["val_auc"]
        va = f"{va:.4f}" if isinstance(va, (int, float)) else "-"
        pr = d["params"]
        pr = f"{int(pr):,}" if isinstance(pr, (int, float)) else "-"
        print(fmt([d["name"], d["state"], va, pr, d["D"], d["L"], d["H"], d["FFN"],
                   d["norm"], d["placement"], d["posenc"], d["pool"],
                   d["ffn_act"], d["softmax_free"]]))

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
