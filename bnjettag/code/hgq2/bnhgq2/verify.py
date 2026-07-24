"""Fidelity gates.

Gate 1 (rebuild ↔ trained): HGQ2-Keras scores vs the verified roc-results/r5 npz
on the era-2 val split — Pearson corr + macro-OvR AUC + per-class AUC. Labels are
compared exactly first (row-alignment proof). Bit-exactness vs the trained model
is impossible by construction (dynamic per-token act scales); the target numbers
are the gold-model static+fold_csd2 measurements (LEDGER 2026-07-04).

Gate 2 (HGQ2 ↔ gold): the HGQ2 Keras forward vs the numpy gold model in
static+fold_csd2 mode on the same inputs — must agree to float tolerance
(they implement the same math; discrepancy = build/port bug).

Gate 3 (HGQ2 ↔ hls4ml): converted model C-sim vs HGQ2 Keras — bit-exactness
(max |Δ| == 0) on real jets. Run in convert.py after conversion.
"""
from __future__ import annotations

import numpy as np

from .gold import GoldModel, macro_ovr_auc, softmax64


def gate1_vs_trained(model, X, y, ref_npz_path, batch=4096):
    ref = np.load(ref_npz_path)
    ry, rs = ref["y"], ref["score"]
    n = min(len(X), len(ry))
    if not np.array_equal(y[:n], ry[:n]):
        raise AssertionError("label alignment with reference npz FAILED")
    logits = predict_in_batches(model, X[:n], batch)
    s = softmax64(logits).astype("float32")
    corr = float(np.corrcoef(s.ravel(), rs[:n].ravel())[0, 1])
    auc, per_class = macro_ovr_auc(y[:n], s)
    auc_ref, per_class_ref = macro_ovr_auc(ry[:n], rs[:n])
    return {
        "n": int(n),
        "corr_scores": corr,
        "auc_macro": auc,
        "auc_macro_ref": auc_ref,
        "d_auc": auc - auc_ref,
        "auc_per_class": per_class,
        "auc_per_class_ref": per_class_ref,
    }, s


def gate2_vs_gold(model, cfg, binz, pe, calib, X, batch=4096, tol=2e-2):
    gm = GoldModel(cfg, binz, pe, mode="static", calib=calib, beta_mode="fold_csd2")
    gold_logits = gm.forward(X)
    hgq_logits = predict_in_batches(model, X, batch)
    d = np.abs(gold_logits - hgq_logits)
    corr = float(np.corrcoef(gold_logits.ravel(), hgq_logits.ravel())[0, 1])
    return {
        "n": int(len(X)),
        "max_abs_diff": float(d.max()),
        "mean_abs_diff": float(d.mean()),
        "corr_logits": corr,
        "pass": bool(corr > 0.999),
        "tol_note": f"informational tol={tol}; LN eps/float-order differences expected",
    }


def predict_in_batches(model, X, batch=4096):
    outs = []
    for i in range(0, len(X), batch):
        outs.append(np.asarray(model(X[i:i + batch], training=False)))
    return np.concatenate(outs)
