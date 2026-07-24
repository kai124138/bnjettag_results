"""AbsMean binarization (BitNet arXiv:2310.11453) + the β-fold bookkeeping.

Replicates qkerasModel.AbsMeanQuantizer (binary branch) exactly:
    alpha = mean(W); Wc = W - alpha; beta = mean(|Wc|) + 1e-6; q = sign(Wc)
Effective inference weight = q * beta with q ∈ {−1, +1} (sign(0)=0 is a STOP
condition — hardware binary has no zero state; checked and reported here).

β-fold plan (decisions.md 2026-07-04, Decision 4) for the per_linear-SubLN graph:
    Wq, Wk : β_q·β_k joins the 1/√d_head score scale  → folded into softmax input scale
    Wv     : β_v dies in Wo's input LayerNorm          → dropped exactly
    fc1    : β_f1 dies in fc2's input LN; bias b/β     → folded into bias
    head_fc1: same as fc1 (head_fc2's LN kills it)     → folded into bias
    input_proj, Wo, fc2, head_fc2 : reach a residual add / the logits
                                                       → explicit affine scale kept
"""
from __future__ import annotations

import numpy as np

# which canonical layers keep an explicit output scale (residual contributors + logits)
EXPLICIT_SCALE = ("input_proj", "_attn_Wo", "_ffn_fc2", "head_fc2")
# which layers fold beta into their own bias (next-LN kills the scale)
BIAS_FOLD = ("_ffn_fc1", "head_fc1")
# which layers' beta joins the attention score scale
SCORE_FOLD = ("_attn_Wq", "_attn_Wk")
# which layers' beta vanishes in the next LayerNorm with no residual path
LN_KILLED = ("_attn_Wv",)


def absmean_binarize(w: np.ndarray):
    """Returns (q int8 in {−1,0,+1}, beta float, alpha float, n_zero int)."""
    w = np.asarray(w, dtype=np.float64)
    alpha = w.mean()
    wc = w - alpha
    beta = np.abs(wc).mean() + 1e-6
    q = np.sign(wc)
    n_zero = int((q == 0).sum())
    return q.astype(np.int8), float(beta), float(alpha), n_zero


def fold_class(layer_name: str) -> str:
    """How this layer's beta is handled: explicit | bias_fold | score_fold | ln_killed."""
    if any(layer_name.endswith(s) or layer_name == s for s in EXPLICIT_SCALE):
        return "explicit"
    if any(layer_name.endswith(s) or layer_name == s for s in BIAS_FOLD):
        return "bias_fold"
    if any(layer_name.endswith(s) for s in SCORE_FOLD):
        return "score_fold"
    if any(layer_name.endswith(s) for s in LN_KILLED):
        return "ln_killed"
    raise ValueError(f"no fold class for layer {layer_name}")


def binarize_checkpoint(layers: dict) -> dict:
    """Binarize every BitLinear. Returns per-layer dict with q/beta/alpha/fold/n_zero
    plus a summary. Raises nothing — the STOP decision belongs to the caller."""
    out, total_zeros = {}, 0
    for name, wb in layers.items():
        q, beta, alpha, n_zero = absmean_binarize(wb["kernel"])
        total_zeros += n_zero
        out[name] = {
            "q": q, "beta": beta, "alpha": alpha, "n_zero": n_zero,
            "bias": None if wb["bias"] is None else np.asarray(wb["bias"], np.float64),
            "fold": fold_class(name),
            "shape": tuple(wb["kernel"].shape),
        }
    out["_summary"] = {
        "n_layers": len(layers),
        "total_sign_zeros": total_zeros,  # must be 0 or the binary claim is void
    }
    return out
