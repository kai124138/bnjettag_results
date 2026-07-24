"""Pure-numpy gold model — the executable spec of the BitNet forward pass.

Two activation-quantization modes:
  * 'dynamic' — exact QKeras training semantics (per-token absmax scale). Used to
    PROVE this reimplementation matches the trained model (vs roc-results/r5 npz).
  * 'static'  — the hardware-honest rebuild semantics: per-tensor fixed-point
    (k=1, i, f) quantizers with power-of-two step, calibrated on real jets. This is
    what HGQ2/hls4ml implement; the gold model measures the substitution cost in
    isolation, before any HGQ2 code exists.

Weights are the {−1,+1} binarized kernels with per-tensor β applied explicitly
(y = β·(x_q @ q) + b) — no folds in v1; folds are separate, individually-measured
optimizations.

Everything float32 matmuls (quantized values are exactly representable), float64
softmax at the head (mirrors make_roc.py).
"""
from __future__ import annotations

import numpy as np

EPS_NORM = 1e-6


def csd2_snap(beta: float) -> float:
    """Best 2-signed-digit (CSD) approximation 2^a ± 2^b of beta (b < a).
    Such constants multiply as one shift-add: DSP-free in Vitis by the
    ≤2-signed-digit rule — PROVIDED they are compile-time constants (affine
    layer weights, Latency-inlined), NOT Resource-mode ROM operands (measured
    2026-07-04: in-weight ±β̃ at RF=256 Resource inferred 256 real DSPs).
    db capped at 8 so every value is a multiple of 2^-16 (exact in f=16 grids)."""
    best, err = None, np.inf
    a0 = int(np.floor(np.log2(beta)))
    for a in (a0, a0 + 1):
        for sgn in (1, -1):
            for db in range(1, 9):
                v = 2.0 ** a + sgn * 2.0 ** (a - db)
                if v <= 0:
                    continue
                e = abs(v - beta)
                if e < err:
                    best, err = v, e
        e = abs(2.0 ** a - beta)  # single digit
        if e < err:
            best, err = 2.0 ** a, e
    return float(best)


def layer_norm(x):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + EPS_NORM)


def act_quant_dynamic(x, bits):
    """QKeras _absmax_quant with axis=-1 (per-token scale). np.round = TF round
    (both half-to-even)."""
    qpos = 2 ** (bits - 1) - 1
    qneg = -(2 ** (bits - 1))
    m = np.maximum(np.abs(x).max(axis=-1, keepdims=True), 1e-5)
    scale = qpos / m
    return np.clip(np.round(x * scale), qneg, qpos) / scale


def act_quant_static(x, bits, i):
    """Fixed-point (k=1, i integer bits, f = bits-1-i fractional): step 2^-f,
    range [-2^i, 2^i - 2^-f], round-half-even, saturate. Matches an HGQ/hls4ml
    ap_fixed<bits, i+1, RND, SAT>."""
    f = bits - 1 - i
    step = 2.0 ** (-f)
    lo = -(2.0 ** i)
    hi = 2.0 ** i - step
    return np.clip(np.round(x / step) * step, lo, hi)


class GoldModel:
    """cfg: the pipeline config dict. binz: binarize_checkpoint() output.
    pe: (n_part, d_model). calib: {layer_name: i_bits} for static mode."""

    def __init__(self, cfg, binz, pe, mode="dynamic", calib=None, beta_mode="exact"):
        self.arch = cfg["arch"]
        self.bits = cfg["quant"]["act_bits"]
        self.binz = binz
        self.pe = pe.astype(np.float32)
        self.mode = mode
        self.calib = calib or {}
        self.beta_mode = beta_mode  # 'exact' | 'pow2' (β snapped to 2^round(log2 β))
        self.d_head = self.arch["d_model"] // self.arch["n_heads"]
        self._taps = None  # optional per-layer activation taps

    def beta(self, name):
        b = self.binz[name]["beta"]
        if self.beta_mode == "pow2":
            return float(2.0 ** np.round(np.log2(b)))
        if self.beta_mode == "fold_csd2":
            fold = self.binz[name]["fold"]
            if fold in ("score_fold", "ln_killed", "bias_fold"):
                return 1.0  # exact: folded into softmax scale / next LN / bias
            return csd2_snap(b)  # explicit residual contributors + logits
        return float(b)

    def bias(self, name):
        e = self.binz[name]
        if e["bias"] is None:
            return None
        b = e["bias"].astype(np.float32)
        if self.beta_mode == "fold_csd2" and e["fold"] == "bias_fold":
            return b / np.float32(e["beta"])  # exact: next LN kills the 1/β scale
        return b

    def score_scale(self, blk):
        """Attention score multiplier: exact β_q·β_k/√d_head in fold mode (free —
        folds into the softmax exp LUT in hardware), else 1/√d_head."""
        s = 1.0 / np.sqrt(self.d_head)
        if self.beta_mode == "fold_csd2":
            s = s * self.binz[f"{blk}_attn_Wq"]["beta"] * self.binz[f"{blk}_attn_Wk"]["beta"]
        return np.float32(s)

    # ---- one BitLinear: LN -> actquant -> (+-1 matmul) -> beta,bias affine ----
    def bitlinear(self, name, x, observe=None):
        e = self.binz[name]
        h = layer_norm(x)
        if observe is not None:
            observe(name, h)
        if self.mode == "dynamic":
            hq = act_quant_dynamic(h, self.bits)
        else:
            hq = act_quant_static(h, self.bits, self.calib[name])
        z = hq.astype(np.float32) @ e["q"].astype(np.float32)
        y = np.float32(self.beta(name)) * z
        bias = self.bias(name)
        if bias is not None:
            y = y + bias
        if self._taps is not None:
            self._taps[name] = y
        return y

    def attention(self, blk, x, observe=None):
        A = self.arch
        B, T, D = x.shape
        H, dh = A["n_heads"], self.d_head
        q = self.bitlinear(f"{blk}_attn_Wq", x, observe)
        k = self.bitlinear(f"{blk}_attn_Wk", x, observe)
        v = self.bitlinear(f"{blk}_attn_Wv", x, observe)

        def split(t):  # (B,T,D) -> (B,H,T,dh)
            return t.reshape(B, T, H, dh).transpose(0, 2, 1, 3)

        qh, kh, vh = split(q), split(k), split(v)
        scores = qh @ kh.transpose(0, 1, 3, 2) * self.score_scale(blk)
        if self._taps is not None:
            self._taps[f"{blk}_scores"] = scores
        scores64 = scores.astype(np.float64)
        attn = np.exp(scores64 - scores64.max(axis=-1, keepdims=True))
        attn = (attn / attn.sum(axis=-1, keepdims=True)).astype(np.float32)
        ctx = (attn @ vh).transpose(0, 2, 1, 3).reshape(B, T, D)
        return self.bitlinear(f"{blk}_attn_Wo", ctx, observe)

    def ffn(self, blk, x, observe=None):
        h = self.bitlinear(f"{blk}_ffn_fc1", x, observe)
        h = np.maximum(h, 0)
        return self.bitlinear(f"{blk}_ffn_fc2", h, observe)

    def forward(self, X, observe=None, taps=False):
        """X (B, n_part, n_feat) -> logits (B, n_classes)."""
        self._taps = {} if taps else None
        x = self.bitlinear("input_proj", X.astype(np.float32), observe)
        x = x + self.pe[None, :, :]
        for i in range(self.arch["n_layers"]):
            blk = f"bit_block_{i}"
            x = x + self.attention(blk, x, observe)
            x = x + self.ffn(blk, x, observe)
        x = x.mean(axis=1)  # GAP
        x = self.bitlinear("head_fc1", x, observe)
        x = np.maximum(x, 0)
        return self.bitlinear("head_fc2", x, observe)

    # ---- calibration ----
    def calibrate(self, X, policy="mse_per_channel", i_candidates=(0, 1, 2, 3, 4, 5)):
        """Two-pass PTQ calibration of every quant site's integer-bit count.

        policy:
          'max'             — per-tensor i = ceil(log2(max|h|)) (covers everything)
          'mse_per_tensor'  — per-tensor i minimizing sum of squared quant error
          'mse_per_channel' — per-channel (last axis) MSE-optimal i (legal in the
                              hls4ml io_parallel flow: per-element wire widths)

        Pass 1 runs with a generous provisional i=4 everywhere; pass 2 re-collects
        with pass-1 choices so downstream sites see realistic upstream quantization.
        Returns {site: int or int-array(D)}.
        """
        bits = self.bits
        old_mode, old_calib = self.mode, dict(self.calib)
        self.mode = "static"

        def collect():
            stats = {}

            def observe(name, h):
                flat = h.reshape(-1, h.shape[-1]).astype(np.float64)
                st = stats.setdefault(name, {
                    "max": np.zeros(h.shape[-1]),
                    "sse": {i: np.zeros(h.shape[-1]) for i in i_candidates},
                    "n": 0,
                })
                st["max"] = np.maximum(st["max"], np.abs(flat).max(axis=0))
                for i in i_candidates:
                    e = act_quant_static(flat, bits, i) - flat
                    st["sse"][i] += (e * e).sum(axis=0)
                st["n"] += flat.shape[0]

            self.forward(X, observe=observe)
            return stats

        def choose(stats):
            calib = {}
            for name, st in stats.items():
                if policy == "max":
                    calib[name] = max(1, int(np.ceil(np.log2(st["max"].max() + 1e-12))))
                else:
                    sse = np.stack([st["sse"][i] for i in i_candidates])  # (C, D)
                    best = np.asarray(i_candidates)[np.argmin(sse, axis=0)]  # (D,)
                    if policy == "mse_per_tensor":
                        tot = sse.sum(axis=1)
                        calib[name] = int(np.asarray(i_candidates)[int(np.argmin(tot))])
                    else:
                        calib[name] = best.astype(np.int64)
            return calib

        self.calib = {}
        # provisional pass: every unseen site quantizes with i=4
        self.calib = {n: 4 for n in self.binz if not n.startswith("_")}
        self.calib = choose(collect())
        self.calib = choose(collect())  # pass 2 with realistic upstream quant
        self.mode = old_mode
        out = dict(self.calib)
        if old_mode == "dynamic":
            self.calib = old_calib
        return out


def softmax64(logits):
    z = logits.astype(np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def macro_ovr_auc(y_onehot, scores):
    from sklearn.metrics import roc_auc_score
    per = [roc_auc_score(y_onehot[:, c], scores[:, c]) for c in range(y_onehot.shape[1])]
    return float(np.mean(per)), [float(p) for p in per]
