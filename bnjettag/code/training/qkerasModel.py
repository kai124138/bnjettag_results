"""
BitNet-style 1-bit Transformer Jet Tagger
==========================================
Drop-in replacement for the QKeras CNN jet tagger.

Data (since 2026-07-01): the public **HLS4ML LHC Jet dataset (150 particles)**
(Zenodo; arXiv:1804.06913 high-level features, arXiv:1908.05318 constituent lists),
already staged on the NRP kai-data PVC at /data/hls4ml_lhc_jet/. All training going
forward uses this dataset (see the internal decisions log, 2026-07-01).

  - Input  shape : (batch, 10, 16)  [top-10 constituents by pT (high→low), 16 features each]
  - Output shape : (batch, 5)       [raw class logits g/q/W/Z/t, no softmax]
  - Loss         : categorical_crossentropy (from_logits)
  - Pruning callbacks and training loop are unchanged.

Architecture overview
---------------------
  Input (10×16)
    │
  BitLinear projection  →  (10×D_MODEL)   [1-bit weights]
    │
  Positional encoding   →  (10×D_MODEL)   [learned, lightweight]
    │
  × N_LAYERS of BitTransformerBlock:
      ├─ RMSNorm
      ├─ 1-bit Multi-Head Self-Attention  (Q/K/V projections are binary)
      ├─ residual add
      ├─ RMSNorm
      ├─ 1-bit FFN  (expand → contract, binary weights)
      └─ residual add
    │
  Global average pool   →  (D_MODEL,)
    │
  BitLinear head        →  (5,)            [class logits g/q/W/Z/t]

BitLinear implementation  (W1A8, per arXiv:2310.11453 "BitNet")
---------------------------------------------------------------
This follows the ORIGINAL BitNet paper (Wang et al., 2023): binary weights
(W~ = Sign(W - alpha), zero-mean-centralized, with absmean scale beta) and
8-bit absmax-quantized activations, with full LayerNorm/SubLN before the
activation quantizer. Straight-through estimators carry gradients through both
the weight Sign and the activation Clip (paper Sec 2.2).

  alpha = mean(W);  W_c = W - alpha
  W_q   = Sign(W_c)               in {-1,+1}        (beta = mean|W_c| scale, STE)
  x~    = Clip(round(x * 127/max|x|), -128, 127)    (8-bit absmax, STE)
  y     = x~ @ W_q * beta

Opt-in BN_TERNARY=1 instead uses the b1.58 follow-up (arXiv:2402.17764)
ternary {-1,0,+1} weight scheme (absmean scale, no centralization).

Opt-in BN_STANDARDIZE=1 (default 0 = unchanged) puts a FIXED per-feature affine
x' = (x - mean)/(std + eps) in front of the model. The 16 raw constituent features span a
2,979x range in std (GeV momenta vs O(1) angular features), which a per-tensor-beta binary
layer cannot compensate for — see the knob block below. The constants are fit on the train
split (non-padded particles only) and PERSISTED beside the checkpoint: they become part of
the model definition, and make_roc.py reads them back rather than re-fitting.
"""

import glob
import h5py
import hashlib
import json
import os
import time
import numpy as np
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Layer, Dense, GlobalAveragePooling1D, Input, Add, MultiHeadAttention
)
from tensorflow.keras.regularizers import l1
# NOTE (2026-07-13): `from sklearn.preprocessing import MinMaxScaler` used to sit here.
# It was NEVER called — a dead import that made this file *look* like it normalised its
# inputs when it did not (and the comment in main() claimed exactly that). Removed, and
# replaced by the explicit, opt-in BN_STANDARDIZE affine below. See the block near L150.
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.sparsity.keras import (
    prune, pruning_callbacks, pruning_schedule
)

# ─────────────────────────────────────────────
# Constants  (HLS4ML LHC Jet dataset, 150 particles — arXiv:1908.05318)
# ─────────────────────────────────────────────
# The dataset stores up to 150 zero-padded constituents × 16 features per jet
# (jetConstituentList). We keep the top-N by constituent pT, sorted high→low.
# N=10 (default, advisor guidance 2026-07-01): small fixed input matching the
# previous 10-particle setup; how input size affects AUC/latency/resources is a
# LATER study — that's exactly what the BN_N_PART knob is for.
N_FEAT          = 16
N_PART_PER_JET  = int(os.environ.get("BN_N_PART", 10))
# 5-class one-hot target (gluon / light-quark / W / Z / top), located BY NAME
# in the 'jets' array via 'jetFeatureNames' — never by hard-coded column index.
N_CLASSES       = 5
CLASS_LABELS    = ["j_g", "j_q", "j_w", "j_z", "j_t"]

# ─────────────────────────────────────────────
# Hyperparameters  (tunable; overridable via env for NRP multi-size runs)
# ─────────────────────────────────────────────
D_MODEL    = int(os.environ.get("BN_D_MODEL",  32))   # embedding dimension
N_HEADS    = int(os.environ.get("BN_N_HEADS",  4))    # attention heads (D_MODEL % N_HEADS == 0)
N_LAYERS   = int(os.environ.get("BN_N_LAYERS", 2))    # transformer blocks
FFN_DIM    = int(os.environ.get("BN_FFN_DIM",  64))   # feed-forward hidden dim (2-4 x D_MODEL)
DROPOUT    = float(os.environ.get("BN_DROPOUT", 0.0)) # set >0 only if overfitting is observed
L1_REG     = float(os.environ.get("BN_L1_REG", 1e-4)) # matches original regularisation
EPOCHS     = int(os.environ.get("BN_EPOCHS", 200))    # max epochs (EarlyStopping may stop sooner)
BATCH_SIZE = int(os.environ.get("BN_BATCH",  50))     # batch size
# Optimizer controls (opt-in via env; defaults reproduce prior behaviour).
# The BitNet paper (arXiv:2310.11453, Tables 5-7) trains 1-bit transformers with a LARGE
# peak LR (~1e-3), Adam beta=(0.9,0.98), polynomial-decay schedule, 750-update warmup,
# weight decay 0.01, and NO gradient clipping. Deep 1-bit stacks can be unstable on this
# jet task, so a mild grad-norm clip is available as a robustness knob (BN_CLIPNORM=0 gives
# the paper-faithful no-clip recipe). See REPORT.md "paper alignment".
LR            = float(os.environ.get("BN_LR", 0.0))          # >0 overrides "adam"; peak LR (paper ~1e-3)
WARMUP_EPOCHS = int(os.environ.get("BN_WARMUP_EPOCHS", 0))   # linear LR warmup over N epochs (paper: 750 updates)
DECAY_EPOCHS  = int(os.environ.get("BN_DECAY_EPOCHS", 0))    # >0: polynomial LR decay over N epochs after warmup
DECAY_POWER   = float(os.environ.get("BN_DECAY_POWER", 1.0)) # polynomial decay power (1.0 = linear)
BETA2         = float(os.environ.get("BN_BETA2", 0.999))     # Adam beta_2 (paper: 0.98)
WEIGHT_DECAY  = float(os.environ.get("BN_WEIGHT_DECAY", 0.0))# AdamW weight decay (paper: 0.01; 0 = off)
CLIPNORM      = float(os.environ.get("BN_CLIPNORM", 1.0))    # global grad-norm clip (paper: OFF; set 0 to disable)
# EarlyStopping controls (defaults reproduce upstream behaviour: monitor val_loss, patience 5).
ES_MONITOR    = os.environ.get("BN_ES_MONITOR", "val_loss") # e.g. "val_auc" to track the tagger metric
ES_MODE       = os.environ.get("BN_ES_MODE", "auto")        # "min"/"max"/"auto" (use "max" with val_auc)
ES_PATIENCE   = int(os.environ.get("BN_ES_PATIENCE", 5))    # epochs w/o improvement before stopping
# Weight precision. Default (0) reproduces upstream BINARY weights (tf.sign -> {-1,+1}).
# BN_TERNARY=1 switches to true BitNet b1.58 TERNARY weights (round/clip -> {-1,0,1}), which
# the docstring describes and which can zero out irrelevant inputs (more expressive). Opt-in
# deviation from upstream, documented in REPORT.md.
TERNARY       = os.environ.get("BN_TERNARY", "0") not in ("0", "", "false", "False", "no")
# Attention core. Default (0) = conventional softmax( QK^T/sqrt(d) ) attention.
# BN_SOFTMAX_FREE=1 swaps softmax for a fully-binarizable, FPGA-friendly score:
# ReLU(QK^T/sqrt(d)) with a CONSTANT 1/N normalization (no exp, no data-dependent
# division). softmax's exp+divide is the one non-binarizable, DSP/LUT-expensive op
# in the attention core; ReLU is a comparator/mux and the 1/N is a fixed shift, so
# this variant keeps the whole MHSA on LUT/logic fabric -- the research abstract's
# "softmax-free attention designed to remain fully binarizable". Opt-in; documented
# in REPORT.md. N (sequence length) = particles-per-jet = 10 here, so attention is
# short-range and the constant normalization is stable.
SOFTMAX_FREE  = os.environ.get("BN_SOFTMAX_FREE", "0") not in ("0", "", "false", "False", "no")
# Activation precision (the research abstract's "varying activation precision" axis).
# Default 8 = paper W1A8 (8-bit absmax activations, Qb=2^7). Lowering this to 6 or 4
# shrinks the activation datapath / on-chip storage and the multiplier bit-width feeding
# the popcount accumulators, at some cost to LLP tagging efficiency -- the sweep maps
# where that efficiency starts to degrade. Symmetric signed b-bit absmax: levels span
# [-2^(b-1), 2^(b-1)-1]. Opt-in; documented in REPORT.md.
ACT_BITS      = int(os.environ.get("BN_ACT_BITS", 8))
assert ACT_BITS >= 2, "BN_ACT_BITS must be >= 2 (need at least sign + 1 magnitude bit)"
# Model variant (the abstract's "1-bit vs the vanilla version" comparison).
#   "bitnet"  (default) : the 1-bit BitNet tagger -- binary {-1,+1} (or ternary if
#                         BN_TERNARY=1) weights + ACT_BITS absmax activations. The target model.
#   "vanilla"           : the FULL-PRECISION baseline -- identical architecture but NO weight
#                         or activation quantization (standard FP32 Dense + LayerNorm + softmax).
#                         The "what did going 1-bit cost us" reference (upper bound on efficiency).
#   "w8a8"              : a CONVENTIONAL 8-bit baseline -- 8-bit absmax weights (per-tensor) and
#                         8-bit absmax activations (per-token). The "1-bit vs ordinary quantization"
#                         reference. Same arch/recipe; only the quantizers change, so the three
#                         runs are a controlled apples-to-apples comparison.
VARIANT       = os.environ.get("BN_VARIANT", "bitnet").strip().lower()
assert VARIANT in ("bitnet", "vanilla", "w8a8"), \
    f"BN_VARIANT must be one of bitnet|vanilla|w8a8 (got {VARIANT!r})"

# ── Per-feature input standardization (BN_STANDARDIZE) — OPT-IN, DEFAULT OFF ──────────
# 0 (default) = exactly the historical behaviour: the raw top-N-by-pT constituent block is
# fed to the model as it comes out of the .h5, unscaled. Every existing checkpoint, job and
# result is therefore bit-identical. 1 = apply a FIXED per-feature affine
#     x' = (x - mean) / (std + eps)
# whose 16 mean / 16 std constants are FIT ON THE TRAIN SPLIT ONLY, over NON-PADDED
# particles only, and PERSISTED next to the checkpoint (they become part of the model
# definition — see save_input_standardizer / make_roc.py).
#
# WHY (measured 2026-07-13 on the real top-10-by-pT inputs, data/val/jetImage_*.h5):
# the 16 constituent features are NOT on a common scale. Raw per-feature std spans
# 0.0394 .. 117.49 — a 2,979x ratio: GeV-scale momenta (px/py/pz/e/pt, std ~78-117) sit
# alongside O(1) angular/relative features (etarel/phirel/ptrel/deltaR/costhetarel,
# std ~0.04-0.075). The model's first op is BitLinear(input_proj) with norm_inside=True,
# i.e. SubLN ACROSS the 16 features of each particle; the LN mean/std are dominated by the
# momenta, so the POST-LN per-feature std comes out ~1.40 for px/py/pz vs ~0.185 for every
# relative feature — an 8.9x imbalance.
#
# Why that specifically kills the BINARY model: a BitLinear's weights are {-beta,+beta}
# with a SINGLE per-tensor beta, so the layer computes beta * sum(+-x_i). There is no
# per-feature gain to learn: a feature arriving with 7.5x less amplitude contributes 7.5x
# less, permanently. FP32 fixes this instantly by learning w ~ 1/std; binary structurally
# cannot. The absmax activation quantizer then sets its range from the dominant momenta,
# costing the small features ~3 more bits of resolution. Standardizing BEFORE the LN (the
# two do NOT commute) drops the post-LN imbalance from 8.9x to 2.1x, lifting the relative
# features from std ~0.185 to ~0.7-1.2.
#
# FPGA cost: this is a FIXED per-feature affine — 16 constant multiplies + 16 adds, known
# at synthesis time — LUT-implementable via the same CSD constant-multiplier approach the
# HLS pipeline already uses for affines. It puts NO DSPs in the binary core. It is NOT a
# data-dependent normalization: no running statistics, no per-batch reduction, nothing to
# compute at inference. The constants are frozen numbers baked into the bitstream, and the
# padded-particle re-zero is a mux on the particle-valid bit the datapath already carries.
STANDARDIZE = os.environ.get("BN_STANDARDIZE", "0") not in ("0", "", "false", "False", "no")
STD_EPS     = float(os.environ.get("BN_STD_EPS", 1e-6))   # the eps in (std + eps); persisted

# Fraction of the train/ files held out INSIDE model.fit as the training monitor. Named
# (was a bare 0.20 literal at the fit call) so the standardizer can be fit on EXACTLY the
# portion Keras trains on and not one jet more — see main().
VALIDATION_SPLIT = 0.20

# Populated by load_hls4ml_jets from the file's `particleFeatureNames`, so the persisted
# standardizer constants are self-describing (constant i belongs to feature name i).
PART_FEATURE_NAMES = []

# ── Architectural-variant knobs (the "a bunch of different transformers" sweep axis) ──
# Every knob DEFAULTS to the current/upstream behaviour, so existing checkpoints and jobs
# are byte-for-byte unchanged; each non-default value is an opt-in architectural ablation
# trained from scratch on GPU. Grounded in the BitNet paper (SubLN, GELU FFN) + Russell's
# original qkerasModel.py. These let one model definition express the whole variant matrix.
#   BN_NORM_TYPE      "layernorm" (default): the paper's mean-subtracting SubLN (Eq 12),
#                       i.e. the existing RMSNorm class. "rmsnorm": TRUE RMSNorm
#                       x*rsqrt(mean(x^2)+eps) with NO mean subtraction (cheaper; drops the
#                       centering term). "none": identity (control: how much does norm buy?).
#   BN_NORM_PLACEMENT "per_linear" (default): a norm inside EVERY BitLinear (upstream).
#                       "shared_prenorm": ONE norm per sub-layer feeding Q/K/V and the FFN
#                       input — the canonical pre-norm transformer placement (~3x fewer norms).
#   BN_POS_ENC        "learned" (default): the upstream Embedding table — NOTE this is applied
#                       to a constant tf.range() so Keras folds it to a FIXED RANDOM offset that
#                       is NOT a tracked/trained weight (kept for back-compat with the lr15 ckpt).
#                       "learned_real": a genuinely trainable (N,D) position table (add_weight).
#                       "none": treat the jet as an unordered set (permutation-invariant).
#                       "sinusoidal": fixed sin/cos (Vaswani).
#   BN_POOL           "gap" (default): GlobalAveragePooling1D. "sum": sum over particles.
#                       "cls": a learnable BERT-style CLS token (sequence 10->11, read row 0).
#   BN_FFN_ACT        "relu" (default) | "gelu" (the BitNet paper's FFN nonlinearity).
NORM_TYPE      = os.environ.get("BN_NORM_TYPE", "layernorm").strip().lower()
NORM_PLACEMENT = os.environ.get("BN_NORM_PLACEMENT", "per_linear").strip().lower()
POS_ENC        = os.environ.get("BN_POS_ENC", "learned").strip().lower()
POOL           = os.environ.get("BN_POOL", "gap").strip().lower()
FFN_ACT        = os.environ.get("BN_FFN_ACT", "relu").strip().lower()
assert NORM_TYPE in ("layernorm", "rmsnorm", "none"), f"bad BN_NORM_TYPE={NORM_TYPE!r}"
assert NORM_PLACEMENT in ("per_linear", "shared_prenorm"), f"bad BN_NORM_PLACEMENT={NORM_PLACEMENT!r}"
assert POS_ENC in ("learned", "learned_real", "none", "sinusoidal"), f"bad BN_POS_ENC={POS_ENC!r}"
assert POOL in ("gap", "sum", "cls"), f"bad BN_POOL={POOL!r}"
assert FFN_ACT in ("relu", "gelu"), f"bad BN_FFN_ACT={FFN_ACT!r}"
_NORM_INSIDE_DEFAULT = (NORM_PLACEMENT == "per_linear")  # BitLinear self-norms iff per_linear


# ══════════════════════════════════════════════════════════════════════════════
# 1-BIT PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

class AbsMeanQuantizer(tf.keras.constraints.Constraint): #inherit from Constraint to apply after each optimizer step
    """
    Straight-through weight quantizer for BitLinear.

    Called INSIDE BitLinear.call (not as a Keras constraint) so the STE runs
    inside the gradient tape against a full-precision latent kernel, per the
    paper's mixed-precision training (latent weights binarized on the fly).

    Default = binary BitNet (arXiv:2310.11453, Eq 1-3,12): centralize to
    zero-mean then Sign, with absmean scale beta:
        W_q = Sign(W - mean(W)) * beta,   beta = mean|W - mean(W)|
    BN_TERNARY=1 = b1.58 (arXiv:2402.17764): absmean-scaled round/clip to
    {-1,0,+1} (no centralization).

    (Still subclasses tf.keras.constraints.Constraint only for backward
    compatibility with older checkpoints; the constraint hook is unused.)
    """
    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def __call__(self, w):
        # Straight-through estimator: quantize in forward, identity in backward.
        if TERNARY:
            # b1.58 (arXiv:2402.17764): absmean-scaled round/clip -> {-1,0,+1}.
            # No mean centralization (the 0 level already drops a weight).
            beta = tf.reduce_mean(tf.abs(w)) + self.eps
            w_scaled = w / beta
            q = tf.clip_by_value(tf.round(w_scaled), -1.0, 1.0)
        else:
            # Binary BitNet (arXiv:2310.11453, Eq 1-3,12): centralize weights to
            # zero-mean (alpha) BEFORE the sign, then absmean scale beta.
            alpha = tf.reduce_mean(w)                    # Eq 3
            w = w - alpha                                # centralize (Eq 1): W~ = Sign(W - alpha)
            beta = tf.reduce_mean(tf.abs(w)) + self.eps  # Eq 12 (on centralized W)
            w_scaled = w / beta
            q = tf.sign(w_scaled)                        # Eq 1-2
        w_q = w_scaled + tf.stop_gradient(q - w_scaled)  # stop_gradient => identity backward
        return w_q * beta  # rescale by the absmean factor beta

    def get_config(self):
        return {"eps": self.eps}

## Quant(x) Function
# b-bit symmetric signed absmax levels (set once from BN_ACT_BITS). At 8-bit this is
# exactly the paper's Qb=2^7 range: qpos=+127, qneg=-128 (refactor is a numerical no-op
# vs the old hard-coded 8-bit path). 6-bit -> [-32,+31]; 4-bit -> [-8,+7].
_ACT_QPOS = float(2 ** (ACT_BITS - 1) - 1)   # +127 (b8) / +31 (b6) / +7 (b4)
_ACT_QNEG = float(-(2 ** (ACT_BITS - 1)))    # -128 (b8) / -32 (b6) / -8 (b4)

def _absmax_quant(x, bits, axis):
    """Symmetric signed b-bit absmax quantizer with a straight-through estimator.

    forward = quantize, backward = identity (paper Sec 2.2). tf.round has zero gradient,
    so without the STE this severs gradients through every input -> the model freezes
    (the val_auc=0.5 bug). axis=-1 -> per-token (activations); axis=None -> per-tensor
    (weights). Used for the ACT_BITS activations AND the conventional W8A8 baseline.
    """
    qpos = float(2 ** (bits - 1) - 1)
    qneg = float(-(2 ** (bits - 1)))
    scale = qpos / tf.maximum(tf.reduce_max(tf.abs(x), axis=axis, keepdims=True), 1e-5)
    q = tf.clip_by_value(tf.round(x * scale), qneg, qpos)
    return (x * scale + tf.stop_gradient(q - x * scale)) / scale

def activation_quant(x):
    # b-bit absmax activation quant (paper Eq 4-5, Qb=2^(b-1)), per-token (axis=-1).
    # NOTE: per-token here; the paper uses per-tensor during training and per-token at
    # inference (Sec 2.1). Per-token is finer-grained, is what the b1.58 follow-up uses,
    # and is more stable on this small-sequence (N=10) task.
    return _absmax_quant(x, ACT_BITS, axis=-1)


class BitLinear(Layer):
    """
    A fully-connected layer with binary {-1, +1} weights.

    Replaces tf.keras.layers.Dense for all projections inside the
    transformer.  Bias uses full float32 (bias contributes negligible
    parameter count and is critical for representational capacity at
    small D_MODEL).

    Args:
        units      : output dimensionality
        use_bias   : whether to add a bias term (default True)
        reg        : L1 regularisation strength on the kernel
        name       : layer name

    Calls activation_quant to quantize RMSNorm activation.
    """
    def __init__(self, units, use_bias=True, reg=L1_REG, norm_inside=None, **kwargs):
        super().__init__(**kwargs)
        self.units    = units
        self.use_bias = use_bias
        self.reg      = reg
        # Does this BitLinear apply its own SubLN before quantizing? Defaults to
        # BN_NORM_PLACEMENT (True=per_linear/upstream; False=shared_prenorm, where the
        # enclosing block owns one shared norm). Saved in get_config so new checkpoints are
        # self-describing; old checkpoints (no key) fall back to the per_linear default.
        self.norm_inside = _NORM_INSIDE_DEFAULT if norm_inside is None else bool(norm_inside)

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        self.kernel = self.add_weight(
            name        = "kernel",
            shape       = (in_dim, self.units),
            initializer = "glorot_uniform",
            regularizer = l1(self.reg),
            trainable   = True,
        )   # FIX: NO constraint -> kernel stays full-precision (latent master weight).
            # Quantization moved into call() so the weight STE runs inside the gradient
            # tape and Adam's sub-threshold updates accumulate (QAT, per the docstring).
        self.quantizer = AbsMeanQuantizer()
        if self.use_bias:
            self.bias = self.add_weight(
                name        = "bias",
                shape       = (self.units,),
                initializer = "zeros",
                regularizer = l1(self.reg),
                trainable   = True,
            )
        self.built = True

    def call(self, x):
        # LayerNorm/SubLN first (shared by all variants), then the variant selects the
        # numeric precision of the matmul. For "bitnet" we quantize the kernel INSIDE the
        # forward pass (STE) against the full-precision latent master weight Adam updates;
        # combined with the activation STE, gradients flow through the whole stack.
        # make_norm() picks the norm type (BN_NORM_TYPE); when norm_inside is False
        # (shared_prenorm) the enclosing block has already normed, so this is a no-op.
        x_norm = make_norm(name=self.name + "_norm")(x) if self.norm_inside else x
        if VARIANT == "vanilla":
            # Full-precision baseline: standard Dense (no weight/activation quantization).
            out = tf.matmul(x_norm, self.kernel)
        elif VARIANT == "w8a8":
            # Conventional 8-bit baseline: 8-bit absmax activations (per-token) x 8-bit
            # absmax weights (per-tensor) -- ordinary fixed-precision quantization.
            out = tf.matmul(_absmax_quant(x_norm, 8, axis=-1),
                            _absmax_quant(self.kernel, 8, axis=None))
        else:  # "bitnet": binary (or ternary) weights x ACT_BITS absmax activations
            out = tf.matmul(activation_quant(x_norm), self.quantizer(self.kernel))
        if self.use_bias:
            out = out + self.bias
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units, "use_bias": self.use_bias,
                    "reg": self.reg, "norm_inside": self.norm_inside})
        return cfg


# ══════════════════════════════════════════════════════════════════════════════
# NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

class RMSNorm(Layer):
    """
    Normalisation applied before activation quantization in each BitLinear.

    NOTE: despite the class name, this subtracts the mean -> it is full
    LayerNorm / SubLN, exactly as the BitNet paper specifies (arXiv:2310.11453,
    Eq 12: LN(x) = (x - E(x)) / sqrt(Var(x) + eps)). The paper introduces LN
    before absmax activation quantization to preserve the output variance
    (Var(y) ~ 1), which stabilises 1-bit training. Kept as LayerNorm to align
    with the paper; the class name is retained for checkpoint/back-compat.

      y = (x - E(x)) / sqrt( Var(x) + eps )
    """
    def __init__(self, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        var = tf.math.reduce_variance(x, axis=-1, keepdims=True)
        return (x - mean) / tf.sqrt(var + self.eps)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"eps": self.eps})
        return cfg


class TrueRMSNorm(Layer):
    """TRUE RMSNorm (Zhang & Sennrich 2019): y = x * rsqrt(mean(x^2) + eps).

    Unlike the (misnamed) ``RMSNorm`` above — which subtracts the mean and is therefore
    full LayerNorm / SubLN — this drops the mean-centering term entirely. It is the
    ``BN_NORM_TYPE=rmsnorm`` variant: an ablation of "does the BitNet paper's mean
    subtraction actually matter for 1-bit jet tagging?" In hardware it also removes the
    running-mean subtract (one fewer reduction per token), leaving only Sx^2 + inv-sqrt.
    No affine (gamma/beta), to match the parameter-free RMSNorm used elsewhere here.
    """
    def __init__(self, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, x):
        ms = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
        return x * tf.math.rsqrt(ms + self.eps)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"eps": self.eps})
        return cfg


class IdentityNorm(Layer):
    """No normalisation (``BN_NORM_TYPE=none``): the lower-bound control. 1-bit stacks
    usually NEED a norm for output-variance control (BitNet Sec 2), so this is expected
    to train worse — it quantifies how much the SubLN actually buys on this task."""
    def call(self, x):
        return x


def make_norm(name=None):
    """Norm-layer factory selected by ``BN_NORM_TYPE``. Default 'layernorm' returns the
    upstream mean-subtracting ``RMSNorm`` with the SAME name pattern, so loading existing
    checkpoints is unaffected. 'rmsnorm' -> TrueRMSNorm; 'none' -> IdentityNorm."""
    if NORM_TYPE == "rmsnorm":
        return TrueRMSNorm(name=name)
    if NORM_TYPE == "none":
        return IdentityNorm(name=name)
    return RMSNorm(name=name)


class CLSToken(Layer):
    """Prepend a single learnable BERT-style classification token (``BN_POOL=cls``).

    Sequence length 10 -> 11; the transformer mixes the CLS row with every particle, and
    the head reads row 0 as the pooled jet representation — an alternative to averaging
    over particles (GlobalAveragePooling1D). The token is a learnable (1,1,D) weight."""
    def build(self, input_shape):
        d = int(input_shape[-1])
        self.cls = self.add_weight(name="cls_vec", shape=(1, 1, d),
                                   initializer="glorot_uniform", trainable=True)
        self.built = True

    def call(self, x):
        B = tf.shape(x)[0]
        tok = tf.tile(self.cls, [B, 1, 1])     # (B,1,D)
        return tf.concat([tok, x], axis=1)     # (B, N+1, D)


def _sinusoidal_pos(n, d):
    """Fixed sinusoidal positional encoding (Vaswani et al.), as a (n,d) tf constant.
    Used for ``BN_POS_ENC=sinusoidal`` — a non-learned alternative to the Embedding table."""
    pos = np.arange(n)[:, None].astype(np.float32)
    i   = np.arange(d)[None, :].astype(np.float32)
    ang = pos / np.power(10000.0, (2.0 * np.floor(i / 2.0)) / float(d))
    pe = np.zeros((n, d), dtype=np.float32)
    pe[:, 0::2] = np.sin(ang[:, 0::2])
    pe[:, 1::2] = np.cos(ang[:, 1::2])
    return tf.constant(pe)


class LearnedPosEnc(Layer):
    """A genuinely TRAINABLE positional encoding (``BN_POS_ENC=learned_real``).

    The upstream default (``BN_POS_ENC=learned``) builds ``Embedding(N,D)(tf.range(N))``;
    because ``tf.range`` is a plain constant (not a KerasTensor wired from the input), Keras
    folds that Embedding into a FIXED RANDOM offset whose weights are never tracked or updated
    — i.e. the "learned" PE never actually learns (verified: it adds 0 trainable params).
    This layer instead allocates the (N,D) table via ``add_weight`` so it IS tracked and
    trained, exactly like the CLS token. Broadcast-added over the batch."""
    def build(self, input_shape):
        n = int(input_shape[-2]); d = int(input_shape[-1])
        self.pos = self.add_weight(name="pos_table", shape=(1, n, d),
                                   initializer="glorot_uniform", trainable=True)
        self.built = True

    def call(self, x):
        return x + self.pos


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER BLOCK
# ══════════════════════════════════════════════════════════════════════════════

class BitMHSA(Layer):
    """
    1-bit Multi-Head Self-Attention.

    Q, K, V projections and the output projection all use BitLinear
    (binary weights).  The attention scores themselves remain in float32 —
    quantising attention logits severely harms performance at small scale.

    Attention core is selectable (env BN_SOFTMAX_FREE):
      * 0 (default): conventional softmax( QK^T / sqrt(d) ).
      * 1: softmax-free ReLU(QK^T / sqrt(d)) with constant 1/N normalization —
        no exp / no row-sum division, so it stays fully binarizable on FPGA
        LUT/logic fabric (the research abstract's binarizable-attention axis).

    Args:
        d_model  : total model dimension
        n_heads  : number of attention heads (d_model % n_heads == 0)
        reg      : L1 regularisation on projection weights
    """
    def __init__(self, d_model, n_heads, reg=L1_REG, **kwargs):
        super().__init__(**kwargs)
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.scale    = tf.math.sqrt(tf.cast(self.d_head, tf.float32))
        self.reg      = reg

    def build(self, input_shape):
        _shared = (NORM_PLACEMENT == "shared_prenorm")
        _ni = not _shared   # sub-linears self-norm only in per_linear (upstream) mode
        self.pre_norm = make_norm(name=self.name + "_prenorm") if _shared else None
        self.W_q = BitLinear(self.d_model, use_bias=False, reg=self.reg,
                             norm_inside=_ni, name=self.name + "_Wq")
        self.W_k = BitLinear(self.d_model, use_bias=False, reg=self.reg,
                             norm_inside=_ni, name=self.name + "_Wk")
        self.W_v = BitLinear(self.d_model, use_bias=False, reg=self.reg,
                             norm_inside=_ni, name=self.name + "_Wv")
        self.W_o = BitLinear(self.d_model, use_bias=True,  reg=self.reg,
                             norm_inside=_ni, name=self.name + "_Wo")
        self.built = True

    def call(self, x, training=False):
        B  = tf.shape(x)[0]
        N  = tf.shape(x)[1]   # sequence length = N_PART_PER_JET (default 10; +1 with CLS)

        # In shared_prenorm, normalise ONCE here and feed the same normed tensor to Q/K/V
        # (whose BitLinears have norm_inside=False); W_o then operates on the attention
        # context directly. In per_linear (default) pre_norm is None and each BitLinear
        # self-norms, exactly as upstream.
        xqkv = self.pre_norm(x) if self.pre_norm is not None else x

        # Project with binary weights  →  (B, N, d_model)
        Q = self.W_q(xqkv)
        K = self.W_k(xqkv)
        V = self.W_v(xqkv)

        # Split into heads  →  (B, n_heads, N, d_head)
        def split_heads(t):
            t = tf.reshape(t, (B, N, self.n_heads, self.d_head))
            return tf.transpose(t, perm=[0, 2, 1, 3])

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # Scaled dot-product scores  →  (B, heads, N, N)
        attn_logits = tf.matmul(Q, K, transpose_b=True) / self.scale

        if SOFTMAX_FREE:
            # Softmax-free, fully-binarizable attention: ReLU scores + CONSTANT 1/N
            # normalization. No exp and no row-sum division (the FPGA-expensive,
            # non-binarizable parts of softmax) — only a ReLU (comparator) and a
            # fixed 1/N scale, so the attention core stays on LUT/logic fabric.
            attn_weights = tf.nn.relu(attn_logits)                       # (B, heads, N, N), ≥0
            ctx = tf.matmul(attn_weights, V) / tf.cast(N, tf.float32)    # (B, heads, N, d_head)
        else:
            # Conventional softmax attention (default / paper core).
            attn_weights = tf.nn.softmax(attn_logits, axis=-1)           # (B, heads, N, N)
            ctx = tf.matmul(attn_weights, V)                            # (B, heads, N, d_head)

        # Merge heads  →  (B, N, d_model)
        ctx = tf.transpose(ctx, perm=[0, 2, 1, 3])
        ctx = tf.reshape(ctx, (B, N, self.d_model))

        # Output projection (binary)
        return self.W_o(ctx)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "n_heads": self.n_heads,
                    "reg": self.reg})
        return cfg


class BitFFN(Layer):
    """
    1-bit Feed-Forward Network.
    Two BitLinear layers with a ReLU in between:
      x → BitLinear(ffn_dim) → ReLU → BitLinear(d_model)
    """
    def __init__(self, d_model, ffn_dim, reg=L1_REG, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.ffn_dim = ffn_dim
        self.reg     = reg

    def build(self, input_shape):
        _shared = (NORM_PLACEMENT == "shared_prenorm")
        _ni = not _shared
        self.pre_norm = make_norm(name=self.name + "_prenorm") if _shared else None
        self.fc1 = BitLinear(self.ffn_dim, reg=self.reg, norm_inside=_ni, name=self.name+"_fc1")
        self.fc2 = BitLinear(self.d_model, reg=self.reg, norm_inside=_ni, name=self.name+"_fc2")
        self.built = True

    def call(self, x):
        if self.pre_norm is not None:
            x = self.pre_norm(x)          # shared_prenorm: norm once at the FFN input
        x = self.fc1(x)
        x = tf.nn.gelu(x) if FFN_ACT == "gelu" else tf.nn.relu(x)
        x = self.fc2(x)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "ffn_dim": self.ffn_dim,
                    "reg": self.reg})
        return cfg


class BitTransformerBlock(Layer):
    """
    One transformer block with pre-norm and residual connections:
      x → BitMHSA → + residual
     → BitFFN  → + residual
    """
    def __init__(self, d_model, n_heads, ffn_dim, reg=L1_REG, **kwargs):
        super().__init__(**kwargs)
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.ffn_dim  = ffn_dim
        self.reg      = reg

    def build(self, input_shape):
        self.attn  = BitMHSA(self.d_model, self.n_heads,
                              reg=self.reg, name=self.name + "_attn")
        self.ffn   = BitFFN(self.d_model, self.ffn_dim,
                             reg=self.reg, name=self.name + "_ffn")
        self.built = True

    def call(self, x, training=False):
        # Self-attention sub-layer
        x = x + self.attn(x, training=training)
        # Feed-forward sub-layer
        x = x + self.ffn(x)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "n_heads": self.n_heads,
                    "ffn_dim": self.ffn_dim, "reg": self.reg})
        return cfg


# ══════════════════════════════════════════════════════════════════════════════
# FULL MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_bitnet_jet_tagger(
    n_particles : int   = N_PART_PER_JET,
    n_features  : int   = N_FEAT,
    d_model     : int   = D_MODEL,
    n_heads     : int   = N_HEADS,
    n_layers    : int   = N_LAYERS,
    ffn_dim     : int   = FFN_DIM,
    reg         : float = L1_REG,
) -> Model:
    """
    Build the 1-bit Transformer jet tagger.

    Input  : (batch, n_particles, n_features)  →  same as QKeras CNN
    Output : (batch, 1)                         →  raw logit, no sigmoid

    The model is permutation-equivariant up to the final GlobalAvgPool,
    which makes it a proper Deep-Sets / set-transformer for jet physics.

    Usage
    -----
    model = build_bitnet_jet_tagger()
    model.summary()
    model.compile(loss="binary_crossentropy", optimizer="adam",
                  metrics=["binary_accuracy"])
    """
    if VARIANT == "vanilla":
        _wstr, _astr = "FULL-PRECISION FP32 (vanilla baseline)", "FP32 (no quant)"
    elif VARIANT == "w8a8":
        _wstr, _astr = "INT8 per-tensor (W8 baseline)", "8-bit absmax (A8)"
    else:
        _wstr = "TERNARY {-1,0,+1}" if TERNARY else "BINARY {-1,+1}"
        _astr = f"{ACT_BITS}-bit absmax [{int(_ACT_QNEG)},{int(_ACT_QPOS)}]"
    print(f"[arch] variant={VARIANT.upper()}  weights={_wstr}  act={_astr}  "
          f"attention={'SOFTMAX-FREE ReLU(QK^T)/N (binarizable)' if SOFTMAX_FREE else 'softmax'}  "
          f"d_model={d_model} heads={n_heads} layers={n_layers} ffn={ffn_dim}")
    print(f"[arch] norm={NORM_TYPE} placement={NORM_PLACEMENT} pos_enc={POS_ENC} "
          f"pool={POOL} ffn_act={FFN_ACT}")

    # ── Input ────────────────────────────────────────────────────────────────
    inputs = Input(shape=(n_particles, n_features), name="input_1")

    # ── Input projection: N_FEAT → D_MODEL  (binary BitLinear) ──────────────
    # Mirrors QConv1D(kernel_size=1) which is mathematically identical
    # to a per-particle Dense / BitLinear applied independently.
    x = BitLinear(d_model, reg=reg, name="input_proj")(inputs)
    # shape: (batch, 10, d_model)

    # ── Positional encoding (BN_POS_ENC)  ─────────────────────────────────────
    # Particles are unordered in principle. "learned" (default) gives the model a
    # learnable position token to discover any residual pT-ordering present in the
    # inputs; "sinusoidal" uses a fixed Vaswani table; "none" treats the jet as a pure
    # set (permutation-invariant up to the final pool) — often the right inductive bias.
    if POS_ENC == "learned":
        pos_emb = tf.keras.layers.Embedding(
            input_dim   = n_particles,
            output_dim  = d_model,
            name        = "pos_embedding"
        )(tf.range(n_particles))                 # shape: (10, d_model)
        x = x + pos_emb                          # broadcast over batch
    elif POS_ENC == "learned_real":
        x = LearnedPosEnc(name="pos_enc_learned")(x)   # genuinely trainable (N,D) table
    elif POS_ENC == "sinusoidal":
        x = x + _sinusoidal_pos(n_particles, d_model)
    # else "none": no positional information added.

    # ── Optional learnable CLS token (BN_POOL=cls)  ───────────────────────────
    if POOL == "cls":
        x = CLSToken(name="cls_token")(x)        # sequence n_particles -> n_particles+1

    # ── Transformer blocks  ───────────────────────────────────────────────────
    for i in range(n_layers):
        x = BitTransformerBlock(
            d_model  = d_model,
            n_heads  = n_heads,
            ffn_dim  = ffn_dim,
            reg      = reg,
            name     = f"bit_block_{i}"
        )(x)
    # shape: (batch, 10, d_model)


    # ── Pool sequence → vector (BN_POOL)  ─────────────────────────────────────
    if POOL == "cls":
        # Read the CLS row (BERT-style); the transformer has mixed it with all particles.
        x = tf.keras.layers.Lambda(lambda t: t[:, 0], name="cls_pool")(x)
    elif POOL == "sum":
        x = tf.keras.layers.Lambda(lambda t: tf.reduce_sum(t, axis=1), name="sum_pool")(x)
    else:  # "gap" (default): aggregate over particles
        x = GlobalAveragePooling1D(name="global_average_pooling1d")(x)
    # shape: (batch, d_model)

    # ── Classification head  ──────────────────────────────────────────────────
    # Two BitLinear layers to mirror two QDense layers.
    x = BitLinear(d_model, reg=reg, name="head_fc1")(x)
    x = tf.keras.layers.Activation("relu", name="head_act")(x)

    outputs = BitLinear(N_CLASSES, reg=reg, name="head_fc2")(x)
    # shape: (batch, 5)  — raw class logits (g/q/W/Z/t), no softmax  ✓

    return Model(inputs=inputs, outputs=outputs, name=f"{VARIANT}_jet_tagger")


# ══════════════════════════════════════════════════════════════════════════════
# DATA — HLS4ML LHC Jet dataset (150 particles)
# ══════════════════════════════════════════════════════════════════════════════

def load_hls4ml_jets(data_dir, n_part=None):
    """Load a directory of jetImage_*_150p_*.h5 files (HLS4ML LHC Jet dataset,
    Zenodo / arXiv:1908.05318; on NRP under /data/hls4ml_lhc_jet/{train/train,val/val}).

    Per file:
      jetConstituentList   (n, 150, 16) float — zero-padded per-particle features
      jets                 (n, n_jetfeat)     — high-level jet features incl. the
                                                one-hot labels j_g/j_q/j_w/j_z/j_t
      jetFeatureNames / particleFeatureNames  — column names (bytes); labels and
                                                pT columns are located BY NAME

    Constituents are re-sorted by constituent pT (descending) before truncation to
    n_part, so "top-N" is guaranteed rather than assumed from file ordering.

    Returns: X (N, n_part, 16) float32, Y (N, 5) float32 one-hot, jet_pt (N,) float32.
    """
    n_part = n_part or N_PART_PER_JET
    files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
    if not files:
        raise FileNotFoundError(f"no .h5 files found in {data_dir}")

    def _names(ds):
        return [n.decode() if isinstance(n, bytes) else str(n) for n in ds]

    Xs, Ys, PTs = [], [], []
    for fp in files:
        with h5py.File(fp, "r") as hf:
            const  = hf["jetConstituentList"][:]
            jets   = hf["jets"][:]
            jnames = _names(hf["jetFeatureNames"][:])
            pnames = _names(hf["particleFeatureNames"][:])
        missing = [l for l in CLASS_LABELS if l not in jnames]
        if missing:
            raise KeyError(f"{fp}: label columns {missing} not in jetFeatureNames={jnames}")
        lab_idx = [jnames.index(l) for l in CLASS_LABELS]
        # Remember the per-feature column names so the standardizer constants are
        # self-describing (no behaviour change; PART_FEATURE_NAMES is read-only elsewhere).
        if not PART_FEATURE_NAMES:
            PART_FEATURE_NAMES.extend(pnames)

        # top-n_part by constituent pT (find the pT column by name; padded rows are 0)
        pt_col = next((i for i, n in enumerate(pnames) if n.endswith("_pt")), None)
        if pt_col is not None:
            order = np.argsort(-const[:, :, pt_col], axis=1, kind="stable")
            const = np.take_along_axis(const, order[:, :, None], axis=1)
        else:
            print(f"[data][warn] {os.path.basename(fp)}: no '*_pt' in "
                  f"particleFeatureNames — keeping stored constituent order")
        Xs.append(const[:, :n_part, :].astype(np.float32))
        Ys.append(jets[:, lab_idx].astype(np.float32))
        jpt = jnames.index("j_pt") if "j_pt" in jnames else None
        PTs.append(jets[:, jpt].astype(np.float32) if jpt is not None
                   else np.zeros(len(jets), dtype=np.float32))
    X, Y, jet_pt = np.concatenate(Xs), np.concatenate(Ys), np.concatenate(PTs)
    assert X.shape[1:] == (n_part, N_FEAT), f"bad X shape {X.shape}"
    assert Y.shape[1] == N_CLASSES and np.all(Y.sum(axis=1) == 1.0), \
        "labels are not one-hot 5-class"
    print(f"[data] {len(files)} files → X{X.shape}  Y{Y.shape}  "
          f"class counts: " + ", ".join(f"{c}={int(n)}" for c, n
                                        in zip(CLASS_LABELS, Y.sum(axis=0))))
    return X, Y, jet_pt


# ══════════════════════════════════════════════════════════════════════════════
# INPUT STANDARDIZATION  (BN_STANDARDIZE; opt-in, default OFF — see the knob at L150)
# ══════════════════════════════════════════════════════════════════════════════
# The 16 mean + 16 std constants ARE PART OF THE MODEL DEFINITION. A checkpoint trained
# with BN_STANDARDIZE=1 is only correct if inference applies the IDENTICAL affine, so the
# constants are persisted next to the .h5 (and to W&B) and read back by make_roc.py.
# They are NEVER re-fit on an eval split.
#
# make_roc.py carries a numpy-only MIRROR of _padding_mask / apply_input_standardizer /
# load_input_standardizer (it must standardize without importing TF, and the ROC job ships
# that file alone via a ConfigMap) — the same convention as bnhgq2/data.py mirroring
# make_roc.load_eval_set. Drift between the two copies is caught HARD, not silently: the
# sidecar carries `probe_sha256`, the hash of this transform applied to a fixed synthetic
# tensor that CONTAINS PADDED ROWS; make_roc recomputes it with its own code and refuses to
# run on a mismatch. If you change the math here, change it there.

_STD_SUFFIX = "_standardizer"     # <stem>_standardizer.npz/.json  beside  <stem>_bitnetJetTagModel.h5


def _padding_mask(X):
    """True where the particle slot is REAL, False where it is zero-padding.

    A padded slot is the all-zero 16-vector (jets with fewer than N constituents; ~0.006%
    of the top-10 slots). This is the ONE definition of padding used in this file.

    It MUST be evaluated on RAW inputs. After the affine a padded row becomes -mean/std,
    which is nonzero, so `X.any(-1)` on already-transformed data returns all-True and is
    silently useless — that is the trap this helper exists to make impossible to fall into.
    """
    return np.asarray(X).any(axis=-1)             # (n_jets, n_part) bool


def fit_input_standardizer(X, feature_names=None, eps=None, chunk=65536):
    """Per-feature mean/std over the NON-PADDED particles of the TRAIN split only.

    Two-pass (mean, then centered sum-of-squares) in float64. A one-pass E[x²]-E[x]²
    would cancel badly here: the GeV columns have mean ~ tens and std ~ 100.

    The padding mask is needed in BOTH passes, for DIFFERENT reasons — worth being explicit,
    because getting either wrong is silent:
      * pass 1 (raw sums): a padded row is all-zero, so it contributes exactly 0 to the
        numerator. The mask is needed only for the DENOMINATOR — divide by the number of
        REAL particles, not by the number of SLOTS (otherwise every mean is scaled by
        n_real/n_slots and every std with it).
      * pass 2 (centered): a padded row contributes (0 - mean)² ≠ 0. It MUST be masked out
        of the numerator or the variance is inflated toward mean².
    Chunked so the float64 working set stays ~80 MB regardless of dataset size.
    """
    X = np.asarray(X)
    assert X.ndim == 3 and X.shape[-1] == N_FEAT, \
        f"expected (n_jets, n_part, {N_FEAT}), got {X.shape}"
    eps = STD_EPS if eps is None else float(eps)

    n_real = 0
    s1 = np.zeros(N_FEAT, dtype=np.float64)
    for i in range(0, len(X), chunk):
        xb = X[i:i + chunk]
        n_real += int(_padding_mask(xb).sum())          # count REAL particles (denominator)
        s1 += xb.sum(axis=(0, 1), dtype=np.float64)     # padded rows add exactly 0.0
    assert n_real > 0, "no non-padded particles in the fit split — is the input all zeros?"
    mean = s1 / n_real

    s2 = np.zeros(N_FEAT, dtype=np.float64)
    for i in range(0, len(X), chunk):
        raw = X[i:i + chunk]
        m   = _padding_mask(raw)                        # mask taken on the RAW rows
        d   = (raw.astype(np.float64) - mean) * m[..., None]   # kill padded rows BEFORE squaring
        s2 += np.einsum("ijk,ijk->k", d, d)
    std = np.sqrt(s2 / n_real)                          # population std (ddof=0, as np.std)

    n_slots = int(X.shape[0]) * int(X.shape[1])
    stats = {
        "mean": mean, "std": std, "eps": eps,
        "n_feat": int(N_FEAT), "n_part": int(X.shape[1]),
        "n_jets_fit": int(X.shape[0]),
        "n_particles_fit": int(n_real),
        "n_padded_slots_excluded": int(n_slots - n_real),
        "feature_names": list(feature_names) if feature_names else [],
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fit_on": f"first {1.0 - VALIDATION_SPLIT:.0%} of the shuffled train/ split — exactly "
                  f"model.fit's training portion; the validation_split tail is EXCLUDED",
        "bn_n_part": int(N_PART_PER_JET),
    }
    stats["probe_sha256"] = _standardizer_probe_sha256(stats)
    return stats


def apply_input_standardizer(X, stats):
    """x' = (x - mean) / (std + eps), WITH THE ZERO-PADDED PARTICLES RE-ZEROED.

    The re-zero is the correctness trap. Without it an all-zero (padded) row maps to
    -mean/(std+eps): a specific, nonzero, perfectly consistent vector — i.e. a FAKE
    PARTICLE. SubLN would normalise it into a fixed unit direction and GlobalAveragePooling
    would average it in, so padding would stop being inert and start carrying signal.
    Today (raw inputs) padding is inert: 0 in → 0 through LN → 0. This keeps it that way.

    Bit-exactness contract (make_roc.py mirrors this): constants are cast to float32 ONCE,
    the arithmetic is float32, the operation order is fixed. `probe_sha256` pins the result.
    Does not mutate X.
    """
    X    = np.asarray(X, dtype=np.float32)
    real = _padding_mask(X)                                  # RAW rows — see _padding_mask
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std  = np.asarray(stats["std"],  dtype=np.float32)
    eps  = np.float32(stats["eps"])
    Xs = (X - mean) / (std + eps)                            # new array; X untouched
    Xs[~real] = np.float32(0.0)                              # ← re-zero the padding
    return Xs


def _standardizer_probe(n_feat):
    """A fixed probe tensor WITH PADDED ROWS — the drift canary for probe_sha256.

    Built with deterministic arithmetic, not np.random: Generator streams are not guaranteed
    stable across numpy versions, and a false alarm on the cluster would be worse than no
    check at all. The padded rows are what make the hash sensitive to a missing re-zero.
    """
    n, p = 8, 10
    x = ((np.arange(n * p * n_feat, dtype=np.float64).reshape(n, p, n_feat) % 37.0) - 18.0
         ).astype(np.float32)
    x[:, -2:, :] = 0.0      # two padded slots in every jet ...
    x[0, 0, :]   = 0.0      # ... and one in the middle, so the mask is not a pure suffix
    return x


def _standardizer_probe_sha256(stats):
    """sha256 of apply_input_standardizer(probe) — pins the EXACT numeric behaviour.

    Catches a different eps, different constants, a different operation order and — the one
    that actually matters — a missing padded-row re-zero. make_roc.py recomputes this with
    ITS OWN copy of the transform and hard-fails on a mismatch, so the training-side and
    eval-side transforms cannot silently drift apart.
    """
    probe = apply_input_standardizer(_standardizer_probe(int(stats["n_feat"])), stats)
    return hashlib.sha256(np.ascontiguousarray(probe, dtype=np.float32).tobytes()).hexdigest()


def standardizer_path_for(model_h5):
    """`<stem>_bitnetJetTagModel.h5` → `<stem>_standardizer.npz` (same dir, same stem)."""
    stem = model_h5
    for suf in ("_bitnetJetTagModel.h5", ".h5", ".keras"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    return stem + _STD_SUFFIX + ".npz"


def save_input_standardizer(stats, path_npz):
    """Persist the constants NEXT TO THE CHECKPOINT. They are part of the model, not a log.

    Writes both `<stem>_standardizer.npz` (canonical; float64, exact) and a human-readable
    `<stem>_standardizer.json` mirror carrying the same numbers. make_roc.py reads either.
    The `meta=json.dumps(...)` field mirrors how make_roc.py already stores meta in its .npz.
    """
    path_npz = os.path.abspath(path_npz)
    os.makedirs(os.path.dirname(path_npz) or ".", exist_ok=True)
    meta = {k: v for k, v in stats.items() if k not in ("mean", "std")}
    np.savez(path_npz,
             mean=np.asarray(stats["mean"], dtype=np.float64),
             std=np.asarray(stats["std"],  dtype=np.float64),
             eps=np.float64(stats["eps"]),
             meta=json.dumps(meta))
    path_json = os.path.splitext(path_npz)[0] + ".json"
    with open(path_json, "w") as fh:
        json.dump({**meta,
                   "mean": [float(v) for v in stats["mean"]],
                   "std":  [float(v) for v in stats["std"]]}, fh, indent=2)
    return path_npz, path_json


def embed_input_standardizer_in_h5(model_h5, stats):
    """Also write the constants INTO the checkpoint, as HDF5 root attributes.

    The sidecar can be separated from the .h5 (these runs write to an emptyDir and ship the
    checkpoint through W&B; someone downloads `model.h5` and nothing else). A checkpoint that
    silently loses its input transform produces a plausible, WRONG AUC — so the constants
    also live inside the file that cannot be separated from itself. make_roc.py reads these
    attrs, so a standardized checkpoint is self-describing even with no sidecar and no env var.

    Verified safe: Keras only reads `model_config` / `keras_version` / `backend` from the
    root attrs and ignores unknown ones; load_model() after this returns byte-identical
    weights and logits.
    """
    with h5py.File(model_h5, "a") as f:
        f.attrs["bn_standardize"]        = 1
        f.attrs["bn_std_mean"]           = np.asarray(stats["mean"], dtype=np.float64)
        f.attrs["bn_std_std"]            = np.asarray(stats["std"],  dtype=np.float64)
        f.attrs["bn_std_eps"]            = float(stats["eps"])
        f.attrs["bn_std_n_feat"]         = int(stats["n_feat"])
        f.attrs["bn_std_probe_sha256"]   = str(stats["probe_sha256"])
        f.attrs["bn_std_n_particles_fit"] = int(stats["n_particles_fit"])
        f.attrs["bn_std_feature_names"]  = json.dumps(list(stats.get("feature_names", [])))
    return model_h5


def load_input_standardizer(path):
    """Read back the constants from .npz (canonical) or .json. Mirrored in make_roc.py."""
    if str(path).endswith(".json"):
        with open(path) as fh:
            src = json.load(fh)
        stats = dict(src)
    else:
        z = np.load(path, allow_pickle=False)
        stats = json.loads(str(z["meta"]))
        src = {"mean": z["mean"], "std": z["std"], "eps": z["eps"]}
    stats["mean"] = np.asarray(src["mean"], dtype=np.float64)
    stats["std"]  = np.asarray(src["std"],  dtype=np.float64)
    stats["eps"]  = float(src["eps"])
    return stats


def _selftest_input_standardizer():
    """Padding-mask + bit-exactness self-test. No data, no GPU, no TF, ~10 ms.

    Runs inside --sanity, which every training job already executes as a preflight gate, so
    a broken standardizer can never reach a real run. Asserts, in order:
      1. the fitted mean/std IGNORE padded rows (vs an explicit boolean-index reference);
      2. the UNMASKED stats would differ — i.e. (1) has teeth and would fail if the mask
         were dropped;
      3. padded rows are EXACTLY 0.0 after the transform (the re-zero);
      4. real rows equal the reference affine bit-for-bit, and X is not mutated;
      5. the constants round-trip through .npz and .json with an identical probe hash.
    """
    rng = np.random.default_rng(0)
    n, p = 512, N_PART_PER_JET
    scale  = np.linspace(0.04, 120.0, N_FEAT).astype(np.float32)   # the real ~3,000x spread
    offset = np.linspace(-5.0, 60.0, N_FEAT).astype(np.float32)
    X = (rng.standard_normal((n, p, N_FEAT)).astype(np.float32) * scale + offset)
    X[:, -1, :] = 0.0        # last slot padded in every jet ...
    X[3, 2, :]  = 0.0        # ... plus one in the middle
    real = X.any(axis=-1)
    n_pad = int((~real).sum())
    assert n_pad == n + 1, f"self-test setup wrong: {n_pad} padded slots"

    st  = fit_input_standardizer(X)
    ref = X[real]                                    # (n_real, 16) — explicit boolean masking
    assert np.allclose(st["mean"], ref.mean(axis=0, dtype=np.float64), rtol=1e-9, atol=1e-6), \
        "fitted mean does not ignore padded particles"
    assert np.allclose(st["std"], ref.std(axis=0, dtype=np.float64), rtol=1e-9, atol=1e-6), \
        "fitted std does not ignore padded particles"
    assert st["n_particles_fit"] == int(real.sum())
    assert st["n_padded_slots_excluded"] == n_pad
    # (2) teeth: if the mask were dropped the stats WOULD change — so (1) is a real check.
    naive = X.reshape(-1, N_FEAT).mean(axis=0, dtype=np.float64)
    assert not np.allclose(st["mean"], naive, rtol=1e-9, atol=1e-6), \
        "self-test is toothless: masked and unmasked means agree"

    Xs = apply_input_standardizer(X, st)
    assert Xs.dtype == np.float32 and Xs.shape == X.shape
    assert np.all(Xs[~real] == 0.0), "PADDED ROWS ARE NOT ZERO after standardization"
    m32 = st["mean"].astype(np.float32)
    s32 = st["std"].astype(np.float32)
    assert np.array_equal(Xs[real], (X[real] - m32) / (s32 + np.float32(st["eps"]))), \
        "real rows differ from the reference affine"
    assert np.array_equal(X[:, -1, :], np.zeros((n, N_FEAT), np.float32)), "X mutated in place"

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        for path in save_input_standardizer(st, os.path.join(d, "t" + _STD_SUFFIX + ".npz")):
            st2 = load_input_standardizer(path)
            assert _standardizer_probe_sha256(st2) == st["probe_sha256"], \
                f"constants did not round-trip through {os.path.basename(path)}"
    print(f"[std][selftest] OK — {n_pad} padded slots excluded from the stats and re-zeroed "
          f"after the affine; npz/json round-trip exact; probe_sha256={st['probe_sha256'][:16]}…")
    return st


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING SCRIPT  (training loop mirrors original train.py)
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    train_dir = args.TrainDir

    # ── Optional reproducibility seed (BN_SEED) ──────────────────────────────
    # Unset (default) = upstream behaviour: unseeded random init + data shuffle, so repeated
    # runs sample run-to-run variance. Set BN_SEED=<int> to fix python/numpy/tf RNGs for
    # reproducible A/B comparisons (used by the variant seed-repeat sweep to separate a real
    # architecture effect from training noise).
    _seed = os.environ.get("BN_SEED", "").strip()
    if _seed:
        import random as _random
        _s = int(_seed)
        _random.seed(_s); np.random.seed(_s); tf.random.set_seed(_s)
        print(f"[seed] BN_SEED={_s}  (python/numpy/tf RNGs fixed for reproducibility)")

    print("Reading HLS4ML LHC Jet (150p) training data from " + train_dir)

    # ── Load data ────────────────────────────────────────────────────────────
    X, Y, jet_pt = load_hls4ml_jets(train_dir)

    # Shuffle jets (labels + jet pT stay aligned). Seeded iff BN_SEED is set.
    idx = np.random.permutation(len(X))
    X, Y, jet_pt = X[idx], Y[idx], jet_pt[idx]

    # ── Tag / output naming ───────────────────────────────────────────────────
    # Kept identical to the old pipeline so downstream tooling (find_model, hls
    # scripts, roc jobs) still matches "bitnet/noNorm_train_bitnetJetTagModel.h5".
    tag = "bitnet/noNorm_train"
    os.makedirs(os.path.dirname(os.getcwd() + f"/{tag}_model.png"),
                exist_ok=True)

    # ── Input standardization (BN_STANDARDIZE; default OFF → nothing changes) ──
    # CORRECTED 2026-07-13. This comment used to read "The dataset ships already
    # preprocessed; no extra normalisation is applied." The second half was true; the
    # FIRST HALF WAS FALSE, and it hid a real bug. The HLS4ML constituent features are
    # NOT on a common scale — measured on the actual top-10-by-pT inputs, per-feature std
    # spans 0.0394 .. 117.49 (a 2,979x ratio): GeV-scale px/py/pz/e/pt next to O(1)
    # etarel/phirel/ptrel/deltaR/costhetarel. Nothing here ever rescaled them (the
    # MinMaxScaler this file imported was never called — now removed). The knob block at
    # L150 explains why that penalises the BINARY model specifically and why it cannot be
    # learned around. Default stays OFF so every existing run reproduces bit-for-bit.
    std_stats = std_npz = std_json = None
    if STANDARDIZE:
        # Fit on EXACTLY the jets model.fit() will train on: Keras's validation_split takes
        # the LAST fraction of the (already shuffled) array, split_at = int(n * (1 - vs)).
        # Fitting on all of X would leak the val-monitor tail into the constants.
        split_at = int(len(X) * (1.0 - VALIDATION_SPLIT))
        std_stats = fit_input_standardizer(X[:split_at],
                                           feature_names=PART_FEATURE_NAMES)
        n_pad_before = int((~_padding_mask(X)).sum())      # padded slots in the RAW inputs
        X = apply_input_standardizer(X, std_stats)
        n_pad_after = int((~_padding_mask(X)).sum())       # ... must still be all-zero
        assert n_pad_after == n_pad_before, (
            f"padding re-zero FAILED: {n_pad_before} all-zero slots before the affine, "
            f"{n_pad_after} after — padded particles have become fake particles")

        std_npz, std_json = save_input_standardizer(
            std_stats, os.getcwd() + f"/{tag}{_STD_SUFFIX}.npz")
        print(f"[std] BN_STANDARDIZE=1 — per-feature affine x'=(x-mean)/(std+eps), "
              f"eps={std_stats['eps']:g}")
        print(f"[std] fit on {std_stats['n_jets_fit']:,} train jets / "
              f"{std_stats['n_particles_fit']:,} NON-PADDED particles "
              f"({std_stats['n_padded_slots_excluded']:,} padded slots excluded); "
              f"{n_pad_after:,} padded slots re-zeroed across the full array")
        _names = std_stats["feature_names"] or [f"f{i}" for i in range(N_FEAT)]
        for i, nm in enumerate(_names[:N_FEAT]):
            print(f"[std]   {i:2d}  {nm:<22s} mean={std_stats['mean'][i]:+13.5f}"
                  f"   std={std_stats['std'][i]:13.5f}")
        print(f"[std] probe_sha256={std_stats['probe_sha256']}")
        print(f"[std] constants → {std_npz} (+ .json). THESE ARE PART OF THE MODEL: "
              f"make_roc.py refuses to evaluate this checkpoint without them, and refuses "
              f"to re-fit them on the eval split.")

    # NOTE (2026-07-01 dataset migration): the old binary sig-vs-bkg flat-pT
    # reweighting and the kinematics_plotter call are dropped — both were wired to
    # the 2-file signal/background format. The 5-class dataset is trained
    # unweighted, matching published hls4ml jet-tagging baselines. Jet pT is kept
    # for reference:
    np.save("{}_ptRange.npy".format(tag), jet_pt)

    # ── Weights & Biases (optional; enabled when WANDB_PROJECT is set) ─────────
    use_wandb = bool(os.environ.get("WANDB_PROJECT"))
    if use_wandb:
        import wandb
        wandb.init(
            project = os.environ["WANDB_PROJECT"],
            name    = os.environ.get("WANDB_RUN_NAME") or None,
            config  = {"variant": VARIANT, "weights": ("ternary" if TERNARY else "binary")
                       if VARIANT == "bitnet" else ("fp32" if VARIANT == "vanilla" else "int8"),
                       "act_bits": ACT_BITS if VARIANT == "bitnet" else (32 if VARIANT == "vanilla" else 8),
                       "softmax_free": SOFTMAX_FREE,
                       "norm_type": NORM_TYPE, "norm_placement": NORM_PLACEMENT,
                       "pos_enc": POS_ENC, "pool": POOL, "ffn_act": FFN_ACT,
                       "d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": N_LAYERS,
                       "ffn_dim": FFN_DIM, "l1_reg": L1_REG, "epochs": EPOCHS,
                       "batch_size": BATCH_SIZE, "n_train": int(len(Y)),
                       "dataset": "hls4ml_lhc_jets_150p",
                       "n_part": N_PART_PER_JET, "n_feat": N_FEAT,
                       "n_classes": N_CLASSES,
                       # The standardizer constants are part of the model definition, so
                       # they go into the run config too — a run's inputs are then fully
                       # reconstructible from W&B even if the sidecar files are lost.
                       "standardize": STANDARDIZE,
                       **({"std_eps": std_stats["eps"],
                           "std_probe_sha256": std_stats["probe_sha256"],
                           "std_n_particles_fit": std_stats["n_particles_fit"],
                           "std_feature_names": std_stats["feature_names"],
                           "std_mean": [float(v) for v in std_stats["mean"]],
                           "std_std":  [float(v) for v in std_stats["std"]]}
                          if std_stats is not None else {}),
                       **{f"n_{c}": int(n) for c, n
                          in zip(CLASS_LABELS, Y.sum(axis=0))}},
        )
        # ... and as files, alongside the checkpoint W&B already receives.
        if std_stats is not None:
            for _p in (std_npz, std_json):
                try:
                    wandb.save(_p)
                except Exception as e:
                    print(f"[warn] wandb.save({_p}) skipped: {e}")

    # ── Build model  ──────────────────────────────────────────────────────────
    model = build_bitnet_jet_tagger()
    model.summary()
    if use_wandb:
        wandb.config.update({"total_params": int(model.count_params())})

    try:
        tf.keras.utils.plot_model(
            model,
            to_file    = os.getcwd() + f"/{tag}_model.png",
            show_shapes= True,
            show_layer_names=True,
        )
    except Exception as e:
        print(f"[warn] plot_model skipped (non-fatal): {e}")

    # ── Pruning  (same schedule as your original) ─────────────────────────────
    # Note: tfmot pruning wraps Dense-like layers. BitLinear is a custom Layer,
    # so we selectively prune only the head Dense equivalents if needed.
    # For the transformer blocks, the binary constraint already achieves ~67%
    # sparsity on average (roughly 1/3 of weights are zero after quantisation).
    # If you want explicit magnitude pruning on top, uncomment the block below.

    # pruning_params = {
    #     "pruning_schedule":
    #         pruning_schedule.ConstantSparsity(0.75, begin_step=2000,
    #                                           frequency=100)
    # }
    # model = prune.prune_low_magnitude(model, **pruning_params)

    # ── Compile  ──────────────────────────────────────────────────────────────
    # Default optimizer is upstream "adam" (1e-3). When BN_LR>0, swap in an explicit
    # Adam(lr=LR) with global grad-norm clipping to stabilise deep 1-bit training.
    if LR > 0:
        opt_kwargs = dict(learning_rate=LR, beta_1=0.9, beta_2=BETA2)
        if CLIPNORM > 0:
            opt_kwargs["global_clipnorm"] = CLIPNORM          # paper: no clipping (set BN_CLIPNORM=0)
        if WEIGHT_DECAY > 0:
            opt_kwargs["weight_decay"] = WEIGHT_DECAY          # paper: 0.01 (AdamW-style)
        optimizer = tf.keras.optimizers.Adam(**opt_kwargs)
        print(f"[opt] Adam(lr={LR}, beta=(0.9,{BETA2}), "
              f"clipnorm={CLIPNORM if CLIPNORM > 0 else 'OFF'}, weight_decay={WEIGHT_DECAY})"
              + (f" + warmup {WARMUP_EPOCHS}ep" if WARMUP_EPOCHS > 0 else "")
              + (f" + poly-decay {DECAY_EPOCHS}ep^{DECAY_POWER}" if DECAY_EPOCHS > 0 else ""))
    else:
        optimizer = "adam"
    # 5-class softmax CE from the raw logits; "auc" is the macro one-vs-rest AUC
    # (multi_label=True averages per-class AUCs). This is the TRAINING monitor —
    # the headline ROC-test AUC still comes from make_roc.py on the held-out val/
    # files (never conflate the two; see decisions.md 2026-06-22).
    model.compile(
        loss      = tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer = optimizer,
        metrics   = ["categorical_accuracy",
                     tf.keras.metrics.AUC(name="auc", multi_label=True,
                                          num_labels=N_CLASSES, from_logits=True)],
    )

    # ── Callbacks  ────────────────────────────────────────────────────────────
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=ES_MONITOR, mode=ES_MODE, verbose=1,
            patience=ES_PATIENCE, restore_best_weights=True
        ),
        # pruning_callbacks.UpdatePruningStep(),  # ← re-enable if pruning above
    ]
    # Optional linear LR warmup (opt-in via BN_LR>0 and BN_WARMUP_EPOCHS>0): ramp LR from
    # LR/WARMUP up to LR over the first WARMUP_EPOCHS epochs, then hold. Stabilises the start
    # of deep 1-bit training; no effect on the default path.
    if LR > 0 and (WARMUP_EPOCHS > 0 or DECAY_EPOCHS > 0):
        def _lr_schedule(epoch, lr):
            # Linear warmup over WARMUP_EPOCHS, then optional polynomial decay to ~0
            # over DECAY_EPOCHS (paper: 750-update warmup + polynomial-decay schedule).
            if WARMUP_EPOCHS > 0 and epoch < WARMUP_EPOCHS:
                return LR * float(epoch + 1) / float(WARMUP_EPOCHS)
            if DECAY_EPOCHS > 0:
                prog = min(1.0, float(epoch - WARMUP_EPOCHS) / float(DECAY_EPOCHS))
                return LR * (1.0 - prog) ** DECAY_POWER
            return LR
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(_lr_schedule, verbose=1))
    if use_wandb:
        class _WandbEpoch(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                wandb.log({**(logs or {}), "epoch": epoch})
        callbacks.append(_WandbEpoch())

    # ── Train  ────────────────────────────────────────────────────────────────
    # validation_split on the train/ files only; the dataset's val/ files stay a
    # pristine held-out set for the ROC-test evaluation (make_roc.py).
    # VALIDATION_SPLIT is the same 0.20 as before, now named so the BN_STANDARDIZE fit can
    # use exactly this split point (Keras trains on X[:int(n*(1-VALIDATION_SPLIT))]).
    history = model.fit(
        X, Y,
        epochs           = EPOCHS,
        batch_size       = BATCH_SIZE,
        verbose          = 2,
        validation_split = VALIDATION_SPLIT,
        callbacks        = callbacks,
    )

    # ── Loss curve  ───────────────────────────────────────────────────────────
    plt.figure(figsize=(7, 5), dpi=120)
    plt.plot(history.history["loss"],     label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("BitNet Model Loss", fontsize=25)
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.getcwd() + "/{}_bitnetLoss.pdf".format(tag), dpi=120)

    # ── Save  ─────────────────────────────────────────────────────────────────
    # model = tfmot.sparsity.keras.strip_pruning(model)  # ← re-enable if pruning
    _ckpt = os.getcwd() + "/{}_bitnetJetTagModel.h5".format(tag)
    model.save(_ckpt)
    print(f"\nModel saved to {tag}_bitnetJetTagModel.h5")

    # Bake the input transform INTO the checkpoint as well (h5 root attrs). The sidecar can
    # get separated from the .h5 — /work is an emptyDir and the checkpoint travels via W&B —
    # and a checkpoint that has quietly lost its input transform yields a plausible, WRONG
    # AUC. Belt and braces: the constants now live in the one file that cannot be separated
    # from itself, so make_roc.py can recover them with no sidecar and no env var.
    if std_stats is not None:
        embed_input_standardizer_in_h5(_ckpt, std_stats)
        print(f"[std] constants also embedded in {os.path.basename(_ckpt)} "
              f"(h5 attrs bn_standardize / bn_std_mean / bn_std_std / bn_std_eps / "
              f"bn_std_probe_sha256) — the checkpoint is self-describing")

    if use_wandb:
        try:
            wandb.save(_ckpt)
        except Exception as e:
            print(f"[warn] wandb.save skipped: {e}")
        wandb.finish()


# ══════════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK  (no data needed)
# ══════════════════════════════════════════════════════════════════════════════

def sanity_check():
    """
    Verify input/output shapes and that binary constraints are applied.
    Run with:  python bitnet_jet_tagger.py --sanity
    """
    print("=" * 60)
    print("BitNet Jet Tagger — sanity check")
    print("=" * 60)

    # Input-standardization preflight. Always run (it is data-free and ~10 ms), regardless
    # of BN_STANDARDIZE, so the padding-mask / re-zero / round-trip guarantees are checked
    # on the cluster before every job — including the jobs that leave the knob OFF.
    print(f"[std] BN_STANDARDIZE={'1 (per-feature affine ON)' if STANDARDIZE else '0 (raw inputs — default)'}")
    _selftest_input_standardizer()

    model = build_bitnet_jet_tagger()
    model.summary()

    try:
        tf.keras.utils.plot_model(
            model,
            to_file    = os.getcwd() + f"/model.png",
            show_shapes= True,
            show_layer_names=True,
        )
    except Exception as e:
        print(f"[warn] plot_model skipped (non-fatal): {e}")

    # Shape check
    dummy_x = np.random.randn(8, N_PART_PER_JET, N_FEAT).astype(np.float32)
    dummy_y = np.eye(N_CLASSES, dtype=np.float32)[
        np.random.randint(0, N_CLASSES, 8)]                      # one-hot (8, 5)
    out   = model(dummy_x, training=False)
    assert out.shape == (8, N_CLASSES), f"Wrong output shape: {out.shape}"

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  optimizer="adam")
    model.train_on_batch(dummy_x, dummy_y)

    print(f"\n✓  Input  shape : {dummy_x.shape}")
    print(f"✓  Output shape : {out.shape}  (raw class logits, no softmax)")

    # Check that BitLinear weights are binary after one build
    binary_ok = True
    for layer in model.layers:
        for w in layer.weights:
            if "kernel" in w.name:
                vals = np.unique(np.round(w.numpy(), 4))
                bad  = [v for v in vals if v not in (np.min(vals), 0.0, np.max(vals))]
                if bad:
                    print(f" {w.name}: non-binary values {bad[:5]}, weights.shape {w.numpy().shape}")
                    binary_ok = False

                #vals = np.unique(np.round(w.numpy(), 4))
                good = [v for v in vals]
                if good:
                    print(f" {w.name}: binary values {good[:5]}, , weights.shape {w.numpy().shape}")
    if binary_ok:
        print("  All BitLinear kernels are binary {-1, +1}")

    # Parameter count comparison
    n_params = model.count_params()
    print(f"\nTotal trainable parameters : {n_params:,}")
    print("(Original QKeras CNN est.  : ~2,000–3,000 params)")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BN 1-bit Transformer jet tagger for CMS L1 trigger"
    )
    parser.add_argument("--sanity", action="store_true",
                        help="Run shape/weight sanity check (no data needed)")
    parser.add_argument("TrainDir", nargs="?", type=str,
                        help="directory of jetImage_*_150p_*.h5 training files "
                             "(NRP: /data/hls4ml_lhc_jet/train/train)")

    args = parser.parse_args()

    if args.sanity:
        sanity_check()
    else:
        if not args.TrainDir:
            parser.error("Provide the training data directory, or use --sanity")
        main(args)
