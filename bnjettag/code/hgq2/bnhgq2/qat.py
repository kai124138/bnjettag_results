"""Native HGQ2 QAT — the FINAL-campaign training model (decisions.md 2026-07-07).

The trained model IS (essentially) the hardware model: binary {-1,+1} weights via a
straight-through absmean estimator on latent floats, and static per-tensor activation
quantizers. This is the QAT counterpart of the (porting-mode) build.py.

Why a CUSTOM weight quantizer (verified 2026-07-07, LEDGER):
  A stock HGQ2 fixed-point weight quantizer at 1-bit repr (KBI k0=1,b0=1,i0=1,SAT_SYM)
  rounds latent floats to the grid {-1,0,+1} — it collapses small weights to ZERO
  (probe T1: 4096/4096 latents -> 0). That is TERNARY and a STOP condition (the thesis
  is binary {-1,+1}; effective weights must be exactly {-beta,+beta}). So bitnet layers
  binarize the latent kernel in the forward pass with the exact BitNet absmean STE
  (identical math to qkerasModel.AbsMeanQuantizer, bipolar-safe: ties -> +1, never 0),
  gradients flowing to the latent via stop_gradient. The layer keeps a 1-bit KBI kq_conf
  purely so EBOPs / hls4ml report 1-bit for the weights (kq.bits == 1, probe T3).

Export path (unchanged rebuild pipeline): at end of training the latent kernels are read
by layer name, absmean-binarized (binarize.py — SAME math), and fed through build.py +
convert.py to emit the pinned {-1,+1}xbeta + static-act-grid hardware model that csynth
consumes UNCHANGED. Fidelity gate: exported vs trained prediction corr ~ 1.0.

Layer names mirror build.py / the QKeras checkpoint so port.py / the export map 1:1.
"""
from __future__ import annotations

import numpy as np
import keras
from keras import ops

from hgq.layers import QEinsumDense, QDense, QSoftmax, QGlobalAveragePooling1D
from hgq.layers.ops.einsum import QEinsum
from hgq.quantizer import QuantizerConfig
from hgq.config import LayerConfigScope

from .subln import PSubLN

EPS_BETA = 1e-6


# --------------------------------------------------------------------------- #
# (a) BitNet absmean straight-through binarizer                               #
# --------------------------------------------------------------------------- #
def bitnet_binary_ste(w, eps: float = EPS_BETA):
    """Effective weight = beta * bipolar_sign(w - mean(w)), STE identity backward.

    Replicates qkerasModel.AbsMeanQuantizer (binary branch), with two deliberate,
    forward-identical hardening changes (LEDGER 2026-07-07):
      * sign uses a strict bipolar rule (>=0 -> +1 else -1) instead of tf.sign, so the
        output is guaranteed in {-beta, +beta} with NO zero state (sign(0)=0 would void
        the binary claim);
      * the normalized weight uses ws = wc / stop_gradient(beta). Forward value is
        byte-identical to qkeras (q*beta), but the backward becomes (I - 1/n) + q*dbeta/dw,
        which is BOUNDED — qkeras's version routes gradient through wc/beta and blows up
        by 1/beta when a kernel's centered weights collapse (beta -> eps). That 1/beta
        explosion produced the A4 training NaN (verified 2026-07-07); this removes it.
    alpha/beta are per-tensor scalars (matches binarize.absmean_binarize exactly, so the
    export re-binarization gives the same +/-beta on the same latents)."""
    w = ops.cast(w, "float32")
    alpha = ops.mean(w)
    wc = w - alpha
    beta = ops.mean(ops.abs(wc)) + eps
    ws = wc / ops.stop_gradient(beta)               # bounded backward (see docstring)
    q = ops.where(wc >= 0.0, 1.0, -1.0)             # strict bipolar, never 0
    wq = ws + ops.stop_gradient(q - ws)             # STE: forward=q
    return wq * beta                                 # in {-beta, +beta}


# --------------------------------------------------------------------------- #
# (b) binary-weight einsum / dense layers                                     #
# --------------------------------------------------------------------------- #
@keras.saving.register_keras_serializable(package="bnhgq2")
class BitQEinsumDense(QEinsumDense):
    """QEinsumDense whose kernel is absmean-binarized in the forward pass (QAT)."""

    def call(self, inputs, training=None):
        qkernel = bitnet_binary_ste(self._kernel)
        if self.enable_iq:
            inputs = self.iq(inputs, training=training)
        x = ops.einsum(self.equation, inputs, qkernel)
        if self.bias is not None:
            assert self.bq is not None
            x = x + self.bq(self.bias, training=training)
        if self.activation is not None:
            x = self.activation(x)
        return x

    @property
    def qkernel(self):
        return bitnet_binary_ste(self._kernel)


@keras.saving.register_keras_serializable(package="bnhgq2")
class BitQDense(QDense):
    """QDense whose kernel is absmean-binarized in the forward pass (QAT)."""

    def call(self, inputs, training=None):
        if self.enable_iq:
            inputs = self.iq(inputs, training=training)
        x = ops.matmul(inputs, bitnet_binary_ste(self._kernel))
        if self.bias is not None:
            x = ops.add(x, self.qbias)
        if self.activation is not None:
            x = self.activation(x)
        return x

    @property
    def qkernel(self):
        return bitnet_binary_ste(self._kernel)


# --------------------------------------------------------------------------- #
# (c) trainable positional encoding (folds into input_proj bias at export)    #
# --------------------------------------------------------------------------- #
@keras.saving.register_keras_serializable(package="bnhgq2")
class AddPositional(keras.layers.Layer):
    """Add a learned (n_part, d_model) positional table (broadcast over batch).

    A constant additive per position at inference -> folds into input_proj's bias table
    exactly like the QKeras `folded_constant` PE (build.py / port.py)."""

    def build(self, input_shape):
        n, d = int(input_shape[-2]), int(input_shape[-1])
        self.pos = self.add_weight(name="pos_table", shape=(1, n, d),
                                   initializer="glorot_uniform", trainable=True)
        super().build(input_shape)

    def call(self, x):
        return x + self.pos

    def compute_output_shape(self, input_shape):
        return input_shape


# --------------------------------------------------------------------------- #
# (c') pairwise-invariant physics features (opt-in attention-logit bias)       #
# --------------------------------------------------------------------------- #
# Canonical HLS4ML LHC-jet particleFeatureNames order (jetConstituentList, 16 feat).
# The data loaders read these from each h5 at runtime (data.py / train.py); the model
# builder has no h5 in hand, so the ordered names are pinned here and the columns we
# need are resolved BY EXACT SUFFIX (so _e != _erel, _pt != _ptrel, _eta != _etarel).
PART_FEATURE_NAMES_CANONICAL = [
    "j1_px", "j1_py", "j1_pz", "j1_e", "j1_erel", "j1_pt", "j1_ptrel", "j1_eta",
    "j1_etarel", "j1_etarot", "j1_phi", "j1_phirel", "j1_phirot", "j1_deltaR",
    "j1_costheta", "j1_costhetarel",
]
# hardcoded fallback column indices (the canonical order above) — used, with a loud
# print, only if suffix resolution fails on a non-canonical name list.
_PAIR_IDX_FALLBACK = {"px": 0, "py": 1, "pz": 2, "e": 3, "pt": 5, "eta": 7, "phi": 10}


def _resolve_pair_indices(names):
    """{px,py,pz,e,pt,eta,phi -> column index} by EXACT suffix match. Falls back to the
    pinned canonical indices (with a loud print) if any of the seven is unresolved."""
    idx = {}
    for key in ("px", "py", "pz", "e", "pt", "eta", "phi"):
        hits = [i for i, n in enumerate(names) if str(n).endswith("_" + key)]
        if len(hits) == 1:
            idx[key] = hits[0]
    if set(idx) != set(_PAIR_IDX_FALLBACK):
        print(f"[pair_bias][WARN] feature-name resolution incomplete (resolved {idx} "
              f"from {names}); FALLING BACK to hardcoded canonical indices "
              f"{_PAIR_IDX_FALLBACK}", flush=True)
        return dict(_PAIR_IDX_FALLBACK)
    return idx


@keras.saving.register_keras_serializable(package="bnhgq2")
class PairFeatures(keras.layers.Layer):
    """Per ordered-pair (i,j) physics features from the RAW constituents: (B,T,F)->(B,T,T,3)
      ln(dR^2 + eps),  dR^2 = (eta_i-eta_j)^2 + (phi_i-phi_j)^2
      ln(kT^2 + eps),  kT^2 = min(pt_i,pt_j)^2 * dR^2
      ln(|m2_ij| + eps),  m2_ij = (E_i+E_j)^2 - (px_i+px_j)^2 - (py_i+py_j)^2 - (pz_i+pz_j)^2
    Parameter-free, no division anywhere; the diagonal is the ln(eps) constant (the net
    learns it). When input_mu/input_sigma are given (the round-8 input_std contract, where
    the model receives offline-standardized inputs) the incoming x is de-standardized to
    physical units first (raw = x*sigma + mu); otherwise x is treated as already-raw."""

    def __init__(self, feat_names=None, input_mu=None, input_sigma=None,
                 eps=1e-6, idx=None, **kw):
        super().__init__(**kw)
        self.feat_names = list(feat_names) if feat_names else list(PART_FEATURE_NAMES_CANONICAL)
        self.idx = dict(idx) if idx else _resolve_pair_indices(self.feat_names)
        self.eps = float(eps)
        self._mu = None if input_mu is None else np.asarray(input_mu, "float32").reshape(-1).tolist()
        self._sigma = None if input_sigma is None else np.asarray(input_sigma, "float32").reshape(-1).tolist()

    def call(self, x):
        x = ops.cast(x, "float32")
        if self._mu is not None and self._sigma is not None:
            x = x * ops.convert_to_tensor(self._sigma, "float32") \
                + ops.convert_to_tensor(self._mu, "float32")
        col = lambda k: ops.take(x, self.idx[k], axis=-1)          # (B,T)
        eta, phi, pt = col("eta"), col("phi"), col("pt")
        px, py, pz, e = col("px"), col("py"), col("pz"), col("e")

        def _pair_i(a):
            return ops.expand_dims(a, 2)                            # (B,T,1) -> broadcast over j

        def _pair_j(a):
            return ops.expand_dims(a, 1)                            # (B,1,T) -> broadcast over i

        deta = _pair_i(eta) - _pair_j(eta)
        dphi = _pair_i(phi) - _pair_j(phi)
        dr2 = deta * deta + dphi * dphi                            # (B,T,T)
        pt_min = ops.minimum(_pair_i(pt), _pair_j(pt))
        kt2 = pt_min * pt_min * dr2
        e_s, px_s = _pair_i(e) + _pair_j(e), _pair_i(px) + _pair_j(px)
        py_s, pz_s = _pair_i(py) + _pair_j(py), _pair_i(pz) + _pair_j(pz)
        m2 = e_s * e_s - px_s * px_s - py_s * py_s - pz_s * pz_s
        eps = self.eps
        return ops.stack([ops.log(dr2 + eps), ops.log(kt2 + eps),
                          ops.log(ops.abs(m2) + eps)], axis=-1)     # (B,T,T,3)

    def compute_output_shape(self, input_shape):
        s = tuple(input_shape)
        return (s[0], s[1], s[1], 3)

    def get_config(self):
        c = super().get_config()
        c.update(feat_names=self.feat_names, idx=self.idx, eps=self.eps,
                 input_mu=self._mu, input_sigma=self._sigma)
        return c


# --------------------------------------------------------------------------- #
# (d) quantizer-config factories                                              #
# --------------------------------------------------------------------------- #
def _binary_kq():
    """1-bit KBI pin — reports 1 bit to EBOPs/hls4ml (forward uses the STE binarizer)."""
    return QuantizerConfig("kbi", "weight", k0=1, b0=1, i0=1,
                           round_mode="RND", overflow_mode="SAT_SYM",
                           trainable=False, heterogeneous_axis=())


def _static_act(act_bits: int, i0: int):
    """Static per-tensor signed fixed<act_bits, 1+i0> with SAT + RND_CONV, FROZEN.
    Same form as build.py.act_q; the round carries an STE so gradients flow."""
    f0 = act_bits - 1 - int(i0)
    return QuantizerConfig("kif", "datalane", k0=1, i0=int(i0), f0=f0,
                           round_mode="RND_CONV", overflow_mode="SAT",
                           trainable=False, heterogeneous_axis=())


def _trainable_act(act_bits: int, i0: int):
    """TRAINABLE-SCALE, FIXED-bit-width activation quantizer (the A4 drift rescue,
    LEDGER 2026-07-07). KBI parametrization: total bits `b` and integer bits `i` are
    separate, so pinning b (via a Constant constraint) fixes the width at `act_bits`
    while `i` (the scale / integer split) trains WITH the model — f = b - i is derived.
    As the activation distributions drift during training the grid re-aligns, which the
    frozen first-batch grid cannot do at A4's 16 levels (the observed 0.58/0.60 plateau).

    Why KBI, not KIF: KIF has independent i AND f (no fixed total). Why SAT (not WRAP):
    SAT clips exactly as the exported hardware grid does, so the checkpoint IS the
    hardware model (train==deploy); i is gradient-trained (HGQ2's native surrogate).
    At export the final (i, b=act_bits-1) reads out as a static SAT<act_bits,1+i> grid —
    identical in form to build.py.act_q, so convert.py is unchanged."""
    from hgq.constraints import Constant, MinMax
    return QuantizerConfig("kbi", "datalane", k0=1, b0=act_bits - 1, i0=int(i0),
                           round_mode="RND_CONV", overflow_mode="SAT",
                           trainable=True, heterogeneous_axis=(),
                           bc=Constant(act_bits - 1),               # pin width -> fixed act_bits
                           ic=MinMax(0, max(0, act_bits - 2)),       # i in [0, bits-2] -> f = b-i >= 1
                           br=None, ir=None)


def _static_w8(i0: int = 2):
    """Static per-tensor 8-bit weight grid (w8a8 baseline): fixed<8,1+i0>, SAT, frozen.
    Latent weights adapt inside the fixed grid; weight_decay keeps them in range."""
    return QuantizerConfig("kif", "weight", k0=1, i0=int(i0), f0=7 - int(i0),
                           round_mode="RND_CONV", overflow_mode="SAT",
                           trainable=False, heterogeneous_axis=())


def _dummy(place="datalane"):
    return QuantizerConfig("dummy", place)


def _table(i0: int, f0: int):
    return QuantizerConfig("kif", "table", k0=0, i0=int(i0), f0=int(f0),
                           round_mode="RND_CONV", overflow_mode="SAT", trainable=False)


# --------------------------------------------------------------------------- #
# (e) the model                                                               #
# --------------------------------------------------------------------------- #
# provisional integer bits used at BUILD time; overwritten by calibrate_activations.
_PROV_I = 4


def build_qat_model(cfg: dict, seed: int, enable_ebops: bool = True,
                    input_mu=None, input_sigma=None):
    """Return (model, taps). `taps` = {site_name: pre-quant KerasTensor} for the
    activation-calibration pass. variant is derived from cfg['quant']['weight']:
      binary_absmean -> bitnet (binary STE weights + static A-bit acts)
      int8_absmax    -> w8a8   (static 8-bit weights + static 8-bit acts)
      none           -> fp32   (float weights + float acts, same skeleton).

    input_mu/input_sigma (per-feature, len n_feat) are only consumed when the opt-in
    arch.pair_bias branch is active: it de-standardizes the model input back to physical
    units before computing the pairwise features. Ignored otherwise, so absent/None keeps
    every existing config byte/hash-identical."""
    keras.utils.set_random_seed(int(seed))

    A = cfg["arch"]
    T, F, D = A["n_part"], A["n_feat"], A["d_model"]
    H, L, FFN, C = A["n_heads"], A["n_layers"], A["ffn_dim"], A["n_classes"]
    E = D // H
    wmode = cfg["quant"]["weight"]
    ab = int(cfg["quant"]["act_bits"])
    fp32 = wmode == "none"
    binv = wmode == "binary_absmean"
    # act_calib: "frozen" (default; static first-batch-calibrated grid) | "trainable"
    # (trainable-scale fixed-bit grid, the drift rescue) | "recalib" (frozen but the
    # train stage re-runs calibration at act_recalib_epochs). Default reproduces the
    # current behavior exactly, so a config WITHOUT this key is byte/hash-unchanged.
    act_calib = str(cfg["quant"].get("act_calib", "frozen")).lower()
    taps: dict = {}

    # Norm placement: 'subln' (default -> PSubLN, existing behavior) | 'none' (identity
    # passthrough: no layer, no params, no DSP -- the norm-ablation science variant).
    # Guarded on arch.norm; a config without norm=='none' is byte/hash-identical to before.
    norm_kind = str(A.get("norm", "subln")).lower()
    if norm_kind not in ("subln", "none"):
        raise ValueError(f"{cfg['name']}: unsupported arch.norm={norm_kind!r} "
                         "(qat stack supports 'subln' or 'none')")

    # Opt-in pairwise-invariant attention-logit bias (guarded on arch.pair_bias; absent =>
    # byte/hash-identical to before). A small binary MLP over per-pair physics features
    # (PairFeatures) is embedded ONCE to (H,T,T) and shared/added to every block's
    # pre-softmax scores. Uses the same binary-weight einsum machinery as the FFN.
    pair_bias = bool(A.get("pair_bias", False))
    pair_hidden = int(A.get("pair_bias_hidden", 8))

    def norm(name, flatten_axes=1):
        if norm_kind == "none":
            return lambda x: x  # identity passthrough: parameter-free, DSP-free
        return PSubLN(flatten_axes=flatten_axes, name=name)

    def act_iq(i0=_PROV_I):
        """A-bit datalane quantizer: trainable-scale if act_calib=='trainable', else the
        frozen static grid (calibrate_activations sets the initial i for both)."""
        return _trainable_act(ab, i0) if act_calib == "trainable" else _static_act(ab, i0)

    def bias_conf():
        return _dummy("bias")  # float bias in all variants (matches qkerasModel)

    def dense_einsum(name, equation, out_shape, x, bias_axes=None):
        """One projection: A-bit input quant + (binary|int8|float) weights."""
        if not fp32:
            taps[name] = x
        if fp32:
            kq, iq = _dummy("weight"), _dummy("datalane")
            cls = QEinsumDense
        elif binv:
            kq, iq = _binary_kq(), act_iq()
            cls = BitQEinsumDense
        else:  # w8a8
            kq, iq = _static_w8(), act_iq()
            cls = QEinsumDense
        return cls(equation, output_shape=out_shape, bias_axes=bias_axes,
                   iq_conf=iq, kq_conf=kq,
                   bq_conf=(bias_conf() if bias_axes is not None else None),
                   name=name)(x)

    def dense(name, units, x, use_bias=True):
        if not fp32:
            taps[name] = x
        if fp32:
            kq, iq, cls = _dummy("weight"), _dummy("datalane"), QDense
        elif binv:
            kq, iq, cls = _binary_kq(), act_iq(), BitQDense
        else:
            kq, iq, cls = _static_w8(), act_iq(), QDense
        return cls(units, use_bias=use_bias, iq_conf=iq, kq_conf=kq,
                   bq_conf=bias_conf(), name=name)(x)

    def stream_iq():
        return _dummy("datalane") if fp32 else act_iq(6)  # provisional

    def einsum(name, equation, xs, iqs):
        if not fp32:
            for i, x in enumerate(xs):
                taps[f"{name}__in{i}"] = x
        return QEinsum(equation, iq_confs=iqs, name=name)(xs)

    def softmax(name, x, scale):
        if fp32:
            return QSoftmax(axis=-1, stable=True, input_scaler=float(scale),
                            exp_iq_conf=_dummy(), inv_iq_conf=_dummy(),
                            exp_oq_conf=_table(1, 20), inv_oq_conf=_table(1, 20),
                            name=name)(x)
        # generous static tables (softmax internals are LN/softmax-bounded; the
        # substitution cost was measured negligible in the rebuild gold experiments)
        return QSoftmax(axis=-1, stable=True, input_scaler=float(scale),
                        exp_iq_conf=_static_act(max(ab, 10), 6),
                        inv_iq_conf=QuantizerConfig("kif", "datalane", k0=0, i0=4, f0=8,
                                                    round_mode="RND_CONV",
                                                    overflow_mode="SAT", trainable=False,
                                                    heterogeneous_axis=()),
                        exp_oq_conf=_table(1, 11), inv_oq_conf=_table(1, 11),
                        name=name)(x)

    with LayerConfigScope(enable_ebops=enable_ebops, beta0=0.0):
        x_in = keras.Input((T, F), name="input_1")

        # ---- input projection ----
        h = norm("ln_input_proj")(x_in)
        h = dense_einsum("input_proj", "btf,fd->btd", (T, D), h, bias_axes="td")
        h = AddPositional(name="pos_enc")(h)

        # ---- pairwise-invariant attention-logit bias (shared across blocks) ----
        pair_bias_tensor = None
        if pair_bias:
            pf = PairFeatures(input_mu=input_mu, input_sigma=input_sigma,
                              name="pair_features")(x_in)          # (B,T,T,3) from RAW input
            pb = dense_einsum("pair_fc1", "btsc,ch->btsh", (T, T, pair_hidden), pf,
                              bias_axes="h")
            pb = keras.layers.ReLU(name="pair_act")(pb)
            pb = dense_einsum("pair_fc2", "btsh,hg->btsg", (T, T, H), pb, bias_axes="g")
            pair_bias_tensor = keras.layers.Permute((3, 1, 2),
                                                    name="pair_bias")(pb)   # (B,H,T,T)

        for li in range(L):
            blk = f"bit_block_{li}"
            # ---- attention ----
            ln = norm(f"ln_{blk}_attn_qkv")(h)
            q = dense_einsum(f"{blk}_attn_Wq", "btd,dhe->bthe", (T, H, E), ln)
            k = dense_einsum(f"{blk}_attn_Wk", "btd,dhe->bthe", (T, H, E), ln)
            v = dense_einsum(f"{blk}_attn_Wv", "btd,dhe->bthe", (T, H, E), ln)
            scores = einsum(f"{blk}_attn_scores", "bthe,bshe->bhts", [q, k],
                            [stream_iq(), stream_iq()])
            if pair_bias_tensor is not None:                        # add shared pair bias
                scores = keras.layers.Add(name=f"{blk}_attn_pairbias")(
                    [scores, pair_bias_tensor])
            attn = softmax(f"{blk}_attn_softmax", scores, 1.0 / np.sqrt(E))
            attn_iq = _dummy("datalane") if fp32 else \
                QuantizerConfig("kif", "datalane", k0=0, i0=1, f0=max(ab, 10) - 1,
                                round_mode="RND_CONV", overflow_mode="SAT",
                                trainable=False, heterogeneous_axis=())
            ctx = QEinsum("bhts,bshe->bthe",
                          iq_confs=[attn_iq, stream_iq()],
                          name=f"{blk}_attn_ctx")([attn, v])
            ctx = norm(f"ln_{blk}_attn_Wo", flatten_axes=2)(ctx)
            wo = dense_einsum(f"{blk}_attn_Wo", "bthe,hed->btd", (T, D), ctx,
                              bias_axes="d")
            h = keras.layers.Add(name=f"{blk}_add_attn")([h, wo])

            # ---- FFN ----
            ln = norm(f"ln_{blk}_ffn_fc1")(h)
            f1 = dense_einsum(f"{blk}_ffn_fc1", "btd,df->btf", (T, FFN), ln,
                              bias_axes="f")
            f1 = keras.layers.ReLU(name=f"{blk}_ffn_act")(f1)
            ln = norm(f"ln_{blk}_ffn_fc2")(f1)
            f2 = dense_einsum(f"{blk}_ffn_fc2", "btf,fd->btd", (T, D), ln,
                              bias_axes="d")
            h = keras.layers.Add(name=f"{blk}_add_ffn")([h, f2])

        # ---- head ----
        h = QGlobalAveragePooling1D(enable_iq=False, name="gap")(h)
        h = norm("ln_head_fc1")(h)
        h = dense("head_fc1", D, h, use_bias=True)
        h = keras.layers.ReLU(name="head_act")(h)
        h = norm("ln_head_fc2")(h)
        out = dense("head_fc2", C, h, use_bias=True)

        model = keras.Model(x_in, out, name=cfg["name"])
    return model, taps


# --------------------------------------------------------------------------- #
# (f) activation calibration (warmup-calibrated-then-frozen static grids)      #
# --------------------------------------------------------------------------- #
def _static_quant_np(x, bits, i):
    """numpy image of the frozen KIF SAT grid (fixed<bits,1+i>, RND_CONV): step 2^-f,
    range [-2^i, 2^i - 2^-f], round-half-even, saturate. Matches build.py/gold."""
    f = bits - 1 - i
    step = 2.0 ** (-f)
    lo, hi = -(2.0 ** i), 2.0 ** i - step
    return np.clip(np.round(x / step) * step, lo, hi)


def calibrate_activations(model, taps: dict, X: np.ndarray, act_bits: int,
                          batch: int = 4096) -> dict:
    """Set every static activation quantizer's integer bits by MSE-optimal per-tensor
    calibration on a real batch.

    All quant inputs are post-SubLN (range-stable in weights), so ONE forward suffices.
    Policy = argmin_i MSE of the static grid (gold-model E4 finding: max+margin over-
    ranges low-bit grids — at A4 it leaves f=0, an integer-only grid that collapses LN
    structure; MSE trades range for fractional resolution). i is floored so f>=1 always.
    Frozen thereafter. Returns {site: i}. No-op for fp32. Sites matched by name (dense)
    or `<layer>__in<k>` (einsum inputs)."""
    if not taps:
        return {}
    probe = keras.Model(model.inputs, {k: v for k, v in taps.items()})
    outs = probe.predict(X[:batch], verbose=0)
    i_hi = max(0, act_bits - 2)                      # guarantee f = bits-1-i >= 1
    cands = list(range(0, i_hi + 1))
    site_i = {}
    for site, arr in outs.items():
        a = np.asarray(arr, dtype=np.float64).ravel()
        sse = [float(np.mean((_static_quant_np(a, act_bits, i) - a) ** 2)) for i in cands]
        site_i[site] = int(cands[int(np.argmin(sse))])

    by_name = {ly.name: ly for ly in model.layers}
    for site, i in site_i.items():
        f = act_bits - 1 - i
        if site.endswith("__in0") or site.endswith("__in1"):
            layer = by_name[site.rsplit("__in", 1)[0]]
            idx = int(site[-1])
            qz = layer.iq[idx].quantizer
        else:
            qz = by_name[site].iq.quantizer
        _assign_if(qz, i, f)
    return site_i


def _assign_if(qz, i, f):
    """Set the integer bits `i` of a datalane quantizer (init for both frozen and
    trainable modes). KIF also carries an explicit `_f` (fractional bits); KBI derives
    f = b - i from the pinned total width `_b`, so only `_i` is set there."""
    iv = getattr(qz, "_i", None)
    if iv is None:
        return
    iv.assign(np.full(iv.shape, i, dtype="float32"))
    fv = getattr(qz, "_f", None)
    if fv is not None:            # KIF: fractional bits are explicit
        fv.assign(np.full(fv.shape, f, dtype="float32"))
    # KBI (trainable-scale): f = b - i is derived; b is pinned by its constraint.


def act_grid_params(model):
    """Snapshot {site: (i, f)} of every datalane activation quantizer — the drift-tracking
    evidence (print before vs after training for the trainable-scale mode)."""
    out = {}
    for ly in model.layers:
        iq = getattr(ly, "iq", None)
        if iq is None:
            continue
        qzs = list(iq) if hasattr(iq, "__len__") else [iq]
        for k, q in enumerate(qzs):
            qz = getattr(q, "quantizer", None)
            iv = getattr(qz, "_i", None)
            if iv is None or getattr(qz, "__dummy__", False):
                continue
            i = float(np.asarray(ops.convert_to_numpy(qz.i)).ravel()[0])
            f = float(np.asarray(ops.convert_to_numpy(qz.f)).ravel()[0])
            out[ly.name if len(qzs) == 1 else f"{ly.name}__in{k}"] = (round(i, 3), round(f, 3))
    return out


def effective_weight_values(model):
    """Return {layer_name: sorted unique effective forward weight values} for every
    binary layer — the binary gate (must be exactly two values, symmetric, no zero)."""
    out = {}
    for ly in model.layers:
        if isinstance(ly, (BitQEinsumDense, BitQDense)):
            w = ops.convert_to_numpy(bitnet_binary_ste(ly._kernel))
            out[ly.name] = np.unique(np.round(w, 10))
    return out
