"""
ebops.py — static EBOPs / BOPs hardware-cost profiler for the BitNet jet tagger
================================================================================
A STATIC, architecture-only accounting tool. It does NOT build TensorFlow, does
NOT load data, and does NOT train — it encodes the per-layer multiply-accumulate
(MAC) and parameter profile of ``build_bitnet_jet_tagger`` (qkerasModel.py) in
closed form, and turns MACs into Bit-Operations (BOPs) for a chosen weight/
activation bit-width.

Why this exists
---------------
EBOPs (Effective Bit-Operations) is the hardware-cost axis for the thesis: a
binary {-1,+1} weight makes every linear-layer MAC a 1xb_a operation, so the
*bit-op* cost of the matmul core collapses vs FP32/W8A8 — the same structural win
that maps the binary matmul onto FPGA LUT/logic (0 DSP) instead of multipliers.

EBOPs definition (HGQ, Sun/Aarrestad et al., arXiv:2405.00645, Eq. 5):

    EBOPs = Σ over every scalar multiply (i,j) of   b_i * b_j

i.e. sum b_i*b_j over every element of every matmul. Two consequences we implement
faithfully:
  * Weight*activation layers (Q/K/V/O, FFN up/down, input proj, head) cost
    b_w * b_a per MAC.  The attention score matmuls Q.K^T and softmax.V put an
    ACTIVATION on BOTH operands, so they cost b_a * b_a per MAC — binarization does
    NOT help attention, only the linear layers. (For the large model attention is
    only 0.65% of MACs, so the binary win is ~7.65x vs W8A8, not a clean 8x.)
  * The ACCUMULATOR / adder-tree is EXCLUDED by design. HGQ, verbatim: "EBOPs
    effectively count only the BOPs conducted during multiplicative processes in a
    network" — accumulation is implicitly counted via the multiply-sum, not added
    as a separate term. So accumulator_bops()=0 is the HGQ DEFINITION, not a stub.
    (The older UNIQ-BOPs, arXiv:1804.10969, DOES add b_a+b_w+log2(fan_in) per MAC;
    that form is documented in accumulator_bops() for reference, off by default.)

Empirical bridge to FPGA resource (HGQ §V.1, io_parallel / unrolled, post-PnR):

    EBOPs  ≈  #LUT  +  55 * #DSP

For binary weights the synthesizer uses LUT sign-select (DSP=0, confirmed by our
2026-06-24 csynth), so for the binary core EBOPs ≈ #LUT directly. NB: this mapping
is for FULLY-UNROLLED designs; our deployed model folds (RF=256), so treat EBOPs as
a cheap, architecture-level RELATIVE efficiency metric, not a literal folded-LUT count.

Elementwise ops (norms, ReLU/GELU, residual adds) and the softmax exp/divide carry
0 MAC and 0 weight bits, so they are tracked separately and excluded from EBOPs.

Provenance of the layer profile (read qkerasModel.py alongside this):
  * BitLinear: kernel (in_dim, units); bias (units,) ONLY if use_bias. The norms
    (RMSNorm/SubLN, TrueRMSNorm, IdentityNorm) are parameter-free.
  * input_proj  : BitLinear(D), use_bias=True, applied per-particle (N tokens).
  * pos_enc default "learned": Embedding(N,D)(tf.range(N)) folds to a FIXED offset
    that is NOT a tracked/trained weight -> contributes 0 params (verified: all
    four size variants reconstruct their known count_params() ONLY with this = 0).
  * BitMHSA: W_q/W_k/W_v use_bias=False; W_o use_bias=True. Standard multi-head:
    d_head = D/H, scores are scaled dot-product over heads.
  * BitFFN : fc1 D->F (bias), ReLU/GELU, fc2 F->D (bias).
  * pool default "gap": GlobalAveragePooling1D over N particles -> (D,). No CLS,
    so seq_len N = N_PART = 10 (BN_POOL=cls would make it N_PART+1 = 11).
  * head: head_fc1 BitLinear(D) (bias) on the pooled vector (ONE token), ReLU,
    head_fc2 BitLinear(N_CLASSES) (bias) — 1 logit in era 1, 5 logits in era 2.

Two dataset eras (RESEARCH.md §4): era 1 = private 2-class (N_FEAT=14, 1-logit
head); era 2 = public HLS4ML LHC Jet 5-class (N_FEAT=16, 5-logit head). Default
is era 2 (current). Cross-check: profile(...).params reproduces the verified
CPU-preflight counts exactly — era 1: tiny 26,529 / small 153,793 / medium
808,065 / large 6,373,633; era 2: large 6,375,173 (round-5 preflight 2026-07-01).

Usage
-----
    python ebops.py                 # era-2 large (fixed main) per-layer + BOPs table
    python ebops.py --era 1         # frozen era-1 accounting (14 feat, 1-logit head)
    python ebops.py --size all      # totals for tiny/small/medium/large
    python ebops.py --size medium

No heavy compute. CPU only. Stdlib only (no numpy / no tensorflow).
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Fixed data-pipeline constants (must match qkerasModel.py)
# ─────────────────────────────────────────────────────────────────────────────
# Two dataset eras (RESEARCH.md §4) — the model I/O differs between them:
#   era 1 (frozen, runs ≤ round-4): private 2-class set — 14 features, 1-logit head
#   era 2 (current, round-5+): public HLS4ML LHC Jet — 16 features, 5-class head
# N_PART = 10 in both (top-10 constituents by pT; GAP pool, no CLS ⇒ seq len 10).
N_PART = 10


@dataclass(frozen=True)
class Era:
    n_feat: int
    n_classes: int
    # Known-good trainable-parameter counts from verified CPU preflights; the
    # profiler asserts against these so accounting drift is caught loudly.
    # Sizes without a verified preflight are absent and not asserted.
    known_params: Dict[str, int]


ERAS: Dict[str, Era] = {
    "1": Era(n_feat=14, n_classes=1, known_params={
        "tiny": 26_529, "small": 153_793, "medium": 808_065, "large": 6_373_633}),
    # era-2: only `large` has a verified preflight (round-5, 2026-07-01).
    "2": Era(n_feat=16, n_classes=5, known_params={"large": 6_375_173}),
}

# Named architecture sizes. The FIXED main model is "large" (the upstream config).
# (d_model, n_heads, n_layers, ffn_dim) — input/head dims come from the era.
SIZES: Dict[str, Tuple[int, int, int, int]] = {
    "tiny":   ( 32, 4, 2,  128),
    "small":  ( 64, 8, 3,  256),
    "medium": (128, 8, 4,  512),
    "large":  (256, 8, 8, 1024),   # <-- FIXED main architecture
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-layer profile
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LayerCost:
    name: str
    params: int
    macs: int            # multiply-accumulates per single-jet forward pass
    kind: str            # "matmul" (weight*act, EBOPs-relevant) | "attn" (act*act) | "elementwise"
    note: str = ""


@dataclass
class Profile:
    size: str
    era: str
    dims: Tuple[int, int, int, int, int, int]   # (n_feat, n_part, D, H, L, F)
    n_classes: int
    layers: List[LayerCost] = field(default_factory=list)

    # ---- aggregate helpers -------------------------------------------------
    @property
    def params(self) -> int:
        return sum(l.params for l in self.layers)

    @property
    def macs_total(self) -> int:
        return sum(l.macs for l in self.layers)

    @property
    def macs_matmul(self) -> int:
        """Weight*activation matmul MACs — the BOPs term scaled by b_w*b_a."""
        return sum(l.macs for l in self.layers if l.kind == "matmul")

    @property
    def macs_attn(self) -> int:
        """Activation*activation attention MACs (Q.K^T, softmax.V) — scaled b_a*b_a."""
        return sum(l.macs for l in self.layers if l.kind == "attn")


def build_profile(size: str, era: str = "2") -> Profile:
    """Encode the closed-form per-layer MAC + param profile for a named size/era."""
    D, H, L, F = SIZES[size]
    e = ERAS[era]
    n_feat, n_part = e.n_feat, N_PART
    assert D % H == 0, f"d_model {D} not divisible by n_heads {H}"
    N = n_part                                  # seq len (GAP -> N_PART; CLS would be +1)
    prof = Profile(size=size, era=era, dims=(n_feat, n_part, D, H, L, F),
                   n_classes=e.n_classes)

    def lin(name, in_dim, units, n_tokens, use_bias, kind="matmul", note=""):
        params = in_dim * units + (units if use_bias else 0)
        macs = n_tokens * in_dim * units
        prof.layers.append(LayerCost(name, params, macs, kind, note))

    # ── Input projection: per-particle BitLinear(D) (use_bias=True) ──────────
    lin("input_proj", n_feat, D, N, use_bias=True,
        note="per-particle N_FEAT->D")

    # ── Positional encoding "learned": Embedding folds to a fixed offset ─────
    # 0 trainable params, 0 MACs (a constant add). Recorded for completeness.
    prof.layers.append(LayerCost(
        "pos_embedding", 0, 0, "elementwise",
        note="learned Embedding folds to FIXED offset -> 0 trainable params"))

    # ── L transformer blocks ────────────────────────────────────────────────
    for i in range(L):
        b = f"bit_block_{i}"
        # --- BitMHSA ---
        lin(f"{b}_attn_Wq", D, D, N, use_bias=False, note="Q proj (binary)")
        lin(f"{b}_attn_Wk", D, D, N, use_bias=False, note="K proj (binary)")
        lin(f"{b}_attn_Wv", D, D, N, use_bias=False, note="V proj (binary)")
        # Q.K^T : per head N*N*d_head, summed over H heads == N*N*D (act x act, FP32 scores)
        prof.layers.append(LayerCost(
            f"{b}_attn_QKt", 0, N * N * D, "attn",
            note="scores = Q.K^T over H heads (sum_h N*N*d_head = N*N*D)"))
        # softmax.V (or ReLU.V/N if softmax-free): same shape, act x act
        prof.layers.append(LayerCost(
            f"{b}_attn_AV", 0, N * N * D, "attn",
            note="ctx = (softmax)A.V over H heads = N*N*D"))
        lin(f"{b}_attn_Wo", D, D, N, use_bias=True, note="output proj (binary)")
        # softmax over (H, N, N) logits — exp + row-sum divide (non-binarizable).
        prof.layers.append(LayerCost(
            f"{b}_attn_softmax", 0, 0, "elementwise",
            note=f"softmax over H*N*N={H*N*N} logits (exp+divide; 0 MAC, not BOPs)"))
        # --- BitFFN ---
        lin(f"{b}_ffn_fc1", D, F, N, use_bias=True, note="FFN up D->F (binary)")
        lin(f"{b}_ffn_fc2", F, D, N, use_bias=True, note="FFN down F->D (binary)")
        # norms (RMSNorm/SubLN x per-linear) + ReLU/GELU + 2 residual adds: parameter-free,
        # negligible MAC; folded into one elementwise marker per block.
        prof.layers.append(LayerCost(
            f"{b}_norm_act_res", 0, 0, "elementwise",
            note="SubLN/RMSNorm + ReLU/GELU + residual adds (param-free)"))

    # ── Pool (GAP over N particles) ─────────────────────────────────────────
    prof.layers.append(LayerCost(
        "global_average_pooling1d", 0, 0, "elementwise",
        note="mean over N particles -> (D,)"))

    # ── Classifier head: BitLinear(D) -> ReLU -> BitLinear(n_classes), 1 token ──
    lin("head_fc1", D, D, 1, use_bias=True, note="head dense D->D (binary)")
    lin("head_fc2", D, e.n_classes, 1, use_bias=True,
        note=f"head logits D->{e.n_classes} (binary)")

    return prof


# ─────────────────────────────────────────────────────────────────────────────
# BOPs / EBOPs
# ─────────────────────────────────────────────────────────────────────────────
def bops(macs: int, b_w: int, b_a: int) -> int:
    """Bit-Operations of a matmul: MACs * b_i * b_j (the per-multiply EBOPs term).

    Per HGQ (arXiv:2405.00645) EBOPs is exactly Σ b_i*b_j over multiplies, with no
    separate accumulator term, so this product IS the EBOPs for a layer.
    """
    return macs * b_w * b_a


def accumulator_bops(macs: int, fan_in: int, b_w: int, b_a: int,
                     include_uniq: bool = False) -> int:
    """Accumulator (partial-sum) bit-growth term.

    HGQ EBOPs (arXiv:2405.00645) EXCLUDES this by design: "EBOPs effectively count
    only the BOPs conducted during multiplicative processes in a network." So for
    the EBOPs metric this is 0 — that is the definition, not a placeholder.

    The OLDER UNIQ-BOPs (arXiv:1804.10969) instead charges the adder tree as
    (b_a + b_w + ceil(log2(fan_in))) per MAC. Set include_uniq=True to get that
    legacy term if you ever want UNIQ-style BOPs instead of HGQ EBOPs.
    """
    if not include_uniq:
        return 0
    from math import ceil, log2
    acc_bits = b_a + b_w + (ceil(log2(fan_in)) if fan_in > 1 else 0)
    return macs * acc_bits


def ebops(prof: Profile, b_w: int, b_a: int) -> Dict[str, int]:
    """EBOPs for a whole profile at a given (weight, activation) bit-width.

    Returns a dict split by term so callers can see what dominates:
      - matmul_bops : Σ weight-matmul MACs * b_w * b_a   (the EBOPs core)
      - attn_bops   : Σ attention (act*act) MACs * b_a * b_a
      - accum_bops  : Σ accumulator term (0 until HGQ is wired in)
      - total       : matmul + attn + accum
    """
    matmul_bops = bops(prof.macs_matmul, b_w, b_a)
    attn_bops = bops(prof.macs_attn, b_a, b_a)        # both operands are activations
    accum = 0
    for l in prof.layers:
        if l.kind == "matmul":
            # fan_in = MACs / (tokens*units) is just in_dim; recover it cheaply.
            accum += accumulator_bops(l.macs, fan_in=0, b_w=b_w, b_a=b_a)
    return {
        "matmul_bops": matmul_bops,
        "attn_bops": attn_bops,
        "accum_bops": accum,
        "total": matmul_bops + attn_bops + accum,
    }


# Quantization presets for the BOPs-per-precision table.
# NOTE: FP32 is treated as 32x32 ONLY as a reference scale — floating-point is NOT
# a true bit-operation (the mantissa/exponent multiply is not a b_w*b_a popcount),
# so the FP32 row is an upper-bound proxy, not a literal BOPs count.
QUANT_PRESETS: List[Tuple[str, int, int]] = [
    ("FP32 (ref, not true bit-ops)", 32, 32),
    ("W8A8",                          8,  8),
    ("W1A8 (binary)",                 1,  8),
    ("W1A6",                          1,  6),
    ("W1A4",                          1,  4),
]


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-printers
# ─────────────────────────────────────────────────────────────────────────────
def _assert_params(prof: Profile) -> None:
    known = ERAS[prof.era].known_params.get(prof.size)
    if known is not None:
        assert prof.params == known, (
            f"PARAM MISMATCH for {prof.size} (era {prof.era}): computed "
            f"{prof.params:,} != known-good {known:,} — the accounting drifted.")


def print_layer_table(prof: Profile) -> None:
    _assert_params(prof)
    n_feat, n_part, D, H, L, F = prof.dims
    print(f"# Per-layer MAC + param profile — size={prof.size} era={prof.era} "
          f"(D={D} H={H} L={L} F={F}, N_PART={n_part}, N_FEAT={n_feat}, "
          f"N_CLASSES={prof.n_classes})\n")
    print(f"| {'layer':28s} | {'kind':11s} | {'params':>11s} | {'MACs/jet':>13s} | note |")
    print(f"|{'-'*30}|{'-'*13}|{'-'*13}|{'-'*15}|------|")
    for l in prof.layers:
        print(f"| {l.name:28s} | {l.kind:11s} | {l.params:>11,} | {l.macs:>13,} | {l.note} |")
    print(f"|{'-'*30}|{'-'*13}|{'-'*13}|{'-'*15}|------|")
    print(f"| {'TOTAL':28s} | {'':11s} | {prof.params:>11,} | {prof.macs_total:>13,} | "
          f"matmul={prof.macs_matmul:,} attn={prof.macs_attn:,} |")
    known = ERAS[prof.era].known_params.get(prof.size)
    if known is None:
        tag = f"(no verified era-{prof.era} preflight for this size — not asserted)"
    elif known == prof.params:
        tag = f"== known-good {known:,}  [OK]"
    else:
        tag = f"!= known-good {known:,}  [MISMATCH]"
    print(f"\nparam cross-check: computed {prof.params:,}  {tag}")
    print(f"MAC split: matmul (weight*act, EBOPs core) = {prof.macs_matmul:,} "
          f"({100*prof.macs_matmul/prof.macs_total:.2f}%) | "
          f"attention (act*act) = {prof.macs_attn:,} "
          f"({100*prof.macs_attn/prof.macs_total:.2f}%)")


def print_ebops_table(prof: Profile) -> None:
    n_feat, n_part, D, H, L, F = prof.dims
    print(f"\n# EBOPs-per-quantization (HGQ arXiv:2405.00645) — size={prof.size} "
          f"era={prof.era} (D={D} H={H} L={L} F={F})")
    print("# EBOPs = matmul-MACs*b_w*b_a + attn-MACs*b_a*b_a  (accumulator excluded by HGQ definition)")
    print(f"\n| {'precision':30s} | {'b_w':>3s} | {'b_a':>3s} | {'matmul EBOPs':>16s} | "
          f"{'attn EBOPs':>13s} | {'total EBOPs':>16s} | {'vs FP32':>8s} | {'vs W8A8':>8s} |")
    print(f"|{'-'*32}|{'-'*5}|{'-'*5}|{'-'*18}|{'-'*15}|{'-'*18}|{'-'*10}|{'-'*10}|")
    ref = ebops(prof, 32, 32)["total"]
    w8 = ebops(prof, 8, 8)["total"]
    for label, b_w, b_a in QUANT_PRESETS:
        e = ebops(prof, b_w, b_a)
        rfp = e["total"] / ref if ref else 0.0
        rw8 = e["total"] / w8 if w8 else 0.0
        print(f"| {label:30s} | {b_w:>3d} | {b_a:>3d} | {e['matmul_bops']:>16,} | "
              f"{e['attn_bops']:>13,} | {e['total']:>16,} | {rfp:>7.3f}x | {rw8:>7.3f}x |")
    e_w1a8 = ebops(prof, 1, 8)["total"]
    print(f"\nΔEBOPs (W1A8 binary): {w8/e_w1a8:.2f}x below W8A8 ({w8:,} -> {e_w1a8:,}); "
          f"{ref/e_w1a8:.1f}x below FP32(ref).")
    print(f"Equal-EBOPs headroom: at the W8A8 EBOPs budget a binary (W1A8) model could carry "
          f"~{w8/e_w1a8:.2f}x more MACs and still cost <= W8A8.")
    print("note: FP32 row uses 32x32 only as a reference scale — floating-point is NOT a true bit-op.")
    print("note: EBOPs ~= #LUT + 55*#DSP (HGQ, unrolled); binary -> DSP=0 -> EBOPs ~= #LUT.")


def print_size_totals(era: str) -> None:
    print(f"# MAC + param TOTALS across size variants, era {era} "
          f"(param cross-check vs preflight)\n")
    print(f"| {'size':7s} | {'D':>4s} | {'H':>2s} | {'L':>2s} | {'F':>5s} | "
          f"{'params':>11s} | {'known':>11s} | {'match':>6s} | {'MACs/jet':>13s} | "
          f"{'matmul MACs':>13s} | {'attn MACs':>10s} |")
    print(f"|{'-'*9}|{'-'*6}|{'-'*4}|{'-'*4}|{'-'*7}|{'-'*13}|{'-'*13}|{'-'*8}|"
          f"{'-'*15}|{'-'*15}|{'-'*12}|")
    for name in ("tiny", "small", "medium", "large"):
        p = build_profile(name, era)
        _, _, D, H, L, F = p.dims
        known = ERAS[era].known_params.get(name)
        if known is None:
            known_s, ok = "n/a", "—"
        else:
            known_s, ok = f"{known:,}", ("OK" if p.params == known else "MISMATCH")
        print(f"| {name:7s} | {D:>4d} | {H:>2d} | {L:>2d} | {F:>5d} | {p.params:>11,} | "
              f"{known_s:>11s} | {ok:>6s} | {p.macs_total:>13,} | {p.macs_matmul:>13,} | "
              f"{p.macs_attn:>10,} |")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Static EBOPs/BOPs profiler for the BitNet jet tagger")
    ap.add_argument("--size", default="large",
                    choices=["tiny", "small", "medium", "large", "all"],
                    help="architecture size (default: large = fixed main); 'all' = totals table")
    ap.add_argument("--era", default="2", choices=sorted(ERAS),
                    help="dataset era: 1 = private 2-class (14 feat, frozen), "
                         "2 = HLS4ML LHC Jet 5-class (16 feat, current; default)")
    args = ap.parse_args()

    if args.size == "all":
        print_size_totals(args.era)
        return

    prof = build_profile(args.size, args.era)
    print_layer_table(prof)
    print_ebops_table(prof)


if __name__ == "__main__":
    main()
