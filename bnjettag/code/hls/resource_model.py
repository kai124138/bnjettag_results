#!/usr/bin/env python3
"""
Per-component FPGA resource model for the binary + softmax-free BitNet jet tagger.

WHY THIS EXISTS
---------------
The research abstract asks for a "per-component breakdown of FPGA resource usage."
The gold-standard route is hls4ml -> Vitis/Vivado HLS C-synthesis (csynth), which
reports exact LUT/FF/DSP/BRAM per layer. We do NOT have a Xilinx HLS backend on the
NRP cluster (the upstream HLS_qk_Roc_Tracing.py hardcodes a Vitis path from a
different cluster). So this script computes the breakdown analytically from the
trained architecture.

It separates HARD structural counts (assumption-free: MACs, multiplies, weight bits,
LayerNorm ops) from DERIVED resource estimates (LUT/DSP/BRAM via clearly-labeled
cost factors). The hard counts are exact and back the qualitative story; the derived
numbers are first-order estimates to be replaced by real csynth when a Vitis backend
is available.

KEY ARCHITECTURAL FACTS
-----------------------
* Every BitLinear = LayerNorm(SubLN) -> A-bit absmax activation quant -> matmul with
  BINARY {-1,+1} weights -> +bias. A binary-weight x A-bit-activation "MAC" needs NO
  multiplier: it is a conditional negate (1 XOR on the sign) followed by an add. So
  these MACs cost LUTs (adder trees), NOT DSPs. (Pure XNOR+popcount is the W1A1 limit;
  at A>1 it is a "sign-select + adder-tree", still DSP-free.)
* The ONLY real (activation x activation) multiplies are the attention score matmuls
  Q.K^T and scores.V. With softmax-free attention the post-score path is just ReLU +
  a constant 1/N shift (no exp, no reciprocal).
"""

import math

# ---- architecture (the trained max model) --------------------------------------
N      = 10     # particles / tokens per jet
F      = 14     # input features per particle
D      = 256    # d_model
H      = 8      # attention heads
DH     = D // H # d_head = 32
L      = 8      # transformer blocks
FFN    = 1024   # ffn hidden dim

# ---- cost-factor assumptions (LABELLED; swap for csynth later) ------------------
# Fully-parallel (reuse factor RF=1, lowest latency = the L1-trigger regime).
def lut_per_binmac(a_bits):
    # binary-weight x a-bit-activation term in an adder tree: ~1 XOR (sign) + an
    # a-bit add. A fixed-point add of width w ~ w/1.5 6-input LUTs; use ~a LUTs/term
    # (conservative, includes tree carry/overhead). Range cited in lit ~1-2.5 LUT for
    # W1A1; scales ~linearly with activation width.
    return a_bits  # LUTs per binary MAC term (parallel)
FF_PER_BINMAC      = 0.0   # combinational adder tree, registered only at layer boundary
DSP_PER_BINMAC     = 0.0   # <-- the whole point: binary weights => no DSP
# real activation x activation multiply (attention scores):
def dsp_per_realmac(a_bits):
    return 1.0 if a_bits >= 7 else 0.0   # A8 -> 1 DSP48; A<=6 packs into LUTs
def lut_per_realmac(a_bits):
    return 0.0 if a_bits >= 7 else (a_bits * a_bits) / 3.0  # LUT multiply at low prec
BRAM18_BITS        = 18 * 1024          # 18 Kb per BRAM18

def fixed_bias_bits():  return 16       # bias / scale stored as ~16-bit fixed
def act_tensor_bits(a_bits, elems):  return a_bits * elems

# ---- component table ------------------------------------------------------------
# each entry: (name, n_in, n_out, tokens, has_bias, instances)
def bitlinear(name, nin, nout, tokens, bias, inst=1):
    return dict(name=name, nin=nin, nout=nout, tokens=tokens, bias=bias, inst=inst,
                binmac=nin*nout*tokens*inst,
                wbits=nin*nout*inst,
                ln_elems=nin*tokens*inst,         # LayerNorm runs on the input, per token
                bias_n=(nout*inst if bias else 0))

comps = []
comps.append(bitlinear("input_proj (14->256)", F, D, N, True))
# per-block components, summed over L via inst=L
comps.append(bitlinear("attn.W_q (256->256)", D, D, N, False, inst=L))
comps.append(bitlinear("attn.W_k (256->256)", D, D, N, False, inst=L))
comps.append(bitlinear("attn.W_v (256->256)", D, D, N, False, inst=L))
comps.append(bitlinear("attn.W_o (256->256)", D, D, N, True,  inst=L))
comps.append(bitlinear("ffn.fc1 (256->1024)", D, FFN, N, True, inst=L))
comps.append(bitlinear("ffn.fc2 (1024->256)", FFN, D, N, True, inst=L))
comps.append(bitlinear("head_fc1 (256->256)", D, D, 1, True))
comps.append(bitlinear("head_fc2 (256->1)",   D, 1, 1, True))

# attention score matmuls (REAL multiplies, not binary): Q.K^T and scores.V
realmac_per_block = (H * N * N * DH) + (H * N * N * DH)   # Q.K^T + scores.V
realmac_total     = realmac_per_block * L

def fmt(n):
    return f"{n:,}"

def main():
    print("="*100)
    print("BINARY + SOFTMAX-FREE BitNet JET TAGGER  |  per-component resource model")
    print(f"arch: N={N} F={F} D={D} H={H} d_head={DH} L={L} FFN={FFN}")
    print("="*100)

    # ---- HARD structural counts (assumption-free) -------------------------------
    print("\n[1] HARD STRUCTURAL COUNTS  (exact; independent of any cost factor)\n")
    hdr = f"{'component':<24}{'inst':>5}{'tokens':>7}{'binary MACs':>16}{'weight bits':>14}{'LN elems':>12}"
    print(hdr); print("-"*len(hdr))
    tot_bin=tot_w=tot_ln=tot_bias=0
    for c in comps:
        print(f"{c['name']:<24}{c['inst']:>5}{c['tokens']:>7}{fmt(c['binmac']):>16}{fmt(c['wbits']):>14}{fmt(c['ln_elems']):>12}")
        tot_bin+=c['binmac']; tot_w+=c['wbits']; tot_ln+=c['ln_elems']; tot_bias+=c['bias_n']
    print("-"*len(hdr))
    print(f"{'TOTAL binary-weight':<24}{'':>5}{'':>7}{fmt(tot_bin):>16}{fmt(tot_w):>14}{fmt(tot_ln):>12}")
    print(f"\n  attention score multiplies (REAL act x act, NOT binary): {fmt(realmac_total)} "
          f"({100*realmac_total/(tot_bin+realmac_total):.2f}% of all MACs)")
    print(f"  -> {100*tot_bin/(tot_bin+realmac_total):.2f}% of all MACs are binary (DSP-free) <-")
    print(f"  total weight storage: {fmt(tot_w)} bits = {tot_w/8/1024:.1f} KiB = {tot_w/1e6:.2f} Mbit "
          f"(~{math.ceil(tot_w/BRAM18_BITS)} BRAM18 if block-stored; usually LUTROM/distributed)")
    print(f"  total bias/scale params (~{fixed_bias_bits()}-bit fixed): {fmt(tot_bias)}")
    print(f"  LayerNorm (SubLN) instances: {sum(1 for _ in comps if True)*0 + (6*L+3)} "
          f"(one per BitLinear), total normalized elements/inference: {fmt(tot_ln)}")

    # ---- DERIVED estimates per activation precision -----------------------------
    print("\n[2] DERIVED RESOURCE ESTIMATES vs ACTIVATION PRECISION  (RF=1, fully parallel)\n")
    print("    cost factors: binMAC -> A LUT / 0 DSP ; realMAC(A>=7) -> 1 DSP ; realMAC(A<=6) -> LUT")
    h2 = f"{'precision':<12}{'binMAC LUTs':>16}{'attn LUTs':>12}{'attn DSPs':>12}{'act-buf bits':>14}{'acc width':>11}"
    print(h2); print("-"*len(h2))
    main_act_elems = N*D + N*FFN        # main token buffer + FFN hidden
    acc_in_max = FFN                    # widest dot product (fc2 input)
    for A in (8, 6, 4):
        bin_luts  = int(tot_bin * lut_per_binmac(A))
        attn_luts = int(realmac_total * lut_per_realmac(A))
        attn_dsps = int(realmac_total * dsp_per_realmac(A))
        actbits   = act_tensor_bits(A, main_act_elems)
        accw      = A + math.ceil(math.log2(acc_in_max))
        print(f"A{A:<11}{fmt(bin_luts):>16}{fmt(attn_luts):>12}{fmt(attn_dsps):>12}{fmt(actbits):>14}{accw:>9}b")
    print("\n    NOTE: RF=1 LUT totals are an UPPER bound (one MAC = one LUT-slice). Real")
    print("    firmware time-multiplexes with a reuse factor RF, dividing parallel LUTs by")
    print("    ~RF and multiplying latency by ~RF. The DSP=0 result for the binary core is")
    print("    structural (no multiplier), not an estimate.")

    # ---- device fit / reuse factor (reality check) ------------------------------
    # XCVU13P (the part the upstream HLS script targets): official capacity.
    VU13P = dict(LUT=1_728_000, FF=3_456_000, DSP=12_288, BRAM18=5_376)
    print("\n[2b] DEVICE FIT on XCVU13P  (LUT=1.728M, FF=3.456M, DSP=12,288, BRAM18=5,376)\n")
    A = 8
    bin_lutops = tot_bin * lut_per_binmac(A)        # parallel LUT-ops at RF=1 (UPPER bound)
    print(f"    binary core LUT-ops @ A8, RF=1 (upper bound): {fmt(int(bin_lutops))} "
          f"= {bin_lutops/VU13P['LUT']:.0f}x a VU13P's LUTs")
    print( "    => RF=1 is INFEASIBLE for this research-size model (63M MACs); a real trigger")
    rf_fit = bin_lutops / (0.5 * VU13P['LUT'])      # target: use <=50% LUTs
    print(f"       time-multiplexes. Reuse factor to fit <=50% LUTs (pre-synth, conservative):")
    print(f"       RF ~ {rf_fit:,.0f}  -> latency ~ RF cycles (~{rf_fit/0.4e3:,.0f} us @ 400 MHz).")
    print( "    NOTE: hls4ml PRE-synth LUT is overstated ~3-10x vs real logic synth (lit.), and")
    print( "    binary jet-taggers measure ~1-8% LUT / 0% DSP / 0 BRAM (arXiv:2003.06308). So the")
    print( "    DSP=0 result is solid; the LUT figure says 'this max model must shrink or fold'")
    print( "    (the activation-precision and model-size knobs are exactly those levers).")

    # ---- softmax vs softmax-free op delta ---------------------------------------
    print("\n[3] SOFTMAX vs SOFTMAX-FREE  (attention nonlinearity, per inference)\n")
    rows  = H * N * L            # attention rows (one softmax/relu-row per query)
    score = H * N * N * L        # score entries
    print(f"    softmax core would need : {fmt(score)} exp() evals (LUT tables/BRAM) + "
          f"{fmt(rows)} reciprocals (DSP/LUT-heavy) + {fmt(score)} norm mults")
    print(f"    softmax-free uses       : {fmt(score)} ReLU (1 comparator each) + a CONSTANT 1/N shift (free)")
    print(f"    => removes ALL {fmt(score)} exp tables and {fmt(rows)} reciprocals; the single")
    print(f"       most expensive, non-binarizable attention op is gone. This is what keeps the")
    print(f"       whole MHSA on LUT/logic and de-risks hls4ml conversion.")
    print("="*100)

if __name__ == "__main__":
    main()
