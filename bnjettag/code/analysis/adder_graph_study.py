"""Adder-graph counting study on the deployed {-1,+1} weight matrices.

Counts, on the exact matrices that were synthesized, what each restructuring
mechanism would save relative to the naive per-output adder tree that Vitis emits
under the Latency strategy:

  naive     - balanced per-row reduction tree (the measured baseline)
  cse       - greedy two-term common-subexpression extraction (da4ml-class sharing)
  affine    - y = +-(T - 2*sum(minority set)), T = total input sum shared across rows
  fr        - Four-Russians / subset-sum block sharing, k in {2,4,8}
  fold      - token-axis time-multiplexing model (constant graph, muxed activations)

Every LUT number is csynth-estimate space, calibrated against the per-instance
census of whole_model_rf8_stdnn.xml. The thresholds each mechanism is judged
against were fixed before the counts were run; the flip-flop model is a bracket
between two measured emission disciplines rather than a point prediction, because
the flip-flop blow-up under distributed arithmetic is Vitis pipelining the emitted
graph, not the backend's own register insertion.

The synthesis check on these counts is Experiment 1, whose sources and raw reports
are under results/adder-graph/e1/; see Section 7 of the top-level README.

Run:  python bnjettag/code/analysis/adder_graph_study.py
Outputs: bnjettag/results/adder-graph/counting_results.json + console summary.
"""

import gzip
import json
import heapq
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]          # bnjettag/
STDNN_PRJ = REPO / "results/adder-graph/inputs"
STDNN_XML = REPO / "r7/results/csynth/whole_model_rf8_stdnn.xml.gz"
DA_XML = REPO / "r7/results/csynth/whole_model_da_rf8attn_stdnn.xml.gz"
OUT_DIR = REPO / "results/adder-graph"

DEVICE_LUT = 1_728_000          # VU13P
CLOCK_NS = 2.157                # stdnn achieved estimate
II_BUDGET = 11                  # floor(25 ns / 2.157 ns)
FOLD = 10                       # token count
MUX_LUT_PER_BIT = 3             # 10:1 mux, two LUT6 levels + margin

# DA emission census, parsed this session from whole_model_da_rf8attn_stdnn.xml
DA_DENSE_FF_MEAS = 5_733_677
DA_DENSE_LUT_MEAS = 2_786_772
PLAIN_DENSE_FF_MEAS = 825_435
PLAIN_DENSE_LUT_MEAS = 2_457_312


# ---------------------------------------------------------------- project parsing

def parse_defines(prj):
    """typedef ap_[u]fixed<W,I,...> name_t;  ->  {name_t: (W, I)}"""
    types = {}
    txt = (prj / "firmware/defines.h").read_text()
    for m in re.finditer(r"typedef\s+ap_u?fixed<(\d+),\s*(-?\d+)[^>]*>\s+(\w+);", txt):
        types[m.group(3)] = (int(m.group(1)), int(m.group(2)))
    return types


def parse_layers(prj):
    """einsum_dense/dense calls in myproject.cpp -> [(layer, cfg, input_t, wfile)]"""
    txt = (prj / "firmware/myproject.cpp").read_text()
    sizes = {}
    for m in re.finditer(r"load_weights_from_txt<\w+,\s*(\d+)>\((w\d+)", txt):
        sizes[m.group(2)] = int(m.group(1))
    layers = []
    pat = (r"nnet::(?:einsum_dense|dense_latency|dense)<(\w+),\s*\w+,\s*"
           r"config(\d+)>\([^)]*\b(w\d+)\b[^)]*\);\s*//\s*(\w+)")
    for m in re.finditer(pat, txt):
        in_t, cfg, wname, lname = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        layers.append(dict(layer=lname, cfg=cfg, input_t=in_t,
                           wfile=wname, wsize=sizes.get(wname)))
    return layers


def load_sign_matrix(prj, wfile, n_in, n_out):
    """Weight txt (flat, kernel[c][f] row-major) -> (m, n) sign matrix, rows=outputs."""
    vals = np.loadtxt(prj / f"firmware/weights/{wfile}.txt",
                      delimiter=",", dtype=np.float64).ravel()
    assert vals.size == n_in * n_out, (wfile, vals.size, n_in, n_out)
    W = vals.reshape(n_in, n_out).T
    s = np.sign(W).astype(np.int8)
    assert np.all(np.abs(s) == 1), f"{wfile}: not pure +-1"
    return s


def read_report(xml_path):
    """Parse a csynth report, gzipped or plain (they ship gzipped: 40 MB each)."""
    path = Path(xml_path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as fh:
        return ET.parse(fh).getroot()


def census(xml_path, res):
    """Per-module own resource from RTLDesignHierarchy (same walk as parse_census.py)."""
    r = read_report(xml_path)
    mres = {}
    for m in r.findall("ModuleInformation/Module"):
        e = m.find(".//AreaEstimates/Resources/" + res)
        mres[m.find("Name").text] = int(e.text) if e is not None and e.text else 0
    own, count = defaultdict(int), defaultdict(int)

    def walk(inst):
        mod = inst.find("ModuleName").text
        il = inst.find("InstancesList")
        ch = il.findall("Instance") if il is not None else []
        own[mod] += mres.get(mod, 0) - sum(mres.get(c.find("ModuleName").text, 0)
                                           for c in ch)
        count[mod] += 1
        for c in ch:
            walk(c)

    top = r.find("RTLDesignHierarchy/TopModule")
    topmod = top.find("ModuleName").text
    il = top.find("InstancesList")
    children = il.findall("Instance") if il is not None else []
    own[topmod] += mres.get(topmod, 0) - sum(mres.get(c.find("ModuleName").text, 0)
                                             for c in children)
    count[topmod] += 1
    for c in children:
        walk(c)
    return own, count


def measured_per_config(xml_path, res="LUT"):
    """config number -> (mean own res per instance, n instances, std) over the
    dense_latency modules (per-token instances are distinct modules, one each)."""
    own, count = census(xml_path, res)
    per_cfg = defaultdict(list)
    for mod, v in own.items():
        if "dense_latency" not in mod and "da_kernel" not in mod:
            continue
        m = re.search(r"config(\d+)", mod)
        if m:
            per_cfg[int(m.group(1))].append(v / max(count[mod], 1))
    return {c: (float(np.mean(v)), len(v), float(np.std(v)))
            for c, v in per_cfg.items()}


# ---------------------------------------------------------------- cost models

def tree_bit_adds(n_terms, w_in, w_acc):
    """(adds, bit-adds, depth) of a balanced reduction over n_terms w_in-bit values,
    widths growing one bit per level, saturating at w_acc."""
    adds = bits = depth = 0
    cnt, w = int(n_terms), w_in
    while cnt > 1:
        pairs, cnt = cnt // 2, (cnt // 2) + (cnt % 2)
        w = min(w + 1, w_acc)
        adds += pairs
        bits += pairs * w
        depth += 1
    return adds, bits, depth


def huffman_bits(widths, w_acc):
    """(adds, bit-adds) of reducing terms of given widths, merging two narrowest."""
    h = list(widths)
    heapq.heapify(h)
    adds = bits = 0
    while len(h) > 1:
        a, b = heapq.heappop(h), heapq.heappop(h)
        w = min(max(a, b) + 1, w_acc)
        adds += 1
        bits += w
        heapq.heappush(h, w)
    return adds, bits


def naive_cost(S, w_in, w_acc):
    m, n = S.shape
    a, b, d = tree_bit_adds(n + 1, w_in, w_acc)          # +1 = bias term
    return dict(adds=m * a, bits=m * b, depth=d)


def affine_cost(S, w_in, w_acc):
    """y_r = +-(T - 2*sum(minority set)); T built once, shared across all m rows.
    Per row: minority tree + one subtract (shift free) + bias add."""
    m, n = S.shape
    aT, bT, dT = tree_bit_adds(n, w_in, w_acc)
    wT = min(w_in + int(np.ceil(np.log2(n))), w_acc)
    adds, bits = aT, bT
    for r in range(m):
        k = min(int((S[r] > 0).sum()), int((S[r] < 0).sum()))
        a, b, _ = tree_bit_adds(max(k, 1), w_in, w_acc)
        adds += a + 2                                    # + subtract + bias
        bits += b + 2 * wT
    return dict(adds=adds, bits=bits, shared_T_adds=aT, depth=dT + 2)


def fr_cost(S, k, w_in, w_acc):
    """Four-Russians: per input-group of k, build all 2^(k-1) canonical signed sums
    (1 add each, incremental Gray-code chaining); every row wires one precomputed
    value per group (compile-time selection, no mux) and reduces g+1 terms.
    Matrix-independent for dense +-1 (that is the point)."""
    m, n = S.shape
    g = int(np.ceil(n / k))
    wg = min(w_in + int(np.ceil(np.log2(k))), w_acc)
    pre_adds = g * (2 ** (k - 1) - 1)
    pre_bits = pre_adds * wg
    a, b, d = tree_bit_adds(g + 1, wg, w_acc)            # +1 bias
    return dict(adds=pre_adds + m * a, bits=pre_bits + m * b,
                depth=int(np.ceil(np.log2(k))) + d, k=k, groups=g)


def greedy_cse(S, w_in, w_acc, cap=6000):
    """Greedy two-term extraction with incremental pair-count maintenance.

    Maintains P = M^T M and N = nnz(M)^T nnz(M) over a preallocated column space;
    agreements a=(N+P)/2, disagreements d=(N-P)/2. Each extraction zeroes the two
    source columns in the matched rows and appends a pseudo-input column.
    Returns adds saved, extraction count, capped flag, and LUT-weighted bits of
    the restructured computation (extraction adds + per-row Huffman reduction).
    """
    m, n = S.shape
    ncap = n + cap + 1
    M = np.zeros((m, ncap), dtype=np.int32)
    M[:, :n] = S
    widths = np.full(ncap, w_in, dtype=np.int32)
    ncols = n
    P = M[:, :n].T @ M[:, :n]
    N = np.abs(M[:, :n]).T @ np.abs(M[:, :n])
    Pf = np.zeros((ncap, ncap), dtype=np.int64)
    Nf = np.zeros((ncap, ncap), dtype=np.int64)
    Pf[:n, :n], Nf[:n, :n] = P, N
    saved = ext = 0
    ext_bits = 0

    def refresh(cols):
        cur = M[:, :ncols]
        for c in cols:
            Pf[c, :ncols] = cur.T @ M[:, c]
            Pf[:ncols, c] = Pf[c, :ncols]
            Nf[c, :ncols] = np.abs(cur).T @ np.abs(M[:, c])
            Nf[:ncols, c] = Nf[c, :ncols]

    while ext < cap:
        A = (Nf[:ncols, :ncols] + Pf[:ncols, :ncols]) // 2
        D = (Nf[:ncols, :ncols] - Pf[:ncols, :ncols]) // 2
        np.fill_diagonal(A, 0)
        np.fill_diagonal(D, 0)
        best = np.maximum(A, D)
        c = int(best.max())
        if c < 2:
            break
        i, j = np.unravel_index(int(best.argmax()), best.shape)
        sigma = 1 if A[i, j] >= D[i, j] else -1
        rows = np.where((M[:, i] != 0) & (M[:, j] == sigma * M[:, i]))[0]
        new = ncols
        M[rows, new] = M[rows, i]
        M[rows, i] = 0
        M[rows, j] = 0
        ncols += 1
        widths[new] = min(max(widths[i], widths[j]) + 1, w_acc)
        ext_bits += widths[new]
        refresh([i, j, new])
        saved += len(rows) - 1
        ext += 1

    row_bits = 0
    for r in range(m):
        terms = list(widths[:ncols][M[r, :ncols] != 0]) + [w_in]  # + bias
        _, b = huffman_bits(terms, w_acc)
        row_bits += b
    return dict(saved=int(saved), extractions=int(ext), capped=bool(ext >= cap),
                bits=int(ext_bits + row_bits))


def cse_pair_stats(S):
    """Duplicate/negated row classes (the July column-sharing metric) and the
    single best pair's reuse count."""
    m, n = S.shape
    rows = {}
    for r in range(m):
        canon = S[r] * S[r, int(np.argmax(S[r] != 0))]
        rows.setdefault(tuple(canon.tolist()), []).append(r)
    dup_classes = sum(1 for v in rows.values() if len(v) > 1)
    P = S.astype(np.int32).T @ S.astype(np.int32)
    np.fill_diagonal(P, 0)
    best_margin = int(np.abs(P).max())
    return dict(rows=int(m), dup_row_classes=int(dup_classes),
                best_pair_reuse=int((m + best_margin) // 2))


# ---------------------------------------------------------------- study drivers

def study_matrix(name, S, w_in, w_acc, greedy_cap, null_seeds=0, null_cap=None):
    m, n = S.shape
    res = dict(name=name, m=int(m), n=int(n), w_in=w_in, w_acc=w_acc,
               balance=float((S > 0).mean()))
    res["naive"] = naive_cost(S, w_in, w_acc)
    res["pairs"] = cse_pair_stats(S)
    res["cse"] = greedy_cse(S, w_in, w_acc, cap=greedy_cap)
    res["affine"] = affine_cost(S, w_in, w_acc)
    res["fr"] = {f"k{k}": fr_cost(S, k, w_in, w_acc) for k in (2, 4, 8)}
    if null_seeds:
        rng = np.random.default_rng(20260724)
        cse_saved, aff_adds = [], []
        for _ in range(null_seeds):
            R = rng.choice(np.array([-1, 1], dtype=np.int8), size=S.shape)
            cse_saved.append(greedy_cse(R, w_in, w_acc,
                                        cap=null_cap or greedy_cap)["saved"])
            aff_adds.append(affine_cost(R, w_in, w_acc)["adds"])
        res["null"] = dict(cse_saved_mean=float(np.mean(cse_saved)),
                           cse_saved_std=float(np.std(cse_saved)),
                           affine_adds_mean=float(np.mean(aff_adds)),
                           seeds=null_seeds)
    return res


# ---------------------------------------------------------------- main

def main():
    OUT_DIR.mkdir(exist_ok=True)
    types = parse_defines(STDNN_PRJ)
    layers = parse_layers(STDNN_PRJ)
    meas = measured_per_config(STDNN_XML, "LUT")
    meas_ff = measured_per_config(STDNN_XML, "FF")

    # ---- small (d32): per-layer counting against the measured census
    small = []
    shapes, per_token_cfgs = {}, set()
    per_token_lut = per_token_ff = 0.0
    mux_lut_total = 0
    for L in layers:
        cfg, lname, wsize = L["cfg"], L["layer"], L["wsize"]
        if wsize == 512:
            n_in, n_out = 16, 32
        elif wsize == 1024:
            n_in, n_out = 32, 32
        elif wsize == 2048:
            n_in, n_out = (32, 64) if lname.endswith("fc1") else (64, 32)
        elif wsize == 160:
            n_in, n_out = 32, 5
        else:
            continue
        w_in = types.get(L["input_t"], (8, 4))[0]
        w_acc = types.get(f"{lname}_accum_t", (max(w_in + 6, 16), 8))[0]
        S = load_sign_matrix(STDNN_PRJ, L["wfile"], n_in, n_out)
        shapes[lname] = (S, w_in, w_acc)
        r = study_matrix(lname, S, w_in, w_acc, greedy_cap=6000, null_seeds=20)
        r["cfg"] = cfg
        if cfg in meas:
            mu, cnt, sd = meas[cfg]
            r["measured_lut"] = dict(per_instance=mu, instances=cnt, std=sd)
            if cnt == 10:
                per_token_cfgs.add(cfg)
                per_token_lut += mu
                per_token_ff += meas_ff.get(cfg, (0, 0, 0))[0]
                mux_lut_total += MUX_LUT_PER_BIT * n_in * w_in
        small.append(r)

    # calibration fit: measured per-instance LUT vs naive bit-adds
    cal_x = np.array([r["naive"]["bits"] for r in small if "measured_lut" in r],
                     dtype=float)
    cal_y = np.array([r["measured_lut"]["per_instance"] for r in small
                      if "measured_lut" in r])
    A = np.vstack([cal_x, np.ones(len(cal_x))]).T
    (alpha, beta), *_ = np.linalg.lstsq(A, cal_y, rcond=None)
    pred = A @ [alpha, beta]
    r2 = 1 - float(np.sum((cal_y - pred) ** 2)) / float(np.sum((cal_y - cal_y.mean()) ** 2))
    max_resid = float(np.max(np.abs(cal_y - pred) / cal_y))
    calib = dict(alpha=float(alpha), beta=float(beta), r2=float(r2),
                 max_residual=max_resid, n_points=int(len(cal_x)),
                 gate_pass=bool(r2 >= 0.95 and max_resid <= 0.15))

    # fused QKV (blk0 Wq/Wk/Wv share the input vector)
    qkv = np.vstack([shapes[f"bit_block_0_attn_W{x}"][0] for x in "qkv"])
    fused_small = study_matrix("blk0_qkv_fused", qkv, 8, 13,
                               greedy_cap=6000, null_seeds=20)

    # ---- FF bracket (amended model, see module docstring)
    naive_bits_total = sum(
        r["naive"]["bits"] * (10 if r["cfg"] in per_token_cfgs else 1) for r in small)
    kappa_plain = PLAIN_DENSE_FF_MEAS / naive_bits_total
    da_bits_total = naive_bits_total * (DA_DENSE_LUT_MEAS / PLAIN_DENSE_LUT_MEAS)
    kappa_da = DA_DENSE_FF_MEAS / da_bits_total
    ff_model = dict(
        note=("kappa = FF per bit-add. plain-Latency discipline (Vitis stage cuts) "
              "vs the DA emission's II=1 deep-pipelined graph. FR/affine/fold FF "
              "projections use kappa_plain (same emission style as Latency); "
              "kappa_da is the cautionary bound. The attribution of the measured "
              "flip-flop growth was corrected to Vitis pipelining the emitted graph."),
        naive_bits_total=int(naive_bits_total),
        kappa_plain=float(kappa_plain), kappa_da=float(kappa_da),
        measured_da_over_plain_ff=DA_DENSE_FF_MEAS / PLAIN_DENSE_FF_MEAS,
        measured_da_over_plain_lut=DA_DENSE_LUT_MEAS / PLAIN_DENSE_LUT_MEAS,
        bracket_coherent=bool(kappa_plain < kappa_da))

    # ---- whole-model token-folding projection from the measured census
    own, count = census(STDNN_XML, "LUT")
    cats = defaultdict(int)
    for mod, v in own.items():
        nl = mod.lower()
        mcfg = re.search(r"config(\d+)", mod)
        cfgno = int(mcfg.group(1)) if mcfg else None
        if "einsum_dense" in nl:
            cats["einsum_dense_wrap"] += v
        elif nl.startswith("einsum_"):
            cats["einsum_actxact"] += v
        elif "softmax" in nl:
            cats["softmax"] += v
        elif "dense_latency" in nl and cfgno in per_token_cfgs:
            cats["dense_per_token"] += v
        elif "dense" in nl:
            cats["dense_head"] += v
        elif "pooling" in nl:
            cats["pooling"] += v
        elif nl.startswith("myproject"):
            cats["top_glue"] += v
        else:
            cats["glue_other"] += v
    unfolded = (cats["einsum_actxact"] + cats["softmax"] + cats["pooling"]
                + cats["top_glue"] + cats["einsum_dense_wrap"] + cats["dense_head"])
    dense_folded = per_token_lut + mux_lut_total + 500        # FSM/control margin
    glue_folded = cats["glue_other"] / FOLD + MUX_LUT_PER_BIT * 2 * 32 * 16
    fold_total = dense_folded + glue_folded + unfolded
    ff_buffers = 3 * FOLD * 32 * 13 + FOLD * 32 * 16          # QKV + residual bufs
    fold = dict(census_categories={k: int(v) for k, v in cats.items()},
                per_token_slice_lut=int(per_token_lut),
                mux_overhead_lut=int(mux_lut_total),
                dense_folded=int(dense_folded), glue_folded=int(glue_folded),
                unfolded=int(unfolded), total=int(fold_total),
                ff_slice=int(per_token_ff), ff_buffers=int(ff_buffers),
                ii=FOLD, ii_budget=II_BUDGET,
                fits=bool(fold_total <= DEVICE_LUT))

    # fold x restructure composition: folded slice restructured with best-of
    # {affine, FR} per layer, in calibrated LUT space
    slice_bits_naive = sum(r["naive"]["bits"] for r in small
                           if r["cfg"] in per_token_cfgs)
    slice_bits_best = sum(
        min(r["affine"]["bits"], min(f["bits"] for f in r["fr"].values()))
        for r in small if r["cfg"] in per_token_cfgs)
    slice_lut_best = alpha * slice_bits_best + beta * len(per_token_cfgs)
    fold_fr_total = fold_total - per_token_lut + slice_lut_best
    compose = dict(slice_bits_naive=int(slice_bits_naive),
                   slice_bits_best=int(slice_bits_best),
                   slice_lut_best=int(slice_lut_best),
                   fold_restruct_total=int(fold_fr_total),
                   fits=bool(fold_fr_total <= DEVICE_LUT))

    out = dict(calibration=calib, ff_model=ff_model, small=small,
               fused_qkv_small=fused_small, fold=fold, fold_restruct=compose,
               constants=dict(device_lut=DEVICE_LUT, clock_ns=CLOCK_NS,
                              ii_budget=II_BUDGET, fold=FOLD,
                              alpha_lut_per_bitadd=float(alpha)))
    (OUT_DIR / "counting_results.json").write_text(json.dumps(out, indent=1) + "\n")

    # ---- console summary
    g = "PASS" if calib["gate_pass"] else "FAIL"
    print(f"calibration: alpha={alpha:.3f} LUT/bit-add beta={beta:.0f} "
          f"R2={r2:.4f} max_resid={max_resid:.3f} gate={g}")
    print(f"FF bracket: kappa_plain={kappa_plain:.3f} kappa_da={kappa_da:.3f} "
          f"(measured DA/plain FF x{ff_model['measured_da_over_plain_ff']:.2f})")
    hdr = (f"{'layer':<26} {'mxn':>9} {'naive':>8} {'cse-sv':>7} {'aff':>8} "
           f"{'frbest':>8} {'fr-lut-x':>8} {'meas':>9}")
    print("\n" + hdr)
    for r in small + [fused_small]:
        frb = min(r["fr"].values(), key=lambda d: d["bits"])
        ratio = r["naive"]["bits"] / min(frb["bits"], r["affine"]["bits"])
        meas_s = (f"{r['measured_lut']['per_instance']:9.0f}"
                  if "measured_lut" in r else "        -")
        cse = r["cse"]
        print(f"{r['name']:<26} {r['m']:>4}x{r['n']:<4} {r['naive']['adds']:>8} "
              f"{cse['saved']:>6}{'*' if cse['capped'] else ' '} "
              f"{r['affine']['adds']:>8} {frb['adds']:>8} {ratio:>7.2f}x {meas_s}")
    print(f"\nfold: total={fold['total']:,} LUT (device {DEVICE_LUT:,}) "
          f"fits={fold['fits']} II={fold['ii']}<={II_BUDGET}")
    print(f"fold+restruct: total={compose['fold_restruct_total']:,} "
          f"fits={compose['fits']}")


if __name__ == "__main__":
    sys.exit(main())
