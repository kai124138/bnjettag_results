# Round-11 capacity ladder — the matched W1-vs-W8 comparison (era-2, n = 260,000, ROC-test)

**What this settles.** Era-2 had a 332× hole on the capacity axis (19,201 → 6,375,173 params) and
the standardized regime the flagship lives in had exactly one point. Round 11 fills 19.2k → 285k
at d_model ∈ {48, 64, 96, 128} (H=4, L=2, FFN=2·d — only d_model moves), each scale at its own
probed peak LR, 3 seeds per arm, 54 jobs, all succeeded.

Critically, it fills it **matched**. Every 8-bit and FP32 arm in this project carried SubLN while
the binary flagship is norm-free, so every previously published W1-vs-W8 comparison crossed the
norm axis. The `w8a8-stdnn` arm below removes that: same weights-quantization, norms removed,
nothing else changed.

Configs `code/hgq2/configs/gen_r11_ladder.py`; jobs `code/jobs/training/variants/gen_round11_jobs.py`;
evaluation `code/hgq2/eval_r11.py`; rejection `code/hgq2/rejection.py`. Arrays here.

## The matched comparison (binary vs norm-free 8-bit)

| scale | params | W1A8-stdnn AUC | W8A8-stdnn AUC | ΔAUC | rejection @ε0.5 W1 | W8 | **binary retention** |
|---|---|---|---|---|---|---|---|
| d32 | 19,201 | 0.8902 ± 0.0067 | 0.9231 ± 0.0009 | **−3.29** | 31.0 | 60.2 | **51.5%** |
| d48 | 41,761 | 0.9091 | 0.9272 | −1.81 | 43.5 | 68.9 | 63.1% |
| d64 | 73,025 | 0.9195 | 0.9354 | −1.60 | 55.6 | 88.6 | 62.8% |
| d96 | 161,665 | 0.9281 ± 0.0013 | 0.9374 ± 0.0007 | −0.93 | 70.0 | 95.2 | 73.6% |
| d128 | 285,121 | 0.9335 ± 0.0006 | 0.9380 ± 0.0008 | **−0.45** | 84.1 | 96.8 | **86.9%** |

## Three findings

**1. Binary is dominated at every scale measured — it converges but never overtakes.** The
penalty falls monotonically from −3.29 AUC pts / 48% rejection loss at the deployable point to
−0.45 pts / 13% at 285k, but the sign never changes. An earlier partial reading of this ladder
(binary above FP32-std at d128) was an artifact of comparing against a *normed* baseline; against
the matched arm the ordering is unambiguous at all five scales.

**2. The binary penalty is a capacity effect, quantified — but the slope does not persist.**
Retention rises **30.3 points per decade** across the ladder (51.5% @19.2k → 86.9% @285k, 1.172
decades). A naive extrapolation puts parity near **10⁶ params [EXTRAPOLATION**, endpoint fit
7.7e5, OLS 1.13e6**]**.

**The FINAL 6.4M campaign does not corroborate that — it contradicts the slope.** Its 94.7%
figure is W1A8 against **FP32**, not against matched 8-bit; the matched analogue there is
**W1A8/W8A8 = 89.7%**. That is only **+2.7 points over the 1.35 decades** beyond d128 — about
**2 points per decade**, against the ladder's 30.3, where the ladder slope would have predicted
~128%. FINAL is also a different regime (d256/H8/L8, raw inputs, and all three arms normed), so
it is not a clean continuation. Read together, the honest statement is that the ladder measures a
steep local slope that **flattens well before 6.4M**, and parity is not demonstrated at any
measured scale.

**3. Removing SubLN is worth a great deal to every arm — and the normed FP32 baseline this
project has quoted throughout is not the ceiling.** At d128: W8A8-stdnn 96.8 rejection vs
W8A8-std 77.0 vs FP32-std 75.1, i.e. norm removal is worth **+25.7%** to the 8-bit arm.
W8A8-std ≈ FP32-std at every scale, so 8-bit weights remain free.

> **CORRECTION 2026-07-26 — an earlier version of this file claimed norm-free 8-bit "beats full
> precision". It does not, and the claim was the project's recurring cross-axis error in a third
> guise:** it set norm-free 8-bit against *normed* FP32. The matched arm now exists
> (`r11-d32-fp32-stdnn`, 3 seeds, ROC-test):
>
> | arm (d32, all norm-free unless noted) | AUC | rejection @ε0.5 |
> |---|---|---|
> | **FP32-stdnn** | **0.9249 ± 0.0004** | **63.5 ± 1.0** |
> | W8A8-stdnn | 0.9231 ± 0.0009 | 60.2 ± 1.2 (94.9%) |
> | FP32-std (normed) | 0.9161 ± 0.0009 | 47.8 ± 0.9 (75.3%) |
> | W1A8-stdnn | 0.8902 ± 0.0067 | 31.0 ± 2.0 (48.8%) |
>
> Full precision is ahead of 8-bit by +0.18 AUC pts and 5.1% of rejection. The durable finding is
> the ordinary one this project has always had — **8-bit is nearly free** (94.9% of matched FP32)
> — not that it is superior. Binary's retention against a *matched* FP32 baseline is **48.8%**,
> worse than the 64.9% quoted against the normed one.

## Why this does not rescue the binary design point on hardware

Folding is **precision-agnostic** — measured, not assumed: 9.50× for 8-bit against 9.31× for ±1
on the same layer, from the same five-emission experiment run on an 8-bit layer (those reports are
not part of this repository). The 8-bit model therefore receives the
same schedule win. Binary cannot buy parity by spending its area advantage on capacity, because
reaching parity needs ~3.5× more parameters than d128 while its whole-model compiled area
advantage over 8-bit is ~1.18× — it would spend more area than it saved.

The one asymmetry that remains real is **FR**: subset-sum restructuring is legal only because the
weights are signs, has no 8-bit counterpart, and is measured at 2.66× on a real layer. That is a
*compiler* asymmetry, not a quantization one.

## Caveats

- 3 seeds per point; d48/d64 spreads not quoted above (all ±0.001–0.002 on AUC).
- Best-epoch pattern is **not** uniform, and the "baselines are under-trained" argument holds
  only at d48/d64. At d128 the *binary* arm peaks latest (w1a8-stdnn 77/95/100 vs fp32-std
  56/56/75, w8a8-std 61/77/100), so there the argument reverses: it is the binary arm that had
  least headroom left in the 101-epoch budget.
- Everything here is accuracy. No ladder scale has been synthesized; the resource side is the
  layer-scale measurements plus the whole-model projections of Section 7.3 of the top-level
  README.
- d64's retention (62.8%) sits marginally below d48's (63.1%) — inside seed noise, not a
  non-monotonicity worth interpreting.
