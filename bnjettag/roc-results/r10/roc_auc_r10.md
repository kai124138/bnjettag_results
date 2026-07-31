# Pairwise attention bias — held-out results (n = 260,000, macro-OvR)

Model: the 19,201-parameter norm-free binary recipe plus a shared pre-softmax attention
bias built from three Lorentz-invariant pair features (ΔR², k_T², m²_ij for the 45
constituent pairs, through a 3→8→n_heads binary MLP; +84 parameters). Evaluated with
standardized inputs on the same held-out split as the baseline (identical class column
sums verified). Arrays: `R10-W1A8-pair-s{1..6}.npz` here.

## Per-seed macro-OvR AUC

| seed | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| AUC | 0.8891 | 0.8943 | 0.8919 | 0.8967 | 0.8822 | 0.8902 |

**6-seed: 0.8907 ± 0.0050 (sample std)** against the norm-free baseline (3 seeds)
0.8902 ± 0.0067 — a difference of +0.06 points: flat. An interim 3-seed reading
(seeds 1–3: 0.8918 ± 0.0026) did not hold at six seeds.

## Background rejection (macro-mean OvR 1/FPR at first TPR ≥ ε)

Spreads below are sample std (ddof = 1); the rejection tables in the README use
population std (ddof = 0), so the same means carry slightly smaller spreads there.

| ε_S | pairwise bias (6 seeds) | baseline (3 seeds) | Δ |
|---|---|---|---|
| 0.3 | 95.4 ± 8.5 | 91.2 ± 7.6 | +4.5 % |
| 0.5 | 32.5 ± 2.2 | 31.0 ± 2.5 | +4.8 % |
| 0.7 | 11.9 ± 0.7 | 11.5 ± 0.7 | +3.6 % |

Per-seed @0.5: 32.2 / 34.0 / 32.3 / 35.6 / 29.0 / 31.9. Per-class @0.5 (g, q, W, Z, t):
biased (23.7, 23.3, 19.7, 24.1, 71.7) against baseline (23.9, 23.2, 18.2, 22.1, 67.7) →
(−1.0, +0.4, +8.4, +9.3, +6.0) %.

**Significance.** Welch t on the macro rejection at ε_S = 0.5, n = 6 vs 3: t = 0.89,
p = 0.43 (Mann-Whitney p = 0.38); the biased arm's mean sits inside the baseline seed
range. The gain is uniform in sign at every working point and remains W/Z-concentrated,
but is not statistically established at this seed count.
