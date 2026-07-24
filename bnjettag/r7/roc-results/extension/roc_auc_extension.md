# round-7 EXTENSION · era-2 data · ROC-test AUC (held-out, n = 260,000)

Recomputed locally from `extension/*.npz` (keys y/score) with
`roc_auc_score(y, score, multi_class="ovr", average="macro")` — verification gate
2026-07-15. Runs: the loose-end sets of experiment-log 2026-07-15 (checkpoints from the
durability-gated retrains where the first cohort's uploads were lost).

## SET 1 — W1A4-small seed extension (seeds 4–6; matrix seeds 1–3 = 0.6102/0.6900/0.6681)

| run | ROC-test AUC |
|---|---|
| r7-small-w1a4-s4 | 0.6575 |
| r7-small-w1a4-s5 | 0.6539 |
| r7-small-w1a4-s6 | 0.6322 |

Six-seed cell: mean **0.6520**, std **0.0264** (was 0.6561 ± 0.0412 on 3 seeds).
Reading: the instability is real but the 3-seed std overstated it; the widened sample
tightens the spread and pulls the mean down slightly. s1's 0.6102 stays the low extreme.

## SET 2 — tiny's LR probe (control: matrix tiny-W1A8 @ 2e-5 = 0.5789 ± 0.0066)

| run | peak LR | ROC-test AUC |
|---|---|---|
| r7lr-tiny-w1a8-lr1e5-s1 | 1e-5 | 0.5455 |
| r7lr-tiny-w1a8-lr5e5-s1 | 5e-5 | **0.7179** |

Reading: **tiny's binary collapse was recipe, not capacity.** The lower-is-better rule
from the small-scale probe inverted at tiny scale; +13.9 ROC pts from LR alone. Round 2/3
probes (5e-5 seeds 2–3 ROC pending; 1e-4 val 0.7309, seeds 2–3 + 2e-4 training) will fix
the final recipe before the curve's 5k point is corrected in the deliverables.

## SET 3 — norm ablation (small W1A8, arch.norm='none'; control 0.7500 ± 0.0083)

| run | ROC-test AUC |
|---|---|
| r7n-small-w1a8-nonorm-s1 | 0.6577 |
| r7n-small-w1a8-nonorm-s2 | 0.6665 |
| r7n-small-w1a8-nonorm-s3 | 0.6466 |

Triple: mean **0.6569**, std **0.0099**. Reading: removing every norm costs
**−9.3 ROC pts** vs the normed control — trainable but expensive. Combined with the
synthesis result (norms = 40% LUT / 58% DSP; kernel remap trades 5.1k DSP for +6.8M LUT),
the full ablation is not the fit answer at this accuracy budget; the next design is
partial (input-norm-only) or a narrowed variance operand in the kernel.

Source npz: W&B run `r7x-roc-artifacts` (project bnjettag-final); fetched 2026-07-15.
