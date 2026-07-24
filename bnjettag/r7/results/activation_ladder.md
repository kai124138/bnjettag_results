# Whole-model silicon vs activation width + the round-8 stdnn rows (2026-07-19)

All rows: one continuous Vitis HLS 2023.2 project, VU13P, 2.5 ns target, io_parallel /
Latency, seed-2 checkpoints. Censuses tree-exact (own = module − Σ children; sums match
profile totals on all 8 files). Raw XML: `csynth/whole_model_*.xml`.

| model (checkpoint) | norm kernel | RF | DSP | LUT | latency | II | est clk |
|---|---|---|---|---|---|---|---|
| W1A8 r7 (0.7500 ROC) | DSP-norm (canonical) | 1 | 33,696 | 6.04M | 466 cyc ≈ 1.00 µs | 1 | 2.157 ns |
| W1A8 r7 | DSP-norm (canonical) | 8 | 6,900 | 5.21M | 615 cyc ≈ 1.33 µs | 8 | 2.157 ns |
| **W1A8-stdnn r8 (0.8902 ROC)** | **none (no norms exist)** | 1 | **14,112** | **4.80M** | **167 cyc ≈ 0.36 µs** | 1 | 2.157 ns |
| **W1A8-stdnn r8** | **none** | 8 | **1,764** | **3.91M** | **218 cyc ≈ 0.47 µs** | 8 | 2.157 ns |
| W1A6 r7 (0.7362 ROC) | fabric-norm† | 1 | 13,984 | 12.24M | 376 cyc | 1 | 3.103 ns |
| W1A6 r7 | fabric-norm† | 8 | 1,748 | 11.47M | 561 cyc | 8 | 3.103 ns |
| W1A4 r7 (0.6561 ROC) | fabric-norm† | 1 | 7,584 | 12.07M | 371 cyc | 1 | 3.103 ns |
| W1A4 r7 | fabric-norm† | 8 | **948** | 11.04M | 557 cyc | 8 | 3.103 ns |

† The A6/A4 emissions (2026-07-18) inherited the BIND_OP SubLN kernel, so their norm LUT
is inflated (~8.9M of each row) and the clock degraded — their LUT column is NOT
comparable to the canonical W1A8 rows. Their DSP column is clean (SubLN = 0 by the
fabric kernel).

## DSP census by activation width (RF=1 / RF=8) — all weightless, at every width

| consumer | A8 | A6 | A4 |
|---|---|---|---|
| binary weight multiplies | **0 / 0** | **0 / 0** | **0 / 0** |
| attention act×act einsums | 12,800 / 1,600 | 12,800 / 1,600 | **6,400 / 800** |
| softmax tables | 800 / 100 | 800 / 100 | 800 / 100 |
| global pooling | 512 / 64 | 384 / 48 | 384 / 48 |

## Readings (all census-verified)

- **The thesis holds at every activation width**: zero DSPs on all fifteen binary weight
  layers in every one of the 8 syntheses. The DSP census is 100% weightless ops always.
- **The one real resource-precision effect: attention DSPs halve at A4** (4-bit operands
  pack two MACs per DSP48). A8→A6 is DSP-flat. At A4/RF=8 the complete transformer costs
  **948 DSPs (7.7% of the device)**.
- **The round-8 stdnn rows are the deployment-quality point**: best binary accuracy
  (0.8902 ± 0.0067 ROC), attention-floor DSPs at the good clock with no kernel trade,
  **sub-microsecond at both operating points** (0.36 / 0.47 µs — the norms were two
  thirds of the pipeline depth), II=8 → 17.3 ns/jet inside the L1 window. Remaining fit
  gap: 3.91M vs 1.728M LUT (2.3×), owned by the inlined binary dense + attention fabric.
- Export-fidelity caveats attached to this table: stdnn GATE1 0.989 (CSD-2 product-γ
  ceiling, characterized 2026-07-18), A6 0.953 / A4 0.883 (static-grid handoff at small
  scale). GATE2 (firmware vs export) passes everywhere; stdnn is bit-exact.
