# Round-7 whole-model operating-point study (synthesis phase CLOSED 2026-07-16)

All rows: one continuous Vitis HLS 2023.2 project, VU13P, 2.5 ns target, io_parallel /
Latency strategy, r7-small W1A8 seed 2. Re-parsed from raw csynth.xml (files in `csynth/`);
censuses tree-exact (`code/hls/parse_census.py`). "DSP kernel" = stock SubLN (variance
squares on DSP48); "fabric kernel" = BIND_OP SubLN (2026-07-15, unit gate 7/7).

| emission | SubLN kernel | DSP | LUT | FF | latency | II | est clk |
|---|---|---|---|---|---|---|---|
| RF=1 (canonical) | DSP | 33,696 | 6.04M | 6.21M | 466 cyc | 1 | 2.157 ns |
| RF=8 (canonical, **the L1 point**) | DSP | 6,900 | 5.21M | 3.09M | 615 cyc | 8 | 2.157 ns |
| RF=1 noSubLNdsp | fabric | **14,112** | 12.89M | 5.07M | 380 cyc | 1 | 3.103 ns |
| RF=8 noSubLNdsp | fabric | **1,764** | 12.04M | 2.92M | 572 cyc | 8 | 3.103 ns |
| RF=10 global | fabric | 1,424 | 11.94M | 2.79M | 596 cyc | 10 | 3.103 ns |
| RF=10 foldmax (norms RF=1) | fabric | 1,424 | 12.09M | 3.00M | 436 cyc | 10 | 3.103 ns |

## What the study established (each row is a measured lever)

1. **The DSP thesis end-state is real.** With the fabric kernel, the census is exactly
   attention einsums + softmax + pooling (RF=1: 12,800 + 800 + 512; RF=8: 1,600 + 100 + 64)
   and SubLN = 0. Binary weight layers are 0 DSP in every row.
2. **The fabric kernel is LUT-fatal.** Norm LUT 2.1M → ~8.9M (the variance squares are
   accumulator-width ~36-bit multiplies, ~1.2k LUT each on fabric) and est clock degrades
   2.157 → 3.103 ns. The DSP-kernel RF=8 point remains the deployable-track emission.
3. **Global folding is exhausted.** RF 8→10 (fabric kernel): LUT 12.04M → 11.94M; binary
   dense LUT LITERALLY identical (2,077,082 → 2,077,382) — inlined ±1 weights do not fold.
   And II=10 × 3.103 ns = 31 ns > 25 ns: RF=10 fails L1 throughput at the achieved clock.
4. **Per-layer maps work but don't change the fit.** foldmax (norms pinned RF=1, MACs+
   einsums RF=10) buys 160 cycles of latency for +150k LUT at identical DSP — the RF map
   plumbing (`--layer-configs`) is proven for future use.
5. **io_stream is closed** (probe 2026-07-15): the HGQ frontend rejects heterogeneously
   quantized activations outside io_parallel (`hgq_proxy_model.py:63`), before codegen.

## Fit verdict

No reuse/kernel configuration reaches the device (1.73M LUT); the gap is structural, owned
by the norms (40% LUT at the L1 point) and the inlined binary dense (40%). The measured
paths forward, in rising effort:
- **Input-norm-only training variant** — full ablation costs 9.3 ROC pts
  (`roc-results/extension/`), but the input norm alone may carry most of the value
  (raw-feature variance ~1e6 is what SubLN was built for). Untrained.
- **Narrowed-variance SubLN kernel** — quantize d[i] to ~8 bits before squaring; small
  numerics change (needs the test_subln gate), collapses both the DSP and fabric cost of
  the squares.
- **Resource-strategy dense emission** — move the ±1 ROMs to BRAM (the large-model
  re-expression, per-shape proven 2026-07-09); attacks the 2.08M dense LUT directly.
