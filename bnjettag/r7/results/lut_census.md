# Round-7 whole-model LUT census (module-verified, 2026-07-15)

Per-module LUT attribution for the two canonical whole-model reports, computed with the
same instance-tree method as the DSP census (own = module − Σ children over
`RTLDesignHierarchy`; `code/hls/parse_census.py`, promoted from the 07-15 scratch script).
The tree sum reproduces the profile total EXACTLY on both files (self-check).

## RF=8 narrow (the L1 point) — total 5,209,569 LUT

| consumer | LUT | share |
|---|---|---|
| SubLN norms (11 instances) | 2,101,371 | 40.3% |
| binary dense (all 15 weight layers) | 2,077,082 | 39.9% |
| attention act×act einsums | 664,276 | 12.8% |
| glue / other | 189,567 | 3.6% |
| top-level glue | 117,064 | 2.2% |
| softmax | 42,140 | 0.8% |
| pooling | 16,596 | 0.3% |
| einsum_dense wrappers | 1,473 | 0.0% |

## RF=1 narrow — total 6,044,148 LUT

| consumer | LUT | share |
|---|---|---|
| binary dense | 2,236,168 | 37.0% |
| SubLN norms | 2,165,182 | 35.8% |
| attention act×act einsums | 908,160 | 15.0% |
| softmax | 337,120 | 5.6% |
| glue / other | 189,567 | 3.1% |
| top-level glue | 112,271 | 1.9% |
| einsum_dense wrappers | 67,200 | 1.1% |
| pooling | 28,480 | 0.5% |

## Readings

- **The norm is the top consumer of BOTH scarce resources.** SubLN owns 19,584/33,696
  DSP (58%) and 2.10M/5.21M LUT (40%) at the L1 fold. The norm-ablation training round
  (`r7-small-w1a8-nonorm`, launched 2026-07-15) is therefore the device-fit experiment,
  not just the DSP experiment: removing the norms takes the RF=8 model from 5.21M LUT
  toward ~3.1M before any reuse tuning.
- **Binary dense barely folds.** RF=1→8 moved dense LUT only 2.24M → 2.08M (−7%):
  at Latency strategy with inlined ±1 weights, "multipliers" are already logic, so reuse
  converts them into muxes of the same order. Deeper global RF is not the fit lever for
  the binary core; strategy/emission changes would be (open).
- **io_stream is closed for this stack** (probe 2026-07-15,
  `results/convert/w1a8-s2/lutfit-iostream-probe/probe_verdict.json`): hls4ml 1.3.0
  raises `NotImplementedError: Heterogenous quantization for activations is only
  supported with IOType=io_parallel` (`backends/fpga/passes/hgq_proxy_model.py:63`)
  before any codegen. Streaming would need a frontend change, not a config.
- Norm LUT per instance is dominated by the wide-input instances (`ln_input_proj`,
  ~390k LUT summed over 10 token instances) — raw detector features force wide
  fixed-point into the variance path, another argument for input standardization
  offline rather than in-fabric normalization.

Sources: `csynth/whole_model_rf8_narrow.xml`, `csynth/whole_model_rf1.xml`
(re-parse with `python code/hls/parse_census.py <xml>`).
