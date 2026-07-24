# bnjettag/ — code, checkpoints, arrays, synthesis reports

All paths below are relative to this folder. The scientific record is
[`../README.md`](../README.md); this file only says where things live.

| Path | What it is |
| --- | --- |
| `code/hgq2/` | The current pipeline: quantization-aware training, hls4ml conversion, the custom SubLN layer, verification gates, synthesis drivers. `run_stage.py` is the CLI over `configs/*.json`. Dated change ledger in `code/hgq2/LEDGER.md`. |
| `code/training/` | The earlier QKeras trainer (`qkerasModel.py`, all variants through `BN_*` environment knobs), the ROC scripts, the EBOPs calculator. |
| `code/hls/` | Synthesis drivers, the Vitis runbook (`RUN_CSYNTH_ON_VITIS.md`), and `parse_census.py`, the instance-tree report parser behind every resource census. |
| `code/jobs/` | Kubernetes job specifications for every training and ROC run, plus the log watchers. |
| `code/plots/` | Figure scripts. |
| `r7/` | The deployable-scale campaign: `roc-results/` (held-out score arrays with their verified AUC tables, including `r7b/`, `r8/` and `extension/`), `results/` (whole-model synthesis reports, the operating-point study, the LUT census, the activation ladder), and four trained checkpoints under `models/` (see `r7/models/README.md`). |
| `roc-results/r10/` | Score arrays for the pairwise-invariant attention bias. |
| `results/hgq2/constraints_map.md` | What converts cleanly through HGQ2 and hls4ml, what needs the custom-layer recipe, and where DSPs hide. Every row established by execution. |

## Conventions

- No AUC is quoted until it has been recomputed from the stored `.npz` arrays; no resource or
  latency number until it has been parsed from the raw C-synthesis XML.
- Variants are selected by configuration, not by forking code: `BN_VARIANT`
  (bitnet / vanilla / w8a8), `BN_ACT_BITS` (8 / 6 / 4), `BN_N_PART`, `BN_TERNARY`,
  `BN_SOFTMAX_FREE`.
- Training runs on the NRP Nautilus cluster; C-synthesis runs on a dedicated Vitis machine.
  Neither is run locally.
