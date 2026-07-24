# The deployable-scale campaign

The tagger sized for the hardware it claims to run on, and everything measured on it:
configurations, trained checkpoints, held-out score arrays, and raw synthesis reports. The
analysis of these results is in the repository's [`README.md`](../../README.md); this file
describes the store.

## The two configurations

| | primary | smaller |
| --- | --- | --- |
| dimensions | d = 32, 4 heads, 2 blocks, FFN 64 | d = 16, 2 heads, 2 blocks, FFN 32 |
| architectural parameters | 18,405 | 4,853 |
| total parameters (with activation-grid scales and positional encoding) | 19,201 | 5,345 |
| operations per jet | ≈ 190 k, of which 6.4 k are activation×activation | ≈ 52 k, of which 3.2 k |
| per-block anatomy | four projections at ≈ 1 k, feed-forward pair at ≈ 4.2 k | four at ≈ 264, pair at ≈ 1.1 k |

Both sit in the parameter class of published FPGA taggers on this dataset, which run between
about 3 k and 34 k parameters. The largest single layer of the primary configuration (32 → 64,
≈ 2 k multiplies) is well under the size at which the Vitis front end stops converging, which
is why the whole model synthesizes as one project with RTL built in minutes.

## Contents

| Path | What it is |
| --- | --- |
| `roc-results/` | Held-out score arrays for the raw-feature matrix — two sizes × five configurations × three seeds — with the verified AUC table in `roc_auc.md`. |
| `roc-results/r7b/` | The smaller column retrained at its own tuned learning rate, with its own verified table. |
| `roc-results/r8/` | The offline-standardization campaign at the primary scale: twelve runs, `roc_auc_r8.md`. |
| `roc-results/extension/` | Loose ends: seeds 4–6 of the 4-bit cell, the learning-rate probe, and the normalization ablation. |
| `results/csynth/` | Sixteen raw Vitis HLS C-synthesis reports, gzipped. Every hardware number in the repository traces to one of these files. |
| `results/operating_points.md` | The closed operating-point study — six whole-model syntheses mapping reuse factor and normalization kernel against resources, latency and throughput. |
| `results/activation_ladder.md` | Whole-model silicon against activation width, with the per-consumer DSP census at each width. |
| `results/lut_census.md` | Module-level LUT attribution, closed exactly against the report totals. |
| `results/convert/` | Conversion and C-simulation evidence per emission: precision maps, fidelity gates, layer inventories. |
| `models/` | Four trained checkpoints, including the norm-free flagship. |

Run configurations are in `../code/hgq2/configs/`; job specifications are in
`../code/jobs/training/variants/`.

## Reading the arrays

Each `.npz` holds `y` (one-hot labels, 260,000 × 5), `score` (softmax scores, same shape) and
`meta` (a provenance string). The claim table beside each store states what those arrays are
asserted to give, so a recomputation can be checked against a written claim:

```bash
python3 -c "
import numpy as np; from sklearn.metrics import roc_auc_score
d = np.load('roc-results/r8/SMALL-W1A8-r8stdnn-s1.npz')
print(d['meta']); print(roc_auc_score(d['y'], d['score'], multi_class='ovr', average='macro'))"
```

Reports are parsed with `../code/hls/parse_census.py`, which walks the instance tree and
subtracts each module's children from its own totals so that nothing is double-counted; it
asserts that the per-consumer attribution sums to the report's own profile total.
