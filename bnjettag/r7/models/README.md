# Checkpoints

Four trained checkpoints from the deployable-scale campaign, seed 2 in each case. These are
the exact artifacts the synthesis reports in `../results/csynth/` were produced from, so a
conversion can be rerun end to end without retraining.

| Directory | Configuration | Held-out macro AUC (this seed) |
| --- | --- | --- |
| `small-stdnn-s2/` | binary weights, 8-bit activations, standardized inputs, no normalization layers — the flagship | 0.8964 |
| `small-w1a8-s2/` | binary weights, 8-bit activations, raw inputs, SubLN | 0.7582 |
| `small-w1a6-s2/` | binary weights, 6-bit activations, raw inputs, SubLN | 0.7459 |
| `small-w1a4-s2/` | binary weights, 4-bit activations, raw inputs, SubLN | 0.6900 |

Each directory holds `model_best.keras`, the best-epoch weights restored at the end of
training. `small-stdnn-s2/` additionally carries `input_std.json`, the per-feature mean and
standard deviation computed on the training split — the model expects inputs standardized with
exactly these constants, so they travel with the checkpoint rather than living in the training
code. `small-w1a8-s2/` carries `train_meta.json` with its run provenance.

The remaining seeds, the full training histories, and the smaller configuration's checkpoints
are in Weights & Biases under the run names recorded in the job specifications
(`../../code/jobs/training/variants/`).

Loading requires the same stack the models were trained in (HGQ2 0.1.9 on Keras 3); the custom
objects are registered by importing `bnhgq2` from `../../code/hgq2/`.
