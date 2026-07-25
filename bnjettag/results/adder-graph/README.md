# adder-graph/

Why the binary dense layers cost what they cost, and what can be done about it. Two things
live here: a counting study on the exact deployed `{−1, +1}` matrices, and Experiment 1, the
synthesis measurement that checks the counting against Vitis HLS on one real layer.

Section 7 of the top-level [README](../../../README.md) is the write-up. This directory is the
evidence.

| Path | What it is |
| --- | --- |
| `counting_results.json` | Full machine-readable output of the counting study: per-layer costs for the naive adder tree, greedy two-term common-subexpression extraction, the affine identity and subset-sum (Four-Russians) sharing at k ∈ {2, 4, 8}, the random-sign null controls, the calibration fit against the measured census, and the token-folding projection. |
| `inputs/firmware/` | The study's inputs: the exact `{−1, +1}` weight files of the synthesized norm-free model, its generated top-level source, and the `defines.h` that fixes every layer's input and accumulator width. |
| `e1/<arm>/` | Experiment 1, one arm per directory: the hand-emitted layer (`myproject.cpp`), its testbench, the build script, and the raw C-synthesis report under `prj_<arm>/sol1/syn/report/` (gzipped). |
| `e1/run_all.sh` | The driver used on the synthesis machine, one arm at a time. |

## The counting study

```bash
pip install numpy
python ../../code/analysis/adder_graph_study.py    # rewrites counting_results.json
```

It reads the weight files and widths under `inputs/`, walks the instance tree of
`../../r7/results/csynth/whole_model_rf8_stdnn.xml.gz` for the measured per-layer LUT cost,
and prints a per-layer table. Every LUT figure is C-synthesis-estimate space. The calibration
gate it reports is the honest one: fitting measured LUT against naive-tree bit-adds gives
0.923 LUT per bit-add with R² = 0.987, but the largest per-layer residual is 24 %, above the
15 % threshold set before the counts were run, so the mechanism numbers are a ranking and a
first-order sizing rather than a prediction. The outlier is the 5×32 output layer, where the
fitted intercept dominates; excluding it the largest residual is 11 %.

The same study was also run on the matrices of a 6.4 M-parameter configuration, which is
outside the scope of this repository and whose rows are therefore not included here.

## Experiment 1

Five emissions of one layer, `bit_block_0_ffn_fc1` (32 → 64, ten per-token instances,
234,930 LUT inside the whole model). Every arm was C-simulated against the reference
implementation and required to be bit-exact before its resource numbers were accepted.

| Arm | Emission |
| --- | --- |
| `p` | Ten instances, naive adder tree. The control: it must reproduce the whole-model census. |
| `a` | One instance time-multiplexed over the ten constituents, II = 10. |
| `a2` | Folded by two, a monotonicity control. |
| `b` | Ten instances, subset-sum restructured with k = 4. |
| `c` | Folded by ten and subset-sum restructured. |

Arms `a` and `c` carry a second report, `csynth_v2_rewind.xml.gz`, from a rerun with
`#pragma HLS PIPELINE II=1 rewind` on the fold loop. Nothing else changed.

Re-parse any of them:

```bash
gunzip -c e1/a/prj_a/sol1/syn/report/csynth_v2_rewind.xml.gz > /tmp/a.xml
python ../../code/hls/parse_census.py /tmp/a.xml LUT
```

The reports were produced by Vitis HLS 2023.2 targeting `xcvu13p-flga2577-2-e` at a 2.5 ns
target clock. Reproducing them needs a Vitis installation; `e1/run_all.sh` is the script that
was used, and each arm's `build.tcl` is self-contained.
