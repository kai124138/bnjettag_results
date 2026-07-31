# A binary-weight transformer jet tagger for the CMS Level-1 trigger

Kai Yamaguchi · University of California, San Diego

The Level-1 trigger of a general-purpose LHC experiment accepts a new event every 25 ns and
must reach a keep-or-reject decision within a few microseconds, using FPGAs as the only
compute substrate fast enough for the task. Any network deployed there competes for two
scarce resources: the dedicated DSP multiplier blocks (12,288 on the Xilinx VU13P used
throughout this work) and the general lookup-table fabric (1.728 M LUTs). Multiplication is
the dominant cost of neural-network inference and DSP blocks are the first resource exhausted
when several algorithms share a device.

This work constrains **every** attention and feed-forward weight of a transformer jet tagger
to the binary set {−1, +1} in the manner of BitNet, pushes the trained network through hls4ml
into real Vitis HLS C-synthesis, and asks two questions with measurements rather than
estimates:

1. do the binary matrix multiplications map onto LUT logic instead of DSPs, and
2. what does the constraint cost in tagging efficiency, and how far can the *activations*
   then be quantized before efficiency, resources, or latency degrade?

Both are answered on a model sized for the hardware it claims to run on — 19,201 parameters,
the size class of published FPGA taggers — and on the whole artifact rather than on isolated
layer probes.

**Summary of what is measured here.** Across eight continuous whole-model syntheses at three
activation widths, all fifteen binary weight layers consume **zero DSP blocks**; every
multiplier that remains belongs to the weightless activation×activation products inside
attention, to the softmax tables, or to pooling. The deployable configuration meets the
Level-1 throughput condition (a new jet every 8 cycles ≈ 17.3 ns, inside the 25 ns window) at
an end-to-end latency of 0.47 µs, and runs to 0.36 µs when fully unrolled. Against a matched
baseline — the identical norm-free architecture at 8-bit weights, trained on the same recipe —
the binary constraint costs 3.3 AUC points (0.8902 ± 0.0067 against 0.9231 ± 0.0009), and in
the trigger's own operating metric the binary model retains 52 % of the matched baseline's
background rejection. Against the normed full-precision reference the gap is 2.6 points, but
that comparison crosses the normalization axis and understates the cost. The model does not
yet fit a single VU13P: it needs 3.91 M LUTs against 1.728 M available, and that gap is the
open problem. On the one layer where a remedy has been synthesized, time-multiplexing a single
constant-weight datapath over the ten constituents instead of instantiating it ten times costs
**9.3× fewer lookup tables**, bit-exactly and at zero DSPs, and 18× when composed with a
subset-sum restructuring of the same dot products. Carried through the measured census, that
projects a model which fits, where no earlier configuration has. The projection is not a
synthesis, and Section 7 states exactly what it assumes.

Everything below is recomputed from the per-jet score arrays and raw synthesis reports stored
in this repository. Section 9 gives the commands.

---

## Contents

1. [The model](#1-the-model)
2. [Data and metrics](#2-data-and-metrics)
3. [Training](#3-training)
4. [Tagging efficiency](#4-tagging-efficiency)
5. [Background rejection at fixed signal efficiency](#5-background-rejection-at-fixed-signal-efficiency)
6. [FPGA implementation](#6-fpga-implementation)
7. [Reducing the lookup-table cost](#7-reducing-the-lookup-table-cost)
8. [Open problems](#8-open-problems)
9. [Reproducing every number](#9-reproducing-every-number)
10. [Repository layout](#10-repository-layout)
11. [Context and references](#11-context-and-references)

---

## 1. The model

A transformer encoder classifier operating on jet constituents. The ten highest-pT
constituents of each jet enter as 16 features each; a linear projection lifts them to a
32-dimensional embedding; two transformer blocks with four attention heads and a
64-dimensional feed-forward network follow; a mean pool over constituents makes the
representation permutation-invariant; a two-layer head emits five class scores.

Every weight matrix in that description — the input projection, the Q, K, V and output
projections of both blocks, both feed-forward layers of both blocks, and both head layers,
fifteen layers in all — is constrained to {−1, +1}. Weights are binarized by an absmean
quantizer, sign(W − α)·β, and trained through the straight-through estimator against
full-precision shadow weights. Activations are quantized to a fixed width of 8, 6 or 4 bits.
The comparison baselines are the identical architecture at full precision (FP32) and at
conventional 8-bit weights and activations (W8A8).

![Architecture and DSP census](figures/fig01_architecture.png)

*Figure 1. Data flow of the tagger, colored by what each operation costs in silicon. Blue
operations carry {−1, +1} weights and use no DSP blocks in any synthesis performed here. Red
operations multiply two activations together and have no weights to binarize; they, the
softmax tables and the pooling accumulator are the entire DSP census of the model.*

Two configurations were trained. The primary one, referred to below by its parameter count,
has d = 32, 4 heads, 2 blocks and FFN width 64: 18,405 architectural parameters, or 19,201
including the trainable activation-grid scales and the positional encoding, and approximately
190 k operations per jet, of which 6.4 k are activation×activation products inside attention.
A smaller configuration (d = 16, 2 heads, 2 blocks, FFN 32; 4,853 architectural and 5,345
total parameters, ≈ 52 k operations per jet) measures the capacity axis.

The networks are trained natively in HGQ2, so each checkpoint *is* the hardware model: the
binary kernels are exactly {−β, +β} with no sign-zeros, and the activation grids are static
per-tensor fixed-point grids whose scales are learned during training at pinned total width.
No separate port or rebuild step stands between the trained network and the firmware.

## 2. Data and metrics

| | |
| --- | --- |
| Dataset | Public HLS4ML LHC Jet dataset, 150-constituent version (Zenodo record 3602260) |
| Task | Five-class jet classification: gluon, light quark, W, Z, top |
| Input | Ten highest-pT constituents × 16 features = 160 values per jet |
| Training monitor | Macro one-vs-rest AUC on a 20 % split of the training set |
| Test set | The dataset's own held-out split, n = 260,000 jets, never seen during training |

The dataset is public and standard, which makes the numbers below directly comparable to
published taggers evaluated on it. Classes are balanced and no reweighting is applied.

Two figures of merit are reported throughout, and never conflated:

- **Macro one-vs-rest AUC** on the held-out split. Quoted as the mean over three training
  seeds with the sample standard deviation of those three values.
- **Background rejection at fixed signal efficiency**, 1/ε_B, computed per class one-vs-rest
  as the reciprocal of the false-positive rate at the threshold where the true-positive rate
  first reaches ε_S, then averaged over the five classes. Spreads on rejection are population
  standard deviations over the three seeds, matching the convention of the verified tables in
  `bnjettag/r7/roc-results/`.

AUC and rejection are both reported because they disagree about how expensive binarization
is, and rejection is the quantity a trigger is actually budgeted against. The same caution
appears in the one published attempt to binarize part of a Particle-Transformer-family
network (arXiv:2508.07431), where AUC parity concealed a 15–17 % rejection loss.

Validation AUC and held-out AUC are different measurements; only held-out numbers appear in
any table below.

## 3. Training

Training runs on the NRP Nautilus Kubernetes cluster; the job specifications are in
`bnjettag/code/jobs/` and the run configurations in `bnjettag/code/hgq2/configs/`. The recipe
is Adam (β₂ = 0.98, weight decay 0.01) with one warm-up epoch and a linear decay over 100,
gradient clipping at 1.0, batch size 256, and early stopping on the validation AUC monitor
with patience 15. Three seeds per configuration.

The model is defined in code rather than in a standalone file. `bnjettag/code/hgq2/bnhgq2/qat.py`
builds the trainable network from a configuration JSON — the architecture of section 1 with the
binary weight quantizers attached — and `bnhgq2/train.py` is the training loop;
`run_stage.py train --config configs/r8-small-w1a8-stdnn.json` is the entry point that produced
the headline model. For hardware, the same architecture is rebuilt as a static graph in
`bnhgq2/build.py`, and the hls4ml conversion call itself (`convert_from_keras_model`, Vitis
backend) is in `bnhgq2/convert.py`, driven end to end with its fidelity gates by
`convert_final.py`. A file-by-file map is in `bnjettag/code/hgq2/README.md`.

The peak learning rate does not transfer between model scales, and the direction of the
correction is not monotonic. A probe at the 19,201-parameter scale placed the optimum at
2 × 10⁻⁵, below the value inherited from larger models. The 5,345-parameter configuration then
inherited *that* value and produced binary AUCs near 0.58, which was initially read as a
capacity limit; a twelve-run probe of its own placed its optimum at 1 × 10⁻⁴, five times
higher, and retraining the whole column there moved its binary AUC from 0.5789 to 0.7321
(`bnjettag/r7/roc-results/extension/` and `r7b/`). The apparent capacity cliff was a
artifact of the inherited learning rate. Every number in this document uses each scale's own tuned rate.

## 4. Tagging efficiency

### 4.1 Input conditioning dominates the binary penalty

The first campaign fed raw detector features to the network. Under that condition the binary
constraint appeared to cost 7.6 AUC points at 19,201 parameters. Applying an offline
per-feature z-score computed on the training split alone — the standard practice, with the
constants shipped alongside each checkpoint — changes both the level and the ordering.

![Effect of input standardization](figures/fig02_input_standardization.png)

*Figure 2. Held-out macro one-vs-rest AUC for four configurations at 19,201 parameters, with
raw features (circles) and with offline-standardized features (squares). Three seeds each;
error bars are the sample standard deviation.*

| Configuration | Raw features | Standardized features |
| --- | --- | --- |
| FP32 | 0.8255 ± 0.0008 | 0.9161 ± 0.0009 |
| W8A8 | 0.8203 ± 0.0019 | 0.9151 ± 0.0009 |
| W8A8, no normalization layers | not run | 0.9231 ± 0.0009 |
| W1A8, with SubLN normalization | 0.7500 ± 0.0083 | 0.8763 ± 0.0035 |
| W1A8, no normalization layers | 0.6569 ± 0.0099 | **0.8902 ± 0.0067** |

Under this conditioning the binary constraint costs 2.6 AUC points against the normed FP32
baseline, not 7.6. That comparison still crosses the normalization axis, however: the binary
arm is norm-free and the FP32 arm is not. The like-for-like control — the identical norm-free
architecture at 8-bit weights, byte-identical recipe, three seeds — reaches
**0.9231 ± 0.0009** (`bnjettag/roc-results/r11/roc_auc_r11_control.md`), so the cost of
binarization proper, everything else matched, is **3.3 AUC points**. Removing the norms helps
both precisions by a comparable amount, which is why the cross-axis comparison flatters the
binary model. The raw-feature measurement remains valid as measured, but its interpretation
was confounded:
8-bit input grids cannot resolve features whose per-feature standard deviation reaches ~118,
and no in-fabric normalization recovers information that input quantization has already
discarded.

Eight-bit quantization stays essentially free under this conditioning, 0.10 points below
full precision.

The best binary configuration is the one with **no normalization layers at all**, 1.4 points
above the normalized binary model. This inverts the raw-feature result, where removing every
normalization layer cost 9.3 points. It matters for hardware because, as Section 6 shows, the
normalization layers were the largest single consumer of both scarce FPGA resources: the
cheapest configuration to build is also the most accurate one, so no accuracy-versus-fit
trade-off has to be made at this point.

![Per-class ROC curves](figures/fig03_roc_by_class.png)

*Figure 3. Per-class one-vs-rest ROC curves on the held-out split with standardized inputs,
seed 1. The mistag axis is logarithmic. Top quarks remain the most separable class and
gluons the least, at every precision.*

### 4.2 The activation-width ladder

With weights fixed at one bit, the activation width is the remaining quantization knob. Both
scales degrade gracefully from 8 to 6 bits and then fall away at 4.

![Quantization axis](figures/fig04_quantization_axis.png)

*Figure 4. Left: held-out AUC against activation width at both model scales, raw features,
each scale at its own tuned learning rate. Right: measured whole-model DSP consumption at the
same three widths, at two reuse factors, from the raw synthesis reports.*

| Activation width | 19,201 parameters | 5,345 parameters |
| --- | --- | --- |
| A8 | 0.7500 ± 0.0083 | 0.7321 ± 0.0183 |
| A6 | 0.7362 ± 0.0095 | 0.7117 ± 0.0229 |
| A4 | 0.6561 ± 0.0412 | 0.6548 ± 0.0080 |
| FP32 reference | 0.8255 ± 0.0008 | 0.8265 ± 0.0041 |

The 4-bit point at 19,201 parameters is seed-unstable. Extending it from three seeds to six
gives 0.6520 ± 0.0279: the instability is real, but the three-seed spread of ± 0.041
overstated it.

### 4.3 Capacity

Full precision saturates by 5,345 parameters on this dataset — the full-precision AUCs at the
two scales agree within their seed spreads — while the binary variants continue to benefit
from capacity. Binarization costs 7.6 points at 19,201 parameters and 9.4 at 5,345 on raw
features. Eight-bit quantization is close to free at both: 0.52 points below full precision at
19,201 parameters, and 0.14 points above it at 5,345, which is inside the seed spread. Only
the one-bit constraint is capacity-hungry, which is the expected shape of a
bits-for-parameters trade rather than a failure mode.

### 4.4 A pairwise-invariant attention bias

One architectural extension has been measured. A shared bias on the pre-softmax attention
logits, built from three Lorentz-invariant functions of each constituent pair (ΔR², k_T², m²)
in the manner of the Particle Transformer, adds 84 parameters to the 19,201-parameter
norm-free recipe and leaves everything else byte-identical. Over six seeds, held-out AUC is
**0.8907 ± 0.0050** against the baseline's 0.8902 ± 0.0067 — statistically flat.

An interim three-seed reading of this experiment had looked better: 0.8918 ± 0.0026, with
rejection up 3.7–7.8 % across working points and a seed spread tighter by a factor of
roughly 2.5. Doubling the seed count removed most of it. The AUC difference vanished, and the
apparent tightening of the seed spread weakened with it (± 0.0026 at three seeds, ± 0.0050 at
six). What survives is a small rejection gain that is uniform in sign across working points —
**+4.8 % at ε_S = 0.5** (32.5 ± 2.0 against 31.0 ± 2.0) — and remains concentrated on W and Z,
but a Welch test on the macro rejection at ε_S = 0.5 gives p = 0.43, so it is not
statistically established at this seed count. The per-seed values are in
`bnjettag/roc-results/r10/roc_auc_r10.md` beside the six arrays.

![Pairwise attention bias](figures/fig05_pairwise_bias.png)

*Figure 5. Per-class effect of the pairwise-invariant attention bias: six seeds for the biased
model, three for the baseline. What gain there is sits on W and Z, the two weakest classes of
the baseline, but it does not reach significance.*

Neither the accuracy case nor the hardware case is established: the AUC effect did not
survive the seed extension, and the pair-feature formation for the 45 constituent pairs has
not been synthesized.

## 5. Background rejection at fixed signal efficiency

AUC integrates over the whole ROC curve, including regions no trigger would operate in. The
same stored arrays give the trigger-native quantity directly.

![Rejection versus efficiency](figures/fig06_rejection.png)

*Figure 6. Background rejection against signal efficiency for the deployable-scale
configurations, macro-averaged over the five classes. Bands are the standard deviation over
seeds — three per configuration, six for the pairwise-bias arm.*

| Configuration (19,201 parameters, standardized inputs) | ε_S = 0.5 | ε_S = 0.7 |
| --- | --- | --- |
| FP32 | 47.8 ± 0.9 | 16.7 ± 0.3 |
| W8A8 | 46.8 ± 0.7 (98 % of FP32) | 16.4 ± 0.2 (98 %) |
| W1A8, no normalization | 31.0 ± 2.0 (65 %) | 11.5 ± 0.6 (69 %) |
| W1A8, with SubLN | 24.6 ± 1.6 (52 %) | 9.7 ± 0.4 (58 %) |
| W1A8, no normalization, + pairwise bias (6 seeds) | 32.5 ± 2.0 (68 %) | 11.9 ± 0.6 (71 %) |

Against the matched norm-free 8-bit control of Section 4.1, which reaches 60.2 ± 1.2 at
ε_S = 0.5 (`bnjettag/roc-results/r11/roc_auc_r11_control.md`), the 3.3-point AUC cost of
binarization corresponds to retaining **51.5 % of the matched baseline's background
rejection** — a factor 1.94 loss. Against the normed FP32 baseline in the table above, a
comparison that crosses the normalization axis, the loss is 35 %. Eight-bit quantization
tracks full precision in this metric as well. Reporting AUC alone would understate the price
of the binary constraint, which is why both metrics accompany every headline number in this
document.

## 6. FPGA implementation

Synthesis targets a Xilinx VU13P (xcvu13p-flga2577-2-e; 1,728,000 LUT, 3,456,000 FF, 12,288
DSP, 5,376 BRAM-18K) with a 2.5 ns clock target, using hls4ml 1.3.0 on HGQ2 0.1.9 and Vitis
HLS 2023.2. Every number in this section is parsed from the C-synthesis reports stored under
`bnjettag/r7/results/csynth/`.

### 6.1 Conversion

hls4ml converts each trained checkpoint into a single project with no manual assembly, and the
whole model — both blocks, the attention cores, the head — synthesizes as one Vitis project
with RTL produced in minutes. Two constraints had to be resolved to get there, both recorded
in `bnjettag/results/hgq2/constraints_map.md`:

- Keras `LayerNormalization` has no working path through this stack, and its stock kernel's
  inverse-square-root table covers only variances below one. The parameter-free SubLN
  normalization was therefore added as a custom hls4ml layer: new IR layer, precision
  propagation hooks, Vitis templates, and a range-reduced inverse-square-root kernel valid for
  any input variance.
- hls4ml parses `keras.layers.ReLU` with an unsigned, wrapping output precision. Fed by the
  wide accumulator of a binary contraction, its negative inputs wrap to ~2¹⁰ instead of
  clamping to zero, which drove whole-model C-simulation correlation to −0.06 before it was
  found. Setting saturation rather than wrapping on relu-family outputs restores agreement.
  The general rule — any ReLU fed by a wide binary or integer accumulator needs saturating
  output arithmetic — is recorded for reuse.

### 6.2 The DSP census

Resources are attributed per consumer by walking the instance tree of the synthesis report and
subtracting each module's children from its own totals, so that no resource is double-counted.
The attribution closes exactly against the report's own profile total on every file.

![DSP census](figures/fig07_dsp_census.png)

*Figure 7. DSP consumption by consumer across all eight whole-model syntheses. The binary
weight layers contribute nothing to any bar.*

| Consumer | A8, RF 1 | A8, RF 8 | A6, RF 1 | A6, RF 8 | A4, RF 1 | A4, RF 8 |
| --- | --- | --- | --- | --- | --- | --- |
| Binary weight multiplies (15 layers) | **0** | **0** | **0** | **0** | **0** | **0** |
| Attention activation×activation | 12,800 | 1,600 | 12,800 | 1,600 | 6,400 | 800 |
| Softmax tables | 800 | 100 | 800 | 100 | 800 | 100 |
| Global pooling | 512 | 64 | 384 | 48 | 384 | 48 |
| Whole model | 14,112 | 1,764 | 13,984 | 1,748 | 7,584 | 948 |

The central claim of this work holds on the complete artifact, at every activation width
tested: **not one DSP block is spent on a binary weight multiplication.** The DSP census is
entirely weightless operations. Where normalization layers are present they dominate it — the
SubLN instances alone account for 19,584 of the 33,696 DSPs of the normalized model at full
unroll — which is a further reason the norm-free configuration of Section 4.1 is the
deployable one.

The single genuine resource-versus-precision effect is that attention DSPs halve at 4-bit
activations, where two multiply-accumulates pack into one DSP48. From 8 to 6 bits the DSP
count is flat. At 4 bits and reuse factor 8 the complete transformer costs 948 DSP blocks,
7.7 % of the device.

### 6.3 Latency and throughput

![Operating points](figures/fig08_operating_points.png)

*Figure 8. Every whole-model synthesis, placed by throughput and latency (left) and by
resource cost (right). A design is deployable when it lies left of 25 ns in the left panel and
below and left of both dashed budgets in the right panel.*

| Emission (19,201 parameters) | DSP | LUT | Latency | II | Est. clock | Per jet |
| --- | --- | --- | --- | --- | --- | --- |
| A8, SubLN, RF 1 | 33,696 | 6.04 M | 466 cyc = 1.005 µs | 1 | 2.157 ns | 2.2 ns |
| A8, SubLN, RF 8 | 6,900 | 5.21 M | 615 cyc = 1.327 µs | 8 | 2.157 ns | 17.3 ns |
| **A8, norm-free, RF 1** | 14,112 | 4.80 M | **167 cyc = 0.360 µs** | 1 | 2.157 ns | 2.2 ns |
| **A8, norm-free, RF 8** | **1,764** | 3.91 M | **218 cyc = 0.470 µs** | 8 | 2.157 ns | 17.3 ns |
| A6, RF 8 | 1,748 | 11.47 M † | 561 cyc = 1.741 µs | 8 | 3.103 ns | 24.8 ns |
| A4, RF 8 | 948 | 11.04 M † | 557 cyc = 1.728 µs | 8 | 3.103 ns | 24.8 ns |

† The A6 and A4 emissions inherited a fabric-bound normalization kernel whose LUT cost is
inflated by roughly 8.9 M and whose achieved clock is degraded; their LUT and clock columns
are not comparable with the canonical 8-bit rows. Their DSP columns are unaffected.

Reuse factor 8 is the Level-1 throughput point: an initiation interval of 8 cycles at the
estimated 2.157 ns clock accepts a new jet every 17.3 ns, inside the 25 ns bunch-crossing
window. The norm-free model reaches it at 0.47 µs end to end, and 0.36 µs when fully
unrolled — sub-microsecond at both operating points, because the normalization layers had been
two thirds of the pipeline depth.

### 6.4 Lookup tables: the constraint that is still open

![LUT budget](figures/fig09_lut_budget.png)

*Figure 9. Lookup-table consumption by consumer, against the device budget. The dashed line is
the whole VU13P.*

| Consumer, norm-free at RF 8 | LUT | Share |
| --- | --- | --- |
| Binary weight layers | 2,458,704 | 62.9 % |
| Attention activation×activation | 789,076 | 20.2 % |
| Glue and buffers | 374,675 | 9.6 % |
| Top-level glue | 218,825 | 5.6 % |
| Softmax | 42,190 | 1.1 % |
| Pooling | 27,045 | 0.7 % |
| Total | 3,910,515 | 226 % of the device |

The model does not fit. Removing the normalization layers took it from 5.21 M to 3.91 M LUT
and removed their 2.10 M-LUT share, but the remaining overhang belongs to the binary dense
layers themselves, which are 63 % of the budget. Everything the model is *not* spending on
those layers — 1.45 M LUT — would already fit, so a fit is arithmetically reachable if the
binary layers can be built more cheaply. It also sets the size of the required win: a
dense-directed remedy has to reach a factor 8.9 to close the gap on its own, so nothing below
roughly an order of magnitude is worth pursuing. Section 7 measures one that is.

Of the 2,458,704 LUT in binary weight layers, 2,419,913 belong to the ten per-token instances
of the same thirteen constant matrices, and the rest to the per-jet head and the input casts.
That distinction matters for what follows: the ten instances are the same matrix applied to
different constituents.

At the same operating point the other resources are comfortable: 1,764 of 12,288 DSP (14 %),
1.92 M of 3.46 M FF (56 %), and 90 of 5,376 BRAM-18K (1.7 %).

**Note added 2026-08-01.** The 3.91 M figure, like every resource number in this section, is a
C-synthesis estimate. A single whole-model Vivado RTL synthesis of the same build —
out-of-context, not placed or routed, and on a smaller part of the same fabric generation than
the VU13P, the only one licensed on the available machine — measured **1,787,637 CLB LUTs**:
C-synthesis over-estimates the LUT cost of these structures by roughly a factor 2.2. At the
same time the worst negative slack at the 2.5 ns constraint implies an achievable clock near
3.6 ns at this stage of the flow. The lookup-table overhang is therefore smaller than the
csynth ratio above suggests, and the binding risk shifts toward timing closure rather than
area. Synthesis-estimate and csynth-estimate numbers are not directly comparable, which is why
the tables of this section remain in csynth units; the raw utilization report is stored at
`bnjettag/results/adder-graph/eb2/util_wholemodel.rpt`.

### 6.5 Measured negative results

Five candidate remedies were tested and did not work. Each is recorded with its evidence,
because each one closes off a direction.

**Distributed arithmetic makes it worse.** `Strategy=distributed_arithmetic` (da4ml 0.5.2)
applied to all fifteen binary layers inflated the whole model from 3.91 M to 4.53 M LUT
(+15.8 %) and multiplied flip-flops by 3.70, from 1.92 M to 7.10 M. It also cost latency and
DSPs, 218 to 615 cycles at the same interval and 1,764 to 2,212 blocks. The emission was
bit-exact against the reference, so this is a resource result and not an accuracy one. The
mechanism is structural rather than a tuning failure: the canonical signed-digit representation
of a ±1 weight is a single digit, so common-subexpression elimination across weights has
nothing to share. Section 7.1 measures that directly — on these matrices a pairwise search
saves what it saves on random signs, and nothing more. Binary weights are the one case
distributed arithmetic cannot help.

**Global reuse folding is exhausted.** Raising the reuse factor from 8 to 10 moved the whole
model from 12.04 M to 11.94 M LUT, and the binary dense layers from 2,077,082 to 2,077,382
LUT — identical to within 0.02 %. With weights inlined as constants, the "multipliers" are
already logic, so reuse converts them into operand-selection multiplexers of the same order.
Reuse folds the wrong axis for a binary core. It also fails the throughput condition:
10 cycles at 3.103 ns is 31 ns, outside the 25 ns window.

**Trading DSPs for fabric in the normalization kernel is anti-fit.** Binding the SubLN
variance multiplies to fabric instead of DSP48s does reach the DSP-free end state — the census
becomes exactly attention plus softmax plus pooling — but normalization LUT rises from 2.10 M
to about 8.9 M and the estimated clock degrades from 2.157 to 3.103 ns. This route was
abandoned in favor of removing the normalization layers entirely, which Section 4.1 shows is
also the more accurate model.

**Streamed I/O is unavailable in this stack.** hls4ml 1.3.0 raises
`NotImplementedError: Heterogenous quantization for activations is only supported with
IOType=io_parallel` before code generation
(`backends/fpga/passes/hgq_proxy_model.py:63`). Streaming would require a frontend change, not
a configuration change.

**Widening the accumulators backfires.** A wide-accumulator emission was tried as a way to
push binary layers off DSPs. It had nothing to fix — they were never on DSPs at this
strategy — and it doubled each attention einsum from 3,200 to 6,400 DSPs by crossing the
DSP48 operand-width boundary, taking the whole model at full unroll from 33,696 to 46,496
blocks. Narrow emission is canonical
(`whole_model_rf{1,8,16}_wide.xml.gz` against `whole_model_rf{1,8}.xml.gz`).

### 6.6 Where this sits against published implementations

The closest published system is a set of sub-microsecond transformers for this same dataset
(arXiv:2510.24784), synthesized for an XCU250. Their numbers below are quoted from that paper's
Table 1; the per-class AUCs are read from its Figure 2 and averaged here, since the paper
reports no macro average.

| System | Input | Weights | LUT | DSP | Latency | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| Sub-µs transformers, most accurate configuration | 64 × 3 | learned bitwidth | 202 k | 0 | 78 ns | 79.8 %; per-class AUC 0.921 to 0.972 over the five classes |
| Same paper, all fifteen DSP-free configurations (its transformers and the set-model baselines it synthesizes alongside them) | ≤ 64 × 3 | learned bitwidth | 47 k–279 k | 0 | 44–140 ns | 64.7–79.8 % |
| This work, 19,201 parameters, interval 8 | 10 × 16 | hard {−1, +1} | 3.91 M | 1,764 | 0.47 µs | macro-OvR AUC 0.8902 |
| This work, 19,201 parameters, interval 1 | 10 × 16 | hard {−1, +1} | 4.80 M | 14,112 | 0.36 µs | macro-OvR AUC 0.8902 |

Three differences account for the gap, and none of them is a synthesis trick that was missed.

**They train to a resource target and this work does not.** Every configuration in that paper is
trained to a fixed budget of 350,000 EBOPs, which its authors describe as roughly one Super
Logic Region of the target device, enforced during training by a controller on the
regularization strength. This work trained for tagging efficiency and is retrofitting fit
afterwards. The same metric measured here is 6,802,571 EBOPs
(`bnjettag/r7/results/convert/r8stdnn-s2/rf8/ebops.json`), roughly a factor 19 above their
350,000 budget. No conversion of EBOPs into lookup tables is attempted; the calibration such a
conversion needs does not hold across the two design styles.

**Their accuracy is higher, and the inputs are not comparable.** Their per-class AUCs at 64
constituents average about 0.95, against 0.9231 for the matched norm-free 8-bit baseline,
0.9161 for normed full precision, and 0.8902 for binary weights here. They use 64 constituents with 3 features each through single-head attention; this work
uses 10 constituents with 16 features each through two blocks with four heads. Neither number
transfers to the other's input, and the comparison is reported in both directions for that
reason.

**Soft and hard quantization are not the same constraint.** Their weights carry learned
per-parameter bitwidths, which may fall to zero or stay multi-bit, and that is the regime in
which distributed arithmetic delivers the lookup-table numbers above. A hard {−1, +1} constraint
is the case where the same technique inverts, measured in Section 6.5 and explained in Section
7.1. The remedy has to come from somewhere else.

For a same-device reference point outside the transformer family, JEDI-linear (arXiv:2508.15468)
reaches DSP-free inference on this dataset at 64 constituents with 16 features on a VU13P, at
O(100 ns) and O(100 k) LUT, using mixed sub-3-bit weights with an extended distributed-arithmetic
backend and a custom pipelining pass. Neither the device nor the input dimensionality is what
stands in the way here.

### 6.7 Export fidelity

The correlation between the exported hardware model and the trained checkpoint is measured and
stored per article rather than assumed. At 8 bits the norm-free flagship reaches 0.989 against
its trained forward pass, a ceiling set by the two-signed-digit representation of the residual
scale constants; the 6-bit and 4-bit exports reach 0.953 and 0.883, where the static-grid
substitution at small model scale costs more. The subsequent gate — compiled firmware against
the exported model — passes everywhere, and the norm-free flagship is bit-exact through it. At
the earlier raw-feature operating point the exported hardware model scored 0.7569 on the
held-out split against 0.7582 for the checkpoint it was exported from — a cost of about
0.001 AUC, negligible in physics units at this scale (`bnjettag/r7/README.md`).

## 7. Reducing the lookup-table cost

Section 6.4 leaves one number outstanding: 3.91 M LUT against 1.728 M, with 63 % of it in the
binary dense layers, and every configuration-level remedy measured and closed in Section 6.5.
This section is what replaced them. It has two parts: a counting study on the exact matrices
that were synthesized, which says where the headroom is and where it is not, and a synthesis
experiment on one real layer, which measures whether the counting survives contact with the
tool.

### 7.1 Why the layers cost what they cost

Fitting the measured per-instance cost of the fifteen layer configurations against the bit-adds
of a naive per-output reduction tree, using each layer's real input and accumulator widths, gives

    LUT = 0.923 × (bit-adds) + 3,412        R² = 0.987

Roughly one lookup table per bit-add is the naive tree priced at face value. Nothing in the flow
restructures it, and had the estimator been pricing block-level sharing already the slope would
sit near 0.3. The threshold set before the fit ran required every per-layer residual under 15 %
and the largest is 24 %, on the 5×32 output layer where the intercept dominates a layer of 4,888
LUT; excluding it the largest is 11 %. Everything counted below is therefore a ranking and a
first-order sizing, not a prediction, and the synthesis experiment is what turns it into a
measurement.

**There is no learned structure for a search to find.** On the deployed matrices, greedy
two-term subexpression extraction, the class of sharing a distributed-arithmetic backend
performs, saves 699 to 712 adds on the 32×32 attention projections. On twenty random `±1`
matrices of the same shape it saves 705.6 ± 4.8. Real is null, to within the noise, and the same
holds for the affine form. That is the mechanism behind Section 6.5's negative result stated
positively: a `±1` weight has a one-digit signed-digit representation and the columns are
uncorrelated by construction, so whatever a pairwise search finds here is generic collision.
The paper behind that backend states the same failure mode for its own algorithm, that matrices
whose columns are rarely correlated yield a trivial decomposition.

The consequence is a change of strategy rather than of tool. Sharing on these matrices has to be
*imposed*, through identities that hold for any `±1` matrix, instead of *discovered*. Two such
identities and one schedule transform were counted (`bnjettag/results/adder-graph/`):

- **The affine form.** Each output is ±(T − 2 Σ minority), where T is the input total, built once
  and shared by every output row and by all three projections that read the same activations.
  About a factor 2 in adds, independent of the matrix.
- **Subset-sum sharing**, the Four-Russians decomposition. Partition the inputs into groups of k,
  build the 2^(k−1) signed group sums once, and let each output row select one precomputed value
  per group; because the matrix is a compile-time constant, that selection is wiring rather than
  a multiplexer. A counted factor 2.85 to 3.19 at k = 4 on these layers, and depth-neutral,
  since the group stage replaces two levels of the reduction tree.
- **Token-axis folding**, which is not adder arithmetic at all but a schedule. Thirteen of the
  fifteen weighted layers exist as ten identical instances applying the same constant matrix to
  different constituents; only the two head layers see a whole jet. Reuse factor cannot fold that axis: it time-multiplexes weights into a
  shared multiplier and so destroys the constancy the emission depends on, which is exactly why
  Section 6.5 measured it as a no-op. Folding time-multiplexes activations through a shared
  constant graph and preserves it.

### 7.2 One layer, five emissions

The counting was checked against the tool on one real layer, `bit_block_0_ffn_fc1`, 32 → 64,
which exists as ten per-token instances costing 234,930 LUT inside the whole model. Five
emissions were synthesized, each C-simulated against the reference implementation and required
to be bit-exact before its resource numbers were accepted. The bands each arm was judged against
were fixed before it ran.

![Token folding](figures/fig10_token_folding.png)

*Figure 10. Left: the five emissions of one 32→64 per-token layer, from the raw reports under
`bnjettag/results/adder-graph/e1/`. Right: the measured whole model beside what the measured fold
ratio projects, against the device budget. The right panel is a projection, not a synthesis.*

| Arm | Emission | LUT | vs arm P | FF | DSP | Latency | II | Clock |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P | ten instances, as synthesized today | 270,382 | 1.00 | 89,459 | 0 | 8 cyc | 1 | 1.583 ns |
| A2 | folded by two | 261,881 | 1.03 | 80,922 | 0 | 13 cyc | 14 | 1.583 ns |
| B | subset-sum restructured, k = 4 | 130,512 | **2.07×** | 64,718 | 0 | 7 cyc | 1 | 1.575 ns |
| A | folded by ten | 29,044 | **9.31×** | 11,597 | 0 | 20 cyc | 21 | 1.652 ns |
| C | folded by ten and restructured | 15,057 | **17.96×** | 9,121 | 0 | 19 cyc | 20 | 1.652 ns |

**Folding is the large effect.** Removing nine of the ten physical instances costs a factor 9.3,
and one layer goes from 270,382 to 15,057 LUT, a factor 18.0, when the restructuring is composed
onto the folded slice. The composition was predicted in advance and holds: arm C is 0.064 of the
whole-model baseline where the product of the two measured mechanisms gives 0.060, inside the
20 % agreement fixed beforehand. No arm used a DSP.

**Restructuring realizes about two thirds of what it counts.** Arm B reaches 2.07× against a
counted 3.07× in bit space. It also improves depth, 7 cycles against 8.

**Folding by two fails, and the reason is the tool.** Arm A2 was a monotonicity control and it
missed its band outright: Vitis unrolls a pipelined loop with a trip count of two, so the design
duplicated hardware while still serializing. Fold factors have to be large enough to defeat
automatic unrolling.

**The control lands on the edge of its band and is treated as a miss.** Arm P reproduces the
whole-model census to 1.151 against a ceiling of 1.15. The offset is accounted for: census
instances run at reuse factor 8, about 8 % cheaper on this fabric than the standalone reuse-1
emission, and the standalone project adds interface logic and the bias add. Every mechanism
ratio above is therefore quoted against arm P, which is like-for-like, rather than against the
census.

**The throughput question is closed.** The folded arms first reported intervals of 21 and 20
cycles against a budget of 11, which at the achieved clock is 33 to 35 ns and outside the 25 ns
window. That figure was not the fold: the scheduler log shows the fold loop itself achieving one
cycle per constituent, and the reported number was the standalone function's restart interval,
latency plus one, because a bare top-level function cannot overlap invocations. With
`#pragma HLS PIPELINE II=1 rewind` on the fold loop and nothing else changed, both folded arms
schedule at an interval of 10 cycles, 16.5 ns at the achieved clock, for 0.1 % less area:
29,029 LUT for arm A and 15,042 for arm C, both still bit-exact. The second report is stored
beside the first as `csynth_v2_rewind.xml.gz`.

### 7.3 What that projects, and what it does not show

Carrying the measured fold through the module census of Section 6.4, with the multiplexer,
control and buffer overheads counted:

| Component | LUT |
| --- | --- |
| Folded per-token dense: one slice, ten-to-one input multiplexers, control | 257,947 |
| Folded per-token glue at one tenth, with its multiplexers | 40,539 |
| Unfolded: attention products, softmax, pooling, head, wrappers | 897,102 |
| Top-level glue, kept in full | 218,825 |
| **Projected whole model** | **1,414,413** |

That is inside the device with about 18 % margin, at an interval of ten cycles against a budget
of eleven; composing the restructuring onto the folded slice projects 1,283,364 LUT, about 26 %.
It is the first configuration of this work projected to fit.

What that sentence is not is a synthesis. The measured part is one layer; the whole-model number
assumes that the multiplexer and control overheads are modeled correctly, that the per-token glue
folds with the dense layers, that reuse inside the folded slice drops to one at the measured 8 %
cost, and that the folded design still closes at or under 2.5 ns, where the 2.157 ns in hand is
an estimate for the unfolded design. A general emitter for the whole per-token stack does not
exist yet, and building it is the next piece of work.

Two further limits belong here rather than in a footnote. The implemented-netlist anchor is
missing: a post-synthesis run on arms P and C was part of the design and could not be completed,
because the synthesis machine has no Vivado synthesis licence for this part. Every ratio in this
section is therefore C-synthesis estimate against C-synthesis estimate, which is internally
consistent but blind to carry-chain packing decided later in the flow, and that blindness cuts
both ways. And the projection folds only the per-token dense stack: how the attention core,
the glue and the folded slices compose in a single design — and therefore which term of the
projected budget binds after folding — is unmeasured.

## 8. Open problems

1. **Device fit.** 3.91 M LUT against 1.728 M available, 63 % of it in the binary dense
   layers. Section 6.5 closes off reuse folding, distributed arithmetic, and streamed I/O.
   Section 7 measures a factor 9.3 from token-axis folding on one layer and projects a fit for
   the whole model, so what remains is engineering rather than search: a general folding
   emitter for the per-token stack, and then the same measurement on the whole artifact.
   Moving the ±1 weights into block RAM remains untested.
2. **The attention floor.** Weight binarization cannot reach the activation-by-activation
   products inside attention, and their share of a folded whole model is unmeasured. Narrowing
   them costs accuracy at the rate Section 4.2 measures, so if they turn out to bind, the
   answer is architectural rather than a precision one.
3. **Whether a 5-bit activation width softens the 6-to-4-bit cliff.** Untested; the ladder has
   only ever been measured at 8, 6 and 4.
4. **The input-size axis.** All results use the ten highest-pT constituents. How accuracy,
   latency and resources move as that count varies has been deferred by design, and the knob
   exists in the code.
5. **The hardware cost of the pairwise attention bias.** The 45-pair feature formation of
   Section 4.4 has not been synthesized, so its accuracy gain currently has no price attached.
6. **An implemented-netlist anchor.** Every resource number in this repository is a
   C-synthesis estimate. Place-and-route was not reachable on the available machine, so how
   much of the binary adder fabric packs into carry chains is unmeasured.
7. **Seed count.** Three seeds per configuration is enough to separate the ladder steps and
   the binary penalty, but not to resolve differences of a few tenths of a point: the
   pairwise-bias result of Section 4.4 stayed unresolved even at six seeds.

## 9. Reproducing every number

No number here needs to be taken on trust. Every accuracy figure comes from a stored array of
per-jet scores, and every hardware figure from a stored synthesis report.

Accuracy, in about a minute:

```bash
pip install numpy scikit-learn
python3 - <<'PY'
import numpy as np
from sklearn.metrics import roc_auc_score
d = np.load("bnjettag/r7/roc-results/r8/SMALL-W1A8-r8stdnn-s1.npz")   # keys: y, score, meta
print(d["meta"])                                                      # provenance string
print(roc_auc_score(d["y"], d["score"], multi_class="ovr", average="macro"))
PY
```

Each store carries a claim table beside its arrays — `roc_auc.md`, `roc_auc_r8.md`,
`roc_auc_extension.md` — stating what the arrays are asserted to give, so a recomputation can
be checked against a written claim rather than against a number in prose.

Hardware, straight from the reports (they are 40 MB each uncompressed, so they are stored
gzipped):

```bash
gunzip -c bnjettag/r7/results/csynth/whole_model_rf8_stdnn.xml.gz | head -60
gunzip -c bnjettag/r7/results/csynth/whole_model_rf8_stdnn.xml.gz > /tmp/report.xml
python bnjettag/code/hls/parse_census.py /tmp/report.xml DSP     # or LUT
```

The same parser reads the single-layer reports of Section 7, which are stored the same way, one
directory per emission:

```bash
gunzip -c bnjettag/results/adder-graph/e1/c/prj_c/sol1/syn/report/csynth_v2_rewind.xml.gz > /tmp/c.xml
python bnjettag/code/hls/parse_census.py /tmp/c.xml LUT
```

The counting study of Section 7.1 rebuilds from the weight files it ships with, and prints its
per-layer table and the fit it is calibrated against:

```bash
python bnjettag/code/analysis/adder_graph_study.py    # rewrites counting_results.json
```

Every figure in this document:

```bash
pip install numpy scikit-learn matplotlib
python figures/make_figures.py
```

The script reads the same arrays and reports, writes the ten figures, and writes
`figures/NUMBERS.txt` listing every value that reached a figure, so a plot can be audited
without rerunning it.

## 10. Repository layout

| Path | Contents |
| --- | --- |
| `figures/` | The ten figures of this document, the script that generates them from the stored data, and the value log it emits. |
| `bnjettag/code/hgq2/` | The current pipeline: quantization-aware training, conversion to hls4ml, the custom SubLN layer and its Vitis templates, verification gates, synthesis drivers, run configurations. |
| `bnjettag/code/training/` | The earlier QKeras trainer and ROC tooling. |
| `bnjettag/code/hls/` | Synthesis drivers and the instance-tree report parser used for every census in Section 6. |
| `bnjettag/code/analysis/` | The counting study of Section 7.1: what each restructuring of a ±1 matrix would save against the adder tree the tool emits. |
| `bnjettag/code/jobs/` | 182 Kubernetes job specifications — every training and synthesis run that produced the results here. |
| `bnjettag/r7/roc-results/` | Per-jet score arrays for the raw-feature campaign, with the verified AUC tables. |
| `bnjettag/r7/roc-results/r7b/` | The 5,345-parameter column, retrained at its own learning rate. |
| `bnjettag/r7/roc-results/r8/` | The standardized-input campaign — the arrays behind Sections 4.1 and 5. |
| `bnjettag/r7/roc-results/extension/` | Seed extensions, the learning-rate probe, and the normalization ablation. |
| `bnjettag/roc-results/r10/` | Arrays for the pairwise attention bias of Section 4.4. |
| `bnjettag/roc-results/r11/` | The capacity ladder: 54 arrays covering d_model 32 to 128 (19,201 to 285,121 parameters), four arms per scale, three seeds each, with the verified AUC and rejection tables. Binary against a matched norm-free 8-bit baseline at every scale. |
| `bnjettag/r7/results/csynth/` | Sixteen raw Vitis C-synthesis reports, gzipped. Every hardware number traces here. |
| `bnjettag/r7/results/` | The operating-point study, the LUT census, and the activation-ladder table. |
| `bnjettag/r7/results/convert/` | Conversion and C-simulation evidence per emission: precision maps, fidelity gates, layer inventories. |
| `bnjettag/r7/models/` | Four trained checkpoints, including the norm-free flagship and the standardization constants it expects. |
| `bnjettag/results/hgq2/constraints_map.md` | What converts cleanly through HGQ2 and hls4ml, what needs a custom layer, and where DSPs hide. Established by execution, not by reading documentation. |
| `bnjettag/results/adder-graph/` | Section 7: the counting study's full output, the weight files and widths it reads, and the five single-layer emissions with their raw reports. |
| `infrastructure/` | Cluster and synthesis-machine setup. |

Not included: the training dataset (public, Zenodo record 3602260), generated HLS project
bundles and firmware trees (regenerable from the checkpoints and configurations in minutes),
and vendored Xilinx headers.

## 11. Context and references

- **BitNet**, arXiv:2310.11453 — the 1-bit transformer training method adapted here; the
  ternary follow-up is arXiv:2402.17764 and the low-bit-activation study arXiv:2411.04965.
- **hls4ml**, arXiv:1804.06913 — the codesign tool and the trigger-ML programme it anchors.
- **HLS4ML LHC Jet dataset**, arXiv:1804.06913 and arXiv:1908.05318, Zenodo 3602260 — the
  public benchmark used throughout.
- **Sub-microsecond transformers for jet tagging on FPGAs**, arXiv:2510.24784 — the closest
  published system: near-full-precision transformers through hls4ml into Level-1 latency, all
  trained to a fixed 350,000-EBOPs budget. The source of the comparison in Section 6.6. The
  differentiator of this work is the one-bit weight core and its DSP-free mapping.
- **JEDI-linear**, arXiv:2508.15468 — DSP-free trigger inference on the same dataset and the same
  device as this work, with mixed sub-3-bit weights and a distributed-arithmetic backend extended
  by a custom pipelining pass.
- **Ultrafast jet classification at the HL-LHC**, arXiv:2402.01876 — a synthesized tagger in
  the same parameter class (20 k–27 k) on the same device, at 2.1 % DSP and 7–9 % LUT.
- **BitParT**, arXiv:2508.07431 — the only published attempt to binarize part of a
  Particle-Transformer-family network. It leaves attention, normalization and the interaction
  pathway in full precision, reports no hardware numbers, and finds a 15–17 % rejection loss
  behind unchanged AUC.
- **Particle Transformer**, arXiv:2202.03772 — the source of the pairwise-invariant attention
  bias of Section 4.4.
- **da4ml**, arXiv:2507.04535 — the distributed-arithmetic backend evaluated in Section 6.5, and
  the source of the correlated-columns condition its own algorithm needs, which Section 7.1
  measures as absent on binary matrices.
- **Unrolling ternary neural networks**, arXiv:1909.04509 — the nearest prior art for compiling a
  low-precision network into shared adder logic rather than multipliers. It is ternary rather than
  binary, convolutional, and three orders of magnitude away from trigger latency, but it is the
  demonstration that the genre works.
- **EBOPs / HGQ**, arXiv:2405.00645 — the synthesis-free hardware cost metric used during
  design.
