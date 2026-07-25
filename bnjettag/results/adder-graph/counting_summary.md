# Counting study — summary

How much of the binary dense layers' lookup-table cost is structure the current toolchain does
not harvest. Counted on the exact `{−1, +1}` matrices that were synthesized, with each layer's
real input and accumulator widths, against the naive per-output adder tree that Vitis emits
under the Latency strategy. Regenerate with
`python ../../code/analysis/adder_graph_study.py`; the full per-layer record is
`counting_results.json`.

Cost unit: bit-adds, meaning adder count weighted by output width and saturated at the
accumulator. All lookup-table figures are C-synthesis-estimate space.

## Calibration: the emission is the naive tree, priced at face value

Fitting measured per-instance LUT against naive-tree bit-adds over the fifteen layer
configurations of the synthesized model gives

    LUT = 0.923 × (bit-adds) + 3,412        R² = 0.987

Roughly one lookup table per bit-add is the naive reduction tree with no sharing. Had the
estimator already been pricing block-level sharing, the slope would sit near 0.3.

The gate set before the fit ran required every per-layer residual under 15 %, and it fails: the
largest is 24 %, on the 5×32 output layer, where the fitted intercept dominates a layer of
4,888 LUT. Excluding it the largest residual is 11 %. The consequence is honored throughout:
what follows is a ranking and a first-order sizing, not a synthesis-grade prediction. The
sizing instrument is Experiment 1 (`e1/RESULTS.md`).

## Mechanisms, per layer

Adds are adder counts; the last column is the measured per-instance cost of that layer in the
whole-model census, for scale. Subset-sum is evaluated at k ∈ {2, 4, 8} and the best is shown.

| Layer | m×n | Naive adds | Two-term CSE saves | Affine adds | Subset-sum adds | Ratio | Measured LUT |
| --- | --- | --- | --- | --- | --- | --- | --- |
| block 0 attn W_q | 32×32 | 1,024 | 710 | 514 | 312 | 2.85× | 13,047 |
| block 0 attn W_k | 32×32 | 1,024 | 703 | 496 | 312 | 2.85× | 12,782 |
| block 0 attn W_v | 32×32 | 1,024 | 705 | 502 | 312 | 2.85× | 13,145 |
| block 0 attn W_o | 32×32 | 1,024 | 705 | 491 | 312 | 3.02× | 17,822 |
| block 0 FFN fc1 | 64×32 | 2,048 | 1,510 | 971 | 568 | 3.07× | 23,493 |
| block 0 FFN fc2 | 32×64 | 2,048 | 1,438 | 1,004 | 624 | 3.00× | 35,498 |
| block 1 attn W_q | 32×32 | 1,024 | 712 | 497 | 312 | 2.85× | 12,792 |
| block 1 attn W_k | 32×32 | 1,024 | 699 | 494 | 312 | 2.85× | 13,011 |
| block 1 attn W_v | 32×32 | 1,024 | 705 | 502 | 312 | 2.85× | 12,916 |
| block 1 attn W_o | 32×32 | 1,024 | 710 | 504 | 312 | 3.03× | 18,321 |
| block 1 FFN fc1 | 64×32 | 2,048 | 1,532 | 1,002 | 568 | 3.07× | 24,143 |
| block 1 FFN fc2 | 32×64 | 2,048 | 1,446 | 1,018 | 624 | 3.01× | 36,985 |
| head fc1 | 32×32 | 1,024 | 704 | 512 | 312 | 2.84× | 14,271 |
| head fc2 | 5×32 | 160 | 100 | 111 | 96 | 1.61× | 4,888 |
| block 0 Q, K, V fused | 96×32 | 3,072 | 2,340 | 1,450 | 824 | 3.19× | not a module |

Two mechanisms are matrix-independent identities rather than searches. The affine form writes
each output as ±(T − 2 Σ minority), where T is the input total built once and shared by every
output row, and by all three projections when Q, K and V read the same activations. Subset-sum
partitions the inputs into groups of k, builds the 2^(k−1) signed group sums once by Gray-code
chaining, and lets each output row select one precomputed value per group; because the matrix
is a compile-time constant that selection is wiring, not a multiplexer. At k = 4 it adds no
depth, since the group stage replaces two levels of the reduction tree.

## Null control: there is no learned structure to find

Twenty random `±1` matrices of identical shape, per layer. On the 32×32 attention projections
the greedy two-term extraction saves **705.6 ± 4.8** adds on random signs against **699 to 712**
on the trained matrices, and the affine form needs 502.1 adds against 491 to 514.

Real is null, everywhere. Whatever a pairwise search finds on these matrices is generic
collision, not learned redundancy, which is why a backend that searches for shareable
subexpressions gains nothing here while one that imposes a fixed decomposition gains a factor
of three. It is also the mechanism behind the measured failure of distributed arithmetic on
this model, reported in Section 6.5 of the top-level README.

## Token-axis folding: the projection

Thirteen of the fifteen weighted layers exist as ten identical instances applying the same
constant matrix to different constituents; only the two head layers see a whole jet. Reuse factor cannot fold that axis, because reuse
time-multiplexes weights into a shared multiplier and so destroys the constancy the emission
depends on; folding time-multiplexes activations through a shared constant graph and preserves
it. From the measured census of the synthesized model (3,910,515 LUT total, per-token slice
2,419,913 across ten instances):

| Component | LUT |
| --- | --- |
| Folded per-token dense: one slice, ten-to-one input multiplexers, control | 257,947 |
| Folded per-token glue at one tenth, with its multiplexers | 40,539 |
| Unfolded: attention products, softmax, pooling, head, wrappers | 897,102 |
| Top-level glue, kept in full | 218,825 |
| **Projected whole model** | **1,414,413** |

That is inside the 1,728,000 LUT of the device with about 18 % margin, at an interval of ten
cycles against a budget of eleven. Composing the best restructuring onto the folded slice
projects 1,283,364 LUT, about 26 % margin.

Four things this projection assumes, none of them synthesized: that the multiplexer and control
overheads are modeled correctly, that the whole per-token glue category folds with the dense
layers, that reuse inside the folded slice drops to one at the measured 8 % cost, and that the
folded design still closes at or under 2.5 ns, where 2.157 ns is the unfolded estimate. The
first two of these carry the ±24 % calibration band, which does not change the verdict but does
set its precision.

After folding, 789,076 of the projected 1,414,413 LUT, about 56 %, is the weightless
activation-by-activation arithmetic inside attention, which no weight-side optimization
touches.
