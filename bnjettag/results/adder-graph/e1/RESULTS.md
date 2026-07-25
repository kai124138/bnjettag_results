# Experiment 1 — five emissions of one binary dense layer

Every number below is parsed from the raw report shipped beside it. Target `xcvu13p-flga2577-2-e`,
Vitis HLS 2023.2, 2.5 ns target clock. The layer is `bit_block_0_ffn_fc1`, 32 → 64, which exists
as ten per-token instances costing **234,930 LUT** inside the whole model
(`../../r7/results/csynth/whole_model_rf8_stdnn.xml.gz`, ten instances of 23,493). Each arm was
C-simulated against the reference implementation and required to be bit-exact before its
resource numbers were accepted.

| Arm | LUT | vs arm P | vs census | FF | DSP | Latency | II | Estimated clock |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P, ten instances | 270,382 | 1.00 | 1.151 | 89,459 | 0 | 8 cyc | 1 | 1.583 ns |
| A2, folded by two | 261,881 | 1.03 | 1.115 | 80,922 | 0 | 13 cyc | 14 | 1.583 ns |
| B, subset-sum k = 4 | 130,512 | **2.07×** | 0.556 | 64,718 | 0 | 7 cyc | 1 | 1.575 ns |
| A, folded by ten | 29,044 | **9.31×** | 0.124 | 11,597 | 0 | 20 cyc | 21 | 1.652 ns |
| C, folded and restructured | 15,057 | **17.96×** | 0.064 | 9,121 | 0 | 19 cyc | 20 | 1.652 ns |

The bands each arm was judged against were fixed before it ran: P 0.85 to 1.15, A 0.10 to 0.17,
A2 0.50 to 0.60, B 0.30 to 0.55, C 0.04 to 0.09, all relative to the census baseline. Folding
was to be abandoned if A exceeded 0.25 while A2 also failed to scale, and restructuring if B
exceeded 0.74.

## What the arms say

**The control lands on the edge of its band and is treated as a miss.** Arm P reproduces the
census to 1.151 against a ceiling of 1.15. The residual is accounted for: census instances run
at reuse factor 8 inside the whole model, which measures about 8 % cheaper on this fabric than
the standalone reuse-1 emission, the standalone project adds its own interface logic, and the
bias add sits inside the kernel here. Because of that offset, every mechanism ratio is quoted
against arm P, which is like-for-like, rather than against the census.

**Folding works and folding by two does not.** Arm A removes nine of the ten physical instances
for a factor 9.3. Arm A2 was the monotonicity control and it failed its band outright, at 1.115:
Vitis unrolls a pipelined loop with a trip count of two, so the design duplicated the hardware
while still serializing the schedule. The lesson is about the tool, not the mechanism. Fold
factors have to be large enough to defeat automatic unrolling.

**Restructuring works, and less well than counted.** Arm B reaches 2.07× against a counted
3.07× in bit space, so realization costs about a third of the paper win, and it lands just
outside the top of its predicted band while clearing its kill line comfortably. It also
*improves* depth, 7 cycles against 8, because the group-sum stage replaces two levels of the
reduction tree.

**The two mechanisms compose.** Arm C is 0.064 of the census where the product of the two
measured mechanisms predicts 0.060, within the 20 % agreement fixed in advance. One layer goes
from 270,382 to 15,057 LUT, a factor 18.0, with no DSPs anywhere and bit-exactness at every arm.

## The rewind rerun: the interval question, closed

The folded arms first reported II = 21 and 20 against a budget of 11, which at the achieved
clock is 33 to 35 ns per invocation and outside the 25 ns window. The scheduler log shows why
this was not the fold: the fold loop itself already achieved an interval of one cycle per
constituent, and the reported figure was the standalone function's restart interval, latency
plus one, because a bare top-level function cannot overlap invocations. Adding
`#pragma HLS PIPELINE II=1 rewind` to the fold loop, and changing nothing else, gives:

| Arm | LUT | Change | II | II × clock | Latency |
| --- | --- | --- | --- | --- | --- |
| A, folded by ten | 29,029 | −0.1 % | **10** (was 21) | **16.5 ns** | 19 cyc |
| C, folded and restructured | 15,042 | −0.1 % | **10** (was 20) | **16.5 ns** | 18 cyc |

Both remain bit-exact, both remain at zero DSPs, and the estimated clock is unchanged at
1.652 ns. Folding is therefore legal against the trigger's throughput contract at layer scale.
Whether it composes into a whole model at II = 10 is a projection until the general emitter
exists.

## What is not measured here

The implemented-netlist anchor is missing. A post-synthesis run on arms P and C was part of the
design and could not be completed: the synthesis machine has no Vivado synthesis licence for
this part. Every ratio above is therefore C-synthesis-estimate space compared against
C-synthesis-estimate space, which is internally consistent but blind to carry-chain packing
decided later in the flow. That cuts both ways: it could flatter the restructured arm, whose win
is bit-level, or the baseline, whose adder trees are exactly what packing targets. Arm C is also
a hand-emitted single layer, not a compiler pass.
