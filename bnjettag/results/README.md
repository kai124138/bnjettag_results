# results/

The synthesis results for the deployable-scale campaign live beside their own score arrays in
[`../r7/results/`](../r7/results/): the whole-model C-synthesis reports, the operating-point
study, the module-level LUT census, and the activation-ladder table.

What remains here is not tied to a single campaign:

| Path | What it is |
| --- | --- |
| `hgq2/constraints_map.md` | The layer-support map for HGQ2 0.1.9 and hls4ml 1.3.0 — what converts and is bit-exact, what converts with caveats, what needs the custom-layer recipe, what is blocked outright, and where DSPs hide. Every row was established by running the conversion, not by reading documentation, so new architectures can be designed against it. |
| `adder-graph/` | Why the binary dense layers cost the lookup tables they cost, and what reduces it: the counting study on the deployed ±1 matrices, and the five single-layer emissions of Experiment 1 with their raw synthesis reports. Written up in Section 7 of the top-level README. |
