# round-8 · era-2 data · ROC-test AUC (offline input standardization; held-out, n = 260,000)

Recomputed locally from `r8/*.npz` (verification gate 2026-07-18, 12/12 exact vs the
job's evaluations). All runs: small architecture (19.2k params), per-feature z-score from
the TRAIN split only (stats travel as `input_std.json`; eval applied per-run stats via
`--input-std`). W&B runs `r8-small-*`, artifacts `r8x-roc-artifacts`.

| arm | inputs | norms | ROC-test AUC (3 seeds) |
|---|---|---|---|
| FP32-std | standardized | SubLN | 0.9161 ± 0.0009 |
| W8A8-std | standardized | SubLN | 0.9151 ± 0.0009 |
| **W1A8-stdnn** | standardized | **none** | **0.8902 ± 0.0067** |
| W1A8-std | standardized | SubLN | 0.8763 ± 0.0035 |

Raw-input controls (round 7, same architecture/data/split): FP32 0.8255 ± 0.0008 ·
W1A8 0.7500 ± 0.0083 · W1A8-nonorm (raw) 0.6569 ± 0.0099.

## Readings (all gate-verified)

1. **Offline standardization lifts everything at this scale**: FP32 +9.1 pts, W1A8 +12.6
   (normed) / +14.0 (norm-free vs its raw control's 0.6569 → +23.3). The 8-bit input
   grids could not resolve raw features (per-feature σ up to ~118); no in-fabric norm
   recovers what input quantization already discarded.
2. **The fair binary gap at 19.2k params is 2.6 pts** (0.9161 → 0.8902), not the 7.6
   measured on raw inputs. The raw-input curve remains valid as measured; its
   interpretation (binarization cost) was confounded with input conditioning.
3. **Norm-free is the BEST binary configuration** (+1.4 pts over normed binary under
   standardization): SubLN was actively hurting the binary model once inputs were
   conditioned — and it is also the top consumer of both scarce FPGA resources. The
   fit-vs-accuracy trade-off dissolved: the cheapest model is the most accurate.
4. The 19.2k stdnn binary (0.8902) exceeds the 6.4M raw-trained binary (0.8503) —
   the earlier scales' numbers are likely also input-conditioning-limited (untested;
   a std-conditioned large-model re-measurement would be a separate campaign).

Contract note: the hardware receives standardized inputs (offline preprocessing —
reference practice per research-log 2026-07-17); the per-feature constants ship with
each checkpoint as `input_std.json`.
