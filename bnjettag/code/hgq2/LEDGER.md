# HGQ2 rebuild — change ledger

Running log of every consequential change in this effort. Dated, newest on top.

## 2026-07-18 (full-synthesis branch) — ROUND-8 stdnn flagship: norm-free GATE1 grid-sizing fix + rf1/rf8 emit
**Why.** The r8 stdnn flagship (small-stdnn-s2, best binary ROC 0.8964) exported with GATE1
corr_scores **0.7535** — a grid-saturation collapse. Diagnosis: without norms the residual
stream grows block-over-block, and the β-carry fold (γ_Wo·β_v, γ_fc2·β_fc1, γ_head_fc2·β_head_fc1;
DONE previously, took GATE1 0.278→0.754) leaves the ctx→Wo / relu(fc1)→fc2 / relu(head_fc1)→head_fc2
INPUTS scaled by 1/β (β dropped upstream, restored only in the downstream affine). Sizing those
grids from the QAT sites' i-bits (`read_qat_act_ibits`) drops a ctx of |max|≈54 into a
SAT<8,1+2>=±4 grid → catastrophic clip (e/q stream std ratio 0.63).

**Code (`convert_final.py`, minimal + guarded).**
- `calibrate_norm_free_grids(export_model, cfg, binz, X, margin=1)` — TWO-PASS, guarded on
  arch.norm=='none' (returns {} + touches nothing for every normed graph): (1) widen all 5
  β-carry-inflated internal-stream input grids permissively so NONE clips (a clip anywhere
  corrupts the downstream residual and mis-measures later sites); (2) tap the true pre-quant
  input max on that non-clipping graph and set i=ceil(log2 max)+1 with f=act_bits-1 KEPT — widen
  TOTAL WIDTH (8→15-17) on these internal streams rather than steal fraction bits (the
  build.py.stream_iq exact-passthrough precedent). Residual/normalized dense inputs
  (input_proj/Q/K/V/fc1/head_fc1) are LEFT at QAT i-bits — they reproduce QAT at 8-bit
  bit-for-bit; widening them slightly HURTS (0.9891→0.9868, measured).
- `build_export(..., calib_jets=None)` — new trailing kwarg; when norm=='none' and calib_jets
  given, runs the two-pass after port/affine/attn and stashes chosen widths on
  `model._bnhgq2_norm_free_grids`. Default None ⇒ prior behaviour byte-identical.
- `_apply_exact_beta(model, binz, pe, cfg)` — now carries β for norm-free (mirrors port.py);
  without it the exact-β reference collapsed to 0.036, making the CSD-2 characterization
  meaningless. Normed: carry=1, byte-identical.
- `export_verify.json` now carries `norm_free_grids` (per-site max/i/f/width).

**Chosen widths (measured export-side, non-clipping):** Wo0<15,i7>@54 · fc2_0<16,i8>@69 ·
Wo1<16,i8>@79 · fc2_1<17,i9>@166 · head_fc2<17,i9>@153. f=7 throughout (act_bits-1).

**Gates (n=4096 real jets, input_std applied; both rf1 & rf8).**
- GATE1 ship (csd2, DSP-free): **0.7535 → 0.98907** (argmax 0.949; repro on 2048 jets 0.98912).
- GATE1 exact-β reference: 0.99423 (CSD-2 Δ = +5.2e-3). copy_qat keras ceiling: 0.99593.
- GATE2 (hls4ml C-sim vs export): **corr 1.0, max|Δ| 0.0, BIT-EXACT, PASS** — the wide grids
  convert cleanly; copy_qat GATE2 demo 0.940 reconfirms hls4ml drops the 1/β datalane scaler.
- weight |Δ| vs QAT forward 1.5e-8 (±β datapath exact). EBOPs 6,802,571.

**CEILING (genuine, characterized — NOT 0.999).** 0.989 is the DSP-free hardware-faithful ceiling.
The limiter is CSD-2 β-snapping on the norm-free PRODUCT gammas (γ=β·β_upstream≈0.02; 2-signed-digit
CSD rel-err 2-7% on such small constants) with NO LayerNorm to absorb it — normed graphs pass 0.999
precisely because each LN kills the snap error. exact-β affines lift it to 0.994 but cost real
per-channel affine multiplies (DSP tradeoff, mulder-only). Even the absolute keras copy ceiling
(exact-β + copied QAT attention grids) is 0.996, so 0.999 is unreachable for THIS norm-free graph;
the residual ~0.4% is softmax/attention table reproduction. Per task: hit a genuine ceiling below
0.999 → characterized + stopped here with evidence.

**Normed path BYTE-IDENTICAL (verified).** r7-small-w1a8-s2 (norm=subln) rf8 into a throwaway
/tmp run-dir → C-sim corr **0.9986957710291886** exactly (frozen reference), GATE2 PASS.

**Emitted (stale stores overwritten):** `bnjettag/r7/results/convert/r8stdnn-s2/rf{1,8}/`
(export_verify/csim_verify/ebops/convert json + hls_prj_rf{1,8} + tarball). Local gates: py_compile
OK (convert_final + probes_final{,_v2} + build/port); build_export/`_apply_exact_beta` signatures
back-compat; normed calibrate returns {}. Nothing trained/synthesized; hand to verification.

## 2026-07-15 (full-synthesis branch) — ROUND-7 loose-end jobs/configs (seed-extension + tiny LR probe)
**Why.** Two logged round-7 caveats become experiments. (1) W1A4-small is seed-unstable — 0.6561 ±
0.0412 over 3 seeds, ~5x the spread of every other cell; extend its seed axis 3 → 6 to characterize.
(2) tiny's binary collapse (~0.58 AUC) must be confirmed FUNDAMENTAL (bits-for-parameters), not a
recipe artifact: tiny inherited small's peak LR 2e-5 without its own probe. Build ready-to-submit
Jobs only — not launched here (training runs on NRP).

**Code (`gen_r7_jobs.py`, minimal + reversible).**
- Refactored `job_yaml(size,variant,seed)` into a generic `render_job(**kw)` (explicit job/label/run/
  out_key/config/size/seed/note) + a thin matrix wrapper. No substitution VALUE contains an @-token,
  so output is order-independent and the 30 matrix YAMLs regenerate BYTE-IDENTICAL (md5 before/after,
  0 diffs — verified).
- New knobs: `W1A4_SMALL_EXTRA_SEEDS=[4,5,6]` (SET 1) and `TINY_LR_PROBE=[("lr1e5",1e-5),("lr5e5",5e-5)]`
  (SET 2). SET 2 also derives `configs/r7-tiny-w1a8-lr{1e5,5e5}.json` from `r7-tiny-w1a8.json` in-code
  (deep-copy, change ONLY `name` + `train.lr`).
- Preflight config-count parameterized 10 → **12** (`@NCONFIGS@`); `launch_r7_staged.sh` untouched — the
  loose ends are launched by hand, not wired into the staged 30-Job matrix.

**Emitted (35 YAMLs + 2 configs).** SET 1: `kai-bn7f-small-w1a4-s{4,5,6}.yaml` (config `r7-small-w1a4.json`,
W&B `r7-small-w1a4-s{4,5,6}`). SET 2: `kai-bn7lr-tiny-{lr1e5,lr5e5}-s1.yaml` (seed 1, W&B
`r7lr-tiny-w1a8-{lr1e5,lr5e5}-s1`); control = existing `kai-bn7f-tiny-w1a8-s1` @ 2e-5.

**Local gates (all PASS, `.venv-hgq2`).** py_compile · 30 matrix YAMLs byte-identical · all 12 r7 configs
build on the QAT stack with EXACT param counts (small 19,201 / tiny 5,345) + binary {−1,+1} gate OK (15L,
both new LR configs included) · 35/35 YAMLs parse (kind=Job) · embedded python compiles (2 preflight
heredocs + inline job snippets) · `bash -n` args valid. New LR configs get distinct cfg-hashes
(1e608b8d / 169221c0 vs base afbc4086).

**REFRESH BEFORE LAUNCH.** Re-run `./make_code_configmap.sh` so the 2 new `r7-tiny-w1a8-lr*.json` ship in
ConfigMap `kai-bnf-code` (pods read configs from the CM). Nothing launched/trained; hand to verification.

## 2026-07-14 (full-synthesis branch) — ROUND-7 whole-model v2: WIDE-ACCUM emission (DSP-0 form)
**Why.** The first whole-model csynth (rf1, plain accums) landed on mulder: 466 cyc @ II=1, est
2.157 ns (timing met), **DSP 33,696 / LUT 6.04M**. Census confirmed risk #2 exactly: ~12.8k DSP are
the structural act×act attention multiplies (3,200×2 per block × 2 blocks, 1 DSP/MAC — expected,
precision-independent), and ~21k are the narrow-accumulator DSP-parking on Wq/Wk/Wv/Wo + fc2/head_fc2
(the 2026-07-09 regime). Fix: apply the measured widen (probes_final_v3) to the WHOLE model.

**Code (minimal, backward-compatible).** `convert_final.py`:
- `widen_weighted_accum(hm, min_frac=16)` — the WHOLE-MODEL generalization of
  `probes_final_v3.widen_dense_accum` (which matched only class_name=='Dense'): now also widens
  **EinsumDense** (both register `TypeAttribute('accum')`, so `get_attr('accum_t')` works), so the
  binary projections/FFN get frac≥16 too. The weightless act×act **QEinsum** score/ctx cores
  (class_name=='Einsum') are deliberately NOT touched — their multiplies are the structural
  attention DSPs.
- `audit_weighted_accums(out_dir, cfg, min_frac)` — reads the GENERATED `firmware/defines.h` and
  reports every weighted-layer `*_accum_t` typedef + frac + a pass/`offenders` list (the evidence
  that "no narrow-frac accumulator remains on any weighted layer").
- `--widen-accum` / `--min-frac` CLI + `widen_accum`/`min_frac` params (default OFF ⇒ prior behavior
  byte-identical); `csim_verify.json` now carries `widen_accum`, `n_layers_widened`, and `accum_audit`.
- `run_convert_r7.sh` extended: `WIDEACC=1` (default) appends `--widen-accum` and stores under
  `rf<N>-wideacc/`; `RFS` default now `"1 8 16"`.

**Emitted THREE RF variants** (VU13P / 2.5 ns / io_parallel / Latency), stored
`bnjettag/r7/results/convert/w1a8-s2/rf{1,8,16}-wideacc/`:
- RF=1 (latency reference), RF=8 and RF=16 (L1-throughput candidates: at 2.5 ns the 40 MHz collision
  rate allows II≤10, so RF 8–16 brackets the deployable envelope).

**Accumulator audit (all three variants): `all_wide=True`, `offenders=[]`.** Every one of the 15
weighted-layer accums now frac 16 — projections `<13,8>→<24,8>`, fc2 `<14,9>→<25,9>` /
`<14,10>→<26,10>`, head_fc2 `<13,8>→<24,8>`; input_proj/fc1/head_fc1 already `<24,8>`/`<25,9>` (frac16).
Structural act×act Einsum accums untouched (scores `<29,19>`, ctx `<40,13>`). Weights still pure ±1;
ReLU outputs still `ap_ufixed<24,8,AP_TRN,AP_SAT,0>`.

**GATE2 (C-sim vs export, n=128): corr = 0.997446, max|Δ| = 0.363 → PASS (≥0.997), identical
RF=1/8/16.** FINDING (flagged, not hidden): this is a small drop from the plain-accum 0.998696, NOT
the "unchanged 0.9987" one might expect. Root cause: the v3 pattern widens the accum AND the layer
**output** precision; in the monolith the widened Wq/Wk/Wv output (`bit_block_*_attn_W*_t`: `<13,8>→
<24,8>`) feeds the downstream act×act score einsum, which re-quantizes a more-precise value — a ~0.0013
perturbation of the table-based softmax/SubLN path. The accum widening itself is bit-exact (±1-MAC over
≤7-frac acts is integer-multiple-of-2^-f). If a cleaner C-sim is wanted, widening the accum ONLY (skip
the output) restores 0.998696 and, per the 2026-07-09 root-cause (accum is the DSP driver), should
still deliver DSP-0 — but the shipped both-widen IS the measured DSP-0 config, so it's the safe ship;
mulder settles it. EBOPs unchanged (5,680,716 — export-side, accum widening is hls-side).

**Expected post-fix DSP (coordinator's model; mulder is the arbiter):** ~12.8k @ RF=1 (all structural
act×act attention), ~1.6k @ RF=8, ~800 @ RF=16. If the reports disagree, that is a finding to flag.
Ship: `scp .../rf<N>-wideacc/hls_prj_rf<N>.tar.gz mulder:~/csynth/ && ssh mulder 'cd ~/csynth &&
./mulder_csynth.sh hls_prj_rf<N>.tar.gz'`.

## 2026-07-14 (full-synthesis branch) — ROUND-7 WHOLE-MODEL hls4ml conversion (the abstract-closing artifact)
**What.** Ran the proven `convert_final.py` chain on the round-7 small binary checkpoint
`r7-small-w1a8-s2` (project `bnjettag-final`, file `small-w1a8-s2/model_best.keras`; cfg
`configs/r7-small-w1a8.json`, hash `ac50da08`, D32/H4/L2/FFN64, 19,201 QAT params, ~190k MACs).
The "one call, whole model" round: every weighted layer ≤ ~2.1k MACs (fc1 = 32×64), ~20x under
the ~40k Latency-einsum LTO wall, so the NATIVE emission (QEinsumDense/QEinsum/QSoftmax as trained,
Latency, io_parallel) converts the whole model in a single project — no Dense re-expression, no
hybrid strategy. Emitted TWO reuse-factor variants, both io_parallel / VU13P (xcvu13p-flga2577-2-e) /
2.5 ns / Latency: **RF=1** (fully parallel, latency-optimal) and **RF=8** (folded). NOTHING here ran
csynth (mulder-only) or the cluster.

**Code (minimal, backward-compatible).**
- `convert_final.py`: parameterized the large-model-specific assumptions — `--config` /
  `config_path` (config JSON path; default `configs/final-<variant>.json` unchanged), `--wandb-subdir`
  (W&B file subdir; `fetch_checkpoint` gained a `subdir=` arg, default `<variant>-s<seed>`),
  `--run-dir`/`--store-root`/`--cache-dir` (results/cache overrides). **All new params default to the
  prior behavior byte-identically** — the FINAL-campaign path is unchanged. No qat/build/port/convert
  semantics touched.
- `run_convert_r7.sh`: new thin wrapper (mirrors `run_convert_final_all.sh`) that drives the two RF
  variants with the r7 config/W&B/store paths; first RF fetches from W&B, later RFs reuse the cached ckpt.

**Gates (both RF variants).**
- GATE2 (hls4ml C-sim vs the exported HGQ2 hardware model, n=128 real jets): **corr = 0.998696**,
  max|Δ| = 0.328, mean|Δ| = 0.0289, bit_exact=False → **PASS** (≥0.997). IDENTICAL for RF=1 and RF=8
  (Latency ⇒ RF sets the einsum `multiplier_limit`/schedule, not the numerics). RF=1 emitted cleanly —
  **no fallback needed**.
- GATE1 (export vs the QAT trained checkpoint, n=4096): corr_scores **0.965**, corr_logits 0.972,
  argmax 0.819 — **below** the 0.9999 threshold. This is the static-grid-substitution + CSD-2-β gap,
  NOT a conversion issue (weight |Δ| vs QAT forward = 2.98e-08 → datapath ±β exact). The FINAL/large
  model has the same status (0.975) and shipped `build_default` anyway (copy_qat's ~0.99 ceiling is
  keras-only — hls4ml drops the 1/β scaler, measured). Small is slightly worse than large (fewer
  heads/dims ⇒ less averaging of the static-grid error). Consequence: full-model C-sim vs the *literal
  trained checkpoint* ≈ 0.965 (export↔QAT gap dominates), NOT the ~0.999 one might expect — see risks.

**Project inspection (rf1 firmware, evidence for the artifact).**
- Automatic attention present: 13 `nnet::einsum_dense` (input_proj + Q/K/V/O + fc1/fc2 ×2) + 4
  `nnet::einsum` (the act×act scores/ctx contractions ×2 blocks) + 2 `nnet::softmax_multidim` + 2
  `nnet::dense` (head) + 11 `nnet::subln`.
- Weight ROMs **pure ±1**: all 15 binary kernels (w4/w9/w11/w17/w23/w29/w33/w39/w41/w47/w53/w59/w63/
  w70/w74) have value-set {−1,+1} only, correct shapes (input_proj 16×32, projections 32×32, fc1/fc2
  32×64/64×32, head_fc2 32×5).
- ReLU **SAT pinned**: all 3 relu outputs `ap_ufixed<24,8,AP_TRN,AP_SAT,0>` (not WRAP) —
  `fix_relu_saturation` confirmed in the emitted defines.h.
- Accumulators: WIDE on the contraction-heavy layers (input_proj `<24,8>` frac16, fc1/head_fc1 `<25,9>`
  frac16, ctx `<40,13>`, gap `<62,14>`, scores `<29,19>`, softmax `<27,8>`); **NARROW-frac `<13,8>`
  (frac5) on the score-feeding projections Wq/Wk/Wv/Wo** and on fc2 `<14,9>`/head_fc2 `<13,8>` — the
  exact BitExact-minimal (score-stream passthrough) regime the 2026-07-09 root-cause tied to DSP-parking
  on the large model. Mulder csynth is the arbiter; the widen-accum fix (`probes_final_v3.widen_dense_accum`)
  applies if DSP shows.

**EBOPs (native HGQ2_native_trace_minmax, exported model).** total = **5,680,716**; the two act×act
attention cores dominate (scores 2×720,000 + ctx 2×1,104,000 = 3,648,000 = 64%); each binary projection
~82k–180k. Recorded in each `ebops.json`.

**Artifacts.** `bnjettag/r7/results/convert/w1a8-s2/{rf1,rf8}/` — `export_verify.json`, `csim_verify.json`,
`ebops.json`, `convert.json`, `hls_prj_rf<N>/`, `hls_prj_rf<N>.tar.gz` (~331 KB, contains `build_prj.tcl`,
no stale `csynth_report.json`). Mulder: `scp .../rf<N>/hls_prj_rf<N>.tar.gz mulder:~/csynth/ && ssh mulder
'cd ~/csynth && ./mulder_csynth.sh hls_prj_rf<N>.tar.gz'`.

## 2026-07-14 (full-synthesis branch) — ROUND-7 held-out ROC eval job BUILT + locally gated
**What.** The ROC test pass for the 30 round-7 checkpoints (r7/README.md). Nothing launched. Files:
- `../jobs/training/kai-roc-r7-gpu.yaml` — mirrors kai-roc-final-gpu.yaml verbatim (python:3.12 +
  tensorflow[and-cuda]==2.21.0 wheels + LD_LIBRARY_PATH-over-nvidia-wheels export, Zenodo val tarball
  with the >1e9 guard, ConfigMap kai-bnf-code code, BAD_NODES-fenced affinity, backoffLimit 2,
  24Gi/4cpu req==lim). Adds the round-7 SIZE axis via **`--run-prefix r7 --sizes small,tiny`**;
  **EVAL_BATCH 4096** (tiny models); deadline **14400**. CKPT_SOURCE=wandb (runs
  `r7-<size>-<variant>-s<seed>`, project bnjettag-final; download-by-basename). Out: 30
  `<SIZE>-<VARIANT>-s<seed>.npz` + roc_auc.md (per-seed + seed-averaged per size×variant, labeled
  "round-7 · era-2 data · ROC-test") + **roc_overlay_SMALL.png + roc_overlay_TINY.png** (FPR log
  axis) -> W&B run **r7-roc-artifacts**.
- `roc_final.py` — extended eval-all with a BACKWARD-COMPATIBLE size axis (NOT a new script): new
  `--sizes` / `--run-prefix` / `--size` / `--table-title` / `--table-source` + helpers `_size_key`,
  `_tag`. size threads through `_wandb_download_ckpt` / `resolve_checkpoint` / `eval_one` /
  `write_auc_table` / `cmd_eval_all` / `cmd_plot`. No --sizes ⇒ size="" everywhere ⇒ the FINAL
  campaign is BYTE-IDENTICAL: npz meta gains no `size` key + campaign stays "final", the roc_auc.md
  has no size column and keeps the FINAL header/source, the single-overlay call is unchanged.
  Round-7 groups seed-averaging per size×variant, adds the size column, and emits one overlay per size.
- `../jobs/training/variants/fetch_r7_results.py` — sibling of the (untouched) fetch_final_results.py;
  pulls `r7-roc-artifacts` -> `bnjettag/r7/roc-results/`.

**Verified locally (`.venv-hgq2`, CPU).** py_compile roc_final.py + fetch_r7_results.py OK; YAML parses
to a single Job with expected name/deadline/backoffLimit/EVAL_BATCH=4096/24Gi-4cpu and all eval-all
args (`--run-prefix r7`, `--sizes small,tiny`, `--expect-n 260000`, round-7 title/source, r7-roc-artifacts);
embedded upload heredoc + 3 inline `python -c` snippets compile; bash -n clean; real argparse `--help`
lists every new flag. BUILD→SAVE→FRESH-LOAD→EVAL round-trip: built r7-small-w1a8 QAT model (count_params
**19,201**) → `model.save` .keras → reload via `roc_final.load_final_model` → `eval_one` → **SMALL-W1A8-s1.npz**
(keys y/score/meta; meta size=SMALL, campaign=round7, key=W1A8, n asserted). FINAL byte-identity proven two
ways: `write_auc_table` output == an embedded golden of the pre-round-7 function, and a size="" `eval_one`
meta has NO `size` key + campaign="final". Round-7 table check: size column present, 10 size×variant
seed-averaged rows, "round-7 · era-2 data · ROC-test" header. NOT launched — hand to verification; the
cluster prereq is `make_code_configmap.sh` (repack kai-bnf-code so the pod gets the new roc_final.py).

## 2026-07-14 (full-synthesis branch) — ROUND-7 deployable-scale QAT matrix BUILT + locally verified
**What.** The round-7 training matrix on THIS QAT stack (decisions.md 2026-07-13, r7/README.md). Two
sizes × 5 variants × 3 seeds = 30 ready-to-submit Jobs; nothing launched. Files:
- `configs/r7-{small,tiny}-{fp32,w8a8,w1a8,w1a6,w1a4}.json` (10) — copied from `final-*.json`, arch
  overridden (small **D32/H4/L2/FFN64**, tiny **D16/H2/L2/FFN32**), recipe overridden (peak LR **2e-5**,
  warmup 1 + linear decay **100**, ES patience **15**, epochs 101). `act_calib=trainable` kept on all
  quantized variants; quant/hls otherwise inherited.
- `../jobs/training/variants/gen_r7_jobs.py` — adapted from gen_final_jobs.py (that file untouched).
  Emits 30 `kai-bn7f-<size>-<variant>-s<n>.yaml` (runs `r7-<size>-<variant>-s<seed>`, project bnjettag-final),
  `preflight_r7.sh` into BOTH variants/ and here (ships in ConfigMap kai-bnf-code), `launch_r7_staged.sh`
  (6 waves of 5 = one full precision sweep per size×seed). Right-sized ask 1 GPU/2 cpu/8Gi/16Gi-eph
  (req==lim), deadline 14400; GPU allowlist widened 13→18 (adds A10, L4, RTX-2080-Ti, Tesla-V100
  SXM2+PCIE-32GB) and PREFERS the abundant mid-tier (A100s de-preferred, flagship last); Pascal excluded,
  BAD_NODES kept. `make_code_configmap.sh` unchanged (tar 132 KB < 1 MB, now carries the r7 configs +
  preflight).
- `run_stage.py`: added optional **`--out-dir`** (default None ⇒ prior behavior byte-unchanged, final
  campaign never passes it) so r7 checkpoints land under `<size>-<variant>-s<seed>/model_best.keras`.
  This QAT preflight_r7.sh SUPERSEDES the QKeras-path one that gen_round7_jobs.py wrote (kai-bn7p/d probes).

**Verified locally (`.venv-hgq2`, CPU).** All 10 build; count_params **quantized 19,201 (small) / 5,345
(tiny)**, fp32 skeleton 19,075 / 5,219 (Δ126 = weight+act quantizer bit-configs) — preflight hard-asserts
these (build-gate exit 0). Binary gate exact `{−β,+β}` symmetric, no zero, on all 15 bit-layers per w1a*.
py_compile OK; 30 YAMLs parse + structural asserts pass; preflight copies identical; embedded python (2
heredocs + 3 inline) compile; bash -n clean. `train --smoke` on r7-small-w1a8 (synthetic era-2 h5, fresh
CLI subprocess w/ `--out-dir`) exit 0. `.keras` reload byte-identical across two independent fresh processes
(predictions + weights, np.array_equal). NOT launched — hand to verification, then cluster.

## 2026-07-10 (full-synthesis branch) — HYBRID-strategy WHOLE-MODEL emission: monolith-v2 + probe_block
**Route A/B of FULL-SYNTHESIS.md, both C-sim-GATED and mulder-ready.** The failed 27M-IR
monolith (convert_final, all QEinsumDense at Latency) was weight-INLINING-dominated. New emission
re-expresses every WEIGHTED layer as the v2/v3 per-shape recipe COMPOSED inside one graph, so
weights go to BRAM ROM (bounded clang-LTO) instead of inlining. New `bnhgq2/monolith.py` +
drivers `monolith_v2.py` (Route A) and `probe_block.py` (Route B); one shared block emitter.

**The mechanism (all measured locally, `.venv-hgq2`, Vitis-backend C-sim):**
- A per-token linear projection = a keras `QDense` on the (T,in) token tensor → hls4ml parses
  it as **PointwiseConv1D** (NOT nnet::dense). At **Strategy=Resource** the ±1 kernel is a weight
  ROM (`load_weights_from_txt`) — the LTO-bounding property (v2's whole point) carries over to
  the composed graph. head_fc1/head_fc2 (post-GAP, 2-D) stay genuine `nnet::dense`.
- Head split/merge that the einsum equations did implicitly (`btd,dhe->bthe`) becomes explicit
  keras `Reshape`, which hls4ml **fuses away in io_parallel** (0 residual layers, bit-exact).
- **Mixed strategy in ONE project WORKS**: Resource pointwise-conv weighted layers + **Latency**
  weightless attention einsums (QKᵀ, attn·V) + softmax, via per-layer `LayerName` overrides. The
  einsum Latency-only assertion fires during optimizer passes (BEFORE post_parse) → strategy MUST
  be set in the hls_config (`convert(layer_configs=…)`, new param), not post_parse.
- input_proj's positional-encoding (T,D) additive — the einsum held it in `bias_axes='td'`, a
  PointwiseConv1D's per-channel bias CANNOT — reinstated by `Reshape (T,D)→(T·D,)` → axis=-1
  QBatchNormalization with (T·D,) per-element params (γ=CSD-2(β), β=(bias_td+PE) flat) → Reshape
  back. Bit-exact form (QBatchNormalization takes only a scalar axis, so the flatten trick is how
  you get a per-(t,d) affine). Leaves a tiny ~3e-4 seed vs the flagship's dense-bias-TABLE
  quantization (affine quantizes the additive directly; flagship pre-divides by β̃ then rescales).
- **PointwiseConv1D at Resource asserts RF ≤ filt_width·n_chan = in_channels** (nnet_conv1d_resource.h):
  input_proj (in=16) needs a per-layer RF cap of 16; the in≥256 layers keep RF=256. New
  `per_layer_configs` emits both the Latency + RF-cap overrides. head denses (2-D) are uncapped.
- `widen_conv_dense_accum` = v3's `widen_dense_accum` generalized to PointwiseConv1D/Conv1D
  (frac≥16 → the ±1 MAC LUT-maps to 0 DSP). ReLU-SAT (`fix_relu_saturation`) unchanged.

**monolith-v2 (Route A) — the ENTIRE flagship in one project, GATE PASS:**
- Full-model **C-sim hls4ml(mono) vs keras(mono) corr 0.99793** (n=96, max|Δ| 0.851, ≥0.997 PASS;
  ReLU-SAT patched 9, wide-accum widened 51 — matches the flagship's own GATE2 ~0.999 class).
- Project inspection (all as intended): **nnet::einsum_dense = 0** (the win — no big-shape Latency
  einsum-dense remains), 49 PointwiseConv1D + 2 Dense at Resource, 16 act×act nnet::einsum + 8
  softmax at Latency, 138 weight ROMs, ReLU outputs `ap_ufixed<26,10,AP_TRN,AP_SAT>`, wide accums
  (Wq/fc1 `ap_fixed<27,11>`, the proven 0-DSP form), weight_t `ap_fixed<2,2>` (pure ±1).
- Keras reformulation fidelity (context, NOT the hardware gate): keras(mono) vs flagship EXPORT
  corr_scores **0.9943** / argmax 0.961; vs QAT 0.9632 — mirrors the flagship's OWN characterized
  export↔QAT GATE1 (0.9697); the reformulation increment is small (per-shape block-iso 0.9999987)
  and accumulates only through the 8 sensitive softmaxes.
- Size indicator vs the v1 (all-einsum-Latency) monolith: weights **105.6 MB → 45.4 MB**;
  top .cpp/.h 8268 → 6945 lines. The decisive delta is ROM-vs-inline (mulder is the synthesis arbiter).
- Deliverable: `results/final/runs/9676418a/monolith-v2/{hls_prj_rf256.tar.gz (6.6 MB),
  monolith_v2_verify.json}`. Mulder: `./mulder_csynth.sh hls_prj_rf256.tar.gz`.

**probe_block (Route B) — flagship block-0 end-to-end, GATE PASS:**
- SubLN→QKV→QKᵀ→QSoftmax→attn·V→SubLN→Wo→residual→SubLN→fc1→ReLU(SAT)→SubLN→fc2→residual.
- **C-sim hls4ml(block) vs the FLAGSHIP's block-0 on REAL tapped activations corr 0.99968**
  (n=48, max|Δ| 0.177, ≥0.997 PASS); keras(block) vs flagship block-0 = 0.9999933.
- A lossless identity affine (γ=1,β=0, 2^-16 grid) conditions the input: hls4ml BitExact raises
  'unexpected input layer chain' on Input→Merge (residual Add fed directly by Input) — only bites
  the standalone block (in the monolith the residual always enters from an internal BatchNorm).
- 6 PointwiseConv1D + 2 einsum + 1 softmax + 4 SubLN, einsum_dense=0, 16 weight ROMs. 8 instances
  + input/head compose the whole model at RTL/IP level (measured, not estimated).
- Deliverable: `results/final/probes/{probe_block0_rf256_hybrid.tar.gz (1.1 MB),
  probe_block0_manifest.json}`. Mulder: `./mulder_csynth.sh probe_block0_rf256_hybrid.tar.gz`.

**One shared-code change, backward-compatible:** `convert()` gained `layer_configs` (per-layer
LayerName merge; default None → single-strategy path byte-identical). build.py/port.py/qat.py/
train.py untouched (monolith.py copies build.py's attention quantizer helpers, guarded by the
C-sim gate). csynth NOT run here (mulder ships/runs).

## 2026-07-07 (mid-campaign) — A4 activation-calibration rescue (trainable-scale grids)
**ROADMAP divergence rule (per-variant rescue, logged, healthy variants untouched).**
Campaign evidence (real runs): w1a4-s2/s3 COMPLETED NaN-free (the clipvalue fix held) but
plateaued at best val_macro_auc **0.5825 / 0.6049** — far below r5's A4 (~0.73); w1a8-s2
healthy (0.8381 @ ep27, r5 range). Diagnosis (coordinator): A4's 16-level activation grids,
MSE-calibrated on random-init activations then FROZEN, go misaligned as training drifts the
distributions; A8's 256 levels absorb the drift, A4 can't.
- **Fix (mechanism 1, primary, HGQ2-native): trainable-scale, FIXED-bit-width act quantizers.**
  New `qat._trainable_act`: KBI datalane, k0=1, `b0=act_bits-1`, SAT, `bc=Constant(act_bits-1)`
  pins total bits, `ic=MinMax(0,act_bits-2)` keeps f>=1; `trainable=True` so the integer split
  `i` trains WITH the model while `f = b - i` derives — the grid re-aligns as distributions
  drift. Why KBI not KIF: KIF's i and f are independent (no fixed total); KBI's (b,i) gives a
  pinned width with a trainable scale. Why SAT not WRAP: SAT clips exactly as the exported
  grid, so the checkpoint IS the hardware model; at export the final (i,b) reads out as a
  static SAT<act_bits,1+i> grid — identical in form to build.py.act_q, convert.py unchanged.
- **Fix (mechanism 2, fallback): scheduled recalibration.** `train._recalib_callback` re-runs
  the MSE calibration on the DRIFTED activations at `act_recalib_epochs` (e.g. [2,5]) then
  re-freezes. Cheap, no quantizer surgery.
- **Knobs (quant section):** `act_calib` = "frozen" (default; ABSENT key => frozen, so healthy
  configs are byte/hash-IDENTICAL — verified fp32/w8a8/w1a8/w1a6 hashes unchanged) | "trainable"
  | "recalib"; `act_recalib_epochs: [int]`. **final-w1a4.json set to "trainable"** (hash
  c90f41bf -> fb70cca0, a NEW experiment); healthy configs untouched.
- **Local gates (CPU, all PASS):** py_compile; trainable adds +150 trainable params (act
  scales); FROZEN grids move 0/83 over 20 steps (default unchanged), TRAINABLE moves 8/83 —
  a badly-init v-stream corrected (i,f) (6,-3)->(2,1); fixed-width invariant i+f==act_bits-1
  holds on ALL 75 trainable KBI sites (0 violations); binary gate still exact {-beta,+beta}
  (51 layers); `.keras` save -> FRESH-process reload byte-identical (max|delta|=0.0); full
  `train()` smoke on synthetic data (act_calib=trainable) exits 0 with the drift-tracking
  print + `act_grid_before/after` in train_meta.json; recalib callback fires at [2,5], skips
  others. NOT launched — relaunch procedure handed to the coordinator.
- **Coordination:** left the convert-driver helpers in qat.py untouched (merged around).

## 2026-07-09 (later) — ROOT-CAUSED the v2 Resource DSP split: it's the ACCUM precision, not the weights
mulder v2 csynth split — fc1/head_fc1/input_proj = **0 non-norm DSP** (fc1 at 262k MACs!), qkv/Wo = **270
(256 dense + 14 SubLN)**. Diagnosed from the emitted projects + the csynth XMLs (scratchpad/v2xml/):
- **NOT the weights** (coordinator's hypothesis): all three ROMs are pure ±1, `weight_t ap_fixed<2,2>`,
  same `ap_fixed<8,3>` input, same `nnet::dense_resource` kernel. **The only difference is the dense
  ACCUMULATOR**: fc1 `ap_fixed<27,11>` (16 frac) → Vitis LUT-maps the ±1 MAC → **0 DSP** (111k LUT);
  qkv/Wo `ap_fixed<16,11>` (5 frac) → DSP48 → **256 DSP** (15k LUT). BitExact sets the accum to the
  DOWNSTREAM request — fc1 feeds a ReLU/LN (wide), the score projections feed the exact-passthrough
  score grid (narrow). **REAL, not a probe artifact:** the FULL flagship's Wq/Wk/Wv/Wo all carry
  `ap_fixed<16,11>` (checked by converting the monolith and reading defines.h).
- **Fix (v3): `widen_dense_accum` (frac ≥ 16, keep integer) → the fc1 0-DSP form, bit-exact.** The
  accum is only more precise; the downstream re-quantizes anyway. Verified: overriding qkv/Wo accum →
  `ap_fixed<27,11>` module (the proven-0-DSP variant), C-sim 0.9999999/0.9999976. So it is a genuine
  **DSP↔LUT trade** (0 DSP/111k LUT vs 256 DSP/15k LUT), NOT the frontend 1-bit-emission gap.
- **Verdict for §6″: the ENTIRE final stack CAN synthesize DSP-free (all DSP in the norm)** by widening
  the 4L attention-projection accumulators — no frontend work needed. New `probes_final_v3.py` emits the
  0-DSP form of qkv/Wo/fc2: `probe3_dense_*_rf256_resource_wideacc.tar.gz` (3), manifest v3, all C-sim
  bit-exact, generated accums qkv/Wo `ap_fixed<27,11>` and fc2 `ap_fixed<30,14>`. v1(9)/v2(6) untouched.
  constraints_map.md 2026-07-09 addendum updated with the named cause. csynth NOT run here.

## 2026-07-09 — v2 csynth probes: big binary shapes as 2-D QDense at RESOURCE (Latency einsum LTO wall)
Overnight mulder csynth of the v1 probes: `probe_bitlinear_head_fc2` (256→5, 1.3k MACs) synthesized
rc=0 in 6 min, but `attn_qkv`/`attn_Wo` (65k MACs) and `ffn_fc1` (262k) ALL hit the 3 h timeout with
`clang -cc1` at 100 % CPU, never reaching a report — **LTO non-convergence, root-caused to EinsumDense
being Latency-only** (at Latency the folded RF doesn't shrink the inlined-constant weight volume).
Recorded as a constraints_map.md 2026-07-09 addendum.

**Fix = new `probes_final_v2.py`: the big binary shapes re-emitted as 2-D QDense at Resource/RF256**
(the §6′-proven route — Resource folds weights into ROM, LTO bounded). Each `...,in->...,out`
flattens the token axis (einsum btf,fd->btd ≡ per-token Dense), carrying the flagship's EXACT config
(±1 sign matrix from binz, KBI act grid via `.kif`, ReLU-SAT). **Resource 1-bit-trap avoidance APPLIED**:
pure ±1 datapath + β in a FROZEN CSD-2 QBatchNormalization affine (bias_fold for fc1/head_fc1) — NEVER
in-weight ±β (the ROM-operand 256-DSP trap). 6 probes, all local-C-sim vs the flagship sublayer on
real activations (hls4ml↔flagship) + a keras↔flagship reproduction check — **all essentially bit-exact:**
- dense_input_proj 16→256 csim 0.9999987 repro 1.0 · attn_qkv 256→256 0.9999999/1.0 ·
  attn_Wo 256→256 0.9999975/1.0 · ffn_fc1 256→1024 0.9999985/1.0 · ffn_fc2 1024→256 0.9999989/1.0 ·
  head_fc1 256→256 0.9999997/1.0
Spot-verified ffn_fc1 tarball: Strategy **Resource** / RF 256, **nnet::dense** (2-D, ZERO nnet::einsum),
±1 weights {−1,+1}, nnet::subln present, ReLU `ap_ufixed<26,10,AP_TRN,AP_SAT,0>`. DSP expectation
recorded BOTH ways per the 1-bit-emission gap: binary dense = 0 DSP IF hls4ml emits true 1-bit weight
types, else ~256 (270 with SubLN) — mulder is the arbiter; SubLN = expected consumer; affine/ReLU = 0.
Output: 6 `probe2_dense_*_rf256_resource.tar.gz` (v1 NOT touched; input_proj/qkv/Wo/fc1/fc2/head_fc1)
+ `probes_manifest_v2.json` in `results/final/probes/`. Mulder cmd in the manifest. csynth NOT run here.
(head_fc2 already synthesized at Latency — no v2 needed; attn_core rf64 left running on mulder.)

## 2026-07-08 — per-shape csynth PROBES emitted from the FINAL flagship (monolith OOM'd)
Whole-model csynth verdict: NEGATIVE — vitis_hls OOM-killed during Unroll/Inline at 27,023,107
instructions (~24 h, mulder 125 GB); even folded RF=256 at D256/L8 exceeds Vitis 2023.2. So the
deployable route is the §6' per-shape probe methodology — new `probes_final.py` generates **9
probes SLICED from `build_export(final-w1a8-s3)`**, so each carries the final stack's EXACT config
(KBI act grids via `.kif`, ReLU unsigned-WRAP→SAT pins, CSD-2/frozen-affine β, range-reduced SubLN)
— no re-porting, the probe IS a sub-graph of the shipped model (keras.Model(subln.input, end.output)).

Probes (Vitis / VU13P / 2.5 ns; RF=256 Latency except attn_core RF=64), each with local C-sim vs the
sliced keras sublayer on REAL tapped activations — **all essentially bit-exact**:
- bitlinear_input_proj (F16→D256, SubLN on raw feats) 0.9999990 · attn_qkv (D256→(H8,E32)) 0.9999998 ·
  attn_Wo ((H,E)→D256) 0.9999975 · ffn_fc1 (256→1024 +ReLU-SAT) 0.9999985 · ffn_fc2 (1024→256)
  0.9999989 · head_fc1 (256→256 +ReLU-SAT) 0.9999997 · head_fc2 (256→5) 0.9999997
- subln_256 (folded 1/√) 1.0000000 · attn_core_rf64 (QKᵀ→QSoftmax→attn·V, T10/H8/E32) 1.0000000

(Per-shape C-sim ~1e-6 — far tighter than the accumulated full-model 0.99892, confirming the
monolith residual is just the 8-block accumulation of these tiny per-layer table deltas.) Spot-verified
in the fc1 tarball: pure ±1 weights {−1,+1}, ReLU `ap_ufixed<26,10,AP_TRN,AP_SAT,0>` (no WRAP),
nnet::einsum_dense + nnet::subln present, Strategy Latency / RF 256, build_prj.tcl present.
DSP expectations per probe (manifest): binary matmuls MUST be 0 DSP (±1 inlined, Latency); CSD-2
affines 0 DSP; ReLU 0 DSP; **SubLN = the expected DSP consumer**; attn_core einsums are act×act so
DSPs there are EXPECTED and counted (§6' rf64 ≈ 820 DSP). Output: 9 tarballs (175–384 KB) + manifest
in `results/final/probes/`; `mulder_csynth.sh <9 tarballs>` list in the manifest. csynth NOT run here
(mulder ships/runs). `probes_final.py` reuses convert_final (fix_relu_saturation, patch_resource_einsum_check).

## 2026-07-07 (convert driver, FLAGSHIP + fidelity table) — w1a8-s3 GATE-2 PASS; prediction held
Flagship `final-w1a8-s3` (val macro-AUC **0.85337**, trainable KBI, current config hash **9676418a**)
+ `final-w1a6-s3` (0.83851, hash b25daa1d) through the full chain (`--wandb-run` mode). Stores in
`results/final/runs/9676418a/w1a8-s3/` and `results/final/runs/b25daa1d/w1a6-s3/`.

**GATE-2 PREDICTION HELD: 8-bit + non-degenerate → PASS.** Three real trainable articles now form a
clean, monotonic fidelity table (n=4096 gate-1 / n=128 gate-2; weight |Δ|=7.5e-9 every time):

**COMPLETE 3-seed × 3-variant table (9 real trainable articles; n=4096 gate-1 / n=128 gate-2;
weight |Δ| = 7.5e-9 exact on all).** GATE1 = export↔QAT (build_default, shipped); GATE2 = hls4ml
C-sim ↔ export (SAT fix, ≥0.997 = PASS):

| article | AUC | GATE1 | exact-β | GATE2 | pass | EBOPs |
|---|---|---|---|---|---|---|
| w1a8-s1 | 0.8540 | 0.9749 | 0.9852 | 0.99886 | ✓ | 661,968,076 |
| w1a8-s2 | 0.8442 | 0.9729 | 0.9670 | 0.99969 | ✓ | 662,016,716 |
| w1a8-s3 | 0.8534 | 0.9697 | 0.9854 | 0.99892 | ✓ | 662,500,556 |
| **w1a8** mean | | 0.9725±0.0022 | 0.979 | **0.99916 ± 0.00038** | **3/3** | 662,161,782 |
| w1a6-s1 | 0.8364 | 0.9780 | 0.9809 | 0.99726 | ✓ | 511,790,094 |
| w1a6-s2 | 0.8359 | 0.9816 | 0.9853 | 0.99652 | ✗ | 511,838,734 |
| w1a6-s3 | 0.8385 | 0.9694 | 0.9837 | 0.99719 | ✓ | 511,838,734 |
| **w1a6** mean | | 0.9763±0.0051 | 0.983 | **0.99699 ± 0.00033** | **2/3** | 511,822,520 |
| w1a4-s1 | 0.7626 | 0.9697 | 0.9702 | 0.99634 | ✗ | 363,153,232 |
| w1a4-s2 | 0.7609 | 0.9702 | 0.9704 | 0.99851 | ✓ | 363,153,232 |
| w1a4-s3 | 0.7634 | 0.9438 | 0.9506 | 0.99709 | ✓ | 363,201,872 |
| **w1a4** mean | | 0.9612±0.0123 | 0.964 | **0.99731 ± 0.00090** | **2/3** | 363,169,445 |

**Reading (corrects the earlier single-seed claim): A8 clears 0.997 on ALL 3 seeds (mean 0.99916)
— the flagship precision is robust. A6 and A4 sit RIGHT ON the 0.997 boundary (means 0.9970 /
0.9973), each with 2/3 seeds passing and one seed dipping to ~0.9965; the per-seed spread
(±0.0003–0.0009) is comparable to the A4-vs-A6 gap, so A4 ≈ A6 < A8 — NOT a clean monotone in act
bits (that was a single-seed artifact).** The C-sim residual is the table-SubLN ↔ fixed-point-grid
straddle, clearly smaller at 8-bit than at 4/6-bit; no wraps anywhere (ReLU-SAT held on all 9).
EBOPs monotonic in act bits and seed-stable (A4 363.2M, A6 511.8M, A8 662.2M; σ ≤ 240k). GATE-1 exact-β and copy_qat also climb with bits
(csd2 β Δ grows to +0.0157 at A8 — worth exact-β if affine DSPs prove free on mulder).

**Automatic-conversion claim CONFIRMED from the converted flagship project** (thesis point): the whole
model incl. attention converted with NO manual per-block assembly — `myproject.cpp` has **16
`nnet::einsum`** (QEinsum: 8 QKᵀ + 8 attn·V), **8 `nnet::softmax_multidim`** (QSoftmax, 1/block),
**49 `nnet::einsum_dense`** (QEinsumDense: input_proj + all Q/K/V/O + fc1/fc2), **35 `nnet::subln`**,
2 `nnet::dense` (head). **Mulder handoff:** Strategy=Latency, ReuseFactor=256, part
xcvu13p-flga2577-2-e, clock 2.5 ns; precision pins that matter — relu-family outputs
`ap_ufixed<26,10,AP_TRN,AP_SAT,0>` (the ReLU-SAT fix is IN the tarball, verified in defines.h),
custom `nnet_subln.h`, softmax tables `ap_ufixed<12,1,AP_SAT>`. Tarballs ready (5.8–6.1 MB each).
Frozencal seeds (act_calib=frozen) skipped — different config hash, off the trainable-cohort table.

## 2026-07-07 (convert driver, first REAL article) — final-w1a4-s1 through the full chain; KBI read-back fixed
First non-degenerate checkpoint (`final-w1a4-s1`, val macro-AUC **0.76257**, config hash fb70cca0,
`act_calib='trainable'`) run end-to-end via `convert_final.py --wandb-run` mode; store in
`results/final/runs/fb70cca0/w1a4-s1/`.

**KBI read-back FIXED (was validated on frozen-KIF only).** The trainable-scale rescue made the act
quantizers KBI (`qat._trainable_act`: k=1, b=act_bits-1 pinned, i TRAINED). `FixedPointQuantizerKBI`
has `_k/_b/_i` (no `_f`), and `_i` is a CONTINUOUS latent (e.g. 1.49), not the integer bits. Fixed
`read_qat_act_ibits` to read the effective grid via **`quantizer.kif`** — the SAME (k,i,f) hls4ml
reads (`keras_v3/hgq2/_base.py`) — so export grid == converted grid == QAT forward grid. Verified on
w1a4-s1: 51/51 layers `round(raw _i) == kif.i`, every width == act_bits(4), effective f = act_bits-1-i
(build.py.act_q reconstructs exactly). Backward-compatible with frozen KIF (kif.i == _i). Distinct
act i across the 51 layers = {1, 2}. Weight export still exact (|Δ| 7.5e-9).

**Real-article gate numbers (n=4096 gate-1 / n=128 gate-2):**
- GATE1 export↔QAT: **0.9697** (build_default) / 0.9702 (exact-β) / **0.9802** (copy_qat ceiling).
  Higher than the degenerate smoke w1a8 (0.9618) — confirms the attention-grid-substitution gap is
  real but the smoke head amplified it; copy_qat demo gate-2 = **0.065** (scaler drop, dramatic at A4).
- GATE2 hls4ml C-sim (SAT fix): **0.99634** (max|Δ| 0.85) — up from smoke 0.9951, confirming the
  degeneracy attribution. **ReLU-SAT HELD** (no wrap on the real model). Per-block bisect: input_proj
  **bit-exact (1.000000)**, all 8 blocks + GAP + head = 0.996–0.999 with NO catastrophic drop — the
  ~0.996 residual is SPREAD UNIFORMLY (not head-localized like the smoke), i.e. **A4's coarse 4-bit
  grids** (16 levels, step ~0.25–0.5) straddling the near-exact-not-bit-exact table-SubLN at every
  SubLN→dense boundary. This is the hardest case (A4) and it is NOT a bug. Finer-grid w1a8 (flagship)
  is expected to clear 0.997.
- EBOPs (HGQ2 native): **363,153,232** (A4; ~0.55× the smoke w1a8 654M — EBOPs scale with act bits).
- copy_qat characterization/demo guarded (try/except) so the optional keras-only path can never block
  the shipped gates; `match_attention_to_qat.set_grid` now reads `src.kif` (KBI-safe).

## 2026-07-07 (convert driver, gate fixes) — GATE-2 ReLU-wrap FIXED; GATE-1 copy_qat measured
Both gates root-caused and addressed on the same `final-smoke` w1a8 article; store refreshed.

**Fix 2 (GATE 2, the big win): the FFN/head ReLU unsigned-WRAP.** `keras.layers.ReLU` is parsed
by hls4ml 1.3.0 as `ParametrizedActivation('thresholdedrelu')` with an UNSIGNED **WRAP** output
precision (`ufixed<26,10>`); a negative input wraps to ~2^i instead of clamping to 0. The binary
FFN's fc1 is a 256-wide ±1 contraction of a ±4-clipped SubLN input → spans ±O(2^10), so ~half the
ReLU outputs (the negatives) explode to ~1024 (element-wise: keras=0, hls=1023.999). **Fix:**
`convert_final.fix_relu_saturation` sets the relu-family output `saturation_mode=SAT` via a new,
backward-compatible `convert(post_parse=…)` hook (default None → rebuild path byte-identical).
Full-model C-sim: **−0.06 → 0.9951** (max|Δ| 0.107). Bisected post-fix: the whole pipeline is
bit-accurate through 8 blocks + GAP (**gap corr 0.99996**); the residual sits only at `head_fc2`
(256→5), where max|Δ| steps 0.91→1.06 — the table-based SubLN's ~1e-3 error straddling the COARSE
smoke head grid (i=3, step 0.0625) across a 256-contraction, amplified by the degenerate smoke head
(logit std 3.4). Not a bug; a real (non-degenerate, finer-grid) checkpoint clears 0.997. Only lever
left is SubLN 1/√ table width (subln.py, shared) — deferred to keep the frozen rebuild bit-identical.

**Fix 1 (GATE 1): `attn_grids='copy_qat'` opt-in flag + the hardware boundary, MEASURED.** copy_qat
reproduces the TRAINED attention/softmax stream grids under the β-fold via quantizer `scaler=1/β`
→ gate-1 **0.9618 → 0.9901**. But hls4ml's HGQ2 frontend `extract_fixed_quantizer_config`
(keras_v3/hgq2/_base.py) reads only the internal `kif` + SAT/RND — it **ignores `q.scaler`**
(read the source AND measured it: converting a copy_qat export gives gate-2 **0.9598**, i.e. hls4ml
reverts to the base grids while keras uses 1/β). A scaler-FREE power-of-2-shifted copy is worse than
default (gate-1 0.949 — reproduces QAT's coarse SATURATING grids and rounds the non-integer β badly).
Net: copy_qat is a genuine keras-side gate-1 gain but NOT hardware-faithful on this hls4ml; the
default `attn_grids='build_default'` ships. Exact hardware reproduction needs non-power-of-2 scale
support in that frontend or ±β projection weights (DSP cost). All logged as constraints_map.md
2026-07-07 addendum rows.

**Non-regression PROVEN:** the two shared-code touches are no-ops for the frozen rebuild —
`convert(post_parse=None)` default is unchanged; `port.py` shape[1]→shape[-1] is identity for the
rebuild (all 18 era2-large explicit-fold kernels are 2-D: shape[1]==shape[-1], verified by extract).

## 2026-07-07 (convert driver) — FINAL hls4ml conversion driver built + smoke-verified
New `convert_final.py` (+ `run_convert_final_all.sh`): trained QAT checkpoint -> W&B fetch
-> load -> hardware-faithful export -> GATE1 (export↔QAT) -> hls4ml convert (Vitis, attention
included) -> GATE2 (C-sim) -> native EBOPs -> mulder tarball, results per config-hash in
`results/final/runs/<hash>/<variant>-s<seed>/`. Ran the WHOLE chain on the real `final-smoke`
w1a8-s1 checkpoint (`.venv-hgq2`, CPU). The export REUSES the proven build.py/port.py/convert.py
path, sourcing binz (== qat.bitnet_binary_ste, exact), the folded PE table, and the STATIC act
grids read back from the trained QAT quantizers.

**One backward-compatible fix to the shared path (NOT qat/train/build):**
- `port.py` affine width `e["shape"][1]` -> `e["shape"][-1]`. QKeras-path kernels are 2D so
  shape[1]==shape[-1]; the QAT export carries einsum-NATIVE kernels (Wo is 3-D (H,E,D)), so
  shape[1]=E=32 mis-sized the Wo affine -> crash. Identity for the rebuild, fixes the QAT export.

**Gate results on the smoke ckpt (n=4096 real era-2 jets), reported honestly:**
- Weight export EXACT: max|β·q_export − bitnet_binary_ste(latent)| = 7.5e-9 (datapath is the
  exact {−β,+β} the QAT forward computes). β-mode is nearly irrelevant to fidelity.
- **GATE1 (export↔QAT) BELOW the 0.9999 target**: corr_scores 0.9618 (csd2) / 0.9631 (exact-β)
  / **0.9901 (attention-matched ceiling)**. Root cause (block-tapped): build.py sizes the
  softmax/attention-stream STATIC grids independently (and finer in places) rather than
  bit-copying the QAT quantizers; per-block attention corr ~0.998 accumulates over 8 blocks and
  is amplified by the near-degenerate smoke head (argmax agreement ~0.81, val AUC 0.53).
  Reproducing the QAT attention grids in the export via quantizer `scaler=1/β` (base(x/scaler)*
  scaler exactly undoes the β-fold) lifts it to 0.9901 — a KERAS-SIDE ceiling only (a scaler on a
  datalane grid is untested through the Vitis BitExact path, so it is NOT carried into the
  converted project). So Decision-1's "gate-1 ≈ 1.0 by construction" holds for WEIGHTS + dense
  act grids but NOT the attention softmax/stream tables under the β-fold.
- **GATE2 (hls4ml C-sim of the FULL monolith) FAILS: corr −0.06.** Two hls4ml realities forced
  the path: (a) `EinsumDense` is **Latency-strategy-only** (Resource raises `EinsumDense layer
  only supports Latency strategy for now`) — so the full model cannot use Resource folding; (b)
  the Vitis `ValidateResourceStrategy` QoR-check crashes on EinsumDense (matches the 'Dense'
  substring, `get_layer_mult_size` returns n_in=None) — guarded in the driver. C-sim bisection
  (rf-independent, identical at rf1 and rf256): input_proj bit-exact (0.9999994); block-0
  ATTENTION bit-exact (Wq 1.0, softmax 0.99986, Wo 0.99998, add_attn 0.99999); **the break is
  the FFN ReLU: fc1 0.99994 -> ReLU corr −0.68, max|Δ| = 1023.999 ≈ 2^10** — the ReLU/downstream
  BitExact output precision WRAPS the wide fc1 contraction. The smoke model's coarse i=2 act grid
  (degenerate calibration) clips the SubLN input to ±4, and the 256-wide ±1 contraction reaches
  ±1024, tripping the wrap. NEXT: pin a wider activation output precision on `*_ffn_act`/`head_act`
  (or fix the BitExact ReLU produce_kif) and re-C-sim; the established per-block/per-shape probe
  path (already C-sim bit-exact, §6′) remains the mulder deliverable until the monolith C-sim is
  green.
- EBOPs (native HGQ2 trace_minmax): 654,303,246 (vs the rebuild A8 634.7M — same class; smoke
  calibration differs). hls4ml compat/env unchanged; no csynth run (mulder-only, later).

## 2026-07-07 (later) — FINAL training jobs converted to PVC-FREE (cephfs blocks GPU nodes)
Cluster reality changed the launch plan. **Evidence (measured 2026-07-07):** the kai-data
cephfs PVC filters training pods OFF the GPU nodes — the smoke pod (with PVC) pended 75+ min,
while an A/B probe pod with IDENTICAL affinity+resources but NO PVC scheduled in ~4 min onto
hcc-chase-shor-c4709.unl.edu (a 3090 node). CPU pods with the PVC schedule fine (preflight/
sync did). Verdict: cephfs-PVC + GPU node = blocked (post-incident CSI tail).
- `gen_final_jobs.py` rewrites the 15 `kai-bnf-*.yaml` + `kai-bnf-smoke.yaml` **PVC-free**
  (pattern: kai-roc-r5-pvcfree): **no PVC volume/mount**; DATA from Zenodo
  (hls4ml_LHCjet_150p_train.tar.gz → /work/data, `>3e9` byte guard, extract, **delete tarball**
  before training, `find` the jetImage_*.h5 dir → `BNHGQ2_TRAIN_DATA`); CODE from ConfigMap
  `kai-bnf-code` (untar /cmcode/hgq2.tar.gz → /work/code, md5 echoed for provenance); OUTPUTS
  to emptyDir `/work/outputs` with durability via train.py's W&B `wandb.save` (verified it
  uploads BOTH model_best.keras + train_meta.json and needs no /data). eph 24Gi req==lim
  (tarball+extract+pip), cpu 4 / mem 24Gi kept. Smoke uses the SAME Zenodo path (real
  data-path test; --smoke just subsets files).
- New emitted `make_code_configmap.sh` (packs code/hgq2 → ConfigMap, md5, 1 MB-limit guard,
  `--dry-run=client | apply`) = **THE training prereq** (replaces the sync as the code source;
  `sync_code_to_pvc_final.sh` kept but only for the PVC ROC job). `preflight_final.sh` now
  PVC-free too (ConfigMap-untar code + Zenodo **HEAD** reachability probe, no 3 GB download).
- Only `/data` assumption found in the stack = configs' `train.data` default, overridden by
  `BNHGQ2_TRAIN_DATA` (set in the job); train.py/data.py have no hardcoded /data.
- Gates: py_compile generator; 16 YAMLs `yaml.safe_load`; embedded/inline python compiles;
  `grep -L persistentVolumeClaim` lists ALL 16 (zero PVC refs); bash -n on 4 scripts; local
  tar-pack = 79,114 B (<1 MB) and `--strip-components=1` untar yields
  /work/code/{run_stage.py,bnhgq2/,configs/,preflight_final.sh,hls_templates/}.

## 2026-07-07 — FINAL-campaign QAT training stack built + locally verified (round-7)
New: `bnhgq2/qat.py` (custom binary-STE layers + model builder + activation calibration),
`bnhgq2/train.py` (the `train` stage), `configs/final-{fp32,w8a8,w1a8,w1a6,w1a4}.json`,
`run_stage.py` `train` stage, and `jobs/training/variants/gen_final_jobs.py` (15 GPU YAMLs +
smoke + preflight_final.sh + launch_final_staged.sh + sync_code_to_pvc_final.sh). All local
gates PASS with `.venv-hgq2` (CPU); NOTHING launched.

**THE quantizer question — resolved empirically (verify-first):**
- CONFIRMED STOP: a stock 1-bit KBI weight quantizer (k0=1,b0=1,i0=1,SAT_SYM) on LATENT
  floats collapses to the grid {-1,0,+1} — probe: 4096/4096 latents -> **0** (ternary; the
  thesis-void condition). So stock fixed-point quantizers cannot do bipolar STE QAT.
- WON: a custom BitNet absmean STE binarizer (`bitnet_binary_ste`) applied inside custom
  layers `BitQEinsumDense`/`BitQDense` (subclass hgq QEinsumDense/QDense, override `call`).
  Effective forward weights = beta·bipolar_sign(W-mean) ∈ **exactly {-beta,+beta}, zero
  zeros** (verified: all 51 bit-layers, 2 symmetric values). The layer keeps a 1-bit KBI
  `kq_conf` so EBOPs/hls4ml still report `kq.bits == 1`. Same absmean math as binarize.py, so
  the export re-binarization of the trained latents reproduces the same ±beta (fidelity ~1.0).
- Two deliberate, FORWARD-identical hardenings vs qkerasModel.AbsMeanQuantizer: (a) strict
  bipolar `where(wc>=0,+1,-1)` (never sign(0)=0); (b) `ws = wc/stop_gradient(beta)` so the
  backward is (I-1/n)+q·dbeta/dw = BOUNDED — qkeras routes grad through wc/beta which blows up
  by 1/beta when a kernel's centered weights collapse (this produced the first A4 NaN).

**Activation quantizers — WON: static per-tensor KIF, MSE-per-tensor calibrated, FROZEN.**
Same form as build.py.act_q (fixed<act_bits,1+i>, SAT, RND_CONV). i is chosen once by a
warmup forward on a real batch (all quant inputs are post-SubLN, so range-stable in weights)
as argmin_i MSE, floored so f>=1. The earlier max+margin policy over-ranged A4 to f=0
(integer-only grid -> collapse); MSE fixed it (A4 -> i∈{1,2}, f∈{1,2}). At export the grids
are static and identical in form to build.py -> the §6' activation-substitution Δ is 0 by
construction.

**Gradient clipping — CHANGED from r5 (evidence-backed, LR-independent bug):** r5's
`global_clipnorm=1.0` is numerically UNSAFE on this deep-SubLN stack. The LayerNorm backward
Jacobian amplifies through the 8-block stack (worst at input_proj) to ~2.5e20 on adversarial
A4 batches; squaring that in the float32 global-norm overflows to Inf -> the clip fails ->
NaN. Verified: clipnorm@5e-5, clipnorm@2.5e-5 (LR/2), clipnorm@1.25e-5 (LR/4) ALL NaN at
step 1 (so the sanctioned LR-halving lever does NOT fix this failure mode). **`clipvalue=1.0`
trains A4 stably** (25 steps, loss decreasing) — element-wise, no squaring, no overflow, and a
near-no-op for the stable variants. Shipped uniformly across all 5 variants (keeps the
A8/A6/A4 comparison on the same clip); `clip_mode` is a config knob.

**Attention:** native HGQ2 QEinsum + QSoftmax (input_scaler=1/sqrt(E), generous static
tables) — trains with finite gradients; softmax internals stay LN/softmax-bounded.

**Local gates (all PASS, CPU, `.venv-hgq2`):** py_compile all files; 16 YAMLs parse; embedded
python compiles; bash -n on 3 scripts; build all 5 (params fp32 6,380,267 / others 6,380,717
total, 6,380,037 trainable — vs QKeras-era-2 ref 6,375,173: +2,560 trainable PE + quantizer
vars); fwd+bwd finite with nonzero latent-kernel grads for all 5; binary gate exact for
w1a8/w1a6/w1a4; `.keras` save -> FRESH-process reload byte-identical (max|Δ|=0.0); full
`train()` smoke (synthetic era-2 data) for all 5 variants exits 0 with the MacroOvR-AUC
callback + best-checkpoint + LR schedule + early-stop + meta all working, w1a4 NaN-free.

## 2026-07-05 (night) — docs reorganized: current-vs-frozen made explicit
- Kai flagged that folder naming made it hard to tell what's current. Fixed at the
  documentation layer (no folder renames — paths are load-bearing in scripts, the cluster
  ConfigMap, and skills): `00-START-HERE.md` gained a "Finding your way" section with a
  current-vs-frozen table and a naming decoder (era-1/era-2, rounds r5/r6s, config hashes,
  probe names); the run README got a dated banner (era-1 table clearly marked, pointers to
  current code/results); new decoders `code/README.md` and `results/README.md`.

## 2026-07-05 (evening) — folded attention core SYNTHESIZED (the deployable-point row)
- **probe_attn_core_rf64** (same core as rf1, multiplier_limit = total/64): 193 cycles
  @ II=64, est. 2.009 ns (0.48 µs at the 2.5 ns target), **DSP 820** — einsums 400 each
  (exactly 25,600/64), softmax 10, transposes 0. Report + raw csynth.xml + modules json
  in `runs/b224a8ea/probe_attn_core_rf64/`.
- **Key reading: folding trades DSP for routing, not for area.** DSP drops 63× vs rf1
  (52,000 → 820) but LUT only 1.7× (4.27M → 2.57M, still 148% of a VU13P): each folded
  einsum burns ~1.2–1.35M LUT in operand-routing muxes that stream 25,600 operands
  through 400 multipliers. The large-model (D256/H8/E32) attention core does not fit
  the device even folded; the small-model (D32/H4/E8) core is 8× smaller in MACs.
- Both attention-core operating points (spatial extreme + folded) are now measured —
  the historically-excluded piece is bounded from both ends.
- Repo re-rooted at project top level and everything pushed to GitHub
  (kai124138/bnjettagfastml): top-level docs, poster/, reports/, nrp runbooks.
- **Store-consistency repair (poster GAPS.md item 11):** the local
  `probe_bitlinear_rf256` dir+tarball had been overwritten by the later v3lat Latency
  build while its csynth.xml was from the earlier Resource build. Mulder's as-built
  Resource project fetched home → rf256 now a consistent firmware↔report pair
  (Strategy: Resource, defines.h 34bdc118…). The Latency firmware (byte-identical to
  mulder's v3lat ship tarball) moved to `probe_bitlinear_v3lat_rf256/` — with NO csynth
  report, that build crashed. The stale contaminated csynth_report.json inside the old
  v3lat ship tarball resurfaced on extraction and was deleted a second time.
- Per-instance re-fetch bonus: `probe_bitlinear_v2_rf256` modules json now in store
  (dense_resource 256 / SubLN 14 / affine 0 = 270) — the pure-±1+affine chain at
  Resource, bracketing the strategy comparison gap 11 asked for.

## 2026-07-05 (later) — per-instance csynth reports in the store; poster gaps 1/2/7 closed
- A parallel poster session (poster/) froze figures against the store and left a gap
  list (poster/GAPS.md); items addressed to the live session are now closed:
- **Gap 1 — per-function DSP attribution is store-backed.** Raw `csynth.xml` fetched
  from mulder into `results/hgq2/runs/b224a8ea/<probe>/` for all four finished probes;
  new `parse_csynth_modules.py` emits `csynth_modules.json` (per-Module resources +
  latency) beside each. Numbers confirm exactly what the ledger had quoted:
  head_fc2_rf32 → SubLN 112 / binary dense 0 / CSD-2 affine 0 (top 112);
  rf256 Resource → dense 256 / SubLN 14 (top 270); subln_rf1 → SubLN 1792.
- **New per-module fact from attn_core_rf1**: each act×act einsum (QKᵀ and attn·V,
  10×10×8×32 = 25,600 MACs) synthesized to **25,600 DSP — exactly 1 DSP per MAC**;
  the softmax module itself is only 10 DSP. The "~51k real multipliers" statement is
  now precise: 51,200 einsum DSPs + ~800 top-level/softmax glue = 52,000.
- **Gap 7 — head count.** `"n_heads": 8` read directly off the `model_config` h5
  attribute of all three r5 large checkpoints (s1 / a6-s1 / a4-s1). Era-2 large = H8, E=32.
- fetch_mulder_reports.sh now also pulls csynth.xml + regenerates csynth_modules.json
  on every fetch, and probe_attn_core_rf64 added to its PAIRS (fetch-ready when it lands).
- rf64 folded attention core still synthesizing on mulder (monitor armed; no crash).

## 2026-07-05 — attention core SYNTHESIZED (the historically-excluded piece)
- **probe_attn_core_rf1** (large-model core: scores QKᵀ → stable table softmax with
  β_qβ_k/√d in the exp LUT → attn·V; T=10, H=8, E=32; 16-bit input grids, RF=1
  fully spatial): **synthesized on Vitis 2023.2** — 31 cycles @ II=1, est. 1.81 ns,
  LUT 4,271,510 (247% VU13P) / FF 9.47M / **DSP 52,000 (423%)** / BRAM 720.
  Reading: the act×act core converts and synthesizes natively now (EinsumDense
  blocker closed), and its cost at the spatial extreme quantifies WHY attention is
  the piece weight-binarization cannot touch — every one of the ~51k act×act products
  is a real multiplier (no weights to binarize). At 0.65% of the model's MACs it is
  disproportionately DSP-expensive per MAC.
- Folded variant probe_attn_core_rf64 (multiplier_limit = total/64) shipped + csynth
  launched — the deployable-point row. For the SMALL (D32/H4/E8) model the core is
  8× fewer MACs; folded, trivial.

## 2026-07-04 (late) — DSP-0 core VERIFIED in csynth on the HGQ2 path; artifact-contamination bug fixed
- **probe_bitlinear_head_fc2_rf32** (pure ±1 + CSD-2 affine, Strategy=Latency, real
  csynth on mulder): total 194,012 LUT / 116,346 FF / **112 DSP** / 100 cyc / II 32 /
  est. 2.03 ns. Per-function split: binary dense **0 DSP** (23,788 LUT adder trees) ·
  CSD-2 affine **0 DSP** (285 LUT) · SubLN **112 DSP** = 100% of the probe's DSPs.
  → The v4 factoring is validated end-to-end: the DSP-0 binary core reproduces on the
  HGQ2-native path when weights are compile-time constants (Latency) — closing the
  Resource-ROM regression found earlier today.
- **Artifact-contamination bug caught & fixed**: pack_for_mulder tarred the project dir
  *including* a previously-fetched csynth_report.json → the stale v1 report rode inside
  the v3lat tarball to mulder and was fetched back masquerading as v3lat's result
  (numbers bit-identical to v1 exposed it; the remote build was genuinely Latency and
  still synthesizing). Fixes: tar now `--exclude csynth_report.json`; the fetch script
  regenerates the json only from a real csynth.xml and deletes stale ones. Bogus row
  purged from the store/table/dashboard.
- Still in flight on mulder: probe_attn_core_rf1 (attention core, heavy elaboration),
  probe_bitlinear_{v3lat,a6lat,a4lat}_rf256 (Latency elaboration of 65k-MAC layers is
  known-slow; the head_fc2 probe already settles the DSP question at small shape).
- **UPDATE (late evening): ALL THREE big-shape Latency probes FAILED identically** —
  `HLS 200-1715 problem during source synthesis` after ~4 h each: v3lat (A8), a6lat (A6),
  a4lat (A4). The Vitis frontend crash on the fully-unrolled 65k-MAC Latency dense is
  **precision-independent** — the big-shape Latency wall is structural, exactly the
  intractability prior art documented. Negative result recorded in constraints_map.md
  ("Latency at big shapes" row): small shapes → Latency/DSP-0 csynth-verified (head_fc2);
  big shapes → Resource + (future) true 1-bit weight-type emission in the HGQ2 frontend,
  or cite the QKeras-path per-shape DSP-0 numbers. Only attn_core still running.

## 2026-07-04 — session close: verified state + what's in flight
- **Verification PASS** (experiment-log 2026-07-04 ✓): all 27 core
  quantities recomputed independently from raw arrays match to full float precision.
- Results record updated: status, the HGQ2 conversion path, provenance rows.
- Dashboard (read-only store view) generated + published as a static page;
  regenerate anytime with `generate_dashboard.py`.
- **csynth still RUNNING on mulder at close** (fetch with `./fetch_mulder_reports.sh`):
  probe_attn_core_rf1 (A8 attention core — THE previously-excluded piece; C-sim corr 1.0
  already proven), probe_bitlinear_v3lat_rf256 + a6lat + a4lat (pure-±1 Latency-strategy
  denses — the expected DSP-0-core validation + the LUT-vs-precision axis). Latency-
  strategy elaboration of 65k-MAC layers is slow; expect hours, not minutes.
- **NOT launched (needs Kai)**: round-6-small training (the directive's literal D32/L2
  era-2 model — no trained checkpoint exists at that scale).
  Launch: `code/jobs/training/variants/launch_r6s_staged.sh` (5 PVC-free jobs,
  server-dry-run validated). Once its W&B checkpoints exist, the same pipeline runs it
  end-to-end (new configs + `run_stage.py all`), and the SMALL model can be synthesized
  as ONE monolith per precision — the deployable-scale table the large model can't give.

## 2026-07-04 — v4: DSP regression caught by per-instance csynth breakdown; affine refactor
- **STOP-grade catch**: probe_bitlinear (v3 factoring: in-weight ±β̃ CSD-2) synthesized
  at RF=256 Resource with **DSP=256 in the "binary" dense** (SubLN only 14). Root
  cause: Resource strategy stores weights in BRAM/ROM → runtime operands → the Vitis
  ≤2-signed-digit constant rule NEVER applies. The QKeras path was safe because its
  weights were literally 1-bit. **The binary-DSP-0 claim requires weights ≤2 bits in
  the datapath, regardless of the constant's digit count.**
- **v4 factoring**: ALL denses carry pure ±1 kernels (2-bit operands → mux/negate);
  the 2L+2 explicit β̃·z+b sites become frozen `QBatchNormalization` affines (ε=0,
  var=1 → HW scale = γ = CSD-2 exactly, compile-time constants). Biases move into
  the affines (input_proj: (bias+PE)/β̃ stays as the dense's (T,D) table). Every
  bias/γ quantizer is an explicit wide frozen SAT grid — v3's silent WRAP-default
  bias quantizers were ALSO costing fidelity.
- **v4 verify (A8, full 260k)**: gate1 corr 0.9589, macro-AUC **0.84933 vs trained
  0.85510 (Δ −0.00578)** — better than v3 (−0.00642); gate2 vs gold corr 0.9937 →
  **0.9978** after the bias-quantizer fix (residual = float-order noise; softmax-
  table and stream-margin ablations both negative).
- A6 v3-semantics verify (math identical to v4), full 260k: gate1 corr 0.8907,
  AUC **0.82150 vs trained 0.83943 (Δ −0.01794)**.
- EBOPs (HGQ2-native, trace_minmax): A8 = **634,685,224** vs analytic HGQ-v1
  convention 530.4M (results/ebops.md) — +19.7%, the accumulator term; conventions
  reported side by side, never mixed.
- SubLN folded probe (inside bitlinear probe): DSP=14, LUT ≈ 170k — LUT-heavy
  (unoptimized wide internals + full unroll), timing 3.03 ns > 2.5 target. Levers:
  narrow diff_t/var_t, fold the norm. Deferred; reported as-is.

## 2026-07-04 — A8 rebuild VERIFIED end-to-end (keras side) + first real csynth
- **v1 → v3 debug trail** (each a real measured failure):
  v1 gate1 AUC 0.500 (garbage) — root cause: every unconfigured HGQ2 datalane
  quantizer defaults to WRAP+uncalibrated (QEinsum inputs, QSoftmax exp/inv grids,
  QGAP). v2 fix (explicit frozen SAT grids + QSoftmax table configs) hit an
  UPSTREAM hgq 0.1.9 bug: `enable_iq=False` crashes multi-input layers
  (`_iqs_confs` typo in QLayerBaseMultiInputs.__init__). v3 workaround: exact-
  passthrough frozen grids on the einsum streams (identity on the integer grid,
  EBOPs stays live) + per-block calibrated score/stream ranges.
- **v3 verify (era2-large-w1a8, full 260k val)**: gate1 HGQ2↔trained corr 0.9578,
  macro-AUC **0.84868 vs ref 0.85510 (Δ −0.0064)** — right at the gold-model
  prediction (0.8475). gate2 HGQ2↔gold corr 0.9937 — residual attributed to the
  table-based softmax vs gold's exact float softmax (ablation pending).
- **First csynth on mulder (HGQ2 path)**: probe_subln_rf1 (dim 256, io_parallel,
  II=1): LUT 165,695 (9.6%) · FF 151,297 · **DSP 1,792** · BRAM 0 · 36 cycles @
  est. 1.818 ns. The range-reduced SubLN at II=1 spends ~7 DSP/lane on the 42-bit
  variance squares — the norm remains the model's DSP consumer (context: old
  LayerNorm census 1,049 DSP TOTAL for 51 instances but folded at RF=256; these
  are different operating points). Lever if needed: narrow diff_t/var_t, or fold.
- probe_bitlinear_rf256 (real block-0 Wo weights + SubLN, RF=256 Resource):
  local C-sim corr 0.9999992. Shipped to mulder with probe_attn_core.

## 2026-07-04 — SubLN custom-layer extension DONE (keras-v3 → Vitis, C-sim gate passed)
- New `bnhgq2/subln.py`: `PSubLN` keras layer (parameter-free per-token LayerNorm,
  biased var, eps=1e-6, optional `flatten_axes=2` for last-two-axes norm), keras-v3
  handler, `SubLN` hls4ml IR class (new class — built-in `LayerNormalization`/QKeras
  path untouched, asserted in the test), `_produce_kif`/`_request_kif` BitExact
  registrations (output pinned fixed<18, 1+ceil(log2 √(dim−1))> per dim; input request
  fixed<31,15>-equiv), Vivado+Vitis config/function templates deriving internal C++
  types per layer from the final input precision, idempotent `register_subln()`.
- New `hls_templates/nnet_subln.h`: RANGE-REDUCED inverse sqrt — var (+eps) shifted by
  an even power of two onto [1/4,1) via MSB scan, 4096-entry 1/√ table over [1/4,1)
  only, exponent re-applied as an exact half-shift on the product. Replaces the shipped
  `nnet_layernorm.h` scheme whose table only covers var ∈ (eps,1] (raw jet features
  reach var ~1e6). Wide pinned internals (var ≈ ap_ufixed<47..67,·>, prod f=24).
- New `bnhgq2/compat.py`: `apply_hls4ml_compat()` (keras-3.15 `EinsumDense.full_output_shape`
  property restored for hgq2 0.1.9; keras-v3 registry alias
  `hgq.layers.attn.mha.QMultiHeadAttention` → stale registered key, same for Linformer)
  and `patch_project_for_macos()` (std::complex forward-decl → `#include <complex>` in
  the two ap_*_special.h of a WRITTEN project; call between `write()` and `_compile()`).
- New `test_subln.py` — the C-sim acceptance gate. ALL PASS locally (Vitis backend,
  io_parallel, 2048 samples/case, corr target ≥0.9999):
  raw (10,16) var 1e4–1e6 corr 0.999999997 max-err 6.8e-4 · (10,256) 0.999999996/1.0e-3 ·
  (10,1024) 0.999999994/1.1e-3 · (256,) 0.999999996/1.0e-3 · (10,4,64) fa=2
  0.999999996/9.8e-4 · +QDense(4, binary frozen ±1) 0.999999972/2.1e-2 ·
  +QEinsumDense abc,cd->abd (binary frozen) 0.999999975/5.9e-3. No NaN/inf anywhere.
- Gotcha: `convert_from_keras_model(..., bit_exact=...)` kwarg silently OVERWRITES
  `hls_config['Model']['BitExact']` — always pass the kwarg (cost one debug round:
  without it the raw-feature input stayed fixed<18,8> and wrapped).
- csynth (mulder) risks, unvalidated locally: `static constexpr double epsilon` in the
  generated config (C++14 odr corner; swap to a mant/shift pair if the Vitis frontend
  balks), full unroll + complete partition at dim=1024 (long csynth, big priority
  encoder over ~67-bit var word), 42×42-bit squares → deep DSP cascades (SubLN is the
  model's known DSP consumer, but budget unmeasured until csynth).

## 2026-07-04 — gold-model experiments fix the rebuild design (all on r5 val subset n=20k vs verified npz)
- **E1 (architecture proof)**: numpy gold model, *dynamic* act quant (exact QKeras
  semantics): corr **0.999948** vs stored A8 scores, macro-AUC 0.852450 vs 0.852420.
  Reimplementation correct; labels row-aligned exactly.
- **E2 (static substitution)**: naive max-calibrated per-tensor static fixed-point:
  A8 corr 0.969 / ΔAUC −0.005 · A6 0.871/−0.027 · A4 0.493/−0.078. MSE-optimal
  **per-channel** calibration recovers: A8 0.980/−0.0015 · A6 0.895/−0.017 ·
  A4 0.687/−0.017. → calibration policy = mse_per_channel, 8192-jet calib set.
- **E3 (β snap)**: naive pow2 β on all 51 layers is DESTRUCTIVE (A8 corr 0.72,
  −6.3 AUC pts). Mantissa sweep: k=2 → −0.015, k=3 → −0.005, k=4 → −0.003 AUC.
- **E4 (fold-aware β)**: exploiting LN scale-invariance: Wq/Wk β exact via softmax
  input_scaler (exp-LUT fold, free), Wv β dropped (LN-killed), fc1/head_fc1 exact
  via bias→b/β; only 18 residual contributors carry CSD-2 β̃ (≤4.5% err, 2-signed-
  digit → DSP-free). Result: A8 static corr 0.959/ΔAUC −0.0049 · A6 0.897/−0.0168 ·
  A4 0.687/−0.0173. **Design locked**: static per-channel MSE calib + fold-aware
  CSD-2. (Upgrade path if mulder shows affines stay DSP-free at higher precision:
  exact-β QBatchNormalization variant recovers ~+0.003 at A8.)
- A4 note: its loss is the static-quant substitution itself (dyn+fold corr 0.987) —
  a trained-static-quant (HGQ2 QAT) run would be the proper fix; logged as future work.
- hgq2/hls4ml source deep-dive (all claims EXECUTED): binary pin =
  KBI(k0=1,b0=1,i0=1,SAT_SYM,frozen) passes ±1 bit-identically, reports 1 bit to
  EBOPs; QSoftmax folds input_scaler into exp LUT; PE+bias fold into QEinsumDense
  bias table (bias_axes='td'); keras LayerNormalization UNCONVERTIBLE 3 ways →
  custom PSubLN extension (in progress); QMultiHeadAttention has 2 hls4ml-1.3.0
  compat bugs (registry key module path + keras-3.15 full_output_shape removal) —
  patched in bnhgq2/compat.py; full MHA+FFN block verified BIT-EXACT (max diff 0.0)
  through Vitis-backend C-sim at RF 1 and 2.

## 2026-07-04 — session start: scaffold + infrastructure
- Scouted repo + HGQ2/hls4ml web state (sweep; findings in
  the internal research log 2026-07-04 entry).
- Local env built: `.venv-hgq2` (py3.12, hgq2 0.1.9, hls4ml 1.3.0, keras 3.15, tf 2.21).
- All 8 round-5 checkpoints fetched from W&B → `models/r5/<key>/bitnet/*.h5` (76.9 MB each).
- Era-2 val split (Zenodo 3602260, 1.14 GB) downloaded → `data/` (project root).
- Confirmed by h5py inspection: r5 `.h5` holds all latent FP32 kernels/biases AND the folded
  positional-encoding constant (serialized in `model_config` → TFOpLambda add `y` kwarg) —
  the port needs no TF-2.11 environment.
- Round-6-small training YAMLs (D32/H4/L2/FFN64, era-2, PVC-free) generated + server-dry-run
  validated: `code/jobs/training/variants/kai-bn6s-*.yaml`. **NOT launched** — permission
  gate requires Kai's approval for new GPU jobs. See decisions.md 2026-07-04.
- Architecture-target decision + binary-pinning verify-first policy + β-fold plan:
  decisions.md 2026-07-04 (Decisions 1–4).
