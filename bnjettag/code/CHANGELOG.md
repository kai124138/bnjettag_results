# Code change ledger — every file we authored or edited

A single running record of **all code in this run**: what's upstream, what we edited,
what we wrote from scratch, and — bluntly — **what broke and how we fixed it** (§3, which
also answers "were there bugs in the hls4ml work?").

This file is the *index + chronology + bug log*: what/why/when for every code change. The
line-level trainer diff it refers to (`qkerasModel.py`, edit groups G1–G5) is kept with the
manuscript sources and is not part of this snapshot.

Conventions: paths are relative to the run root. Dates are file mtimes on this machine
(there is no git history for this run, so mtimes are the timeline of record).

---

## 1. File inventory

### 1a. Training (`code/training/`)

| file | origin | what it is | status |
| --- | --- | --- | --- |
| `qkerasModel.py` | upstream + **our edits** | THE trainer; all precisions via `BN_VARIANT`. Edits catalogued as groups G1–G5. G5 = opt-in `BN_STANDARDIZE` per-feature input affine (default OFF ⇒ every prior run bit-identical). | **edited** (last Jul 13) |
| `qkerasModel.patch` | **ours** | Phase-0 upstream diff (data path / env / W&B) captured as a standalone patch. | new (Jun 22) |
| `ebops.py` | **ours** | STATIC EBOPs/BOPs profiler. Encodes the closed-form per-layer MAC + param profile of `build_bitnet_jet_tagger` (stdlib only, no TF, no training); exposes `bops(macs,b_w,b_a)` and a `ebops()` aggregator. Param totals reconstruct the verified preflight counts exactly (tiny 26,529 / small 153,793 / medium 808,065 / large 6,373,633). Accumulator (HGQ) term is a 0-stub `accumulator_bops()` hook pending the HGQ formula. | new (Jun 29) |
| `HLS_qk_Roc_Tracing.py` | upstream | upstream HLS+ROC tracing; hardcodes a Vitis path from another cluster. Reference only — not run here. | vendored, unmodified |
| `ROC.py` | upstream | upstream ROC plotting. Not invoked inside the training jobs (ROC lives in W&B). | vendored, unmodified |
| `README_upstream.md` | upstream | the upstream project README. | vendored, unmodified |
| `environment.yml` | upstream | upstream conda environment. | vendored, unmodified |
| `dataForgeScripts/dataForge.py` | upstream | data prep (root → tensors). | vendored, unmodified |
| `dataForgeScripts/removeBackground.py` | upstream | data prep helper. | vendored, unmodified |
| `util/plotting/kinematics_plotter.py` | upstream | the kinematics diagnostic plots. | vendored, unmodified |

### 1b. hls4ml / Vitis (`code/hls/`) — all authored by us

| file | what it is | status |
| --- | --- | --- |
| `run_csynth.py` | **The Vitis C-synthesis driver that produced §B** of `results/hls_resource_table.md`. Builds the binary FFN block, runs `hls_model.build(synth=True)` on hls4ml 1.x, unwraps `CSynthesisReport` (+ `csynth.xml` fallback). | new (Jun 24) |
| `full_model_csynth.py` | **The Vitis driver that produced §B′** — the *full trained transformer* end-to-end. Loads `lr15_bitnetJetTagModel.h5`, rebuilds each `BitLinear` as `LayerNormalization→QActivation→QDense(binary)`, ports trained weights via the BitNet `AbsMeanQuantizer`. Three modes (`HLS_MODE`): `fidelity` (rebuild vs trained, corr 0.99998), `convert` (QKeras↔Vitis bit-accuracy 0.9967–0.9999), `csynth` (per-shape synth + 51-instance composition). **Result: binary matmul = 0 DSP; all 1,049 DSP = LayerNorm.** | new (Jun 26) |
| `RUN_CSYNTH_ON_VITIS.md` | Turnkey runbook for a Vitis-2023.2 box (the steps actually used on `mulder`). | new (Jun 24) |
| `sweep_precision.py` | Quantization-aggressiveness × hls4ml **firmware** sweep: QKeras → HLS C++, g++ bit-accurate emulation (corr), inspects `defines.h`/`parameters.h` to confirm binary weights type as `ap_uint<1>` → **0 DSP**. | new (Jun 23) |
| `resource_model.py` | **Analytical** per-component resource model (MACs / weight-bits / LUT·DSP·BRAM via labeled cost factors). The pre-csynth first-order estimate; separates exact structural counts from derived estimates. Superseded for §B by real csynth, kept for the per-component story. | new (Jun 23) |
| `convert_probe.py` | hls4ml **convertibility** probe (0.8.1 era): proves the binary core (Dense + FFN + head) converts and emulates bit-accurately; probes the two hard pieces (LayerNorm, softmax-free EinsumDense attention). | new (Jun 23) |
| `full_transformer_probe.py` | STRETCH: the **whole** BitNet block through hls4ml ≥ 1.2 (Python ≥ 3.10) — closes the Phase-1 gap (SubLN LayerNorm + EinsumDense attention that 0.8.1 could not convert). | new (Jun 23) |
| `stage_a_fix.py` | Stage-A **rerun with a correctly-sized accumulator** — the fix for the bit-accuracy bug (see §3.2). | new (Jun 23) |

### 1c. Plots (`code/plots/`) — authored by us

| file | what it is | status |
| --- | --- | --- |
| `make_results_plots.py` | Generates the 3 publication figures (`results/plots/results_*.png`). AUCs cited from `RESULTS.md`; FPGA resources read **live** from `results/csynth/*.json`. No invented numbers. | new (Jun 24) |

### 1d. NRP Jobs & watchers (`code/jobs/`) — authored by us

| group | files | what |
| --- | --- | --- |
| `jobs/training/` (10) | `kai-bn-train-paper-binary-lr15`, `-paper-binary-sm-a{8,6,4}`, `-paper-binary-sffree{,-a6,-a4}`, `-paper-ternary`, `-vanilla-fp32`, `-w8a8` | the Jobs that produced the headline + sweep + baseline + appendix numbers. |
| `jobs/hls/` (4) | `kai-hls-{csynth,full,inspect,sweep}.yaml` | NRP Jobs wrapping the `code/hls/` scripts (csynth Job retained for any Vitis-equipped cluster). |
| `jobs/` (2) | `watch_act_sweep.sh`, `watch_paper_runs.sh` | log-watchers. |

### 1e. Archived (superseded — kept for history, `archive/`)

| file(s) | what | why archived |
| --- | --- | --- |
| `qkerasModel_ste.py` | pre-STE intermediate trainer snapshot | superseded by `code/training/qkerasModel.py`. |
| `jobs/*.yaml` (16) | earlier job YAMLs (pre-paper-recipe, smoke tests, dev/setup pods) | off the final paper recipe. |
| `upstream_samples/*.root` | original upstream `.root` samples | replaced by the PVC dataset. |

---

## 2. Phase chronology

**Phase 0 — vendor + data/env/W&B patch (Jun 22).** Vendored the upstream files (1a, unmodified)
and captured the minimal data-path / env / W&B wiring as `qkerasModel.patch` (G1).

**Phase 1 — gradient fix + paper-faithful binary trainer (Jun 22 → Jun 23).** STE gradient fix (G2)
and the paper-faithful binary quantizer + training recipe (G3). `archive/qkerasModel_ste.py` is the
intermediate snapshot from this phase; `qkerasModel.py` is the result.

**Phase 2 — baselines + sweep knobs + Jobs (Jun 23 → Jun 24).** Added the `BN_VARIANT` /
`BN_ACT_BITS` / `BN_TERNARY` / `BN_SOFTMAX_FREE` knobs (G4) and wrote the 10 training YAMLs —
binary headline, A8/A6/A4 canonical-softmax sweep, vanilla FP32 + W8A8 baselines, ternary +
softmax-free appendix.

**Phase 3 — hls4ml convert + bit-accurate emulation (Jun 23).** `convert_probe.py`,
`resource_model.py`, `stage_a_fix.py`, then `full_transformer_probe.py` + `sweep_precision.py`.
Result: the binary core converts and emulates **bit-accurately**, with binary weights typing as
`ap_uint<1>` → **0 DSP** (firmware-confirmed on NRP; no Xilinx synthesis backend there).

**Phase 4 — Vitis C-synthesis on `mulder` (Jun 24).** `run_csynth.py` + `kai-hls-csynth.yaml` +
`RUN_CSYNTH_ON_VITIS.md`. Synthesized the binary FFN at A8/A6/A4, RF=256, on Vitis HLS 2023.2 →
**DSP = 0 confirmed in silicon estimates**; LUT/FF/BRAM/latency filled into `results/hls_resource_table.md` §B;
raw reports → `results/csynth/csynth_report_a{8,6,4}_rf256.json`.

**Phase 5 — publication figures (Jun 24).** `make_results_plots.py` → the 3 `results_*.png`.

**Phase 6 — full trained transformer, synthesized end-to-end (Jun 26).** `full_model_csynth.py`: loaded the real
`lr15_bitnetJetTagModel.h5`, rebuilt every custom layer (`BitLinear`/`RMSNorm`/attention projections) from
hls4ml-supported primitives, **ported the trained binary weights**, validated (fidelity corr 0.99998; QKeras↔Vitis
bit-accuracy 0.9967–0.9999), and C-synthesized the 5 distinct layer shapes at A8 (RF=256) on Vitis HLS 2023.2.
**Headline: binary matmul = 0 DSP, re-confirmed on the real model; the entire transformer's 1,049 DSP (8.5 % of a
VU13P) is 100 % LayerNorm.** Filled `results/hls_resource_table.md` **§B′**; raw → `results/csynth/full_model_*_a8_rf256.json`.
A6/A4 sweep launched same night (backfill). This directly answers the "are you just synthesizing an FFN?" critique.

**Phase 7 — FINAL-campaign QAT training stack (Jul 7).** New native-HGQ2 QAT trainer so the
trained model IS the hardware model (decisions.md 2026-07-07). `code/hgq2/bnhgq2/qat.py`:
custom serializable layers `BitQEinsumDense`/`BitQDense` that absmean-binarize the latent
kernel in the forward pass (strict bipolar STE, `ws=wc/stop_gradient(beta)` for a bounded
backward) → effective weights **exactly {-beta,+beta}** (probe-proven a stock 1-bit KBI
instead collapses latent floats to {-1,0,+1} = ternary, the thesis-void STOP), plus a static
MSE-per-tensor activation-grid calibrator and `build_qat_model` (variant from
`quant.weight`: binary_absmean / int8_absmax / none). `code/hgq2/bnhgq2/train.py`: the `train`
stage (era-2 data, MacroOvR-AUC callback + best-checkpoint + LR warmup/poly-decay + early-stop
+ wandb + `train_meta.json`), gradient clip = **clipvalue=1.0** (r5's global_clipnorm overflows
float32 on the deep-SubLN backward → A4 NaN, LR-independent — see §3 bug log). `train` stage in
`run_stage.py`; `config.py` accepts train configs; `store.py` honors `BNHGQ2_STORE`. Configs
`configs/final-{fp32,w8a8,w1a8,w1a6,w1a4}.json`. Job generator
`jobs/training/variants/gen_final_jobs.py` → 15 `kai-bnf-<variant>-s<n>.yaml` + `kai-bnf-smoke.yaml`
+ `preflight_final.sh` (`PREFLIGHT_ALL_PASS`) + `launch_final_staged.sh` (≤3 concurrent, 5 waves)
+ `sync_code_to_pvc_final.sh` (tar → PVC `/data/BNJetTag-hgq2/`, md5-verified both ends). All
local gates pass (see LEDGER 2026-07-07 + experiment-log): 5 configs build (params 6,380,267 /
6,380,717), binary gate exact, `.keras` fresh-reload byte-identical, full smoke train exits 0.
**Not launched** (build + local verify only). **Follow-up same day — jobs converted PVC-FREE:**
the kai-data cephfs PVC filters training pods off the GPU nodes (A/B probe: no-PVC pod
scheduled in ~4 min, PVC pod pended 75+ min). The 15+smoke YAMLs now take DATA from Zenodo,
CODE from ConfigMap `kai-bnf-code`, write to an emptyDir, durability via W&B; new emitted
`make_code_configmap.sh` is the training prereq (`sync_*` kept for the PVC ROC job only). See
LEDGER 2026-07-07 (later).

**Phase 7 — FINAL-campaign ROC/eval driver (Jul 7).** New standalone eval driver
`code/hgq2/roc_final.py` for the round-7 HGQ2-native checkpoints (decisions.md 2026-07-07):
loads each `model_best.keras` (`import hgq` + `bnhgq2.subln.register_subln()` +
`apply_keras_compat()` + `keras.models.load_model`), evaluates the full era-2 held-out split
`/data/hls4ml_lhc_jet/val/val` (asserts n=260000), writes per-model `.npz` (keys `y`/`score`/
`meta`, same convention as `roc-results/r5/*.npz`; npz name `<VARIANT_UPPER>-s<seed>.npz`),
and emits `roc_auc.md` (per-seed + per-variant seed-averaged mean±std, macro-OvR + 5 per-class
AUCs, **recomputed from the stored arrays** so the verification procedure reproduces them) and a HEP-style
`roc_overlay.png` (TPR linear x, mistag-rate FPR log y). Reuses `bnhgq2.data.load_eval_set`
(byte-identical to make_roc). Since HGQ2 `.keras` files are self-describing (no BN_* env
globals), the whole 15-model matrix (5 variants × 3 seeds) is evaluated in ONE process
(`eval-all`), loading val data once; checkpoints resolve PVC-first with a per-missing W&B
fallback (knob `CKPT_SOURCE=pvc|wandb`), skip-if-missing. Job `jobs/training/kai-roc-final.yaml`
(CPU-only, `python:3.12-slim` + pins mirroring `.venv-hgq2`: keras 3.15.0 / tensorflow-cpu
2.21.0 / hgq2 0.1.9 / hls4ml 1.3.0 / quantizers 1.2.2 / numpy 2.5.0 / sklearn 1.9.0 / h5py
3.14.0 / matplotlib 3.11.0 / wandb 0.28.0), code+data from the `kai-data` PVC, results → PVC
`/data/outputs/final/roc/` **and** W&B run `final-roc-artifacts` (project `bnjettag-final`).
Fetch helper `jobs/training/variants/fetch_final_results.py`. Local gates: py_compile + YAML
parse + embedded-heredoc compile + a **build→save→fresh-process-load→npz→AUC→table→plot**
round-trip (model built from `configs/era2-large-w1a8.json` via `bnhgq2.build`, synthetic
binz/calib + synthetic era-2 h5) — all pass; npz `roc_auc_score` reproduces the table value
exactly. New result namespaces `roc-results/final/` and `results/final/` (README stubs).
**Not run on the cluster** (build + local verify only).

**Phase 7b — GPU/PVC-free ROC job (Jul 7).** The CPU/PVC job above hit its 8 h deadline
mid-matrix (15 QAT KBI-quantizer models eval far slower on 4 CPUs than r5 QKeras), and the
final checkpoints trained PVC-free (cephfs PVC blocks GPU-node scheduling) so they live only
on W&B. New **primary** ROC job `jobs/training/kai-roc-final-gpu.yaml` — GPU + PVC-free end to
end, reusing `gen_final_jobs.py`'s hardened GPU blocks verbatim (image `python:3.12` +
`tensorflow[and-cuda]==2.21.0` pip wheels — the `2.21.0-gpu` image is broken; LD_LIBRARY_PATH
over the nvidia wheels; BAD_NODES exclusion (k8s-chase-ci-07 + suncave-*); backoffLimit 2;
1 GPU / 4 cpu / 24Gi / 24Gi-eph requests==limits; code from ConfigMap `kai-bnf-code` via
strip-components untar + md5 echo), plus `matplotlib==3.11.0` added to the pins (the ROC
overlay needs it; training pins omit it). Data = Zenodo `hls4ml_LHCjet_150p_val.tar.gz`
(~1.14 GB, >1e9 guard, deleted post-extract) — the SAME val split as the PVC copy, so the
`n=260000` assert is unchanged; `CKPT_SOURCE=wandb`; outputs → W&B run `final-roc-artifacts`;
GPU predict batch is an env knob `EVAL_BATCH` (default 16384). **No `roc_final.py` change
needed** — `--batch` is already a CLI arg and the load path was already patched to
`import bnhgq2.qat` (verified importable). `kai-roc-final.yaml` kept as the CPU/PVC variant
with a header pointing here as primary. Gates: py_compile + YAML parse + heredoc compile +
patched-load-path import smoke — all pass.

**FINAL hls4ml conversion driver (Jul 7).** `code/hgq2/convert_final.py` (+ thin
`run_convert_final_all.sh`): W&B-fetch a trained QAT ckpt → load → hardware-faithful export
(REUSES build.py/port.py/convert.py; binz == `qat.bitnet_binary_ste` exact, PE folded, STATIC
act grids read back from the trained QAT quantizers) → GATE1 export↔QAT → hls4ml Vitis convert
(attention included) → GATE2 C-sim → native EBOPs → mulder tarball; results per config-hash in
`results/final/runs/<hash>/<variant>-s<seed>/` (export_verify/csim_verify/ebops/convert JSONs +
project + tarball). One backward-compatible fix to `bnhgq2/port.py` (affine width shape[1]→
shape[-1] for 3-D einsum-native QAT kernels; identity for the QKeras rebuild). Ran the WHOLE
chain on the real `final-smoke` w1a8-s1 ckpt (`.venv-hgq2`, CPU) — weights export exact, both
fidelity gates root-caused (see §3 boundary), no csynth (mulder-only). fp32/w8a8 skipped
(no binary story; w8a8 = EBOPs-only).

**HYBRID-strategy WHOLE-MODEL emission — monolith-v2 + probe_block (Jul 10, `full-synthesis` branch).**
`code/hgq2/bnhgq2/monolith.py` (new; shared hybrid emitter) + drivers `code/hgq2/monolith_v2.py`
(Route A: the ENTIRE flagship in one hls4ml project) and `code/hgq2/probe_block.py` (Route B: one
encoder block per project, composable at RTL/IP). Every WEIGHTED layer is re-expressed as a keras
`QDense` on the (T,in) token tensor → hls4ml **PointwiseConv1D** at **Strategy=Resource** (±1 weights
→ BRAM ROM, bounded LTO — the property that un-blocks single-project synthesis vs the 27M-IR
all-einsum-Latency OOM); the weightless attention einsums + softmax stay **Latency** via per-layer
`LayerName` overrides; head split/merge = keras `Reshape` (fused away in io_parallel); input_proj's
per-position PE via a flatten→(T·D) axis=-1 affine (PointwiseConv1D can't hold a (T,D) bias). Both
C-sim-GATED locally (`.venv-hgq2`, Vitis backend): **monolith-v2 0.99793 vs keras(mono)** (einsum_dense
= 0, 49 pointwise + 2 dense @ Resource, 16 einsum + 8 softmax @ Latency, 138 weight ROMs, ReLU AP_SAT,
wide accums), **probe_block 0.99968 vs the flagship's real block-0** — both ≥0.997. One
backward-compatible shared-code change: `bnhgq2/convert.py` gained `layer_configs` (per-layer
LayerName merge; default None → byte-identical). build.py/port.py/qat.py/train.py untouched. csynth is
mulder-only. Full detail + measured numbers in `code/hgq2/LEDGER.md` 2026-07-10 and FULL-SYNTHESIS.md.

**Monolith emitter made ARCHITECTURE-AGNOSTIC (Jul 13, `full-synthesis` branch) — unblocks Route C.**
The whole-model emitter was pinned to the flagship: `probes_final_v2.load_flagship()` hardcoded
`configs/final-w1a8.json` + `FLAGSHIP_VARIANT/SEED/RUN`, and `monolith_v2` hardcoded a v1-firmware path
and `rf=256`. Round-7 retrains at deployable scale (D32/H4/L2/FFN64), so the emitter must take ANY
config. Changes (3 driver files; **`bnhgq2/` package untouched** — build.py/monolith.py/gold.py/port.py/
extract.py/binarize.py already derived every dim from `cfg['arch']`):
* `probes_final_v2.py`: new `resolve_source(config, variant, seed, wandb_run)` (naming convention:
  variant = config minus `final-`; seed = 3 for the flagship else 1; run = `<config>-s<seed>` — these
  REPRODUCE the FLAGSHIP_* constants exactly) + new general `load_model(config_name, …)`. `load_flagship()`
  kept as a thin wrapper == `load_model(FLAGSHIP_CONFIG)`, so `probes_final_v3` and this file's own
  per-shape `main()` are unaffected. Checkpoint precedence: `--checkpoint` > `cfg['checkpoint']` > W&B.
* `monolith_v2.py`: `--config/--variant/--seed/--wandb-run/--rf`; `rf` now defaults to `cfg['hls']['rf']`
  (= 256 for the flagship, unchanged); flagship-pinned `V1_FIRMWARE` constant → `_v1_firmware(h,variant,
  seed,rf)` (resolves to the identical path for the flagship, skipped when absent); new arch-derived
  structural self-check `expected_from_arch` = {pointwise 1+6L, dense 2, einsum 2L, softmax L, subln 4L+3,
  einsum_dense 0} + `structure_matches_arch`. Manifest key `flagship_run` → `wandb_run` (+ `variant`,
  `seed`, `arch`); nothing parsed the old key.
* `probe_block.py`: same `--config` plumbing; hardcoded "8 block instances" note → `{L}`.
**Behavior-preservation evidence (local, `.venv-hgq2`, no Vitis/GPU):** re-emitted the flagship through
the refactored path with the REAL w1a8-s3 checkpoint (C-sim skipped, written to a scratch dir so the
shipped store is untouched) — every counter matches the stored 2026-07-10 `monolith_v2_verify.json`:
49 PointwiseConv1D + 2 Dense @ Resource, 0 einsum_dense, 16 einsum + 8 softmax @ Latency, 35 SubLN,
138 weight ROMs, ReLU AP_SAT, 24 Latency overrides, RF cap {input_proj:16}, post-parse (relu_sat 9 /
wide_accum 51), full 12-class layer census, `weight_t = ap_fixed<2,2>` ×51, accum_t `<27,11>` ×41 with a
16-bit fraction floor everywhere. A synthetic-weight dry emission of the *not-yet-trained* small config
(D32/H4/L2/FFN64) converts cleanly: 13 pointwise + 2 dense, 0 einsum_dense, 4 einsum + 2 softmax,
11 SubLN, 42 ROMs, `weight_t = ap_fixed<2,2>` ×15, `structure_matches_arch` true. **Known:** at small
dims hls4ml auto-snaps head_fc2's RF 256→160 (valid set 1,2,4,8,16,32,160) with a WARNING — set
`"rf": 32` in the small config's `hls` block to avoid the serialization.

**Opt-in per-feature INPUT STANDARDIZATION `BN_STANDARDIZE` (Jul 13) — fixes a real, measured bug.**
`qkerasModel.py` claimed (main(), the tag block) *"The dataset ships already preprocessed; no extra
normalisation is applied."* **The first half was false**, and it hid the bug: it imported
`MinMaxScaler` and never called it, so the 16 constituent features went into the model on wildly
different scales. Measured on the real top-10-by-pT inputs (`data/val/jetImage_*.h5`, 2026-07-13):
raw per-feature std spans **0.0394 … 117.49 — a 2,979× ratio** (GeV-scale px/py/pz/e/pt beside O(1)
etarel/phirel/ptrel/deltaR/costhetarel). The model's first op is `BitLinear(input_proj)` with
`norm_inside=True`, i.e. SubLN **across** the 16 features, so the LN moments are dominated by the
momenta and the **post-LN** per-feature std lands at **1.40 (px/py/pz) vs ~0.185 (every relative
feature) — an 8.9× imbalance**. That is fatal *specifically for the binary model*: BitLinear weights
are `{−β,+β}` with a **single per-tensor β**, so there is no per-feature gain to learn — a feature
arriving with 7.5× less amplitude contributes 7.5× less, permanently. FP32 escapes instantly by
learning `w ~ 1/std`; **binary structurally cannot**. The A8 absmax quantizer then sets its range from
the dominant momenta, costing the small features ~3 more bits. Standardizing **before** the LN (they
do **not** commute) drops the post-LN imbalance **8.9× → 2.1×**. Context: the deployable-scale
18,405-param binary model plateaus at ~42 % train acc / train_auc ~0.72.
* **`BN_STANDARDIZE`** (default **`0` = today's behaviour, bit-for-bit**) applies
  `x' = (x − mean)/(std + eps)`; **`BN_STD_EPS`** (default 1e-6) is persisted with the constants.
  The 16 mean + 16 std are fit **on the train split only** — and on the *train portion* of it: the
  new named `VALIDATION_SPLIT` (the same 0.20 that was a bare literal at the `model.fit` call) means
  the stats are fit on exactly `X[:int(n·0.8)]`, Keras's own `split_at`, so the val-monitor tail
  never leaks in. Only **non-padded** particles contribute.
* **The padded-particle re-zero is the correctness trap.** A padded slot is the all-zero 16-vector; a
  naive affine maps it to `−mean/std` — a *specific, consistent, nonzero* vector, i.e. a **fake
  particle** that SubLN normalises into a fixed direction and GlobalAveragePooling then averages in.
  `apply_input_standardizer` re-zeros it, keeping padding inert (0 in → 0 out) exactly as raw inputs
  are today. The mask (`~X.any(-1)`) is *only* valid on **raw** inputs — after the affine it is
  all-True and useless — so `_padding_mask()` is the single definition and both call sites take it
  pre-transform. The fit needs the mask in **both** passes for different reasons (pass 1: padded rows
  add exactly 0, so the mask is needed only for the *denominator*; pass 2: they contribute
  `(0−mean)² ≠ 0`, so they must be masked out of the *numerator*).
* **The constants are part of the model definition**, not a log line. They are persisted **three
  ways**: beside the checkpoint as `<stem>_standardizer.npz` (+ a `.json` mirror), to W&B (files *and*
  `wandb.config`), and — because a sidecar *can* be separated from the `.h5` (these jobs write to an
  `emptyDir` and ship the checkpoint through W&B) — **inside the checkpoint itself**, as HDF5 root
  attrs (`bn_standardize`, `bn_std_mean/std/eps/probe_sha256/…`). So a standardized checkpoint is
  **self-describing**: `make_roc` recovers the transform from the `.h5` alone, with no sidecar and no
  env var. (Verified additive: the 24 weight datasets and `model_config` are byte-identical before/
  after embedding; Keras ignores unknown root attrs.) `make_roc.py` only ever **reads** them —
  re-fitting on the eval split is refused. Whichever source is found is **authoritative**: found →
  applied even if the launcher forgot `BN_STANDARDIZE` (the dominant failure is *omitting* the
  transform, which is silent). An explicit `BN_STANDARDIZE` that contradicts it, or a sidecar that
  disagrees with the checkpoint's own attrs, is a **hard error** — never a quiet fallback.
* **Anti-drift:** `make_roc` carries a numpy-only *mirror* of the transform (it must standardize
  without importing TF, and the ROC job ships that file alone via a ConfigMap — the same convention as
  `bnhgq2/data.py` mirroring `load_eval_set`). To make the duplication safe, the sidecar stores
  `probe_sha256` = sha256 of the transform applied to a fixed synthetic tensor **containing padded
  rows**; `make_roc` recomputes it with *its own* code and hard-fails on mismatch. A missing re-zero,
  a different eps or a different op order therefore cannot produce a plausible-but-wrong AUC.
* **FPGA:** a **fixed** per-feature affine — 16 constant multiplies + 16 adds, known at synthesis
  time, CSD/LUT-implementable like the affines the HLS pipeline already handles. **0 DSP added to the
  binary core.** It is **not** a data-dependent normalization (no running stats, nothing computed at
  inference); the re-zero is a mux on the particle-valid bit the datapath already carries.
* Files: `code/training/qkerasModel.py`, `code/training/make_roc.py`,
  `code/jobs/training/kai-roc-r5.yaml` (ConfigMap re-embedded via `reembed_make_roc.py`), new job
  `code/jobs/training/variants/kai-bn8-small-a8-std.yaml` (single-knob A/B twin of the round-7 control
  `kai-bn7d-small-a8-long` — verified identical on all 20 other `BN_*` exports).
* **Verified locally (no training, no GPU; `.venv-hgq2` + a tfmot stub):** `BN_STANDARDIZE=0` is
  **bit-identical to HEAD** — `load_hls4ml_jets` returns byte-identical `X`/`Y`/`jet_pt`, and across
  **all four** `BN_POS_ENC` modes the model's params (18,405), layer/weight names+shapes and
  serialized graph are identical, with **byte-identical logits under forced-identical weights** for
  `none`/`learned_real`/`sinusoidal`. (Under the default `learned` PE, logits differ ~0.11 — but
  **HEAD-vs-HEAD differs by the same 0.1164**: that PE is a *folded random constant* untracked by
  `model.weights`, a pre-existing documented quirk, not this change. Likewise "same seed → same
  weights" is *not* a valid premise here: TF's per-op seed counter survives `clear_session()`, so one
  unmodified module built twice already differs on 15/24 tensors — a trap worth not re-entering.)
  Also verified: the qkerasModel↔make_roc transforms are byte-identical; the `probe_sha256` canary
  *fires* when the re-zero is deleted; and all six cells of the sidecar/env truth table behave as
  specified. A numpy-only `_selftest_input_standardizer()` now runs inside `--sanity`, which every
  training job already executes as a preflight gate.

**Phase 8 — round-7 deployable-scale QAT matrix (Jul 14, `full-synthesis`).** Built the round-7
training matrix on the HGQ2 QAT stack: 10 `hgq2/configs/r7-{small,tiny}-*.json`, generator
`jobs/training/variants/gen_r7_jobs.py` (30 `kai-bn7f-*` Jobs + `preflight_r7.sh` + `launch_r7_staged.sh`).
One behavioral code change: `hgq2/run_stage.py` gained an **optional `--out-dir`** for the `train` stage
(default None ⇒ prior behavior byte-unchanged; the final campaign never passes it) so r7 checkpoints land
under `<size>-<variant>-s<seed>/`. Full detail + local-verification gates in `hgq2/LEDGER.md` (2026-07-14)
and the internal experiment log. Nothing launched.

**Phase 9 — round-7 WHOLE-MODEL hls4ml conversion (Jul 14, `full-synthesis`).** Ran the proven
`hgq2/convert_final.py` chain on `r7-small-w1a8-s2` (cfg `r7-small-w1a8.json`, D32/H4/L2/FFN64) to
emit the whole model as ONE native project (QEinsumDense/QEinsum/QSoftmax as trained, Latency,
io_parallel), in two RF variants (RF=1 fully-parallel, RF=8 folded), VU13P / 2.5 ns. Two
behavioral code changes, both backward-compatible byte-for-byte on the FINAL path: `convert_final.py`
gained `--config`/`config_path`, `--wandb-subdir` (+ `fetch_checkpoint(subdir=)`), and
`--run-dir`/`--store-root`/`--cache-dir` overrides (all default to the prior FINAL-campaign layout);
new wrapper `hgq2/run_convert_r7.sh` (mirrors `run_convert_final_all.sh`). Local C-sim gate PASS
(hls4ml vs exported hardware model corr 0.9987, ≥0.997, identical RF=1/RF=8); export↔QAT gate 0.965
(static-grid-substitution gap, same status as the large model). Then, after the first mulder csynth
showed the narrow-accumulator DSP-parking (rf1 plain: DSP 33,696 = ~12.8k structural act×act + ~21k
parking), a **v2 WIDE-ACCUM re-emission**: `convert_final.py` gained `widen_weighted_accum` (the
whole-model generalization of `probes_final_v3.widen_dense_accum` — also handles EinsumDense, leaves
the weightless act×act QEinsum cores alone) + `audit_weighted_accums` + `--widen-accum`/`--min-frac`
(default OFF ⇒ byte-identical); `run_convert_r7.sh` WIDEACC mode → `rf{1,8,16}-wideacc/`. Audit
all_wide=True (every weighted accum frac≥16); C-sim 0.9974 (PASS; small drop from 0.9987 = the v3
output-precision widening perturbing the downstream act×act grid, flagged). Artifacts + full detail in
`hgq2/LEDGER.md` (2026-07-14) and the internal experiment log. No csynth, no cluster.

**Phase 10 — round-10 pairbias: pairwise-invariant attention-logit bias (Jul 24, `full-synthesis`).**
Added an opt-in ParT-style pairwise attention-logit bias to the HGQ2 **training** model
(`hgq2/bnhgq2/qat.py`), guarded on two new `arch` keys (`pair_bias`, `pair_bias_hidden`; absent/false
⇒ byte/param-identical to before — verified: `r8-small-w1a8-stdnn` `count_params` **19,201 unchanged**
vs pristine code). When enabled, a parameter-free `PairFeatures` layer computes per ordered constituent
pair `ln(ΔR²+ε)`, `ln(kT²+ε)`, `ln(|m²ᵢⱼ|+ε)` from the RAW 16-feature inputs (de-standardized inside
the layer via the train-split `input_std` stats, which `train.py` now forwards to `build_qat_model`);
a small **binary** MLP (3→8→ReLU→`n_heads`, same absmean-STE einsum machinery as the FFN) embeds them
once to `(H,T,T)` and adds the shared bias to **every** block's pre-softmax scores. Feature columns are
resolved by exact suffix from the canonical HLS4ML `particleFeatureNames` (hardcoded fallback + loud
print). New config `hgq2/configs/r10-small-w1a8-pairbias.json` (exact clone of `r8-small-w1a8-stdnn`
+ the two keys); 3 Jobs `jobs/training/variants/kai-bn10-pair-s{1,2,3}.yaml` (new ConfigMap
`kai-bn10-code` — `kai-bnf-code` left frozen, app `bnjet-r10`, stale-ConfigMap gate greps the unpacked
code for `pair_bias`) + `launch_r10.sh`. Conversion/verification stages untouched this round; the ROC
job loads the self-describing `.keras` (`PairFeatures` auto-registers via `import bnhgq2.qat`) — cold
process load verified bit-exact. `r10` `count_params` 19,285 (+84 vs r8); binary gate 17 bit-layers
exactly ±β. Local gates only; nothing launched.

### Docs & results artifacts touched (not code, logged for completeness)
`README.md`, `results/REPORT.md`, `results/RESULTS.md`, `results/hls_resource_table.md`,
`results/plots/README.md` — updated to past tense + figure pointers once csynth landed and figures
were generated. `results/csynth/*.json` — the three measured reports copied back from `mulder`.

---

## 3. The hls4ml / csynth bug & fix log  ← "were there bugs?"

**Short answer: yes, a handful — all in *our* glue code, all found and fixed; none in hls4ml itself.**
The final emulation is bit-accurate (corr = 1.000) and the synthesis is clean. We also corrected one
of *our own* documentation overclaims. Several scary-looking log lines turned out to be expected, not bugs.

### Real bugs we hit and fixed

1. **csynth driver written against the wrong hls4ml API.** `run_csynth.py` was first written for the
   0.8.1-style `build(csynth=…)` with a `DSP48E`/`ap_fixed<…>` report shape. `mulder` runs hls4ml
   **1.4.0**, where C-synthesis is `build(synth=True)`, the report nests under
   `report["CSynthesisReport"]`, the key is `DSP` (not `DSP48E`), and precision strings drop the `ap_`
   prefix (`fixed<32,16>`). **Fix:** rewrote the driver for the 1.x API + added a `csynth.xml` fallback parser.

2. **Accumulator overflow → bit-*in*accuracy.** First FFN emulation diverged (corr ≈ 0.24). Cause: the
   default `fixed<16,6>` accumulator saturates at ±32, but `fc1` sums 256 signed terms reaching ≈ ±50.
   **Fix:** pin `ap_fixed<32,16>` on the dense path (`HLS_WIDE`, documented in `stage_a_fix.py`).

3. **Over-widening the activation undid the fix.** Naively widening *everything* (including the
   `quantized_relu(8,2)` result) removed its `[0,4)` saturation; HLS activations blew up ~15×, `fc2`
   diverged (corr 0.85, max|diff| 1639). **Fix:** widen accum/result/bias on the Dense layers **only**,
   leave `act` native so the clip is preserved. Net result after 2+3: corr = **1.000**.

4. **protobuf / TF import crash on `mulder`.** TensorFlow wouldn't import until
   `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` was set. **Fix:** export it in the runbook/driver env.

5. **QAT activation calibration over-ranged the 4-bit grid (Jul 7).** The first
   `calibrate_activations` used max+margin then clipped `i` to `act_bits-1` — at A4 that forces
   `i=3, f=0`, an integer-only grid that collapses the LN-normalized sub-integer structure and
   drives training to NaN. **Fix:** MSE-per-tensor policy (argmin_i quant-MSE) with `i` floored
   so `f>=1` always (A4 → i∈{1,2}). Also matches the gold-model E4 finding that max+margin
   over-ranges low-bit grids.

6. **r5's `global_clipnorm=1.0` is numerically unsafe on the HGQ2/keras-3 deep-SubLN stack
   (Jul 7).** The LayerNorm backward Jacobian amplifies through the 8-block stack to ~2.5e20 on
   adversarial A4 batches; squaring that in the float32 global-norm computation overflows to Inf
   → the clip fails → NaN at step 1, **independent of LR** (verified: clipnorm at 5e-5, 2.5e-5,
   1.25e-5 all NaN — so the sanctioned LR-halving lever does NOT fix this mode). **Fix:**
   `clipvalue=1.0` (element-wise, no squaring, no overflow; near-no-op for the stable variants),
   shipped uniformly so the A8/A6/A4 comparison stays on one clip. `clip_mode` is a config knob.

9. **hls4ml 1.3.0 Vitis `ValidateResourceStrategy` crashes on EinsumDense (Jul 7, convert_final).**
   The QoR-warning pass matches `QEinsumDense` via the `'Dense'` substring of `'EinsumDense'`, but
   `get_layer_mult_size` returns `n_in=None` for einsum layers, so `rf > n_in` raises `TypeError`
   and kills the whole conversion (never hit before: rebuild probes used Latency or plain QDense).
   **Fix:** `convert_final.patch_resource_einsum_check()` guards the pass against `None` (it only
   prints a urem-core note). Note also: **EinsumDense is Latency-strategy-ONLY** in this hls4ml
   (`EinsumDense layer only supports Latency strategy for now`) — the full model cannot use
   Resource folding; convert_final defaults `strategy=Latency` (rf → einsum multiplier_limit).

10. **FFN/head ReLU unsigned-WRAP → 2^10 wrap (Jul 7, convert_final GATE 2).** `keras.layers.ReLU`
    is parsed by hls4ml 1.3.0 as `ParametrizedActivation('thresholdedrelu')` whose BitExact output
    precision is UNSIGNED + **WRAP**; a negative input wraps to ~2^i instead of clamping to 0. The
    binary FFN's fc1 is a 256-wide ±1 contraction of a ±4-clipped SubLN input → spans ±O(2^10), so
    ~half the ReLU outputs (the negatives keras zeroes) exploded to ~1024 (keras=0 vs hls=1023.999)
    → full-model C-sim corr **−0.06**, max|Δ|=1023.999=2^10. **Fix:** `convert_final.fix_relu_saturation`
    sets the relu-family output `saturation_mode=SAT` (negatives saturate to 0, bit-exact) via a new
    backward-compatible `convert(post_parse=…)` hook. **−0.06 → 0.9951**; post-fix the pipeline is
    bit-accurate through 8 blocks + GAP (gap corr 0.99996), the small residual being the table-SubLN
    ↔ coarse smoke-head-grid interaction at head_fc2 (degeneracy-amplified, clears 0.997 on a real
    checkpoint). Rule: **a ReLU whose input is a wide binary/int accumulator needs SAT, not WRAP.**

11. **Norm-free (round-8 stdnn) export GATE-1 grid saturation (Jul 18, convert_final).** The r8
    `arch.norm='none'` flagship exported at GATE-1 **0.7535** (grid collapse, not weights: |Δ|=1.5e-8).
    Root cause: with no LayerNorm the residual stream grows block-over-block AND the β-carry fold
    (γ_Wo·β_v etc.) leaves the ctx→Wo / relu(fc1)→fc2 / relu(head_fc1)→head_fc2 INPUTS scaled by 1/β
    (β dropped upstream, restored only in the downstream affine). Sizing those grids from the QAT
    sites' i-bits (`read_qat_act_ibits`) drops |ctx|≈54 into a SAT<8,1+2>=±4 grid → saturates hard
    (e/q stream std ratio 0.63). **Fix:** `convert_final.calibrate_norm_free_grids` (two-pass, guarded
    on arch.norm=='none', NO-OP for normed graphs): widen those 5 internal-stream input grids to cover
    the measured non-clipping export-side max (i=ceil(log2 max)+1) while KEEPING f=act_bits-1 — widen
    total width 8→15-17, not steal fraction bits (the `build.py.stream_iq` exact-passthrough precedent).
    Residual/normalized dense inputs untouched (reproduce QAT at 8-bit bit-for-bit). `_apply_exact_beta`
    also now carries β for norm-free (else the exact-β reference collapses to 0.036). **0.7535 → 0.98907**
    (n=4096); GATE-2 C-sim stays **1.0 bit-exact** (wide grids convert cleanly), both rf1 & rf8. **Genuine
    ceiling, characterized:** 0.989 is the DSP-free hardware-faithful limit — CSD-2 β-snap on the norm-free
    PRODUCT gammas (γ=β·β_upstream≈0.02, 2-digit CSD rel-err 2-7%) has no LN to absorb it (exact-β → 0.994,
    a DSP tradeoff; keras copy ceiling 0.996). Normed path verified byte-identical (r7-small-w1a8-s2 rf8
    C-sim 0.9986957710291886 exactly). Rule: **a norm-free β-carry inflates the post-fold internal streams
    by 1/β — size those input grids from measured export-side ranges, not the QAT i-bits.**

### A documentation overclaim we corrected (our claim, not an hls4ml bug)

7. **"DSP = 0 *and* BRAM = 0."** True only for the fully-unrolled **RF=1 / Latency** design. A folded,
   deployable design (**RF=256 / Resource**) parks the binary weights in **~1.2% BRAM**. We reconciled
   every doc to the folded numbers. **DSP = 0 is the fold-independent structural win** (binary weights →
   no multipliers); BRAM = 0 was not.

### Design choices that look like bugs but aren't

8. **RF=1 is intractable**, not broken — fully unrolling the 262144-MAC layers is the issue. We synthesize
   at **RF=256** (II=256), which is tractable *and* deployable.

### Scary log lines that are **expected**, not bugs

- `config_array_partition -maximum_size … not supported` (Vitis HLS 200-642) — benign warning, ignored.
- `Vivado synthesis report not found / Cosim report not found / Timing report not found` — **expected**:
  we ran `synth=True` (C-synthesis) only, not `csim`/`cosim`/`vsynth`/Vivado logic synthesis.
- Keras `HeNormal` unseeded + TF autograph warnings — cosmetic.

### Honest scope boundary (a capability limit, not a bug)

- **FULL-MONOLITH C-sim of the FINAL QAT model — GATE 2 fixed, GATE 1 boundary characterized
  (Jul 7, convert_final).** `convert_final.py` runs the whole chain on the real `final-smoke` w1a8
  ckpt (fetch→load→export→GATE1→convert→GATE2→EBOPs→tarball). Weights export exact (|Δ|=7.5e-9).
  (a) **GATE 2** hit the ReLU unsigned-WRAP (§3 bug 10) — **fixed** (relu output SAT): full-model
  C-sim **−0.06 → 0.9951**, and the pipeline is bit-accurate through 8 blocks + GAP (gap corr
  **0.99996**); the small logit residual is the table-SubLN ↔ coarse smoke-head-grid interaction at
  head_fc2 (degeneracy-amplified — the 5-class head of a val-AUC-0.53 model — clears 0.997 on a real
  checkpoint; only lever is the shared SubLN 1/√ table, deferred to keep the rebuild bit-identical).
  (b) **GATE 1** export↔QAT = **0.9618** (weights exact; residual = build.py's independently-sized
  attention softmax/stream grids not bit-copying the QAT quantizers under the β-fold). Opt-in
  `attn_grids='copy_qat'` reproduces them via `scaler=1/β` → **0.9901**, but hls4ml's HGQ2 frontend
  ignores that datalane scaler (source-read + measured: copy_qat gate-2 collapses to 0.96), so it is
  a KERAS-only ceiling; `build_default` ships. Exact hardware reproduction would need non-power-of-2
  scale support in `extract_fixed_quantizer_config` or ±β projection weights (DSP cost). The
  per-block/per-shape probe path (§6′) remains the deployable mulder route at large D256/L8 scale.
- hls4ml 0.8.1 could **not** convert LayerNorm (SubLN) or EinsumDense attention; hls4ml ≥ 1.2 added that
  support (`full_transformer_probe.py`). LayerNorm is convertible-but-fragile (`io_parallel` only). §B's numbers
  cover the binary FFN block (the dominant primitive); **§B′ (Phase 6) now extends this to the full trained
  transformer** — all 51 BitLinears + 51 SubLN norms + the 4 weighted attention projections, synthesized with
  real trained weights.
- **The one remaining gap (Phase 6):** hls4ml's parser only converts layer *types* with a registered handler, so
  the custom `BitLinear`/`RMSNorm`/`BitMHSA` subclasses can't be ingested from the `.h5` directly — they were
  rebuilt as `LayerNormalization→QActivation→QDense(binary)` with weights ported in. The attention **score core**
  (Q·Kᵀ / softmax / ·V — *weightless*, 0.65 % of MACs) uses `EinsumDense`, which **does not convert** on this
  hls4ml (verified: probe Stages C & D fail). So §B′ covers **100 % of the weights and 99.35 % of the MACs**; the
  weightless score core is handled analytically (`resource_model.py`), not synthesized. This is a real, documented
  boundary — not a silent omission.
