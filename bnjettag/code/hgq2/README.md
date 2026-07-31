# HGQ2 rebuild pipeline (`code/hgq2/`)

Config-driven pipeline that takes a trained BitNet QKeras checkpoint through:

```
(a) config ──▶ (b) HGQ2 rebuild (binary PINNED {−1,+1}) ──▶ (c) weight port +
fidelity gate (corr + AUC vs the verified r5 .npz) ──▶ (d) EBOPs ──▶
(e) hls4ml convert (+ csynth on mulder) ──▶ (f) ROC, log-FPR ──▶ (g) results store
```

Every stage is keyed by the **config hash**; results land in
`../../results/hgq2/runs/<hash8>/` with a manifest in `../../results/hgq2/manifest.json`.

> Note on paths in this file: the `roc-results/r5/*.npz` reference arrays used by the
> stage-(c) fidelity gate, and the per-config `runs/` stores, belong to an earlier campaign
> and are not part of this snapshot. The gate and the store layout are documented here
> because the code still implements them; the results that are published live under
> `../../r7/` and `../../roc-results/`.
A new model = a new JSON in `configs/` through the same stages — nothing re-derived.

## Where the model is defined

There is no single `model.py`; the network is built from a config JSON in code:

- `bnhgq2/qat.py` — the trainable model (`build_qat_model`): input projection, the two
  encoder blocks (einsum-dense QKV and output projections, softmax attention, feed-forward),
  pooling and head, with the binary {−1,+1} weight quantizers attached. This file plus the
  config JSON *is* the model; the headline configuration is `configs/r8-small-w1a8-stdnn.json`.
- `bnhgq2/train.py` — the training loop (optimizer, schedule, early stopping, W&B logging),
  entered through `run_stage.py train --config configs/<name>.json --seed N`.
- `bnhgq2/build.py` — the same architecture rebuilt as the static hardware graph for
  conversion (static activation quantizers in place of the trainable ones).
- `bnhgq2/convert.py` — the hls4ml call itself (`hls4ml.converters.convert_from_keras_model`,
  Vitis backend, `bit_exact=True`) and the project write for synthesis.
- `convert_final.py` — the end-to-end driver: fetch checkpoint → static export → gate 1
  (export against the trained model, correlation ≥ 0.9999) → hls4ml → gate 2 (C simulation
  against the export on real jets, correlation ≥ 0.997) → synthesis tarball.

## Layout

| Path | Role |
| --- | --- |
| `configs/*.json` | One model+quantization spec per file (arch, checkpoint, act bits, quantizer policy). |
| `bnhgq2/config.py` | Config load/validate/hash. |
| `bnhgq2/qat.py` | Config → trainable QAT model (`build_qat_model`) — the model definition. |
| `bnhgq2/train.py` | Training loop (schedule, early stopping, W&B); entered via `run_stage.py train`. |
| `bnhgq2/extract.py` | h5py-only extraction of latent kernels/biases + the folded PE constant from a QKeras `.h5` (no TF-2.11 needed). |
| `bnhgq2/binarize.py` | AbsMean binarization math (α/β/sign) + the β-fold bookkeeping (which β is folded where, exactly). |
| `bnhgq2/data.py` | numpy/h5py-only era-2 loader replicating `make_roc.py` exactly (sorted glob, per-jet pT re-sort, top-N). |
| `bnhgq2/build.py` | config → HGQ2 model (binary pinned, static act quantizers). |
| `bnhgq2/port.py` | extracted weights → HGQ2 model (+ datalane calibration). |
| `bnhgq2/verify.py` | fidelity gates: rebuild↔trained corr + macro-OvR AUC vs `roc-results/r5/*.npz`; HGQ2↔hls4ml bit-exactness. |
| `bnhgq2/ebops_calc.py` | native HGQ2 EBOPs per config. |
| `bnhgq2/convert.py` | hls4ml (Vitis backend) conversion + project write for mulder csynth. |
| `bnhgq2/subln.py` | SubLN hls4ml extension: `PSubLN` keras layer + keras-v3 handler + `SubLN` IR + BitExact kif hooks + Vitis/Vivado templates + `register_subln()`. |
| `bnhgq2/compat.py` | Shims: keras-3.15 `EinsumDense.full_output_shape`, hgq MHA registry alias, `patch_project_for_macos()` for local C-sim. |
| `hls_templates/nnet_subln.h` | SubLN C++ kernel with range-reduced 1/√ table (valid for any var magnitude, unlike stock `nnet_layernorm.h`). |
| `test_subln.py` | SubLN C-sim acceptance gate: shapes × (corr, max-err) vs keras float; run before any csynth handoff. |
| `bnhgq2/store.py` | results store (JSON per stage + manifest). |
| `probe_binary_pinning.py` | the empirical test matrix that established the binary recipe (run once, kept as evidence). |
| `run_stage.py` | CLI entry: `python run_stage.py <stage> --config configs/X.json`. |
| `LEDGER.md` | running change ledger for this effort (dated, newest on top). |

## Environment

Local venv at repo root: `../../../.venv-hgq2` (Python 3.12, hgq2 0.1.9, hls4ml 1.3.0,
keras 3.15, TF 2.21 backend). Synthesis runs on mulder only (see `../hls/RUN_CSYNTH_ON_VITIS.md`).

## Fidelity gates (what "verified" means here)

1. **rebuild ↔ trained**: Pearson corr of softmax scores + recovered macro-OvR AUC on the
   full 260k-jet era-2 val split, against the *stored, verified* `roc-results/r5/*.npz`.
   Bit-exactness vs the trained model is impossible by construction (the QKeras model uses
   dynamic per-token activation scales; hardware needs static) — the substitution is the
   same one `code/hls/full_model_csynth.py` made, and the correlation is reported, not hidden.
2. **HGQ2 ↔ hls4ml**: bit-exactness of the converted model against the HGQ2 forward pass
   (HGQ2's design guarantee; checked on real jets, not random noise).
