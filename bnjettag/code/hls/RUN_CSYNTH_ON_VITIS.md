# Run the csynth on the group's Vitis box — step-by-step

This fills the only pending cells in `results/hls_resource_table.md` §B: the exact **LUT / FF /
Latency-in-cycles / II** for the binary FFN at A8/A6/A4. NRP has no Xilinx backend, so this runs on the
group's **Vitis 2023.2** machine. **C-synthesis needs no license** (only bitstream place-and-route does).

> **VERIFIED 2026-06-24 on `mulder` (mulder.t2.ucsd.edu).** The exact procedure below was run end-to-end:
> Vitis HLS 2023.2 lives at `/data/software/xilinx/Vitis_HLS/2023.2/`, and the existing `bnjet` micromamba env
> already ships **hls4ml 1.4.0** (NOT 0.8.1) + qkeras + tf 2.11 — `run_csynth.py` is written for that 1.x API
> (`build(synth=True)`, no `csynth=` kwarg). Two gotchas that bit us and are now baked into the steps:
> **(1)** set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` or the TF import dies on a protobuf version clash;
> **(2)** RF=1 fully unrolls the 256→1024 layer and is intractable here — synthesize at a **folded** operating
> point (`HLS_RF=256` → II=256, ~50 min/precision, 8.5 GB, fits VU13P at ~25% LUT). Sample A8 result:
> `DSP=0, BRAM_18K=64, FF=251331, LUT=440882, Latency=520 cyc, II=256, Fmax≈568 MHz`.

`run_csynth.py` is **self-contained**: it rebuilds the binary FFN from scratch and synthesizes it, so you do
**not** need the PVC or the pre-emitted `binary_ffn_a*_prj` projects — just this one script, a Python env, and
`vitis_hls` on `PATH`.

> **What gets synthesized:** the dominant primitive — one binary FFN block `fc1 (256→1024) → relu →
> fc2 (1024→256)`, binary `{−1,+1}` weights, activations at A8/A6/A4 — identical to `sweep_precision.py`
> (the design we already proved bit-accurate, corr = 1.0). Not the full transformer; the FFN is the cost
> driver and what the table reports.

---

## 1. Get the script onto the box
Copy the repo (or just `code/hls/run_csynth.py`) to the Vitis machine. Everything below assumes you're in the
repo root.

## 2. Python env
**On `mulder`:** just use the existing `bnjet` micromamba env — it already has hls4ml 1.4.0 / qkeras / tf 2.11:
```bash
PY=/home/users/kayamaguchi/micromamba/envs/bnjet/bin/python
```
**On any other box:** make a fresh env. `run_csynth.py` targets the **hls4ml 1.x** API, so install 1.x (not 0.8.1):
```bash
conda create -n csynth python=3.10 -y && conda activate csynth      # or python3.10 -m venv
pip install "hls4ml>=1.0" qkeras "tensorflow==2.11.1" "pyparsing<3" scikit-learn keras-tuner "numpy<2"
PY=python
```

## 3. Put Vitis HLS on PATH

> **CORRECTION 2026-07-02 (verified by smoke test on mulder):** hls4ml 1.4.0.dev's Vitis
> backend launches synthesis through **`vitis-run`** (the unified CLI), which lives in the
> **full Vitis** install — NOT in `Vitis_HLS/2023.2/bin`. Sourcing only the Vitis_HLS
> settings64.sh makes `hls_model.build(synth=True)` die with
> `Exception: Vitis installation not found. Make sure "vitis-run" is on PATH.`
> The June-25/26 full-model runs worked because they had the full Vitis bin on PATH
> (see `out/csynth_a8.log`: `vitis-run v2023.2 ... Launching vitis_hls`). Use:

```bash
source /data/software/xilinx/Vitis/2023.2/settings64.sh       # FULL Vitis (provides vitis-run)
command -v vitis-run && command -v vitis_hls                  # BOTH must print a path
# (vitis_hls also resolves from the full-Vitis tree; the Vitis_HLS-only source is not enough)
```

## 4. Run it
```bash
export CUDA_VISIBLE_DEVICES=-1                       # CPU-only; no GPU needed for csynth
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python # REQUIRED: else TF import dies on a protobuf clash
export HLS_OUT="$PWD/out"                            # where the JSONs land
export HLS_RF=256                                    # folded operating point (II=256). RF=1 is intractable here.
export HLS_ABITS=8,6,4                               # the three precisions (or "8" to smoke-test one)
# optional overrides (defaults shown):
# export HLS_PART=xcvu13p-flga2577-2-e   # the VU13P part the thesis targets
# export HLS_CLOCK_NS=2.5                # 2.5 ns = 400 MHz
# export HLS_BACKEND=Vitis               # use Vivado if that's what's installed
$PY code/hls/run_csynth.py               # on mulder run from ~/csynth where run_csynth.py was scp'd
```
C-synthesis at RF=256 is **~50 min per precision** (8.5 GB peak) — budget ~2.5 h for all three. (RF=1 would be
"lowest latency / II=1" but the 256→1024 layer fully unrolls and won't synthesize at this model size; the table
reports the folded RF=256 operating point, which already fits the VU13P at ~25 % LUT.)

## 5. What it produces
```
$HLS_OUT/csynth_report_a8_rf256.json
$HLS_OUT/csynth_report_a6_rf256.json
$HLS_OUT/csynth_report_a4_rf256.json
```
Each `row` has exactly the table columns:
```json
{ "precision":"A8", "BRAM_18K":64, "DSP":0, "FF":251331, "LUT":440882,
  "LatencyCyclesWorst":520, "LatencyCyclesBest":518, "IntervalII":256,
  "ReuseFactor":256, "part":"xcvu13p-flga2577-2-e", "clock_ns":2.5 }
```

## 6. Send back
Either the three JSON files, or just paste the numbers per precision
(`BRAM_18K / DSP / FF / LUT / LatencyCyclesWorst / IntervalII`). They drop straight into the bold-able cells of
`results/hls_resource_table.md` §B. **Sanity check we expect:** **`DSP = 0` at every precision** — this is the
structural, fold-independent result (binary weights can't drive a multiplier port); if DSP comes back non-zero,
the binary mapping didn't hold and we should look. **`BRAM_18K` is small-but-nonzero (~64) at RF=256** and that
is *expected* — the Resource-strategy fold stores the folded weights in block RAM. (BRAM only hits 0 in the
fully-unrolled RF=1/Latency design, which we don't synthesize because it's intractable at this size.)

---

### Troubleshooting
- **`vitis_hls: command not found`** → you didn't source `settings64.sh` (step 3), or the install path
  differs — find it with `find /tools /opt -name settings64.sh 2>/dev/null | grep -i vitis`.
- **"part not found / unsupported"** → your Vitis install may not ship VU13P; set `HLS_PART` to an installed
  UltraScale+ part (e.g. `xcvu9p-flga2104-2-e`, which matches the published Ngadiuba anchor) and note which
  part you used so the table can cite it.
- **`build()` returns nothing / empty report** → the script already falls back to
  `hls4ml.report.read_vivado_report(<prj>)`; if a row still has `null` LUT/FF, send me the
  `csynth_out/binary_ffn_a*_prj/myproject_prj/solution1/syn/report/*_csynth.rpt` and I'll parse it.
- **Vivado instead of Vitis** → set `HLS_BACKEND=Vivado`; `run_csynth.py` honors it.
