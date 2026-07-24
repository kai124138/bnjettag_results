# code/ — which code is current (decoder, 2026-07-05)

Five subfolders, two of them current. If you're asking "what do we actually run?":

| Folder | What it is | Status |
| --- | --- | --- |
| **`training/`** | **THE trainer** — `qkerasModel.py`, one file, all variants via `BN_*` env knobs (`BN_VARIANT`, `BN_ACT_BITS`, `BN_D_MODEL`, …). Runs on NRP as ConfigMap `kai-qkerasmodel-r5` (md5-verified against this file 2026-07-05). Also: `make_roc.py`/`ROC.py` (ROC eval), `ebops.py` (analytic EBOPs). | **CURRENT** |
| **`hgq2/`** | **The conversion → verification → EBOPs → synthesis pipeline** (HGQ2 + hls4ml, era 2). Entry point `run_stage.py` over `configs/*.json`; `probes.py` builds the mulder synthesis probes; `fetch_mulder_reports.sh` brings csynth numbers into the store; `LEDGER.md` is the dated change trail. Start at `hgq2/README.md`. | **CURRENT** |
| `jobs/` | Kubernetes job YAMLs. **Current: `training/variants/kai-bn6s-*.yaml`** (round-6-small, fire-ready) + `launch_r6s_staged.sh`; `variants/kai-bn5-*` = round 5 (done). YAMLs directly in `training/` = era-1 (frozen). `hls/` = old NRP HLS jobs (superseded — synthesis moved to mulder). | mixed — see left |
| `hls/` | Era-1 QKeras-path synthesis scripts (`run_csynth.py`, probes). Produced the era-1-shape DSP-0 numbers quoted in the archived table `../../archive/era1/results/hls_resource_table.md`. Superseded for new work by `hgq2/`, kept as the QKeras-path reference. | frozen reference |
| `plots/` | Era-1 figure generator (`make_results_plots.py`). Current figures come from `../../figures/make_figures.py`. | frozen |

`CHANGELOG.md` here covers the era-1 run's code changes; the current pipeline's ledger is
`hgq2/LEDGER.md`.
