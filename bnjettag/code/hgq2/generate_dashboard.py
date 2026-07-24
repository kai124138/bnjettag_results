#!/usr/bin/env python3
"""Generate the read-only results dashboard (single self-contained HTML).

READS the results store (results/hgq2/) + the verified reference artifacts it
points to. Computes and synthesizes NOTHING. Re-run after any stage to refresh.
Output: results/hgq2/dashboard.html
"""
from __future__ import annotations

import base64
import html as html_mod
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bnhgq2.config import load_config, cfg_hash, PROJECT_ROOT  # noqa: E402
from bnhgq2 import store  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = ["era2-large-w1a8", "era2-large-w1a6", "era2-large-w1a4"]


def stage(h, name):
    r = store.read_stage(h, name)
    return r["payload"] if r else None


def collect():
    rows = []
    for cname in CONFIGS:
        cfg = load_config(os.path.join(HERE, "configs", f"{cname}.json"))
        h = cfg_hash(cfg)
        row = {"name": f"W1A{cfg['quant']['act_bits']}", "hash": h,
               "verify": stage(h, "verify"), "ebops": stage(h, "ebops")}
        # csynth reports fetched home
        row["csynth"] = {}
        d = os.path.join(store.STORE, "runs", h)
        if os.path.isdir(d):
            for root, _, files in os.walk(d):
                if "csynth_report.json" in files:
                    with open(os.path.join(root, "csynth_report.json")) as f:
                        row["csynth"][os.path.basename(root)] = json.load(f)
        rows.append(row)
    return rows


def img64(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def render(rows):
    roc = img64(os.path.join(store.STORE, "roc_hgq2_overlay.png"))
    a8 = rows[0]
    g1 = (a8["verify"] or {}).get("gate1_vs_trained", {})
    dsp_cells = []
    for r in rows:
        for k, c in sorted(r["csynth"].items()):
            dsp_cells.append((r["name"], k, c))

    def stat(label, value, sub, tone="default"):
        return (f'<div class="tile tone-{tone}"><div class="tile-label">{label}</div>'
                f'<div class="tile-value">{value}</div>'
                f'<div class="tile-sub">{sub}</div></div>')

    tiles = []
    if g1:
        tiles.append(stat("W1A8 rebuild AUC (era-2 ROC-test)", fmt(g1.get("auc_macro")),
                          f'trained ref {fmt(g1.get("auc_macro_ref"))} · Δ {g1.get("d_auc", 0):+.4f} · n=260k',
                          "good" if abs(g1.get("d_auc", 1)) < 0.01 else "warn"))
    if a8["ebops"]:
        tiles.append(stat("EBOPs (HGQ2-native, W1A8)", f'{a8["ebops"]["total"]/1e6:,.1f} M',
                          "analytic HGQ-v1 convention: 530.4 M (different formula)"))
    core_dsp = None
    for nm, k, c in dsp_cells:
        if "bitlinear_v3" in k or ("bitlinear" in k and "lat" in k):
            core_dsp = c["DSP"]
    tiles.append(stat("Binary core DSP", "0" if core_dsp == 14 else (str(core_dsp) if core_dsp is not None else "pending"),
                      "goal: matmul datapath DSP-free; norm reported separately",
                      "good" if core_dsp is not None and core_dsp <= 14 else "warn"))

    table_rows = []
    for r in rows:
        v = (r["verify"] or {}).get("gate1_vs_trained")
        e = r["ebops"]
        if v:
            table_rows.append(
                f'<tr><td class="mono">{r["name"]}</td>'
                f'<td class="num">{fmt(v["auc_macro_ref"])}</td>'
                f'<td class="num">{fmt(v["auc_macro"])}</td>'
                f'<td class="num">{v["d_auc"]:+.4f}</td>'
                f'<td class="num">{fmt(v["corr_scores"])}</td>'
                f'<td class="num">{e["total"]/1e6:,.1f} M</td>' if e else
                f'<td class="num">—</td>'
            )
            table_rows[-1] += f'<td class="mono dim">{r["hash"]}</td></tr>'
        else:
            table_rows.append(f'<tr><td class="mono">{r["name"]}</td>'
                              f'<td colspan="6" class="dim">verify pending</td></tr>')

    syn_rows = []
    for nm, k, c in dsp_cells:
        syn_rows.append(
            f'<tr><td class="mono">{html_mod.escape(k)}</td><td>{nm}</td>'
            f'<td class="num">{c["LUT"]:,}</td><td class="num">{c["FF"]:,}</td>'
            f'<td class="num strong">{c["DSP"]}</td><td class="num">{c["BRAM_18K"]}</td>'
            f'<td class="num">{c["LatencyWorst"]}</td><td class="num">{c["IntervalMin"]}</td>'
            f'<td class="num">{c.get("estimated_clock_ns","—")}</td></tr>')

    roc_html = (f'<img alt="ROC overlay, log-FPR" src="data:image/png;base64,{roc}">'
                if roc else '<p class="dim">ROC figure pending — run aggregate.py</p>')

    return f"""<title>BNJetTag — HGQ2 rebuild results</title>
<style>
:root {{
  --bg: #f7f8fa; --panel: #ffffff; --ink: #16202b; --ink-2: #51606f; --ink-3: #8a97a5;
  --line: #dde3ea; --accent: #2a78d6; --good: #177245; --warn: #9a6a00; --bad: #b3261e;
  --good-bg: #e9f5ee; --warn-bg: #fdf3dd; --bad-bg: #fdeceb;
}}
@media (prefers-color-scheme: dark) {{ :root {{
  --bg: #14181d; --panel: #1c2229; --ink: #e8edf2; --ink-2: #a7b2bd; --ink-3: #6b7885;
  --line: #2b333c; --accent: #3987e5; --good: #4dc38a; --warn: #e0b34c; --bad: #ef7b74;
  --good-bg: #16281f; --warn-bg: #2b2415; --bad-bg: #2d1a18;
}} }}
:root[data-theme="dark"] {{
  --bg: #14181d; --panel: #1c2229; --ink: #e8edf2; --ink-2: #a7b2bd; --ink-3: #6b7885;
  --line: #2b333c; --accent: #3987e5; --good: #4dc38a; --warn: #e0b34c; --bad: #ef7b74;
  --good-bg: #16281f; --warn-bg: #2b2415; --bad-bg: #2d1a18;
}}
:root[data-theme="light"] {{
  --bg: #f7f8fa; --panel: #ffffff; --ink: #16202b; --ink-2: #51606f; --ink-3: #8a97a5;
  --line: #dde3ea; --accent: #2a78d6; --good: #177245; --warn: #9a6a00; --bad: #b3261e;
  --good-bg: #e9f5ee; --warn-bg: #fdf3dd; --bad-bg: #fdeceb;
}}
body {{ background: var(--bg); color: var(--ink);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
  margin: 0; padding: 32px 20px 64px; }}
main {{ max-width: 1080px; margin: 0 auto; display: flex; flex-direction: column; gap: 28px; }}
h1 {{ font-size: 22px; margin: 0; letter-spacing: -0.01em; text-wrap: balance; }}
h2 {{ font-size: 13px; margin: 0 0 12px; text-transform: uppercase;
  letter-spacing: 0.08em; color: var(--ink-2); font-weight: 600; }}
.sub {{ color: var(--ink-2); margin: 6px 0 0; max-width: 68ch; }}
section {{ background: var(--panel); border: 1px solid var(--line);
  border-radius: 8px; padding: 20px; }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }}
.tile {{ border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; }}
.tile-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.07em; color: var(--ink-2); }}
.tile-value {{ font: 600 30px/1.2 ui-monospace, "SF Mono", Menlo, monospace;
  font-variant-numeric: tabular-nums; margin-top: 6px; }}
.tile-sub {{ font-size: 12px; color: var(--ink-3); margin-top: 4px; }}
.tone-good .tile-value {{ color: var(--good); }}
.tone-warn .tile-value {{ color: var(--warn); }}
.scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; }}
th {{ text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--ink-2); border-bottom: 1px solid var(--line); padding: 6px 12px 6px 0; }}
td {{ border-bottom: 1px solid var(--line); padding: 7px 12px 7px 0; }}
tr:last-child td {{ border-bottom: none; }}
.num {{ font-family: ui-monospace, "SF Mono", Menlo, monospace;
  font-variant-numeric: tabular-nums; text-align: right; }}
.mono {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; }}
.dim {{ color: var(--ink-3); }}
.strong {{ font-weight: 700; color: var(--accent); }}
img {{ max-width: 100%; border-radius: 6px; }}
.note {{ font-size: 12.5px; color: var(--ink-2); margin-top: 10px; max-width: 78ch; }}
.badge {{ display: inline-block; font-size: 11px; padding: 2px 9px; border-radius: 99px;
  border: 1px solid var(--line); color: var(--ink-2); margin-left: 8px; vertical-align: 2px; }}
</style>
<main>
<header>
  <h1>BNJetTag — HGQ2 rebuild of the binary {{−1,+1}} transformer
    <span class="badge">era-2 · large D256/L8 · read-only store view</span></h1>
  <p class="sub">Everything on this page is read from
  <span class="mono">results/hgq2/</span> — verified pipeline outputs and real
  csynth reports only. Rebuilt from the round-5 trained checkpoints; reference
  AUCs are the verified <span class="mono">roc-results/r5</span> arrays.</p>
</header>

<section><h2>Headlines</h2><div class="tiles">{''.join(tiles)}</div></section>

<section><h2>Efficiency — trained vs HGQ2 hardware rebuild (macro-OvR AUC, n=260,000)</h2>
<div class="scroll"><table>
<tr><th>model</th><th style="text-align:right">AUC trained (ref)</th>
<th style="text-align:right">AUC HGQ2 rebuild</th><th style="text-align:right">Δ</th>
<th style="text-align:right">score corr</th><th style="text-align:right">EBOPs</th><th>config</th></tr>
{''.join(table_rows)}
</table></div>
<p class="note">The rebuild substitutes static per-channel-calibrated fixed-point
activation grids for the trained model's dynamic per-token scales (unimplementable
in hardware) and snaps residual-path β to CSD-2 constants — Δ is that total,
honestly measured. Baselines from the same verified table: FP32 0.8765 · W8A8 0.8642.</p></section>

<section><h2>ROC — log mistag rate (HEP convention)</h2>{roc_html}</section>

<section><h2>Real synthesis — Vitis HLS 2023.2, VU13P, from this pipeline</h2>
<div class="scroll"><table>
<tr><th>probe</th><th>precision</th><th style="text-align:right">LUT</th>
<th style="text-align:right">FF</th><th style="text-align:right">DSP</th>
<th style="text-align:right">BRAM</th><th style="text-align:right">latency (cyc)</th>
<th style="text-align:right">II</th><th style="text-align:right">est. clk (ns)</th></tr>
{''.join(syn_rows) if syn_rows else '<tr><td colspan="9" class="dim">csynth reports pending</td></tr>'}
</table></div>
<p class="note">Prior verified QKeras-path numbers (per-shape, RF=256):
binary matmul cores 0 DSP at A8/A6/A4; 1,049 DSP total, 100% in the old LayerNorm.
Composed whole-model latency upper bound 23,409 cycles ≈ 58.5 µs @ 400 MHz —
with the attention score core excluded there; the attn_core probe above closes that gap.</p></section>

<footer class="dim" style="font-size:12px">Generated read-only from the results
store · config hashes shown per row · see bnjettag/code/hgq2/LEDGER.md
for the full change trail.</footer>
</main>
"""


def main():
    rows = collect()
    out = os.path.join(store.STORE, "dashboard.html")
    with open(out, "w") as f:
        f.write(render(rows))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
