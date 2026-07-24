#!/usr/bin/env python
"""
run_csynth.py — exact LUT/FF/DSP/BRAM + latency-in-cycles via Vitis HLS C-synthesis.

Rebuilds the binary FFN block used in `sweep_precision.py` (the dominant primitive:
fc1 256->1024 -> relu -> fc2 1024->256, binary {-1,+1} weights, A8/A6/A4 activations),
runs hls4ml C-synthesis, and writes one JSON per precision:

    $HLS_OUT/csynth_report_a{8,6,4}.json
    { "precision":"A8", "BRAM_18K":0, "DSP":0, "FF":<int>, "LUT":<int>,
      "LatencyCyclesWorst":<int>, "LatencyCyclesBest":<int>, "IntervalII":<int>,
      "ReuseFactor":<int>, "part":..., "clock_ns":2.5 }

API NOTE: written for the hls4ml 1.x installed on mulder (1.4.0.dev). C-synthesis is
`build(synth=True)` (there is NO `csynth=` kwarg); the report comes back nested under
`report["CSynthesisReport"]` with keys LUT/FF/DSP/BRAM_18K/BestLatency/WorstLatency/
IntervalMin/IntervalMax. A csynth.xml fallback parser is included for robustness.

KNOBS (env):
  HLS_OUT     output dir            (default ./csynth_out)
  HLS_PART    FPGA part             (default xcvu13p-flga2577-2-e, the thesis target)
  HLS_BACKEND Vitis | Vivado        (default Vitis)
  HLS_CLOCK_NS clock period ns      (default 2.5 = 400 MHz)
  HLS_WIDE    accum/result precision(default fixed<32,16>, 1.x form; pinned for bit-accuracy)
  HLS_RF      reuse factor / fold   (default 1 = fully parallel, II=1; the 256x1024 layer
                                     fully unrolls at RF=1 and can be slow/large to synth, so
                                     set e.g. HLS_RF=256 for a tractable folded operating point)
  HLS_ABITS   activation precisions (default "8,6,4"; set "8" to smoke-test one point)
"""
import os, json, glob
import xml.etree.ElementTree as ET
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import hls4ml
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from qkeras import QDense, QActivation, binary, quantized_bits

OUT     = os.environ.get("HLS_OUT", os.path.abspath("csynth_out"))  # local default; NRP Job sets HLS_OUT=/data/outputs/hls
D, FFN  = 256, 1024
PART    = os.environ.get("HLS_PART", "xcvu13p-flga2577-2-e")
BACKEND = os.environ.get("HLS_BACKEND", "Vitis")
CLOCK   = float(os.environ.get("HLS_CLOCK_NS", "2.5"))
WIDE    = os.environ.get("HLS_WIDE", "fixed<32,16>")               # 1.x precision form (no ap_ prefix)
RF      = int(os.environ.get("HLS_RF", "1"))
ABITS   = tuple(int(a) for a in os.environ.get("HLS_ABITS", "8,6,4").split(","))


def build_ffn(abits):
    """Same binary FFN block as sweep_precision.py (the dominant primitive)."""
    inp = Input(shape=(D,), name="ffn_in")
    x = QDense(FFN, kernel_quantizer=binary(alpha=1),
               bias_quantizer=quantized_bits(8, 3, alpha=1), name="fc1")(inp)
    x = QActivation(f"quantized_relu({abits},2)", name="act")(x)
    x = QDense(D, kernel_quantizer=binary(alpha=1),
               bias_quantizer=quantized_bits(8, 3, alpha=1), name="fc2")(x)
    return Model(inp, x, name=f"binary_ffn_a{abits}")


def make_cfg(model):
    cfg = hls4ml.utils.config_from_keras_model(
        model, granularity="name", backend=BACKEND,
        default_precision=WIDE, default_reuse_factor=RF)
    cfg.setdefault("Model", {})
    cfg["Model"]["ReuseFactor"] = RF
    cfg["Model"]["Strategy"]    = "Latency" if RF == 1 else "Resource"
    for ln in ("ffn_in", "fc1", "fc2"):
        p = cfg["LayerName"].setdefault(ln, {}).setdefault("Precision", {})
        for k in ("result", "accum", "bias"):
            p[k] = WIDE
    for ln in ("fc1", "fc2"):
        cfg["LayerName"].setdefault(ln, {})["ReuseFactor"] = RF
    return cfg


def _flat(report):
    """hls4ml 1.x nests the numbers under CSynthesisReport."""
    if isinstance(report, dict) and "CSynthesisReport" in report:
        return report["CSynthesisReport"]
    return report if isinstance(report, dict) else {}


def _pull(report, *keys, default=None):
    for k in keys:
        if isinstance(report, dict) and k in report and report[k] not in (None, ""):
            return report[k]
    return default


def _xml_fallback(prj):
    """Parse the Vitis csynth.xml directly (stable schema) if the dict came back empty."""
    cands = glob.glob(f"{prj}/**/syn/report/csynth.xml", recursive=True)
    if not cands:
        return {}
    root = ET.parse(sorted(cands)[0]).getroot()
    g = lambda path: (root.find(path).text if root.find(path) is not None else None)
    return {
        "LUT":          g("./AreaEstimates/Resources/LUT"),
        "FF":           g("./AreaEstimates/Resources/FF"),
        "DSP":          g("./AreaEstimates/Resources/DSP") or g("./AreaEstimates/Resources/DSP48E"),
        "BRAM_18K":     g("./AreaEstimates/Resources/BRAM_18K"),
        "WorstLatency": g("./PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency"),
        "BestLatency":  g("./PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency"),
        "IntervalMax":  g("./PerformanceEstimates/SummaryOfOverallLatency/Interval-max"),
        "IntervalMin":  g("./PerformanceEstimates/SummaryOfOverallLatency/Interval-min"),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"[cfg] backend={BACKEND} part={PART} clock={CLOCK}ns wide={WIDE} RF={RF} abits={ABITS}", flush=True)
    for A in ABITS:
        prj = f"{OUT}/binary_ffn_a{A}_rf{RF}_prj"
        print(f"\n===== A{A} (RF={RF}): csynth in {prj} =====", flush=True)
        model = build_ffn(A)
        hls_model = hls4ml.converters.convert_from_keras_model(
            model, hls_config=make_cfg(model), backend=BACKEND,
            output_dir=prj, part=PART, clock_period=CLOCK, io_type="io_parallel",
        )
        # C-synthesis only: synth=True (NO csynth kwarg in hls4ml). Skip csim/cosim/vsynth.
        report = hls_model.build(reset=True, csim=False, synth=True, cosim=False, vsynth=False)
        flat = _flat(report)
        if not _pull(flat, "LUT"):                       # dict empty -> try the report file, then xml
            try:
                flat = _flat(hls4ml.report.parse_vivado_report(prj))
            except Exception as e:
                print(f"[A{A}] parse_vivado_report failed: {e}", flush=True)
        if not _pull(flat, "LUT"):
            flat = {**flat, **_xml_fallback(prj)}

        row = {
            "precision":          f"A{A}",
            "BRAM_18K":           _pull(flat, "BRAM_18K", "BRAM", default=0),
            "DSP":                _pull(flat, "DSP", "DSP48E", default=0),
            "FF":                 _pull(flat, "FF"),
            "LUT":                _pull(flat, "LUT"),
            "LatencyCyclesWorst": _pull(flat, "WorstLatency", "LatencyWorst", "LatencyMax", "Latency"),
            "LatencyCyclesBest":  _pull(flat, "BestLatency", "LatencyBest", "LatencyMin"),
            "IntervalII":         _pull(flat, "IntervalMax", "IntervalMin", "Interval", "II", default=RF),
            "ReuseFactor":        RF,
            "part":               PART,
            "clock_ns":           CLOCK,
        }
        out_json = f"{OUT}/csynth_report_a{A}_rf{RF}.json"
        with open(out_json, "w") as f:
            json.dump({"row": row, "raw_report": report}, f, indent=2, default=str)
        print(f"[A{A}] -> {out_json}: {json.dumps(row)}", flush=True)

    print("\n[done] csynth complete; fill results/hls_resource_table.md §B from the csynth_report_a*.json rows.")


if __name__ == "__main__":
    main()
