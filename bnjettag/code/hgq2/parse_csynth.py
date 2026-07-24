#!/usr/bin/env python3
"""Parse a Vitis HLS csynth.xml into the project's standard JSON row.
Stdlib only (runs on mulder's system python3). Usage: parse_csynth.py <csynth.xml>"""
import json
import sys
import xml.etree.ElementTree as ET

xml = ET.parse(sys.argv[1]).getroot()

def txt(path, default=None):
    e = xml.find(path)
    return e.text if e is not None else default

res = {r.tag: r.text for r in xml.find("AreaEstimates/Resources") or []}
avail = {r.tag: r.text for r in xml.find("AreaEstimates/AvailableResources") or []}
lat = xml.find("PerformanceEstimates/SummaryOfOverallLatency")
timing = xml.find("PerformanceEstimates/SummaryOfTimingAnalysis")

out = {
    "part": txt("UserAssignments/Part"),
    "top": txt("UserAssignments/TopModelName"),
    "target_clock_ns": txt("UserAssignments/TargetClockPeriod"),
    "estimated_clock_ns": (timing is not None and timing.findtext("EstimatedClockPeriod")) or None,
    "LUT": int(res.get("LUT", -1)),
    "FF": int(res.get("FF", -1)),
    "DSP": int(res.get("DSP", res.get("DSP48E", -1))),
    "BRAM_18K": int(res.get("BRAM_18K", -1)),
    "URAM": int(res.get("URAM", -1)),
    "avail": {k: int(v) for k, v in avail.items()},
    "LatencyBest": (lat is not None and lat.findtext("Best-caseLatency")) or None,
    "LatencyWorst": (lat is not None and lat.findtext("Worst-caseLatency")) or None,
    "IntervalMin": (lat is not None and lat.findtext("Interval-min")) or None,
    "IntervalMax": (lat is not None and lat.findtext("Interval-max")) or None,
}
for k in ("LatencyBest", "LatencyWorst", "IntervalMin", "IntervalMax"):
    if out[k] is not None:
        out[k] = int(out[k])
print(json.dumps(out, indent=1))
