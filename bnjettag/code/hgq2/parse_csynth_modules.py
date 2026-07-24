#!/usr/bin/env python3
"""Per-instance resource attribution from a Vitis csynth.xml.

Reads ModuleInformation/Module entries (the per-function synthesis table) and
emits csynth_modules.json next to the xml, so DSP/LUT/FF/BRAM attribution is
re-derivable from raw store data (poster gate requirement, GAPS.md item 1).

Usage: parse_csynth_modules.py <csynth.xml> [more.xml ...]
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

RES_KEYS = ("BRAM_18K", "DSP", "FF", "LUT", "URAM")

ROLE_PREFIXES = [
    ("dense_", "dense"),
    ("normalize_", "affine"),
    ("subln", "subln"),
    ("softmax", "softmax"),
    ("einsum", "einsum"),
    ("relu", "activation"),
    ("linear", "activation"),
]


def role_of(name: str) -> str:
    low = name.lower()
    for prefix, role in ROLE_PREFIXES:
        if low.startswith(prefix):
            return role
    return "other"


def parse(xml_path: Path) -> dict:
    root = ET.parse(xml_path).getroot()
    modules = []
    for mod in root.iter("Module"):
        name = mod.findtext("Name")
        if name is None:
            continue
        res = {k: None for k in RES_KEYS}
        area = mod.find("AreaEstimates/Resources")
        if area is not None:
            for k in RES_KEYS:
                v = area.findtext(k)
                if v is not None:
                    res[k] = int(v)
        lat = mod.find("PerformanceEstimates/SummaryOfOverallLatency")
        latency = {
            "worst_cycles": lat.findtext("Worst-caseLatency") if lat is not None else None,
            "ii": lat.findtext("PipelineInitiationInterval") if lat is not None else None,
        }
        modules.append({"module": name, "role": role_of(name), **res, "latency": latency})

    top = root.find("AreaEstimates/Resources")
    top_res = {}
    if top is not None:
        for k in RES_KEYS:
            v = top.findtext(k)
            if v is not None:
                top_res[k] = int(v)

    return {
        "source": str(xml_path),
        "top_total": top_res,
        "modules": modules,
        "note": "per-Module (per-function) synthesis table from ModuleInformation; "
        "top_total is the whole-probe AreaEstimates. Instances of the same module "
        "share one entry (Vitis reports unique modules).",
    }


def main() -> None:
    for arg in sys.argv[1:]:
        xml_path = Path(arg)
        out = parse(xml_path)
        out_path = xml_path.parent / "csynth_modules.json"
        out_path.write_text(json.dumps(out, indent=2) + "\n")
        dsp_split = ", ".join(
            f"{m['role']}:{m['module'][:40]}={m['DSP']}" for m in out["modules"]
        )
        print(f"[{xml_path.parent.name}] top DSP={out['top_total'].get('DSP')} | {dsp_split}")


if __name__ == "__main__":
    main()
