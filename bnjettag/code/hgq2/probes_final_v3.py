#!/usr/bin/env python3
"""v3 csynth probes: the DSP-0 form of the narrow-accum binary denses (qkv / Wo / fc2).

Root-cause of the v2 DSP split (measured on the flagship + csynth XMLs, constraints_map 2026-07-09):
it is NOT the weights (all three ROMs are pure ±1, weight_t ap_fixed<2,2>) and NOT frontend 1-bit
emission — it is the DENSE ACCUMULATOR PRECISION. Identical `nnet::dense_resource` kernel, same ±1
weights, same ap_fixed<8,3> input:
  * wide-frac accum  ap_fixed<27,11> (fc1, input_proj) -> Vitis maps the ±1 MAC to LUT  -> 0 DSP (LUT-heavy)
  * narrow-frac accum ap_fixed<16,11> (Wq/Wk/Wv/Wo)   -> Vitis maps it to a DSP48        -> 256 DSP (LUT-light)
The attention projections get the narrow accum because BitExact minimizes it to the downstream
request (the exact-passthrough score-stream grid needs only ~5 frac). Verified in the FULL flagship:
Wq/Wk/Wv/Wo all carry ap_fixed<16,11> -> the 256 DSP is REAL, not a probe artifact.

Fix = WIDEN the dense accumulator (frac -> >=16, keep the integer range), forcing the ±1 MAC into
LUTs -> 0 DSP. Numerically harmless: the accum is only more precise; the downstream re-quantizes to
its grid anyway (C-sim stays bit-exact — measured 0.9999999). This is a genuine DSP<->LUT TRADE
(fc1: 0 DSP / 111k LUT vs qkv-narrow: 256 DSP / 15k LUT), not a bug. So: the ENTIRE final stack CAN
synthesize DSP-free (all DSP in the norm) by widening the attention-projection accumulators; the
default BitExact-minimal accum parks 256 DSP in each of the 4L attention projections.

Same VU13P / 2.5 ns; Resource/RF256; pure ±1 datapath + CSD-2 affine (v2 trap avoidance retained).
Distinct names probe3_dense_*_rf256_resource_wideacc; manifest v3. csynth is mulder-only.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import convert_final as cf  # noqa: E402
import probes_final_v2 as pv2  # noqa: E402
from bnhgq2.config import cfg_hash, PROJECT_ROOT  # noqa: E402

V3_SHAPES = ("attn_qkv", "attn_Wo", "ffn_fc2")  # the narrow-accum / timed-out binary denses
MIN_FRAC = 16  # fc1's proven 0-DSP point is ap_fixed<27,11> = 16 frac


def widen_dense_accum(hm, min_frac=MIN_FRAC):
    """Widen every Dense accum + output precision so its fractional width >= min_frac (keeping
    the integer range). This flips Vitis's ±1-MAC mapping from DSP48 to LUT (0 DSP). Mutates the
    existing precision objects (do NOT replace — they carry definition_cpp)."""
    n = 0
    for nd in hm.get_layers():
        if nd.class_name == "Dense":
            for p in (nd.get_attr("accum_t").precision, nd.get_output_variable().type.precision):
                frac = int(p.width) - int(p.integer)
                if frac < min_frac:
                    p.width = int(p.integer) + min_frac
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--n-csim", type=int, default=64)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()

    cfg, exp, qat, binz, calib, pe, X, ckpt_rel = pv2.load_flagship(a.checkpoint)
    byn = {l.name: l for l in exp.layers}
    cf.patch_resource_einsum_check()
    from bnhgq2.convert import convert, pack_for_mulder

    def post_parse(hm):
        cf.fix_relu_saturation(hm)
        widen_dense_accum(hm)

    out_root = os.path.join(PROJECT_ROOT, "bnjettag", "results", "final", "probes")
    os.makedirs(out_root, exist_ok=True)
    want = set(a.only.split(",")) if a.only else set(V3_SHAPES)
    manifest = {"source": {"flagship": pv2.FLAGSHIP_RUN, "config": cfg["name"],
                           "config_hash": cfg_hash(cfg), "checkpoint": ckpt_rel,
                           "note": "v3: the DSP-0 form of the narrow-accum binary denses (qkv/Wo/fc2). "
                                   "Same v2 2-D QDense at Resource/RF256, pure ±1 + CSD-2 affine, PLUS "
                                   "widen_dense_accum (frac>=16) — flips the ±1 MAC from DSP48 to LUT. "
                                   "Root cause: dense ACCUM precision, not weights (all ±1)."},
                "hls": {"part": cfg["hls"].get("part"), "clock_ns": cfg["hls"].get("clock_ns"),
                        "backend": "Vitis", "io": "io_parallel", "strategy": "Resource", "rf": 256},
                "fix": "widen_dense_accum frac>=16 (accum ap_fixed<16,11> -> <27,11>); DSP<->LUT trade",
                "written": time.strftime("%Y-%m-%dT%H:%M:%S"), "probes": {}}
    tarballs = []
    for spec in pv2.SHAPES:
        if spec["name"] not in want:
            continue
        m, in_dim, out_dim = pv2.build_probe(cfg, binz, calib, spec)
        xin, yref = pv2.flagship_ref(exp, byn, spec, binz, pe, X, a.n_csim, in_dim, out_dim)
        proj = os.path.join(out_root, f"probe3_dense_{spec['name']}_rf256_resource_wideacc")
        shutil.rmtree(proj, ignore_errors=True)
        _, report = convert(m, cfg, proj, rf=256, strategy="Resource",
                            csim_X=xin, keras_ref=yref, post_parse=post_parse)
        # read back the generated accum precision (evidence the wide-accum module was emitted)
        acc = "?"
        for line in open(os.path.join(proj, "firmware", "defines.h")):
            if f"dense_{spec['name']}_accum_t" in line:
                acc = line.strip().replace("typedef ", "")
                break
        tar = pack_for_mulder(proj, proj + ".tar.gz")
        tarballs.append(os.path.basename(tar))
        cs = report.get("csim", {})
        corr = cs.get("corr")
        ok = bool(corr is not None and corr >= 0.997)
        manifest["probes"][f"dense_{spec['name']}_wideacc"] = {
            "shape": f"{in_dim}->{out_dim}", "desc": spec["desc"], "fold": spec["fold"],
            "rf": 256, "strategy": "Resource",
            "generated_accum": acc,
            "csim_vs_flagship": {"n": cs.get("n"), "corr": corr,
                                 "max_abs_diff": cs.get("max_abs_diff"), "sanity_pass": ok},
            "expected_checks": ("binary dense = 0 DSP (wide accum -> LUT-mapped ±1 MAC, the fc1 "
                                "form which synthesized 0 DSP / 111k LUT). Trade vs the v2 narrow "
                                "accum (256 DSP / 15k LUT). SubLN = expected DSP consumer; affine/"
                                "ReLU = 0 DSP. If mulder still shows DSP>0, the accum width threshold "
                                "differs — report the number."),
            "tarball": os.path.relpath(tar, PROJECT_ROOT),
        }
        shutil.rmtree(proj, ignore_errors=True)
        print(f"[probe3] dense_{spec['name']:10s} Resource/RF256 wide-accum  csim={corr}  "
              f"accum={acc}  sanity={'PASS' if ok else 'CHECK'}", flush=True)

    manifest["mulder_command"] = "./mulder_csynth.sh " + " ".join(sorted(tarballs))
    mpath = os.path.join(out_root, "probes_manifest_v3.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[done] {len(manifest['probes'])} v3 probes -> {os.path.relpath(out_root, PROJECT_ROOT)}")
    print(f"       manifest: {os.path.relpath(mpath, PROJECT_ROOT)}")
    print(f"       mulder:   {manifest['mulder_command']}")


if __name__ == "__main__":
    main()
