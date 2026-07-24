"""hls4ml conversion (Vitis backend) + local C-sim bit-exactness gate + project
packaging for mulder csynth.

The generated project dir is self-contained C++/tcl; csynth runs on mulder as
  source /data/software/xilinx/Vitis/2023.2/settings64.sh && vitis_hls -f build_prj.tcl
(see code/hls/RUN_CSYNTH_ON_VITIS.md for the vitis-run correction history).
"""
from __future__ import annotations

import os

import numpy as np

from .compat import apply_hls4ml_compat, patch_project_for_macos
from .subln import register_subln


def convert(model, cfg: dict, out_dir: str, rf: int | None = None,
            strategy: str = "Latency", csim_X=None, keras_ref=None,
            post_parse=None, layer_configs=None):
    """Returns (hls_model, report dict). csim gate: max|hls − keras| on csim_X.

    post_parse: optional callable(hls_model) run AFTER parsing/BitExact and BEFORE
    write() — the sanctioned place to repair BitExact-assigned precisions (e.g. the
    ReLU unsigned-WRAP fix, convert_final._fix_relu_saturation). Default None keeps the
    frozen rebuild path byte-identical.

    layer_configs: optional {layer_name: {hls_key: value}} merged into hls_config['LayerName']
    for the HYBRID-strategy monolith (monolith_v2). Two uses, both MUST be set here (not in
    post_parse) because they drive optimizer passes that run during conversion:
      * Strategy: the weighted layers (pointwise/2-D Dense) run at the Model default `strategy`
        (Resource, weights→BRAM); the weightless attention einsums MUST stay 'Latency'
        (hls4ml 1.3.0 asserts Einsum/EinsumDense are Latency-only).
      * ReuseFactor: a PointwiseConv1D at Resource asserts RF ≤ filt_width·n_chan = in_channels
        (nnet_conv1d_resource.h), so input_proj (in=16) needs a per-layer RF ≤ 16 while the
        in≥256 layers keep RF=256. Default None keeps the single-strategy path byte-identical."""
    import hls4ml

    apply_hls4ml_compat()
    register_subln()

    hlscfg = cfg["hls"]
    rf = rf or hlscfg.get("rf", 1)
    config = {
        "Model": {
            "Precision": "fixed<24,12>",  # fallback only; BitExact overrides the quantized path
            "ReuseFactor": rf,
            "Strategy": strategy,
        },
    }
    if layer_configs:
        config["LayerName"] = {name: dict(kv) for name, kv in layer_configs.items()}
    hm = hls4ml.converters.convert_from_keras_model(
        model,
        backend=hlscfg.get("backend", "Vitis"),
        io_type=hlscfg.get("io", "io_parallel"),
        output_dir=out_dir,
        part=hlscfg.get("part", "xcvu13p-flga2577-2-e"),
        clock_period=hlscfg.get("clock_ns", 2.5),
        hls_config=config,
        bit_exact=True,  # the kwarg silently overwrites hls_config — must be passed here
    )
    if post_parse is not None:
        post_parse(hm)
    hm.write()
    report = {"output_dir": out_dir, "rf": rf, "strategy": strategy,
              "backend": hlscfg.get("backend", "Vitis")}

    if csim_X is not None:
        import platform
        if platform.system() == "Darwin":
            # must patch AFTER write() and compile via _compile() — plain
            # compile() re-writes the project and undoes the patch
            patch_project_for_macos(out_dir)
        hm._compile()
        if isinstance(csim_X, (list, tuple)):
            xs = [np.ascontiguousarray(x.astype(np.float32)) for x in csim_X]
        else:
            xs = np.ascontiguousarray(csim_X.astype(np.float32))
        y_hls = hm.predict(xs)
        ref = keras_ref if keras_ref is not None else np.asarray(
            model(csim_X, training=False))
        y_hls = y_hls.reshape(ref.shape)
        d = np.abs(y_hls - ref)
        n_csim = len(csim_X[0]) if isinstance(csim_X, (list, tuple)) else len(csim_X)
        report["csim"] = {
            "n": int(n_csim),
            "max_abs_diff": float(d.max()),
            "mean_abs_diff": float(d.mean()),
            "bit_exact": bool(d.max() == 0.0),
            "corr": float(np.corrcoef(y_hls.ravel(), ref.ravel())[0, 1]),
        }
    return hm, report


def pack_for_mulder(out_dir: str, tar_path: str):
    import subprocess
    # exclude any previously-fetched csynth report: shipping one inside the
    # tarball made a stale result masquerade as the new run's (caught 2026-07-04)
    subprocess.run(["tar", "--exclude", "csynth_report.json", "-czf", tar_path,
                    "-C", os.path.dirname(out_dir), os.path.basename(out_dir)],
                   check=True)
    return tar_path
