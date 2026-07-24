#!/usr/bin/env bash
# ROUND-7 preflight — PVC-FREE, CPU-only (no GPU burned). Builds all 26 r7 configs from the
# SAME ConfigMap the training jobs mount and hard-gates on the exact QAT-stack param counts +
# the binary {-1,+1} gate + Zenodo data-source reachability. Run in a CPU pod on image
# python:3.12 with the code ConfigMap kai-bnf-code mounted at /cmcode (apply it first with the
# EXISTING ./make_code_configmap.sh, which now also packs the r7-*.json configs + this script):
#   kubectl -n cms-ml run kai-r7f-preflight --restart=Never -it --image=python:3.12 \
#     --overrides='{"spec":{"containers":[{"name":"p","image":"python:3.12","command":["bash","-lc",
#       "tar xzf /cmcode/hgq2.tar.gz -C /tmp && bash /tmp/hgq2/preflight_r7.sh"],
#       "volumeMounts":[{"name":"cm","mountPath":"/cmcode"}]}],
#       "volumes":[{"name":"cm","configMap":{"name":"kai-bnf-code"}}]}}'
# Require the literal PREFLIGHT_ALL_PASS in the output before launching any GPU job.
set -uo pipefail
export KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES=-1
CODE=${BNF_CODE:-/work/code}
mkdir -p "$CODE"
if [ -f /cmcode/hgq2.tar.gz ]; then tar -xzf /cmcode/hgq2.tar.gz -C "$CODE" --strip-components=1; fi
fail=0
echo "=== BNJetTag ROUND-7 preflight (PVC-free) $(date) ==="

echo "[deps] pinned install (mirror .venv-hgq2)"
pip install -q --no-cache-dir "tensorflow[and-cuda]==2.21.0" "keras==3.15.0" "hgq2==0.1.9" "quantizers==1.2.2" "scikit-learn==1.9.0" "h5py==3.14.0" "wandb==0.28.0" "hls4ml==1.3.0" "numpy==2.5.0" || { echo "  FAIL deps"; fail=1; }

echo "--- data source reachability: Zenodo train tarball (HEAD, no download) ---"
if python - <<'PYEOF'
import urllib.request
req = urllib.request.Request("https://zenodo.org/records/3602260/files/hls4ml_LHCjet_150p_train.tar.gz?download=1", method="HEAD")
r = urllib.request.urlopen(req, timeout=90)
cl = int(r.headers.get("Content-Length", "0"))
assert r.status == 200 and cl > 2_500_000_000, f"status={r.status} content-length={cl}"  # real tarball = 2,725,115,104 B
print(f"  OK Zenodo train tarball reachable: {cl} bytes")
PYEOF
then echo "  PASS data-source"; else echo "  FAIL data-source"; fail=1; fi

echo "--- build all 26 r7 configs (CPU) + HARD-ASSERT param counts + binary {-1,+1} gate ---"
if python - <<PYEOF
import sys, glob, numpy as np
sys.path.insert(0, "$CODE")
from bnhgq2.compat import apply_keras_compat; apply_keras_compat()
from bnhgq2.subln import register_subln; register_subln()
from bnhgq2.config import load_config, cfg_hash
from bnhgq2 import qat

# QAT-stack (quantized) vs fp32-skeleton count_params, per size (measured 2026-07-14).
PARAMS_QAT  = {"small": 19201, "tiny": 5345}   # w8a8 / w1a8 / w1a6 / w1a4
PARAMS_FP32 = {"small": 19075, "tiny": 5219}   # fp32 skeleton
ok = True
# r7-* matrix/probes + r7b-* tiny recal all gate here
configs = sorted(glob.glob("$CODE/configs/r7*.json") + glob.glob("$CODE/configs/r8*.json"))
if len(configs) != 26:
    print(f"  FAIL expected 26 r7 configs, found {len(configs)}"); ok = False
for p in configs:
    c = load_config(p)
    name = c["name"]                              # r7-<size>-<variant>
    size = "small" if "-small-" in name else "tiny"
    fp32 = c["quant"]["weight"] == "none"
    exp = PARAMS_FP32[size] if fp32 else PARAMS_QAT[size]
    m, taps = qat.build_qat_model(c, seed=1)
    got = int(m.count_params())
    pmatch = (got == exp)
    ok = ok and pmatch
    tag = ""
    if c["quant"]["weight"] == "binary_absmean":
        effs = qat.effective_weight_values(m)
        binok = all(len(v) == 2 and not (v == 0).any()
                    and abs(abs(v[0]) - abs(v[1])) < 1e-9 for v in effs.values())
        tag = f" binary_gate={'OK' if binok else 'FAIL'}({len(effs)} layers)"
        ok = ok and binok
    print(f"  {name:16s} [{cfg_hash(c)}] params={got:>6,} exp={exp:>6,} "
          f"{'OK' if pmatch else 'MISMATCH!'}{tag}")
# headline: the two QAT-stack counts the round is gated on
print(f"  headline QAT-stack: small={PARAMS_QAT['small']:,}  tiny={PARAMS_QAT['tiny']:,}")
sys.exit(0 if ok else 1)
PYEOF
then echo "  PASS build"; else echo "  FAIL build"; fail=1; fi

if [ "$fail" = 0 ]; then echo "PREFLIGHT_ALL_PASS"; else echo "PREFLIGHT_HAD_FAILURES"; fi
