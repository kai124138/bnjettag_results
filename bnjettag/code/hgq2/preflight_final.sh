#!/usr/bin/env bash
# FINAL-campaign preflight — PVC-FREE. Run in a CPU pod on image python:3.12 with the code
# ConfigMap kai-bnf-code mounted at /cmcode (apply it first: ./make_code_configmap.sh):
#   kubectl -n cms-ml run kai-bnf-preflight --restart=Never -it --image=python:3.12 \
#     --overrides='{"spec":{"containers":[{"name":"p","image":"python:3.12","command":["bash","-lc",
#       "tar xzf /cmcode/hgq2.tar.gz -C /tmp && bash /tmp/hgq2/preflight_final.sh"],
#       "volumeMounts":[{"name":"cm","mountPath":"/cmcode"}]}],
#       "volumes":[{"name":"cm","configMap":{"name":"kai-bnf-code"}}]}}'
# Builds all 5 configs on CPU (prints exact param counts + binary gate) and probes the Zenodo
# data source is reachable (HEAD, no 3 GB download). No PVC needed.
set -uo pipefail
export KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES=-1
CODE=${BNF_CODE:-/work/code}
mkdir -p "$CODE"
if [ -f /cmcode/hgq2.tar.gz ]; then tar -xzf /cmcode/hgq2.tar.gz -C "$CODE" --strip-components=1; fi
fail=0
echo "=== BNJetTag FINAL preflight (PVC-free) $(date) ==="

echo "[deps] pinned install (mirror .venv-hgq2)"
pip install -q --no-cache-dir "tensorflow[and-cuda]==2.21.0" "keras==3.15.0" "hgq2==0.1.9" "quantizers==1.2.2" "scikit-learn==1.9.0" "h5py==3.14.0" "wandb==0.28.0" "hls4ml==1.3.0" "numpy==2.5.0" || { echo "  FAIL deps"; fail=1; }

echo "--- data source reachability: Zenodo train tarball (HEAD, no download) ---"
if python - <<'PYEOF'
import urllib.request
req = urllib.request.Request("https://zenodo.org/records/3602260/files/hls4ml_LHCjet_150p_train.tar.gz?download=1", method="HEAD")
r = urllib.request.urlopen(req, timeout=90)
cl = int(r.headers.get("Content-Length", "0"))
assert r.status == 200 and cl > 2_500_000_000, f"status={r.status} content-length={cl}"  # real tarball = 2,725,115,104 B (measured 2026-07-07)
print(f"  OK Zenodo train tarball reachable: {cl} bytes")
PYEOF
then echo "  PASS data-source"; else echo "  FAIL data-source"; fail=1; fi

echo "--- build all 5 configs (CPU) + param counts + binary gate ---"
if python - <<PYEOF
import sys, glob, numpy as np
sys.path.insert(0, "$CODE")
from bnhgq2.compat import apply_keras_compat; apply_keras_compat()
from bnhgq2.subln import register_subln; register_subln()
from bnhgq2.config import load_config, cfg_hash
from bnhgq2 import qat
ok = True
for p in sorted(glob.glob("$CODE/configs/final-*.json")):
    c = load_config(p)
    m, taps = qat.build_qat_model(c, seed=1)
    nt = int(sum(np.prod(v.shape) for v in m.trainable_variables))
    tag = ""
    if c["quant"]["weight"] == "binary_absmean":
        effs = qat.effective_weight_values(m)
        binok = all(len(v) == 2 and not (v == 0).any() for v in effs.values())
        tag = f" binary_gate={'OK' if binok else 'FAIL'}({len(effs)} layers)"
        ok = ok and binok
    print(f"  {c['name']:14s} [{cfg_hash(c)}] params={m.count_params():,} "
          f"trainable={nt:,} taps={len(taps)}{tag}")
sys.exit(0 if ok else 1)
PYEOF
then echo "  PASS build"; else echo "  FAIL build"; fail=1; fi

if [ "$fail" = 0 ]; then echo "PREFLIGHT_ALL_PASS"; else echo "PREFLIGHT_HAD_FAILURES"; fi
