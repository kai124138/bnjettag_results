#!/usr/bin/env python3
"""
FINAL campaign (round-7) job generator — native HGQ2 QAT (decisions.md 2026-07-07).

The definitive run: 5 variants x 3 seeds = 15 GPU training Jobs, trained natively in
HGQ2 so the trained model IS the hardware model (binary {-1,+1} STE weights + static
activation grids, code in bnjettag/code/hgq2/bnhgq2/{qat,train}.py). New W&B project
`bnjettag-final`; job names kai-bnf-<variant>-s<n>; run names final-<variant>-s<seed>.

PVC-FREE (2026-07-07): the kai-data cephfs PVC filters training pods OFF the GPU nodes
(post-incident CSI tail) — so training/smoke take CODE from a ConfigMap (kai-bnf-code) and
DATA from Zenodo, write to an emptyDir under /work, and the durable copy of
model_best.keras + train_meta.json is train.py's W&B upload (wandb.save). Pattern per the
proven kai-roc-r5-pvcfree.yaml.

Recipe carried from r5 (peak LR 5e-5, Adam beta2=0.98, warmup 1ep + poly decay 40ep,
weight_decay 0.01, batch 256, early-stop patience 10 on val macro-OvR AUC). Gradient
clipping = clipvalue=1.0 (NOT r5's global_clipnorm — verified 2026-07-07 to overflow
float32 on this deep-SubLN stack and NaN A4 regardless of LR; see hgq2/LEDGER.md).

Everything is GENERATED — never hand-edit a kai-bnf-*.yaml (regenerate here). Writes:
  * kai-bnf-<variant>-s<n>.yaml   (15) + kai-bnf-smoke.yaml  (PVC-free: ConfigMap code + Zenodo data)
  * make_code_configmap.sh        (THE training prereq: pack code/hgq2 -> ConfigMap kai-bnf-code)
  * preflight_final.sh            (build all 5 configs + Zenodo HEAD probe -> PREFLIGHT_ALL_PASS)
  * launch_final_staged.sh        (<=3 concurrent, wave-by-wave)
  * sync_code_to_pvc_final.sh     (kept for the PVC-based ROC job ONLY; not a training prereq)

Run:  python3 gen_final_jobs.py
"""
import os
import stat

HERE = os.path.dirname(os.path.abspath(__file__))

# Container image: mirrors the local .venv-hgq2 (Python 3.12 / TF 2.21). The exact pin
# set is pip-installed on top so the trained stack == the locally-verified stack.
# Image history (2026-07-07, all measured on-cluster):
#  * tensorflow/tensorflow:2.21.0-gpu — BROKEN CUDA packaging: cuDNN version probe dies
#    with CUDNN_STATUS_INTERNAL_ERROR on two independent healthy nodes (suncave-15 AND
#    k8s-3090-02.clemson.edu, driver 595/CUDA 13.2) → TF registers no GPU. Also ships
#    python 3.11, which made numpy==2.5.0 uninstallable.
#  * Fix: plain python:3.12 base + `tensorflow[and-cuda]` pip extras — CUDA/cuDNN come
#    as NVIDIA pip wheels (the TF-official path, driver-forward-compatible), and the env
#    now mirrors the local .venv-hgq2 EXACTLY (py3.12, TF 2.21, numpy 2.5.0).
IMAGE = "python:3.12"
PIP_PINS = ('"tensorflow[and-cuda]==2.21.0" "keras==3.15.0" "hgq2==0.1.9" "quantizers==1.2.2" '
            '"scikit-learn==1.9.0" "h5py==3.14.0" "wandb==0.28.0" "hls4ml==1.3.0" "numpy==2.5.0"')
PVC_CODE = "/data/BNJetTag-hgq2"          # sync_code_to_pvc_final.sh still puts code here (the ROC job uses PVC code)
# PVC-FREE training (2026-07-07): the kai-data cephfs PVC filters training pods OFF the GPU
# nodes (post-incident CSI tail) — A/B probe: an identical no-PVC pod scheduled in ~4 min, the
# PVC pod pended 75+ min. So training/smoke take code from a ConfigMap and data from Zenodo,
# outputs to an emptyDir, durability via train.py's W&B uploads.
DATA_URL = "https://zenodo.org/records/3602260/files/hls4ml_LHCjet_150p_train.tar.gz?download=1"
CODE_CM = "kai-bnf-code"                   # ConfigMap holding the packed code/hgq2 (key hgq2.tar.gz)
OUT_ROOT = "/work/outputs"                 # emptyDir; the durable copy is on W&B (wandb.save)

VARIANTS = ["fp32", "w8a8", "w1a8", "w1a6", "w1a4"]
SEEDS = [1, 2, 3]

NOTES = {
    "fp32": "BASELINE FP32 (float weights + acts, same skeleton) — the abstract's comparison point.",
    "w8a8": "BASELINE W8A8 (8-bit static weights + acts) — 1-bit-vs-ordinary-quantization reference.",
    "w1a8": "THESIS binary {-1,+1} weights + static 8-bit acts (the target model).",
    "w1a6": "QUANT AXIS binary weights + static 6-bit acts.",
    "w1a4": "QUANT AXIS binary weights + static 4-bit acts (clipvalue-clipped: deep-1-bit A4 is unstable).",
}

# Nodes that eat pods (2026-07-07 smoke campaign, all measured):
#  * k8s-chase-ci-07: UnexpectedAdmissionError on 3/3 attempts — device plugin advertises
#    free GPUs precisely because nothing can start there, so the scheduler keeps picking it.
#  * suncave-*: UCSD CAVE display wall — 3090s show "free" but cuDNN init fails
#    (CUDNN_STATUS_INTERNAL_ERROR on suncave-15; display nodes, not compute nodes).
# Append new offenders here and regenerate.
BAD_NODES = ["k8s-chase-ci-07.calit2.optiputer.net",
             # ry-gpu-08: healthy at 16:20 (debug pod + smoke ran fine), degraded by
             # 18:46 (CUDA_ERROR_NOT_INITIALIZED), full admission-error attractor by
             # 23:5x (3/3 UnexpectedAdmissionError) — nodes go bad MID-CAMPAIGN.
             "ry-gpu-08.sdsc.optiputer.net"] + [f"suncave-{i}" for i in range(1, 21)]

NODE_AFFINITY = f"""      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: kubernetes.io/hostname
                operator: NotIn
                values: [{", ".join(BAD_NODES)}]
              - key: kubernetes.io/arch
                operator: In
                values: [amd64]
              - key: nvidia.com/gpu.product
                operator: In
                values:
                - NVIDIA-A100-SXM4-80GB
                - NVIDIA-A100-80GB-PCIe
                - NVIDIA-A100-PCIE-40GB
                - NVIDIA-H100-80GB-HBM3
                - NVIDIA-H200-NVL
                - NVIDIA-RTX-PRO-6000-Blackwell-Max-Q-Workstation-Edition
                - NVIDIA-GeForce-RTX-4090
                - NVIDIA-L40S
                - NVIDIA-L40
                - NVIDIA-RTX-A6000
                - NVIDIA-A40
                - NVIDIA-RTX-A5000
                - NVIDIA-GeForce-RTX-3090
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            preference:
              matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values: [NVIDIA-H200-NVL, NVIDIA-H100-80GB-HBM3, NVIDIA-RTX-PRO-6000-Blackwell-Max-Q-Workstation-Edition]
          - weight: 90
            preference:
              matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values: [NVIDIA-A100-SXM4-80GB, NVIDIA-A100-80GB-PCIe, NVIDIA-GeForce-RTX-4090, NVIDIA-L40S]
          - weight: 80
            preference:
              matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values: [NVIDIA-A100-PCIE-40GB, NVIDIA-L40, NVIDIA-RTX-A6000]
          - weight: 60
            preference:
              matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values: [NVIDIA-A40, NVIDIA-RTX-A5000, NVIDIA-GeForce-RTX-3090]"""

# PVC-FREE job template (@-token substitution — avoids brace-escaping the embedded shell).
JOB_TMPL = r'''apiVersion: batch/v1
kind: Job
metadata:
  name: @JOB@
  labels:
    app: bnjet-final
    bnf: "@LABEL@"
spec:
  # backoffLimit 2 (was 0): first smoke attempt died to UnexpectedAdmissionError — a
  # node-level GPU device-plugin race where the container never starts (zero side
  # effects), which with backoffLimit 0 kills the whole Job. Real training crashes are
  # what the smoke gate exists to catch; duplicate-W&B-run risk from a mid-train retry
  # is accepted (the ROC fetcher must prefer the finished run if duplicates appear).
  backoffLimit: 2
  activeDeadlineSeconds: @DEADLINE@
  template:
    metadata:
      labels:
        app: bnjet-final
        bnf: "@LABEL@"
    spec:
      restartPolicy: Never
@AFFINITY@
      containers:
      - name: train
        image: @IMAGE@
        command: ["bash", "-c"]
        args:
        - |
          set -eo pipefail
          export KERAS_BACKEND=tensorflow MPLBACKEND=Agg TF_FORCE_GPU_ALLOW_GROWTH=true
          mkdir -p /work/code /work/data /work/outputs
          CODE=/work/code
          OUT=@OUT@
          mkdir -p "$OUT"
          echo "[code] $(date) unpack code from ConfigMap @CODE_CM@ (PVC-free)"
          tar -xzf /cmcode/hgq2.tar.gz -C "$CODE" --strip-components=1
          md5sum /cmcode/hgq2.tar.gz | awk '{print "[code] configmap tar md5:", $1}'
          export PYTHONPATH="$CODE:$PYTHONPATH"
          echo "[setup] $(date) pinned deps (mirror .venv-hgq2)"
          pip install -q --no-cache-dir @PINS@
          # TF 2.21 pip wheel's RPATH discovery of the nvidia-*-cu12 wheel libs is broken
          # (debug-pod evidence 2026-07-07: every lib dlopens by absolute path, TF still
          # says "Cannot dlopen"; explicit LD_LIBRARY_PATH over the wheel lib dirs fixes
          # it — GPU matmul + keras fit then verified on an RTX 3090 / driver 595).
          NVLIBS=$(python -c "import glob; print(':'.join(sorted(glob.glob('/usr/local/lib/python*/site-packages/nvidia/*/lib'))))")
          export LD_LIBRARY_PATH="$NVLIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          echo "[gpu] $(date) nvidia-smi + TF backend assertion"
          nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "[fatal] no nvidia-smi / GPU"; exit 1; }
          python -c "import os; os.environ.setdefault('KERAS_BACKEND','tensorflow'); import tensorflow as tf; g=tf.config.list_physical_devices('GPU'); assert g, 'NO GPU visible to TensorFlow'; print('[gpu] TF sees', g)"
          echo "[data] $(date) fetch HLS4ML LHC Jet 150p TRAIN split from Zenodo (~3 GB, PVC-free)"
          python -u -c 'import urllib.request; urllib.request.urlretrieve("@DATA_URL@", "/work/data/train.tar.gz")'
          sz=$(stat -c %s /work/data/train.tar.gz); echo "[data] tarball bytes: $sz"
          [ "$sz" -gt 2500000000 ] || { echo "[fatal] train tarball too small ($sz < 2.5GB; real size 2,725,115,104 B measured 2026-07-07)"; exit 1; }
          tar -xzf /work/data/train.tar.gz -C /work/data && rm -f /work/data/train.tar.gz
          DATA=$(dirname "$(find /work/data -name 'jetImage_*.h5' -print -quit)")
          [ -n "$DATA" ] || { echo "[fatal] no jetImage_*.h5 after extract"; exit 1; }
          echo "[data] DATA=$DATA ($(ls "$DATA" | wc -l) files)"
          export BNHGQ2_TRAIN_DATA="$DATA"
          export BNHGQ2_OUT_ROOT=@OUT_ROOT@
          export BNHGQ2_STORE=@OUT_ROOT@/_store
          export WANDB_PROJECT=bnjettag-final
          export WANDB_RUN_NAME=@RUN@
          # ---- @NOTE@ ----
          echo "[train] $(date) @RUN@  config=@CONFIG@ seed=@SEED@@EXTRA_DESC@"
          cd "$CODE"
          python -u "$CODE/run_stage.py" train \
            --config "$CODE/configs/@CONFIG@" --seed @SEED@@EXTRA_FLAGS@ \
            2>&1 | tee "$OUT/train.log"
          echo "[done] $(date) @RUN@"
        env:
        - name: WANDB_API_KEY
          valueFrom:
            secretKeyRef:
              name: kai-wandb
              key: WANDB_API_KEY
        resources:
          # PVC-FREE (2026-07-07): cephfs-PVC + GPU node is blocked (post-incident CSI tail).
          # A/B probe: identical no-PVC pod scheduled in ~4 min on hcc-chase-shor-c4709 (3090);
          # the PVC pod pended 75+ min. Data<-Zenodo, code<-ConfigMap, outputs->emptyDir + W&B.
          # cpu 4 / mem 24Gi kept (6cpu/32Gi failed on free-node CPU); eph 24Gi for the ~3-7G
          # tarball (deleted post-extract) + extracted data + pip. requests==limits.
          limits:
            nvidia.com/gpu: "1"
            cpu: "4"
            memory: 24Gi
            ephemeral-storage: 24Gi
          requests:
            nvidia.com/gpu: "1"
            cpu: "4"
            memory: 24Gi
            ephemeral-storage: 24Gi
        volumeMounts:
        - name: cmcode
          mountPath: /cmcode
        - name: work
          mountPath: /work
      volumes:
      - name: cmcode
        configMap:
          name: @CODE_CM@
      - name: work
        emptyDir:
          sizeLimit: 24Gi
'''


def job_yaml(variant, seed, *, smoke=False):
    label = "smoke" if smoke else f"{variant}-s{seed}"
    job = "kai-bnf-smoke" if smoke else f"kai-bnf-{variant}-s{seed}"
    run = "final-smoke" if smoke else f"final-{variant}-s{seed}"
    out = f"{OUT_ROOT}/{'smoke' if smoke else f'{variant}-s{seed}'}"
    note = ("SMOKE: w1a8, <=3 epochs on a 2-file subset (same code path; full Zenodo fetch "
            "so the data path itself is exercised)." if smoke else NOTES[variant])
    subs = {
        "@JOB@": job, "@LABEL@": label, "@RUN@": run, "@OUT@": out, "@OUT_ROOT@": OUT_ROOT,
        "@AFFINITY@": NODE_AFFINITY, "@IMAGE@": IMAGE, "@PINS@": PIP_PINS,
        "@DATA_URL@": DATA_URL, "@CODE_CM@": CODE_CM, "@CONFIG@": f"final-{variant}.json",
        "@SEED@": str(seed), "@NOTE@": note,
        "@EXTRA_FLAGS@": " --smoke" if smoke else "",
        "@EXTRA_DESC@": "  [SMOKE]" if smoke else "",
        "@DEADLINE@": str(14400 if smoke else 172800),
    }
    y = JOB_TMPL
    for k, v in subs.items():
        y = y.replace(k, v)
    return y


PREFLIGHT = r'''#!/usr/bin/env bash
# FINAL-campaign preflight — PVC-FREE. Run in a CPU pod on image @IMAGE@ with the code
# ConfigMap @CODE_CM@ mounted at /cmcode (apply it first: ./make_code_configmap.sh):
#   kubectl -n cms-ml run kai-bnf-preflight --restart=Never -it --image=@IMAGE@ \
#     --overrides='{"spec":{"containers":[{"name":"p","image":"@IMAGE@","command":["bash","-lc",
#       "tar xzf /cmcode/hgq2.tar.gz -C /tmp && bash /tmp/hgq2/preflight_final.sh"],
#       "volumeMounts":[{"name":"cm","mountPath":"/cmcode"}]}],
#       "volumes":[{"name":"cm","configMap":{"name":"@CODE_CM@"}}]}}'
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
pip install -q --no-cache-dir @PINS@ || { echo "  FAIL deps"; fail=1; }

echo "--- data source reachability: Zenodo train tarball (HEAD, no download) ---"
if python - <<'PYEOF'
import urllib.request
req = urllib.request.Request("@DATA_URL@", method="HEAD")
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
'''


def preflight_sh():
    return (PREFLIGHT.replace("@IMAGE@", IMAGE).replace("@CODE_CM@", CODE_CM)
            .replace("@DATA_URL@", DATA_URL).replace("@PINS@", PIP_PINS))


MAKE_CM = r'''#!/usr/bin/env bash
# Pack code/hgq2 into ConfigMap @CODE_CM@ (key hgq2.tar.gz) — the PVC-FREE code source that
# training pods untar to /work/code. THE TRAINING PREREQ (replaces sync_code_to_pvc_final.sh,
# which is only for the PVC-based ROC job). Idempotent (--dry-run=client | apply). Prints the
# tar md5 — pods echo the same md5 for provenance. code/hgq2 packs to ~80 KB (< the 1 MB
# ConfigMap limit); same excludes as the sync script.
set -euo pipefail
CTX="nautilus"; NS="cms-ml"
SRC="$(cd "$(dirname "$0")/../../../hgq2" && pwd)"
TAR="/tmp/bnjettag-hgq2-cm.tar.gz"
tar --exclude='__pycache__' --exclude='*.pyc' --exclude='*.tar.gz' --exclude='*.keras' \
    --exclude='.DS_Store' -czf "$TAR" -C "$(dirname "$SRC")" "$(basename "$SRC")"
MD5=$(md5sum "$TAR" 2>/dev/null | awk '{print $1}' || md5 -q "$TAR")
SZ=$(wc -c < "$TAR")
echo "[cm] packed $SRC -> hgq2.tar.gz  md5=$MD5  bytes=$SZ"
[ "$SZ" -lt 1048576 ] || { echo "[fatal] tar $SZ bytes exceeds the 1 MB ConfigMap limit — trim excludes"; exit 1; }
kubectl --context "$CTX" -n "$NS" create configmap @CODE_CM@ \
  --from-file=hgq2.tar.gz="$TAR" --dry-run=client -o yaml \
  | kubectl --context "$CTX" -n "$NS" apply -f -
echo "[cm] applied ConfigMap @CODE_CM@ (key hgq2.tar.gz, md5 $MD5)"
echo "[cm] training pods echo this md5 as '[code] configmap tar md5: $MD5' for provenance"
'''


def make_cm_sh():
    return MAKE_CM.replace("@CODE_CM@", CODE_CM)


def launch_sh(waves, key2job):
    ls = ["#!/usr/bin/env bash",
          "# Launch the FINAL campaign on NRP, STAGED <=3 concurrent Jobs.",
          "# Usage:  ./launch_final_staged.sh            # wave-by-wave (needs laptop alive)",
          "#         ./launch_final_staged.sh delete     # tear everything down",
          "# PVC-FREE (cephfs-PVC + GPU node is blocked). PRE-REQS (in order):",
          "#   1. ./make_code_configmap.sh      # THE training code source (ConfigMap kai-bnf-code)",
          "#   2. preflight_final.sh -> PREFLIGHT_ALL_PASS   (in a pod with the ConfigMap mounted)",
          "# (sync_code_to_pvc_final.sh is NOT needed for training — it is only for the PVC ROC job.)",
          "set -euo pipefail",
          'CTX="nautilus"; NS="cms-ml"',
          'HERE="$(cd "$(dirname "$0")" && pwd)"',
          "",
          'ALL=(' + " ".join(key2job.values()) + ')',
          'if [ "${1:-apply}" = "delete" ]; then',
          '  for j in "${ALL[@]}"; do kubectl --context "$CTX" -n "$NS" delete job "$j" --ignore-not-found; done',
          '  exit 0',
          'fi',
          "",
          "wait_for_wave () {",
          '  for j in "$@"; do',
          '    echo "[wait] $j"',
          '    while true; do',
          '      s=$(kubectl --context "$CTX" -n "$NS" get job "$j" -o jsonpath="{.status.conditions[?(@.status==\\"True\\")].type}" 2>/dev/null || true)',
          '      case "$s" in *Complete*|*Failed*) echo "[done] $j -> $s"; break;; esac',
          '      sleep 120',
          '    done',
          '  done',
          "}",
          ""]
    for i, wave in enumerate(waves, 1):
        jobs = [key2job[k] for k in wave]
        ls.append(f'echo "=== wave {i}: {" ".join(wave)} ==="')
        for j in jobs:
            ls.append(f'kubectl --context "$CTX" -n "$NS" delete job "{j}" --ignore-not-found >/dev/null 2>&1 || true')
            ls.append(f'kubectl --context "$CTX" -n "$NS" apply -f "$HERE/{j}.yaml"')
        ls.append(f'wait_for_wave {" ".join(jobs)}')
        ls.append("")
    ls.append('echo "=== FINAL campaign complete ==="')
    ls.append('kubectl --context "$CTX" -n "$NS" get jobs -l app=bnjet-final')
    return "\n".join(ls) + "\n"


SYNC = r'''#!/usr/bin/env bash
# Sync code/hgq2 -> PVC @PVC_CODE@ (what actually trains). md5-verified BOTH ends via a
# throwaway util pod. Stale PVC code has bitten us before — always run this before launch.
set -euo pipefail
CTX="nautilus"; NS="cms-ml"
POD="kai-bnf-sync"
# code/hgq2 is three levels up from this script (variants/ -> training/ -> jobs/ -> code/hgq2)
SRC="$(cd "$(dirname "$0")/../../../hgq2" && pwd)"
TAR="/tmp/bnjettag-hgq2.tar.gz"

echo "[sync] packing $SRC"
tar --exclude='__pycache__' --exclude='*.pyc' --exclude='*.tar.gz' --exclude='*.keras' \
    --exclude='.DS_Store' -czf "$TAR" -C "$(dirname "$SRC")" "$(basename "$SRC")"
LOCAL_MD5=$(md5sum "$TAR" 2>/dev/null | awk '{print $1}' || md5 -q "$TAR")
echo "[sync] local  md5=$LOCAL_MD5  ($(wc -c < "$TAR") bytes)"

cleanup() { kubectl --context "$CTX" -n "$NS" delete pod "$POD" --ignore-not-found >/dev/null 2>&1 || true; }
trap cleanup EXIT
kubectl --context "$CTX" -n "$NS" delete pod "$POD" --ignore-not-found >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" apply -f - <<'YAML'
apiVersion: v1
kind: Pod
metadata:
  name: kai-bnf-sync
  labels: {app: kai-util}
spec:
  restartPolicy: Never
  containers:
  - name: util
    image: alpine:3.19
    command: ["sh", "-c", "sleep 3600"]
    resources: {limits: {memory: 512Mi, cpu: "500m"}, requests: {memory: 256Mi, cpu: "200m"}}
    volumeMounts: [{name: data, mountPath: /data}]
  volumes:
  - name: data
    persistentVolumeClaim: {claimName: kai-data}
YAML
echo "[sync] waiting for $POD ..."
kubectl --context "$CTX" -n "$NS" wait --for=condition=Ready "pod/$POD" --timeout=180s

kubectl --context "$CTX" -n "$NS" cp "$TAR" "$POD:/data/bnjettag-hgq2.tar.gz"
REMOTE_MD5=$(kubectl --context "$CTX" -n "$NS" exec "$POD" -- md5sum /data/bnjettag-hgq2.tar.gz | awk '{print $1}')
echo "[sync] remote md5=$REMOTE_MD5"
if [ "$LOCAL_MD5" != "$REMOTE_MD5" ]; then echo "[fatal] tarball md5 MISMATCH (transfer corrupt)"; exit 1; fi
echo "[sync] tarball md5 MATCH -> extracting to @PVC_CODE@"
kubectl --context "$CTX" -n "$NS" exec "$POD" -- sh -c '
  rm -rf @PVC_CODE@ && mkdir -p @PVC_CODE@ &&
  tar -xzf /data/bnjettag-hgq2.tar.gz -C @PVC_CODE@ --strip-components=1 &&
  echo "[remote] extracted:" && ls @PVC_CODE@ | tr "\n" " " && echo'

# extraction integrity: md5 a few key files on both ends
for f in run_stage.py bnhgq2/qat.py bnhgq2/train.py bnhgq2/config.py; do
  L=$(md5sum "$SRC/$f" 2>/dev/null | awk '{print $1}' || md5 -q "$SRC/$f")
  R=$(kubectl --context "$CTX" -n "$NS" exec "$POD" -- md5sum "@PVC_CODE@/$f" | awk '{print $1}')
  if [ "$L" = "$R" ]; then echo "[sync] OK  $f"; else echo "[fatal] MISMATCH $f (local $L != pvc $R)"; exit 1; fi
done
kubectl --context "$CTX" -n "$NS" exec "$POD" -- rm -f /data/bnjettag-hgq2.tar.gz
echo "[sync] DONE — @PVC_CODE@ is byte-verified against local code/hgq2"
'''


def sync_sh():
    return SYNC.replace("@PVC_CODE@", PVC_CODE)


def main():
    written = []
    key2job = {}
    # smoke first
    p = os.path.join(HERE, "kai-bnf-smoke.yaml")
    with open(p, "w") as f:
        f.write(job_yaml("w1a8", 1, smoke=True))
    written.append(p)
    key2job["smoke"] = "kai-bnf-smoke"

    for v in VARIANTS:
        for s in SEEDS:
            y = job_yaml(v, s)
            path = os.path.join(HERE, f"kai-bnf-{v}-s{s}.yaml")
            with open(path, "w") as f:
                f.write(y)
            written.append(path)
            key2job[f"{v}-s{s}"] = f"kai-bnf-{v}-s{s}"

    # staged waves: <=3 concurrent. Group by seed so each wave is a full precision axis
    # slice + a baseline (spreads variant risk across waves).
    waves = [
        ["w1a8-s1", "w1a6-s1", "w1a4-s1"],
        ["fp32-s1", "w8a8-s1", "w1a8-s2"],
        ["w1a6-s2", "w1a4-s2", "fp32-s2"],
        ["w8a8-s2", "w1a8-s3", "w1a6-s3"],
        ["w1a4-s3", "fp32-s3", "w8a8-s3"],
    ]

    # preflight must ALSO live inside code/hgq2/ — the sync ships only that tree to the
    # PVC, and the script is run as /data/BNJetTag-hgq2/preflight_final.sh (integration
    # gap caught 2026-07-07: variants/-only copy never reached the cluster).
    hgq2_dir = os.path.abspath(os.path.join(HERE, "..", "..", "..", "hgq2"))

    for fname, content, execbit, dirs in [
        # preflight ALSO lives inside code/hgq2/ so it ships in the ConfigMap and can be run as
        # /work/code/preflight_final.sh inside a pod (PVC-free); it is also handy in variants/.
        ("preflight_final.sh", preflight_sh(), True, [HERE, hgq2_dir]),
        ("make_code_configmap.sh", make_cm_sh(), True, [HERE]),   # PVC-free training code source (THE prereq)
        ("launch_final_staged.sh", launch_sh(waves, {k: key2job[k] for k in sum(waves, [])}), True, [HERE]),
        ("sync_code_to_pvc_final.sh", sync_sh(), True, [HERE]),   # kept for the PVC-based ROC job only
    ]:
        for d in dirs:
            path = os.path.join(d, fname)
            with open(path, "w") as f:
                f.write(content)
            if execbit:
                os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
            written.append(path)

    for p in written:
        print("wrote", p)
    print(f"\n{len(VARIANTS)*len(SEEDS)} training YAMLs + smoke + preflight + launch + sync written.")


if __name__ == "__main__":
    main()
