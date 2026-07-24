#!/usr/bin/env python3
"""
Round-6-SMALL — the DEPLOYABLE-SCALE era-2 sweep (D32/H4/L2/FFN64, the code-default
architecture), trained PVC-FREE.

Why this round exists (decision 2026-07-04): the HGQ2 rebuild + full-model hls4ml
synthesis effort needs an era-2 model that (a) matches the deployable-scale spec
(10x16=160 inputs, D=32, L=2, 5-class) and (b) actually FITS a VU13P as one monolith
(the frozen `large` D256/L8 composes to ~516% LUT spatially — it can never be the
L1 deployment point). No trained era-2 checkpoint exists at this scale — round-5
was all `large`. This round creates them: same era-2 data, same tuned r5 recipe
(peak LR 5e-5 — carried over untuned for the small model; flag if unstable), the
architecture knobs simply left at the code defaults.

PVC-FREE (pattern proven by kai-roc-r5-pvcfree, 2026-07-02, during the rook-cephfs
outage — PVC mountability is still not trusted):
  * train data : hls4ml_LHCjet_150p_train.tar.gz from Zenodo record 3602260 (2.7 GB)
  * code       : ConfigMap kai-qkerasmodel-r5 (md5 4d0a3fed... = the code that trained r5)
  * checkpoint : wandb.save from inside qkerasModel.py -> W&B runs r6s-*
  * no volume  : emptyDir only

5 jobs, 1 seed each (seed-2 queued only if the HGQ2 effort promotes this scale):
  W1A8 / W1A6 / W1A4 (thesis axis) + FP32 vanilla + W8A8 (baselines).

Run:  python3 gen_round6_small_jobs.py    # writes kai-bn6s-*.yaml + launch_r6s_staged.sh
"""
import os
import stat

HERE = os.path.dirname(os.path.abspath(__file__))

NODE_AFFINITY = """      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
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

# The r5 recipe verbatim (gen_round5_jobs.py BASE_RECIPE). Peak LR 5e-5 was tuned
# for `large` on the OLD dataset and is carried over here untested at D32 — flag
# for a spot-check if val_auc looks unstable.
BASE_RECIPE = dict(
    BN_LR="0.00005", BN_WARMUP_EPOCHS="1", BN_DECAY_EPOCHS="40", BN_DECAY_POWER="1.0",
    BN_BETA2="0.98", BN_WEIGHT_DECAY="0.01", BN_CLIPNORM="1.0",
    BN_L1_REG="0", BN_BATCH="256",
    BN_ES_MONITOR="val_auc", BN_ES_MODE="max", BN_ES_PATIENCE="10",
)

# Deployable-scale architecture = the code defaults (qkerasModel.py:89-92).
D, H, L, FFN = 32, 4, 2, 64

ROUND6S = [
    dict(key="small-a8-s1", job="kai-bn6s-a8-s1", run="r6s-small-a8-s1",
         env={"BN_VARIANT": "bitnet", "BN_ACT_BITS": "8", "BN_SEED": "1"},
         note="THESIS: binary W1A8 at deployable scale D32/L2, seed=1."),
    dict(key="small-a6-s1", job="kai-bn6s-a6-s1", run="r6s-small-a6-s1",
         env={"BN_VARIANT": "bitnet", "BN_ACT_BITS": "6", "BN_SEED": "1"},
         note="QUANT AXIS: W1A6 at deployable scale, seed=1."),
    dict(key="small-a4-s1", job="kai-bn6s-a4-s1", run="r6s-small-a4-s1",
         env={"BN_VARIANT": "bitnet", "BN_ACT_BITS": "4", "BN_SEED": "1"},
         note="QUANT AXIS: W1A4 at deployable scale, seed=1 (era-2 large showed a cliff here)."),
    dict(key="small-fp32-s1", job="kai-bn6s-fp32-s1", run="r6s-small-fp32-s1",
         env={"BN_VARIANT": "vanilla", "BN_ACT_BITS": "8", "BN_SEED": "1"},
         note="BASELINE: FP32 vanilla at deployable scale (the abstract's comparison point)."),
    dict(key="small-w8a8-s1", job="kai-bn6s-w8a8-s1", run="r6s-small-w8a8-s1",
         env={"BN_VARIANT": "w8a8", "BN_ACT_BITS": "8", "BN_SEED": "1"},
         note="BASELINE: W8A8 at deployable scale."),
]

JOB_TMPL = """apiVersion: batch/v1
kind: Job
metadata:
  name: {job}
  labels:
    app: bnjet-r6-small
    bnv: "{key}"
spec:
  backoffLimit: 0
  activeDeadlineSeconds: 86400
  template:
    metadata:
      labels:
        app: bnjet-r6-small
        bnv: "{key}"
    spec:
      restartPolicy: Never
{affinity}
      containers:
      - name: train
        image: tensorflow/tensorflow:2.11.1-gpu
        command: ["bash", "-c"]
        args:
        - |
          set -eo pipefail
          export MPLBACKEND=Agg
          export TF_FORCE_GPU_ALLOW_GROWTH=true
          mkdir -p /work/code /work/data /work/out/bitnet
          cp /qkcode/qkerasModel.py /work/code/
          # py3.8 in the TF 2.11 image: make sure modern annotations don't crash import
          grep -q "from __future__ import annotations" /work/code/qkerasModel.py || \\
            sed -i '1i from __future__ import annotations' /work/code/qkerasModel.py
          export PYTHONPATH="/work/code:$PYTHONPATH"
          echo "[setup] $(date) deps"
          apt-get update -qq && apt-get install -y -qq graphviz
          pip install -q qkeras==0.9.0 "tensorflow==2.11.1" "matplotlib<3.8" pandas seaborn mplhep pydot scikit-learn h5py wandb
          nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
          export WANDB_PROJECT=bnjettag-bitnet
          export WANDB_RUN_NAME={run}

          echo "[data] $(date) fetching HLS4ML LHC Jet 150p TRAIN split from Zenodo (2.7 GB)"
          python -u -c 'import urllib.request; urllib.request.urlretrieve(
            "https://zenodo.org/records/3602260/files/hls4ml_LHCjet_150p_train.tar.gz?download=1",
            "/work/data/train.tar.gz")'
          sz=$(stat -c %s /work/data/train.tar.gz); echo "[data] tarball bytes: $sz"
          [ "$sz" -gt 2500000000 ] || {{ echo "[fatal] train tarball too small"; exit 1; }}
          tar -xzf /work/data/train.tar.gz -C /work/data && rm /work/data/train.tar.gz
          DATA=$(dirname "$(find /work/data -name 'jetImage_*.h5' -print -quit)")
          [ -n "$DATA" ] || {{ echo "[fatal] no jetImage_*.h5 after extract"; exit 1; }}
          echo "[data] DATA=$DATA ($(ls "$DATA" | wc -l) files)"

          # ---- {note} ----
          # Deployable-scale architecture = code defaults (decision 2026-07-04).
          export BN_D_MODEL={D} BN_N_HEADS={H} BN_N_LAYERS={L} BN_FFN_DIM={FFN}
          export BN_N_PART=10
{recipe_exports}{env_exports}
          echo "[preflight] $(date) --sanity build"
          env CUDA_VISIBLE_DEVICES=-1 python -u /work/code/qkerasModel.py --sanity 2>&1 | tail -5
          echo "[train] $(date) {run}  D={D} H={H} L={L} FFN={FFN}  {knobdesc}"
          cd /work/out
          python -u /work/code/qkerasModel.py "$DATA" 2>&1 | tee /work/out/train_{key}.log
          echo "[done] $(date) {run}"
        env:
        - name: WANDB_API_KEY
          valueFrom:
            secretKeyRef:
              name: kai-wandb
              key: WANDB_API_KEY
        resources:
          limits:
            nvidia.com/gpu: "1"
            memory: 24Gi
            cpu: "4"
            ephemeral-storage: 24Gi
          requests:
            nvidia.com/gpu: "1"
            memory: 24Gi
            cpu: "4"
            ephemeral-storage: 20Gi
        volumeMounts:
        - name: qkcode
          mountPath: /qkcode
        - name: work
          mountPath: /work
      volumes:
      - name: qkcode
        configMap:
          name: kai-qkerasmodel-r5
      - name: work
        emptyDir:
          sizeLimit: 24Gi
"""


def exports_block(d):
    if not d:
        return ""
    line = " ".join(f"{k}={v}" for k, v in d.items())
    return f"          export {line}\n"


def knobdesc(v):
    return ",".join(f"{k}={val}" for k, val in v["env"].items())


def main():
    written = []
    for v in ROUND6S:
        y = JOB_TMPL.format(
            job=v["job"], key=v["key"], run=v["run"], note=v["note"],
            D=D, H=H, L=L, FFN=FFN,
            affinity=NODE_AFFINITY,
            recipe_exports=exports_block(BASE_RECIPE),
            env_exports=exports_block(v["env"]),
            knobdesc=knobdesc(v),
        )
        path = os.path.join(HERE, f"{v['job']}.yaml")
        with open(path, "w") as f:
            f.write(y)
        written.append(path)
        print(f"wrote {path}")

    # staged launcher, <=3 concurrent (cluster etiquette carried over from r5)
    waves = [
        ["small-a8-s1", "small-fp32-s1", "small-w8a8-s1"],
        ["small-a6-s1", "small-a4-s1"],
    ]
    key2job = {v["key"]: v["job"] for v in ROUND6S}
    ls = ["#!/usr/bin/env bash",
          "# Launch round-6-small on NRP, STAGED <=3 concurrent Jobs.",
          "# Usage:  ./launch_r6s_staged.sh            # wave-by-wave",
          "#         ./launch_r6s_staged.sh delete     # tear down",
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
    ls.append('echo "=== round-6-small complete ==="')
    ls.append('kubectl --context "$CTX" -n "$NS" get jobs -l app=bnjet-r6-small')
    ls_path = os.path.join(HERE, "launch_r6s_staged.sh")
    with open(ls_path, "w") as f:
        f.write("\n".join(ls) + "\n")
    os.chmod(ls_path, os.stat(ls_path).st_mode | stat.S_IXUSR)
    print(f"wrote {ls_path}")
    print(f"\n{len(written)} round-6-small job yamls + launch_r6s_staged.sh written.")


if __name__ == "__main__":
    main()
