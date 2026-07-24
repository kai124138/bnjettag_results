#!/usr/bin/env python3
"""
Round-3 of the BitLinear-transformer sweep.  Round-2 established that the OPTIMIZER (peak LR),
not the architectural knobs, governs AUC at small scale: `small`@5e-5 hit 0.7540 (matching the
~3M "large" model), while the round-1 clspool/gelu "wins" evaporated under seed-averaging.

Round-3 nails down the recommended config:
  LR ceiling - small @ 2.5e-5 / 3.5e-5 (lr05 peaked late at ep17, so there may be more headroom).
  Confirm    - small @ 5e-5 with BN_SEED=1,2 (put error bars on the 0.7540 recommendation).
  Size x LR  - medium @ 5e-5 / 7.5e-5 (medium also collapsed at ep3 under lr15, so it is LR-limited
               too; does the size gap re-open once medium gets a proper LR?).

Everything else (binary weights, A8, poly-decay 40ep, beta2=0.98, wd=0.01, clip=1.0, batch=256,
ES val_auc/max/p10, EPOCHS=200) is held fixed.

Run:  python3 gen_round3_jobs.py     # writes kai-bn3-*.yaml + preflight_r3.sh here
"""
import os

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

BASE_RECIPE = dict(
    BN_LR="0.00015", BN_WARMUP_EPOCHS="1", BN_DECAY_EPOCHS="40", BN_DECAY_POWER="1.0",
    BN_BETA2="0.98", BN_WEIGHT_DECAY="0.01", BN_CLIPNORM="1.0",
    BN_L1_REG="0", BN_BATCH="256",
    BN_ES_MONITOR="val_auc", BN_ES_MODE="max", BN_ES_PATIENCE="10",
)

ROUND3 = [
    # ---- LR ceiling on small (lr05 peaked late at ep17 -> probe lower) ----
    dict(key="small-lr025", job="kai-bn3-small-lr025", run="r3-small-lr025",
         D=64, H=8, L=3, FFN=256, arch={}, recipe={"BN_LR": "0.000025"},
         note="LR ceiling: small @ 2.5e-5 (lower than the round-2 best 5e-5)."),
    dict(key="small-lr035", job="kai-bn3-small-lr035", run="r3-small-lr035",
         D=64, H=8, L=3, FFN=256, arch={}, recipe={"BN_LR": "0.000035"},
         note="LR ceiling: small @ 3.5e-5 (between 2.5e-5 and the round-2 best 5e-5)."),
    # ---- confirm the round-2 best (small @ 5e-5 = 0.7540) with seeds ----
    dict(key="small-lr05-s1", job="kai-bn3-small-lr05-s1", run="r3-small-lr05-s1",
         D=64, H=8, L=3, FFN=256, arch={}, recipe={"BN_LR": "0.00005", "BN_SEED": "1"},
         note="CONFIRM: small @ 5e-5, seed=1 (error bar on the 0.7540 recommendation)."),
    dict(key="small-lr05-s2", job="kai-bn3-small-lr05-s2", run="r3-small-lr05-s2",
         D=64, H=8, L=3, FFN=256, arch={}, recipe={"BN_LR": "0.00005", "BN_SEED": "2"},
         note="CONFIRM: small @ 5e-5, seed=2."),
    # ---- size x LR: is medium also LR-limited?  Re-run below lr15. ----
    dict(key="medium-lr05", job="kai-bn3-medium-lr05", run="r3-medium-lr05",
         D=128, H=8, L=4, FFN=512, arch={}, recipe={"BN_LR": "0.00005"},
         note="SIZExLR: medium @ 5e-5 (medium collapsed at ep3 under lr15 -> also LR-limited)."),
    dict(key="medium-lr075", job="kai-bn3-medium-lr075", run="r3-medium-lr075",
         D=128, H=8, L=4, FFN=512, arch={}, recipe={"BN_LR": "0.000075"},
         note="SIZExLR: medium @ 7.5e-5."),
]

JOB_TMPL = """apiVersion: batch/v1
kind: Job
metadata:
  name: {job}
  labels:
    app: bnjet-variant-r3
    bnv: "{key}"
spec:
  backoffLimit: 0
  activeDeadlineSeconds: 172800
  template:
    metadata:
      labels:
        app: bnjet-variant-r3
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
          CODE=/data/BNJetTag
          OUT=/data/outputs/qk-variants-r3/{key}
          mkdir -p "$OUT/bitnet" "$OUT/v1/bitnet"
          echo "[setup] $(date) deps"
          apt-get update -qq && apt-get install -y -qq graphviz
          pip install -q qkeras==0.9.0 "tensorflow==2.11.1" "matplotlib<3.8" pandas seaborn mplhep pydot scikit-learn h5py wandb
          ( grep -rl " | " --include="*.py" "$CODE" 2>/dev/null | while read f; do grep -q "from __future__ import annotations" "$f" || sed -i '1i from __future__ import annotations' "$f"; done ) || true
          nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
          export PYTHONPATH="$CODE:$PYTHONPATH"
          export WANDB_PROJECT=bnjettag-bitnet
          export WANDB_RUN_NAME={run}
          # ---- {note} ----
          # BINARY weights (BN_TERNARY unset).  A8 activations (BN_ACT_BITS default 8).
          export BN_D_MODEL={D} BN_N_HEADS={H} BN_N_LAYERS={L} BN_FFN_DIM={FFN}
{recipe_exports}{arch_exports}          echo "[train] $(date) {run}  D={D} H={H} L={L} FFN={FFN}  {knobdesc}"
          cd "$OUT"
          python -u "$CODE/qkerasModel.py" \\
            /data/bnjet/train_merged/merged_trainPart.h5 \\
            /data/bnjet/QCD_Pt15To3000_Flat_PU200/Bkg_train.h5 \\
            /data/bnjet/train_merged/merged_trainJet.h5 \\
            /data/bnjet/QCD_Pt15To3000_Flat_PU200/Bkg_trainJets.h5 \\
            2>&1 | tee "$OUT/train_{key}.log"
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
          requests:
            nvidia.com/gpu: "1"
            memory: 24Gi
            cpu: "4"
        volumeMounts:
        - name: data
          mountPath: /data
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: kai-data
"""


def exports_block(d):
    if not d:
        return ""
    line = " ".join(f"{k}={v}" for k, v in d.items())
    return f"          export {line}\n"


def knobdesc(v):
    recipe = ",".join(f"{k}={val}" for k, val in v["recipe"].items()) or "lr15"
    arch = ",".join(f"{k}={val}" for k, val in v["arch"].items()) or "default-arch"
    return f"recipe[{recipe}] arch[{arch}]"


def main():
    written = []
    for v in ROUND3:
        recipe = {**BASE_RECIPE, **v["recipe"]}
        y = JOB_TMPL.format(
            job=v["job"], key=v["key"], run=v["run"], note=v["note"],
            D=v["D"], H=v["H"], L=v["L"], FFN=v["FFN"],
            affinity=NODE_AFFINITY,
            recipe_exports=exports_block(recipe),
            arch_exports=exports_block(v["arch"]),
            knobdesc=knobdesc(v),
        )
        path = os.path.join(HERE, f"{v['job']}.yaml")
        with open(path, "w") as f:
            f.write(y)
        written.append(path)
        print(f"wrote {path}")

    pf = ["#!/usr/bin/env bash",
          "# Round-3 preflight: CPU --sanity build of every round-3 variant.",
          "set -uo pipefail",
          "CODE=/data/BNJetTag",
          "fail=0",
          'echo "=== BitNet round-3 preflight $(date) ==="']
    for v in ROUND3:
        env = f"BN_D_MODEL={v['D']} BN_N_HEADS={v['H']} BN_N_LAYERS={v['L']} BN_FFN_DIM={v['FFN']}"
        for k, val in {**v["recipe"], **v["arch"]}.items():
            env += f" {k}={val}"
        pf.append(f'echo "--- {v["run"]} ({knobdesc(v)}) ---"')
        pf.append(
            f'if env CUDA_VISIBLE_DEVICES=-1 {env} python -u "$CODE/qkerasModel.py" --sanity '
            f'> /tmp/pf3_{v["key"]}.log 2>&1 '
            f'&& grep -q "Total trainable parameters" /tmp/pf3_{v["key"]}.log '
            f'&& ! grep -qE "Traceback|Exception" /tmp/pf3_{v["key"]}.log; then '
            f'p=$(grep "Total trainable parameters" /tmp/pf3_{v["key"]}.log | tail -1); '
            f'echo "  PASS {v["key"]}  ($p)"; '
            f'else echo "  FAIL {v["key"]}"; tail -30 /tmp/pf3_{v["key"]}.log; fail=1; fi'
        )
    pf.append('if [ "$fail" = 0 ]; then echo "PREFLIGHT_ALL_PASS"; else echo "PREFLIGHT_HAD_FAILURES"; fi')
    pf_path = os.path.join(HERE, "preflight_r3.sh")
    with open(pf_path, "w") as f:
        f.write("\n".join(pf) + "\n")
    print(f"wrote {pf_path}")
    print(f"\n{len(written)} round-3 job yamls + 1 preflight written.")


if __name__ == "__main__":
    main()
