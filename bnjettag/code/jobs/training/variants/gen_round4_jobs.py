#!/usr/bin/env python3
"""
Round-4 of the BitLinear-transformer sweep.  Round-3 left ONE thread open in the size-vs-recipe
story: `medium`@5e-5 = 0.7592 is the sweep-best, beating the ~3M "large" model (0.7530) at ~4x
fewer params -- BUT the large 0.7530 anchor was trained with the lr15 recipe (peak LR 1.5e-4),
the very recipe that we proved is TOO HOT at smaller sizes.  If large is ALSO LR-limited, then
"medium beats large" is comparing a tuned medium against an UN-tuned large, which is not a fair
size comparison.

Round-4 closes that thread:
  Confirm  - medium @ 5e-5 with BN_SEED=1,2 (error bars on the 0.7592 sweep-best, just like we
             put error bars on small@5e-5 in round-3).
  Tune large - large @ 5e-5 / 7.5e-5 (D256/L8/H8/FFN1024, ~3M params).  Is the ~3M lr15 anchor
             also LR-limited?  If large@5e-5 jumps well above 0.7530, then the size gap was real
             all along and we need a tuned-large baseline; if it stalls ~0.753, then medium is a
             genuine efficiency sweet spot.

Everything else (binary weights, A8, poly-decay 40ep, beta2=0.98, wd=0.01, clip=1.0, batch=256,
ES val_auc/max/p10, EPOCHS=200) is held fixed.

NB: `large` is ~3M params and ~4x slower per epoch than medium; give the waiter a generous budget.

Run:  python3 gen_round4_jobs.py     # writes kai-bn4-*.yaml + preflight_r4.sh here
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

ROUND4 = [
    # ---- confirm the sweep-best (medium @ 5e-5 = 0.7592) with seeds ----
    dict(key="medium-lr05-s1", job="kai-bn4-medium-lr05-s1", run="r4-medium-lr05-s1",
         D=128, H=8, L=4, FFN=512, arch={}, recipe={"BN_LR": "0.00005", "BN_SEED": "1"},
         note="CONFIRM: medium @ 5e-5, seed=1 (error bar on the 0.7592 sweep-best)."),
    dict(key="medium-lr05-s2", job="kai-bn4-medium-lr05-s2", run="r4-medium-lr05-s2",
         D=128, H=8, L=4, FFN=512, arch={}, recipe={"BN_LR": "0.00005", "BN_SEED": "2"},
         note="CONFIRM: medium @ 5e-5, seed=2."),
    # ---- tune large: is the ~3M lr15 anchor (0.7530) ALSO LR-limited? ----
    dict(key="large-lr05", job="kai-bn4-large-lr05", run="r4-large-lr05",
         D=256, H=8, L=8, FFN=1024, arch={}, recipe={"BN_LR": "0.00005"},
         note="TUNE-LARGE: large D256/L8/H8/F1024 (~3M) @ 5e-5 (lr15 anchor was 0.7530; LR-limited?)."),
    dict(key="large-lr075", job="kai-bn4-large-lr075", run="r4-large-lr075",
         D=256, H=8, L=8, FFN=1024, arch={}, recipe={"BN_LR": "0.000075"},
         note="TUNE-LARGE: large @ 7.5e-5."),
]

JOB_TMPL = """apiVersion: batch/v1
kind: Job
metadata:
  name: {job}
  labels:
    app: bnjet-variant-r4
    bnv: "{key}"
spec:
  backoffLimit: 0
  activeDeadlineSeconds: 172800
  template:
    metadata:
      labels:
        app: bnjet-variant-r4
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
          OUT=/data/outputs/qk-variants-r4/{key}
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
    for v in ROUND4:
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
          "# Round-4 preflight: CPU --sanity build of every round-4 variant.",
          "set -uo pipefail",
          "CODE=/data/BNJetTag",
          "fail=0",
          'echo "=== BitNet round-4 preflight $(date) ==="']
    for v in ROUND4:
        env = f"BN_D_MODEL={v['D']} BN_N_HEADS={v['H']} BN_N_LAYERS={v['L']} BN_FFN_DIM={v['FFN']}"
        for k, val in {**v["recipe"], **v["arch"]}.items():
            env += f" {k}={val}"
        pf.append(f'echo "--- {v["run"]} ({knobdesc(v)}) ---"')
        pf.append(
            f'if env CUDA_VISIBLE_DEVICES=-1 {env} python -u "$CODE/qkerasModel.py" --sanity '
            f'> /tmp/pf4_{v["key"]}.log 2>&1 '
            f'&& grep -q "Total trainable parameters" /tmp/pf4_{v["key"]}.log '
            f'&& ! grep -qE "Traceback|Exception" /tmp/pf4_{v["key"]}.log; then '
            f'p=$(grep "Total trainable parameters" /tmp/pf4_{v["key"]}.log | tail -1); '
            f'echo "  PASS {v["key"]}  ($p)"; '
            f'else echo "  FAIL {v["key"]}"; tail -30 /tmp/pf4_{v["key"]}.log; fail=1; fi'
        )
    pf.append('if [ "$fail" = 0 ]; then echo "PREFLIGHT_ALL_PASS"; else echo "PREFLIGHT_HAD_FAILURES"; fi')
    pf_path = os.path.join(HERE, "preflight_r4.sh")
    with open(pf_path, "w") as f:
        f.write("\n".join(pf) + "\n")
    print(f"wrote {pf_path}")
    print(f"\n{len(written)} round-4 job yamls + 1 preflight written.")


if __name__ == "__main__":
    main()
