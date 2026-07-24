#!/usr/bin/env python3
"""
Round-2 of the BitLinear-transformer sweep.  Round-1 (gen_variant_jobs.py) found that EVERY
small model peaked at epoch 2-6 then collapsed under the lr15 recipe (peak LR 1.5e-4 was tuned
for the ~3M "large" model and is too hot at these sizes).  So round-2:

  LR sweep   - re-train the `small` anchor at gentler peak LRs (5e-5 / 7.5e-5 / 1e-4) + a
               longer-warmup variant, to find small's true ceiling instead of a transient peak.
  COMBO      - stack the three round-1 winning knobs (CLS pool + GELU FFN + no-PE) on small and
               on medium (best-arch x best-size).
  SEEDS      - seed-repeats (BN_SEED) of the small baseline and the clspool ablation, so the
               ~0.013 knob gaps get error bars.

Everything else (binary weights, A8 activations, poly-decay 40ep, beta2=0.98, wd=0.01, clip=1.0,
batch=256, ES val_auc/max/p10, EPOCHS=200) is held fixed, exactly as round-1.

Run:  python3 gen_round2_jobs.py     # writes kai-bn2-*.yaml + preflight_r2.sh here
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Broad GPU affinity (identical to round-1 / the canonical lr15 job).
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

# The lr15 recipe as a dict so per-variant `recipe` overrides can patch single keys.
BASE_RECIPE = dict(
    BN_LR="0.00015", BN_WARMUP_EPOCHS="1", BN_DECAY_EPOCHS="40", BN_DECAY_POWER="1.0",
    BN_BETA2="0.98", BN_WEIGHT_DECAY="0.01", BN_CLIPNORM="1.0",
    BN_L1_REG="0", BN_BATCH="256",
    BN_ES_MONITOR="val_auc", BN_ES_MODE="max", BN_ES_PATIENCE="10",
)

# arch = architectural knobs (BN_POOL/BN_FFN_ACT/...); recipe = optimizer overrides on BASE_RECIPE.
ROUND2 = [
    # ---- LR sweep on the small anchor (fix the lr15-too-hot early-peak collapse) ----
    dict(key="small-lr05", job="kai-bn2-small-lr05", run="r2-small-lr05",
         D=64, H=8, L=3, FFN=256, arch={}, recipe={"BN_LR": "0.00005"},
         note="LR sweep: small @ 5e-5 (3x gentler than lr15) to stop the early-peak collapse."),
    dict(key="small-lr075", job="kai-bn2-small-lr075", run="r2-small-lr075",
         D=64, H=8, L=3, FFN=256, arch={}, recipe={"BN_LR": "0.000075"},
         note="LR sweep: small @ 7.5e-5."),
    dict(key="small-lr10", job="kai-bn2-small-lr10", run="r2-small-lr10",
         D=64, H=8, L=3, FFN=256, arch={}, recipe={"BN_LR": "0.0001"},
         note="LR sweep: small @ 1e-4."),
    dict(key="small-lr10-warm3", job="kai-bn2-small-lr10-warm3", run="r2-small-lr10-warm3",
         D=64, H=8, L=3, FFN=256, arch={}, recipe={"BN_LR": "0.0001", "BN_WARMUP_EPOCHS": "3"},
         note="LR sweep: small @ 1e-4 with 3-epoch warmup (extra stabilization)."),
    # ---- combo-best architecture: stack round-1 winners (lr15, for round-1 comparability) ----
    dict(key="combo-small", job="kai-bn2-combo-small", run="r2-combo-small",
         D=64, H=8, L=3, FFN=256,
         arch={"BN_POOL": "cls", "BN_FFN_ACT": "gelu", "BN_POS_ENC": "none"}, recipe={},
         note="COMBO: stack the 3 round-1 winners (CLS pool + GELU FFN + no-PE) on small."),
    dict(key="combo-medium", job="kai-bn2-combo-medium", run="r2-combo-medium",
         D=128, H=8, L=4, FFN=512,
         arch={"BN_POOL": "cls", "BN_FFN_ACT": "gelu", "BN_POS_ENC": "none"}, recipe={},
         note="COMBO: the 3 winners on medium (D128/L4) -- best arch x best size."),
    # ---- seed repeats: error bars on the ~0.013 knob gaps (paired small vs clspool) ----
    dict(key="small-s1", job="kai-bn2-small-s1", run="r2-small-s1",
         D=64, H=8, L=3, FFN=256, arch={}, recipe={"BN_SEED": "1"},
         note="SEED repeat: small baseline, seed=1 (round-1 unseeded best was 0.7335)."),
    dict(key="small-s2", job="kai-bn2-small-s2", run="r2-small-s2",
         D=64, H=8, L=3, FFN=256, arch={}, recipe={"BN_SEED": "2"},
         note="SEED repeat: small baseline, seed=2."),
    dict(key="clspool-s1", job="kai-bn2-clspool-s1", run="r2-clspool-s1",
         D=64, H=8, L=3, FFN=256, arch={"BN_POOL": "cls"}, recipe={"BN_SEED": "1"},
         note="SEED repeat: clspool, seed=1 (paired with small-s1; round-1 best was 0.7475)."),
    dict(key="clspool-s2", job="kai-bn2-clspool-s2", run="r2-clspool-s2",
         D=64, H=8, L=3, FFN=256, arch={"BN_POOL": "cls"}, recipe={"BN_SEED": "2"},
         note="SEED repeat: clspool, seed=2 (paired with small-s2)."),
]

JOB_TMPL = """apiVersion: batch/v1
kind: Job
metadata:
  name: {job}
  labels:
    app: bnjet-variant-r2
    bnv: "{key}"
spec:
  backoffLimit: 0
  activeDeadlineSeconds: 172800   # 48h safety cap (EarlyStopping usually stops sooner)
  template:
    metadata:
      labels:
        app: bnjet-variant-r2
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
          OUT=/data/outputs/qk-variants-r2/{key}
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
    """One `export K=V ...` line (indented to match the YAML literal block) or ''."""
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
    for v in ROUND2:
        recipe = {**BASE_RECIPE, **v["recipe"]}   # per-variant overrides patch the lr15 base
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

    # ---- preflight: CPU --sanity build of every round-2 variant (no data, no GPU) ----
    pf = ["#!/usr/bin/env bash",
          "# Round-2 preflight: construct + forward + train_on_batch each variant via --sanity.",
          "set -uo pipefail",
          "CODE=/data/BNJetTag",
          "fail=0",
          'echo "=== BitNet round-2 preflight $(date) ==="']
    for v in ROUND2:
        env = f"BN_D_MODEL={v['D']} BN_N_HEADS={v['H']} BN_N_LAYERS={v['L']} BN_FFN_DIM={v['FFN']}"
        for k, val in {**v["recipe"], **v["arch"]}.items():
            env += f" {k}={val}"
        pf.append(f'echo "--- {v["run"]} ({knobdesc(v)}) ---"')
        pf.append(
            f'if env CUDA_VISIBLE_DEVICES=-1 {env} python -u "$CODE/qkerasModel.py" --sanity '
            f'> /tmp/pf2_{v["key"]}.log 2>&1 '
            f'&& grep -q "Total trainable parameters" /tmp/pf2_{v["key"]}.log '
            f'&& ! grep -qE "Traceback|Exception" /tmp/pf2_{v["key"]}.log; then '
            f'p=$(grep "Total trainable parameters" /tmp/pf2_{v["key"]}.log | tail -1); '
            f'echo "  PASS {v["key"]}  ($p)"; '
            f'else echo "  FAIL {v["key"]}"; tail -30 /tmp/pf2_{v["key"]}.log; fail=1; fi'
        )
    pf.append('if [ "$fail" = 0 ]; then echo "PREFLIGHT_ALL_PASS"; else echo "PREFLIGHT_HAD_FAILURES"; fi')
    pf_path = os.path.join(HERE, "preflight_r2.sh")
    with open(pf_path, "w") as f:
        f.write("\n".join(pf) + "\n")
    print(f"wrote {pf_path}")
    print(f"\n{len(written)} round-2 job yamls + 1 preflight written.")


if __name__ == "__main__":
    main()
