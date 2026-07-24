#!/usr/bin/env python3
"""
Generate NRP (k8s) Job yamls for the BitLinear-transformer architecture sweep.

All variants share the *binary* BitNet weight quantization and the empirically-best
lr15 training recipe (LR=1.5e-4, warmup=1, poly-decay=40, beta2=0.98, wd=0.01,
clip=1.0, batch=256, ES val_auc/max/patience=10, EPOCHS=200, A8 activations).
They differ ONLY along the axes Kai's prior sweep never explored: model SIZE and
norm/architecture STRUCTURE.  This isolates "what small architecture is best" while
holding the optimizer + data + precision fixed.

Two groups:
  SIZE   - shrink D/L/H/FFN together (constant ~4x FFN ratio).  The existing lr15 run
           (D256/L8/H8/FFN1024) is the "large" anchor; reuse its W&B numbers.
  ABL    - anchor on the "small" config and flip exactly ONE architectural knob, so
           every ablation is a controlled A/B vs. var-size-small.

Run:  python3 gen_variant_jobs.py        # writes kai-bnv-*.yaml + preflight.sh here
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ----- shared bits ----------------------------------------------------------
# Broad GPU affinity (same list as the canonical lr15 job) so k8s can place each
# job on whatever accelerator is free.
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

# base training recipe shared by ALL variants (the lr15 recipe)
BASE_RECIPE = (
    "export BN_LR=0.00015 BN_WARMUP_EPOCHS=1 BN_DECAY_EPOCHS=40 BN_DECAY_POWER=1.0\n"
    "          export BN_BETA2=0.98 BN_WEIGHT_DECAY=0.01 BN_CLIPNORM=1.0\n"
    "          export BN_L1_REG=0 BN_BATCH=256\n"
    "          export BN_ES_MONITOR=val_auc BN_ES_MODE=max BN_ES_PATIENCE=10"
)

# ----- the variant matrix ---------------------------------------------------
# extra = dict of additional env exports (the single architectural knob flipped).
VARIANTS = [
    # SIZE sweep (all-default knobs: layernorm / per_linear / learned-posenc / gap / relu, softmax)
    dict(key="tiny",       job="kai-bnv-tiny",       run="var-size-tiny-D32L2H4F128",
         D=32,  H=4, L=2, FFN=128,  extra={},
         note="SIZE sweep: tiny.  Smallest transformer; floor of the size axis."),
    dict(key="small",      job="kai-bnv-small",      run="var-size-small-D64L3H8F256",
         D=64,  H=8, L=3, FFN=256,  extra={},
         note="SIZE sweep: small.  ANCHOR for all ablations below (one-knob A/B baseline)."),
    dict(key="medium",     job="kai-bnv-medium",     run="var-size-medium-D128L4H8F512",
         D=128, H=8, L=4, FFN=512,  extra={},
         note="SIZE sweep: medium.  Midpoint between small and the lr15 large anchor."),
    # ABLATIONS (anchor = small D64/L3/H8/FFN256, flip exactly ONE knob)
    dict(key="rmsnorm",    job="kai-bnv-rmsnorm",    run="var-abl-rmsnorm",
         D=64, H=8, L=3, FFN=256, extra={"BN_NORM_TYPE": "rmsnorm"},
         note="ABLATION vs small: true RMSNorm (no mean-subtraction) instead of full LayerNorm/SubLN."),
    dict(key="sharednorm", job="kai-bnv-sharednorm", run="var-abl-sharednorm",
         D=64, H=8, L=3, FFN=256, extra={"BN_NORM_PLACEMENT": "shared_prenorm"},
         note="ABLATION vs small: ONE pre-norm per sublayer (classic pre-LN) vs per-BitLinear norm."),
    dict(key="noposenc",   job="kai-bnv-noposenc",   run="var-abl-noposenc",
         D=64, H=8, L=3, FFN=256, extra={"BN_POS_ENC": "none"},
         note="ABLATION vs small: drop positional encoding (jet constituents are a set)."),
    dict(key="trueposenc", job="kai-bnv-trueposenc", run="var-abl-trueposenc",
         D=64, H=8, L=3, FFN=256, extra={"BN_POS_ENC": "learned_real"},
         note="ABLATION vs small: a GENUINELY trainable (N,D) position table. The upstream "
              "'learned' PE is applied to a tf.range constant -> folded to a fixed random offset "
              "that trains 0 params; this tests whether a real learned PE actually helps."),
    dict(key="clspool",    job="kai-bnv-clspool",    run="var-abl-clspool",
         D=64, H=8, L=3, FFN=256, extra={"BN_POOL": "cls"},
         note="ABLATION vs small: learnable CLS token + CLS pooling instead of global-average-pool."),
    dict(key="gelu",       job="kai-bnv-gelu",       run="var-abl-gelu",
         D=64, H=8, L=3, FFN=256, extra={"BN_FFN_ACT": "gelu"},
         note="ABLATION vs small: GELU FFN activation instead of ReLU (BitNet/transformer default)."),
    dict(key="sffree",     job="kai-bnv-sffree",     run="var-abl-sffree",
         D=64, H=8, L=3, FFN=256, extra={"BN_SOFTMAX_FREE": "1"},
         note="ABLATION vs small: softmax-free linear attention (hardware-cheaper for L1)."),
]

JOB_TMPL = """apiVersion: batch/v1
kind: Job
metadata:
  name: {job}
  labels:
    app: bnjet-variant
    bnv: "{key}"
spec:
  backoffLimit: 0
  activeDeadlineSeconds: 172800   # 48h safety cap (EarlyStopping usually stops sooner)
  template:
    metadata:
      labels:
        app: bnjet-variant
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
          OUT=/data/outputs/qk-variants/{key}
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
{extra_exports}          {recipe}
          echo "[train] $(date) {run}  D={D} H={H} L={L} FFN={FFN}  knob={knobdesc}"
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


def extra_exports_block(extra):
    if not extra:
        return ""
    line = " ".join(f"{k}={v}" for k, v in extra.items())
    return f"          export {line}\n"


def knobdesc(extra):
    if not extra:
        return "defaults(layernorm/per_linear/learned/gap/relu,softmax)"
    return ",".join(f"{k}={v}" for k, v in extra.items())


def main():
    written = []
    for v in VARIANTS:
        y = JOB_TMPL.format(
            job=v["job"], key=v["key"], run=v["run"], note=v["note"],
            D=v["D"], H=v["H"], L=v["L"], FFN=v["FFN"],
            affinity=NODE_AFFINITY,
            extra_exports=extra_exports_block(v["extra"]),
            recipe=BASE_RECIPE,
            knobdesc=knobdesc(v["extra"]),
        )
        path = os.path.join(HERE, f"{v['job']}.yaml")
        with open(path, "w") as f:
            f.write(y)
        written.append(path)
        print(f"wrote {path}")

    # ---- preflight: build every variant on CPU via --sanity (no data, no GPU) ----
    pf = ["#!/usr/bin/env bash",
          "# Preflight: construct + forward + train_on_batch each variant via --sanity.",
          "# Runs on CPU inside the TF image; no data files, no GPU needed.",
          "set -uo pipefail",
          "CODE=/data/BNJetTag",
          "fail=0",
          'echo "=== BitNet variant preflight $(date) ==="']
    for v in VARIANTS:
        env = f"BN_D_MODEL={v['D']} BN_N_HEADS={v['H']} BN_N_LAYERS={v['L']} BN_FFN_DIM={v['FFN']}"
        for k, val in v["extra"].items():
            env += f" {k}={val}"
        pf.append(f'echo "--- {v["run"]} ({knobdesc(v["extra"])}) ---"')
        # Success = sanity_check ran to completion (it prints "Total trainable parameters"
        # as its LAST line, after build + forward + train_on_batch) AND no Python traceback.
        # NB: do NOT gate on "All BitLinear kernels are binary" — that line only prints for an
        # idealized kernel; real BitLinear stores LATENT fp weights (binarized on-the-fly via
        # STE), so it legitimately never prints. Param count is the meaningful build signal.
        pf.append(
            f'if env CUDA_VISIBLE_DEVICES=-1 {env} python -u "$CODE/qkerasModel.py" --sanity '
            f'> /tmp/pf_{v["key"]}.log 2>&1 '
            f'&& grep -q "Total trainable parameters" /tmp/pf_{v["key"]}.log '
            f'&& ! grep -qE "Traceback|Exception" /tmp/pf_{v["key"]}.log; then '
            f'p=$(grep "Total trainable parameters" /tmp/pf_{v["key"]}.log | tail -1); '
            f'echo "  PASS {v["key"]}  ($p)"; '
            f'else echo "  FAIL {v["key"]}"; tail -30 /tmp/pf_{v["key"]}.log; fail=1; fi'
        )
    pf.append('if [ "$fail" = 0 ]; then echo "PREFLIGHT_ALL_PASS"; else echo "PREFLIGHT_HAD_FAILURES"; fi')
    pf_path = os.path.join(HERE, "preflight.sh")
    with open(pf_path, "w") as f:
        f.write("\n".join(pf) + "\n")
    print(f"wrote {pf_path}")
    print(f"\n{len(written)} job yamls + 1 preflight written.")


if __name__ == "__main__":
    main()
