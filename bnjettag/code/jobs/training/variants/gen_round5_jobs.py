#!/usr/bin/env python3
"""
Round-5 of the BitLinear-transformer sweep — the QUANTIZATION round (arch is FIXED).

Per the 2026-06-29 decision the architecture is frozen at the upstream main config
`large` = D256/L8/H8/FFN1024 (6,373,633 params).  From here we vary *quantization*,
not shape.  Round-4 proved the lr15 recipe (peak 1.5e-4) was too hot even for large
(large@5e-5 = 0.7672 vs the lr15 anchor 0.7530, +0.0142) — which invalidates THREE
things round-5 must now repair:

  1. SEED-CONFIRM   large@5e-5 = 0.7672 is a SINGLE unseeded run (medium's seed
                    spread is ~0.010).  s1/s2 here + the existing r4 run = 3 samples.
  2. RE-BASE A6/A4  the thesis activation axis (A8→A6→A4) was only ever measured
                    under lr15.  Those numbers are stale lower bounds.  Re-run A6/A4
                    at the tuned LR, 2 seeds each.
  3. FAIR BASELINES the FP32 (val 0.7703) and W8A8 (0.7719) baselines were never
                    LR-tuned — the exact tuned-vs-untuned trap round-4 caught for
                    "medium beats large".  Re-train both at 5e-5.  The honest
                    baseline for the report is max(old recipe, 5e-5) per variant —
                    baselines may only get STRONGER.

8 jobs, all D256/L8/H8/F1024, all peak LR 5e-5, everything else = the r1-r4 recipe
(poly-decay 40ep, beta2=0.98, wd=0.01, clip=1.0, batch=256, ES val_auc/max/p10).
Launch STAGED, <=3 concurrent (see launch_r5_staged.sh).

DATASET MIGRATION (2026-07-01): round-5 is the FIRST round trained on the public
HLS4ML LHC Jet dataset (150 particles) at /data/hls4ml_lhc_jet/train/train on the
kai-data PVC (5-class g/q/W/Z/t, top-10 constituents by pT × 16 features = 160
inputs; input-size effects on AUC/latency/resources are a LATER study via the
BN_N_PART knob). Consequences:
  * every AUC quoted above (0.7672, 0.7530, 0.7703, 0.7719, ...) was measured on
    the OLD private 2-file dataset — round-5 numbers are NOT directly comparable
    to them, they start a fresh 5-class (macro-OvR AUC) baseline table;
  * the tuned peak LR 5e-5 is CARRIED OVER as an assumption from the old-dataset
    LR finding, not re-derived — flag it for a spot-check if round-5 looks unstable;
  * "val_auc" is now the macro one-vs-rest AUC over 5 classes, and the ROC-test
    set is the dataset's own held-out val/ split (kai-roc-r5.yaml).

Run:  python3 gen_round5_jobs.py     # writes kai-bn5-*.yaml + preflight_r5.sh + launch_r5_staged.sh
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

# r1-r4 recipe with the tuned peak LR baked in (5e-5 everywhere in round-5).
BASE_RECIPE = dict(
    BN_LR="0.00005", BN_WARMUP_EPOCHS="1", BN_DECAY_EPOCHS="40", BN_DECAY_POWER="1.0",
    BN_BETA2="0.98", BN_WEIGHT_DECAY="0.01", BN_CLIPNORM="1.0",
    BN_L1_REG="0", BN_BATCH="256",
    BN_ES_MONITOR="val_auc", BN_ES_MODE="max", BN_ES_PATIENCE="10",
)

# Fixed architecture (decision 2026-06-29): large, 6,373,633 params.
D, H, L, FFN = 256, 8, 8, 1024

ROUND5 = [
    # ---- 1. seed-confirm large@5e-5 (r4's 0.7672 was single + unseeded) ----
    dict(key="large-lr05-s1", job="kai-bn5-large-lr05-s1", run="r5-large-lr05-s1",
         env={"BN_VARIANT": "bitnet", "BN_ACT_BITS": "8", "BN_SEED": "1"},
         note="SEED-CONFIRM: binary W1A8 large @ 5e-5, seed=1 (r4 unseeded run = 0.7672)."),
    dict(key="large-lr05-s2", job="kai-bn5-large-lr05-s2", run="r5-large-lr05-s2",
         env={"BN_VARIANT": "bitnet", "BN_ACT_BITS": "8", "BN_SEED": "2"},
         note="SEED-CONFIRM: binary W1A8 large @ 5e-5, seed=2."),
    # ---- 2. the thesis quantization axis at the TUNED recipe ----
    dict(key="large-lr05-a6-s1", job="kai-bn5-large-lr05-a6-s1", run="r5-large-lr05-a6-s1",
         env={"BN_VARIANT": "bitnet", "BN_ACT_BITS": "6", "BN_SEED": "1"},
         note="QUANT AXIS: W1A6 @ tuned 5e-5, seed=1 (old A6 val 0.7507 was under too-hot lr15)."),
    dict(key="large-lr05-a6-s2", job="kai-bn5-large-lr05-a6-s2", run="r5-large-lr05-a6-s2",
         env={"BN_VARIANT": "bitnet", "BN_ACT_BITS": "6", "BN_SEED": "2"},
         note="QUANT AXIS: W1A6 @ tuned 5e-5, seed=2."),
    dict(key="large-lr05-a4-s1", job="kai-bn5-large-lr05-a4-s1", run="r5-large-lr05-a4-s1",
         env={"BN_VARIANT": "bitnet", "BN_ACT_BITS": "4", "BN_SEED": "1"},
         note="QUANT AXIS: W1A4 @ tuned 5e-5, seed=1 (old A4 val 0.7437 was under too-hot lr15)."),
    dict(key="large-lr05-a4-s2", job="kai-bn5-large-lr05-a4-s2", run="r5-large-lr05-a4-s2",
         env={"BN_VARIANT": "bitnet", "BN_ACT_BITS": "4", "BN_SEED": "2"},
         note="QUANT AXIS: W1A4 @ tuned 5e-5, seed=2."),
    # ---- 3. FAIR baselines: same tuned recipe for FP32 + W8A8 ----
    dict(key="fp32-lr05", job="kai-bn5-fp32-lr05", run="r5-fp32-lr05",
         env={"BN_VARIANT": "vanilla", "BN_ACT_BITS": "8", "BN_SEED": "1"},
         note="FAIR BASELINE: FP32 vanilla @ 5e-5 (old 0.7703 was never LR-tuned; report max of both)."),
    dict(key="w8a8-lr05", job="kai-bn5-w8a8-lr05", run="r5-w8a8-lr05",
         env={"BN_VARIANT": "w8a8", "BN_ACT_BITS": "8", "BN_SEED": "1"},
         note="FAIR BASELINE: W8A8 @ 5e-5 (old 0.7719 was never LR-tuned; report max of both)."),
]

JOB_TMPL = """apiVersion: batch/v1
kind: Job
metadata:
  name: {job}
  labels:
    app: bnjet-variant-r5
    bnv: "{key}"
spec:
  backoffLimit: 0
  activeDeadlineSeconds: 172800
  template:
    metadata:
      labels:
        app: bnjet-variant-r5
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
          OUT=/data/outputs/qk-variants-r5/{key}
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
          # (NB: any AUC in the note above is an OLD-dataset number — round-5 is the
          #  first round on the HLS4ML LHC Jet 150p dataset; see decisions.md 2026-07-01.)
          # Architecture FIXED at large (decision 2026-06-29): D256/L8/H8/FFN1024.
          # Param count differs slightly from the old 6,373,633 (input is now 16
          # feats, head 5 classes) — preflight_r5.sh prints the exact new count.
          export BN_D_MODEL={D} BN_N_HEADS={H} BN_N_LAYERS={L} BN_FFN_DIM={FFN}
          # top-10 constituents by pT (advisor guidance 2026-07-01; 10x16=160 inputs)
          export BN_N_PART=10
{recipe_exports}{env_exports}          echo "[train] $(date) {run}  D={D} H={H} L={L} FFN={FFN}  {knobdesc}"
          cd "$OUT"
          python -u "$CODE/qkerasModel.py" \\
            /data/hls4ml_lhc_jet/train/train \\
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
    return ",".join(f"{k}={val}" for k, val in v["env"].items())


def main():
    written = []
    for v in ROUND5:
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

    # ---- preflight: CPU --sanity build of every round-5 variant ----
    pf = ["#!/usr/bin/env bash",
          "# Round-5 preflight: dataset check + CPU --sanity build of every round-5 variant.",
          "set -uo pipefail",
          "CODE=/data/BNJetTag",
          "fail=0",
          'echo "=== BitNet round-5 preflight $(date) ==="',
          "",
          "# ---- HLS4ML LHC Jet (150p) dataset check (migration 2026-07-01) ----",
          'echo "--- data check: /data/hls4ml_lhc_jet ---"',
          "if python - <<'PYEOF'",
          "import glob",
          "import h5py",
          'for d in ("/data/hls4ml_lhc_jet/train/train", "/data/hls4ml_lhc_jet/val/val"):',
          '    fs = sorted(glob.glob(d + "/*.h5"))',
          '    assert fs, f"no .h5 files in {d}"',
          '    with h5py.File(fs[0], "r") as hf:',
          '        for k in ("jetConstituentList", "jets",',
          '                  "jetFeatureNames", "particleFeatureNames"):',
          '            assert k in hf, f"{fs[0]} missing key {k}"',
          '        names = [n.decode() if isinstance(n, bytes) else str(n)',
          '                 for n in hf["jetFeatureNames"][:]]',
          '        for l in ("j_g", "j_q", "j_w", "j_z", "j_t"):',
          '            assert l in names, f"label {l} not in jetFeatureNames"',
          '        print(f"  OK {d}: {len(fs)} files, "',
          '              f"jetConstituentList {hf[\'jetConstituentList\'].shape}")',
          "PYEOF",
          'then echo "  PASS data"; else echo "  FAIL data"; fail=1; fi',
          ""]
    for v in ROUND5:
        env = f"BN_D_MODEL={D} BN_N_HEADS={H} BN_N_LAYERS={L} BN_FFN_DIM={FFN} BN_N_PART=10"
        for k, val in v["env"].items():
            env += f" {k}={val}"
        pf.append(f'echo "--- {v["run"]} ({knobdesc(v)}) ---"')
        pf.append(
            f'if env CUDA_VISIBLE_DEVICES=-1 {env} python -u "$CODE/qkerasModel.py" --sanity '
            f'> /tmp/pf5_{v["key"]}.log 2>&1 '
            f'&& grep -q "Total trainable parameters" /tmp/pf5_{v["key"]}.log '
            f'&& ! grep -qE "Traceback|Exception" /tmp/pf5_{v["key"]}.log; then '
            f'p=$(grep "Total trainable parameters" /tmp/pf5_{v["key"]}.log | tail -1); '
            f'echo "  PASS {v["key"]}  ($p)"; '
            f'else echo "  FAIL {v["key"]}"; tail -30 /tmp/pf5_{v["key"]}.log; fail=1; fi'
        )
    pf.append('if [ "$fail" = 0 ]; then echo "PREFLIGHT_ALL_PASS"; else echo "PREFLIGHT_HAD_FAILURES"; fi')
    pf_path = os.path.join(HERE, "preflight_r5.sh")
    with open(pf_path, "w") as f:
        f.write("\n".join(pf) + "\n")
    os.chmod(pf_path, os.stat(pf_path).st_mode | stat.S_IXUSR)
    print(f"wrote {pf_path}")

    # ---- staged launcher: <=3 concurrent (round-4 'Next' constraint) ----
    waves = [
        ["large-lr05-s1", "large-lr05-s2", "fp32-lr05"],
        ["large-lr05-a6-s1", "large-lr05-a6-s2", "w8a8-lr05"],
        ["large-lr05-a4-s1", "large-lr05-a4-s2"],
    ]
    key2job = {v["key"]: v["job"] for v in ROUND5}
    ls = ["#!/usr/bin/env bash",
          "# Launch the ROUND-5 quantization sweep on NRP, STAGED <=3 concurrent Jobs.",
          "# Usage:  ./launch_r5_staged.sh            # launch wave-by-wave, waiting between waves",
          "#         ./launch_r5_staged.sh delete     # tear everything down",
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
          "wait_for_wave () {   # block until every job named in $@ is Complete or Failed",
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
    ls.append('echo "=== round-5 complete ==="')
    ls.append('kubectl --context "$CTX" -n "$NS" get jobs -l app=bnjet-variant-r5')
    ls_path = os.path.join(HERE, "launch_r5_staged.sh")
    with open(ls_path, "w") as f:
        f.write("\n".join(ls) + "\n")
    os.chmod(ls_path, os.stat(ls_path).st_mode | stat.S_IXUSR)
    print(f"wrote {ls_path}")
    print(f"\n{len(written)} round-5 job yamls + preflight_r5.sh + launch_r5_staged.sh written.")


if __name__ == "__main__":
    main()
