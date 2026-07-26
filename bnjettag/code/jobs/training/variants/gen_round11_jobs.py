#!/usr/bin/env python3
"""Round-11 capacity-ladder job YAMLs.

Emits one NRP Job per (config, seed) from the r11 configs written by
`code/hgq2/configs/gen_r11_ladder.py`.  Structure is the round-10 YAML verbatim
except for: the round label/name, the ConfigMap name (kai-bn11-code -- r10's
kai-bn10-code stays frozen, per the one-ConfigMap-per-round convention), the
stale-ConfigMap gate (r11 requires its own configs to be present in the unpacked
tree), the config path, the seed, and the W&B run name.

Stage 0 is the LR probe (one seed); stage 1 is the ladder (three seeds) and is
generated only after the probe has picked each scale's peak LR -- r7b's lesson
was that an inherited LR silently invalidated a whole column.

Usage (from code/jobs/training/variants/):
    python gen_round11_jobs.py stage0
    python gen_round11_jobs.py stage1
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = os.path.abspath(os.path.join(HERE, "..", "..", "..", "hgq2", "configs"))

import glob as _glob

# Derived from the config dir so it tracks gen_r11_ladder.py's probe grid, which
# was extended after d64's first-pass optimum landed on the edge of {1e-5,2e-5,5e-5}.
STAGE0 = sorted((os.path.basename(p)[:-5], 1)
                for p in _glob.glob(os.path.join(CONFIGS, "r11-lrprobe-*.json")))

STAGE1 = [(f"r11-d{d}-{arm}", s)
          for d in (48, 64, 96, 128)
          for arm in ("w1a8-stdnn", "w8a8-std", "fp32-std", "w8a8-stdnn")
          for s in (1, 2, 3)] + [(f"r11-d32-{a}-stdnn", s) for a in ("w8a8", "fp32") for s in (1, 2, 3)]

TEMPLATE = """apiVersion: batch/v1
kind: Job
metadata:
  name: {job}
  labels:
    app: bnjet-r11
    bn11: "{leaf}"
spec:
  # backoffLimit 2: an UnexpectedAdmissionError (node GPU device-plugin race, zero side
  # effects) must not kill the Job. Duplicate-W&B-run risk from a mid-train retry is
  # accepted (the ROC fetcher prefers the finished run if duplicates appear).
  backoffLimit: 2
  activeDeadlineSeconds: 14400
  template:
    metadata:
      labels:
        app: bnjet-r11
        bn11: "{leaf}"
    spec:
      restartPolicy: Never
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: kubernetes.io/hostname
                operator: NotIn
                values: [k8s-chase-ci-07.calit2.optiputer.net, k8s-chase-ci-10.calit2.optiputer.net, k8s-haosu-22.sdsc.optiputer.net, ry-gpu-08.sdsc.optiputer.net, suncave-1, suncave-2, suncave-3, suncave-4, suncave-5, suncave-6, suncave-7, suncave-8, suncave-9, suncave-10, suncave-11, suncave-12, suncave-13, suncave-14, suncave-15, suncave-16, suncave-17, suncave-18, suncave-19, suncave-20]
              - key: kubernetes.io/arch
                operator: In
                values: [amd64]
              - key: nvidia.com/gpu.product
                operator: In
                values:
                - NVIDIA-H200-NVL
                - NVIDIA-H100-80GB-HBM3
                - NVIDIA-RTX-PRO-6000-Blackwell-Max-Q-Workstation-Edition
                - NVIDIA-A100-SXM4-80GB
                - NVIDIA-A100-80GB-PCIe
                - NVIDIA-A100-PCIE-40GB
                - NVIDIA-GeForce-RTX-4090
                - NVIDIA-L40S
                - NVIDIA-L40
                - NVIDIA-RTX-A6000
                - NVIDIA-A10
                - NVIDIA-L4
                - NVIDIA-GeForce-RTX-2080-Ti
                - Tesla-V100-SXM2-32GB
                - Tesla-V100-PCIE-32GB
                - NVIDIA-A40
                - NVIDIA-RTX-A5000
                - NVIDIA-GeForce-RTX-3090
          preferredDuringSchedulingIgnoredDuringExecution:
          # PREFER the abundant mid-tier (A10/L4/2080-Ti/V100/A40/A5000/3090) — plentiful and
          # more than enough for a <=300k-param model. De-prefer A100s, flagship = last resort.
          - weight: 100
            preference:
              matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values: [NVIDIA-A10, NVIDIA-L4, NVIDIA-GeForce-RTX-2080-Ti, Tesla-V100-SXM2-32GB, Tesla-V100-PCIE-32GB, NVIDIA-A40, NVIDIA-RTX-A5000, NVIDIA-GeForce-RTX-3090]
          - weight: 60
            preference:
              matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values: [NVIDIA-GeForce-RTX-4090, NVIDIA-L40S, NVIDIA-L40, NVIDIA-RTX-A6000]
          - weight: 20
            preference:
              matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values: [NVIDIA-A100-SXM4-80GB, NVIDIA-A100-80GB-PCIe, NVIDIA-A100-PCIE-40GB]
          - weight: 10
            preference:
              matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values: [NVIDIA-H200-NVL, NVIDIA-H100-80GB-HBM3, NVIDIA-RTX-PRO-6000-Blackwell-Max-Q-Workstation-Edition]
      containers:
      - name: train
        image: python:3.12
        command: ["bash", "-c"]
        args:
        - |
          set -eo pipefail
          export KERAS_BACKEND=tensorflow MPLBACKEND=Agg TF_FORCE_GPU_ALLOW_GROWTH=true
          mkdir -p /work/code /work/data /work/outputs
          CODE=/work/code
          OUT=/work/outputs/{leaf}
          mkdir -p "$OUT"
          echo "[code] $(date) unpack code from ConfigMap kai-bn11-code (PVC-free)"
          tar -xzf /cmcode/hgq2.tar.gz -C "$CODE" --strip-components=1
          md5sum /cmcode/hgq2.tar.gz | awk '{{print "[code] configmap tar md5:", $1}}'
          # ---- STALE-CONFIGMAP GATE: round-11 REQUIRES its own ladder configs in the code ----
          [ -f "$CODE/configs/{config}.json" ] || {{
            echo "[fatal] unpacked code has no configs/{config}.json -> ConfigMap kai-bn11-code is STALE."
            echo "[fatal] rebuild it (make_code_configmap.sh with name kai-bn11-code) and re-apply."
            exit 1
          }}
          echo "[gate] configs/{config}.json present (ConfigMap kai-bn11-code is fresh)"
          export PYTHONPATH="$CODE:$PYTHONPATH"
          echo "[setup] $(date) pinned deps (mirror .venv-hgq2)"
          pip install -q --no-cache-dir "tensorflow[and-cuda]==2.21.0" "keras==3.15.0" "hgq2==0.1.9" "quantizers==1.2.2" "scikit-learn==1.9.0" "h5py==3.14.0" "wandb==0.28.0" "hls4ml==1.3.0" "numpy==2.5.0"
          # TF 2.21 pip wheel's RPATH discovery of the nvidia-*-cu12 wheel libs is broken;
          # explicit LD_LIBRARY_PATH over the wheel lib dirs fixes it (verified 2026-07-07).
          NVLIBS=$(python -c "import glob; print(':'.join(sorted(glob.glob('/usr/local/lib/python*/site-packages/nvidia/*/lib'))))")
          export LD_LIBRARY_PATH="$NVLIBS${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
          echo "[gpu] $(date) nvidia-smi + TF backend assertion"
          nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {{ echo "[fatal] no nvidia-smi / GPU"; exit 1; }}
          python -c "import os; os.environ.setdefault('KERAS_BACKEND','tensorflow'); import tensorflow as tf; g=tf.config.list_physical_devices('GPU'); assert g, 'NO GPU visible to TensorFlow'; print('[gpu] TF sees', g)"
          echo "[data] $(date) fetch HLS4ML LHC Jet 150p TRAIN split from Zenodo (~3 GB, PVC-free)"
          python -u -c 'import urllib.request; urllib.request.urlretrieve("https://zenodo.org/records/3602260/files/hls4ml_LHCjet_150p_train.tar.gz?download=1", "/work/data/train.tar.gz")'
          sz=$(stat -c %s /work/data/train.tar.gz); echo "[data] tarball bytes: $sz"
          [ "$sz" -gt 2500000000 ] || {{ echo "[fatal] train tarball too small ($sz < 2.5GB; real size 2,725,115,104 B measured 2026-07-07)"; exit 1; }}
          tar -xzf /work/data/train.tar.gz -C /work/data && rm -f /work/data/train.tar.gz
          DATA=$(dirname "$(find /work/data -name 'jetImage_*.h5' -print -quit)")
          [ -n "$DATA" ] || {{ echo "[fatal] no jetImage_*.h5 after extract"; exit 1; }}
          echo "[data] DATA=$DATA ($(ls "$DATA" | wc -l) files)"
          export BNHGQ2_TRAIN_DATA="$DATA"
          export BNHGQ2_OUT_ROOT=/work/outputs
          export BNHGQ2_STORE=/work/outputs/_store
          export WANDB_PROJECT=bnjettag-final
          export WANDB_RUN_NAME={leaf}
          # ---- ROUND-11 capacity ladder: the r8 arms at larger d_model, standardized inputs.
          # Fills the era-2 gap between 19,201 and 6,375,173 params so background rejection at
          # fixed signal efficiency can be read against capacity at fixed preprocessing. ----
          echo "[train] $(date) {leaf}  config={config}.json seed={seed}"
          cd "$CODE"
          python -u "$CODE/run_stage.py" train \\
            --config "$CODE/configs/{config}.json" --seed {seed} --out-dir "$OUT" \\
            2>&1 | tee "$OUT/train.log"
          echo "[done] $(date) {leaf}"
        env:
        - name: WANDB_API_KEY
          valueFrom:
            secretKeyRef:
              name: kai-wandb
              key: WANDB_API_KEY
        resources:
          # RIGHT-SIZED (decisions.md 2026-07-13): deployable-scale jobs are data-bound, not
          # FLOP-bound. 2 cpu / 8Gi mem / 16Gi eph (tarball 2.7 GB deleted post-extract +
          # extracted h5 + pip wheels). requests==limits.
          limits:
            nvidia.com/gpu: "1"
            cpu: "2"
            memory: 8Gi
            ephemeral-storage: 16Gi
          requests:
            nvidia.com/gpu: "1"
            cpu: "2"
            memory: 8Gi
            ephemeral-storage: 16Gi
        volumeMounts:
        - name: cmcode
          mountPath: /cmcode
        - name: work
          mountPath: /work
      volumes:
      - name: cmcode
        configMap:
          name: kai-bn11-code
      - name: work
        emptyDir:
          sizeLimit: 16Gi
"""


def leaf_of(config: str, seed: int) -> str:
    return f"{config}-s{seed}"


def job_of(config: str, seed: int) -> str:
    # k8s names: lowercase alnum + '-', <= 63 chars. The config names already comply.
    return f"kai-bn11-{config.removeprefix('r11-')}-s{seed}"


def emit(pairs, stage: str) -> None:
    names = []
    for config, seed in pairs:
        if not os.path.exists(os.path.join(CONFIGS, f"{config}.json")):
            sys.exit(f"[fatal] missing config {config}.json — run gen_r11_ladder.py first")
        job = job_of(config, seed)
        if len(job) > 63:
            sys.exit(f"[fatal] job name too long ({len(job)}): {job}")
        text = TEMPLATE.format(job=job, leaf=leaf_of(config, seed),
                               config=config, seed=seed)
        with open(os.path.join(HERE, f"{job}.yaml"), "w") as f:
            f.write(text)
        names.append(job)
        print(f"wrote {job}.yaml")

    sh = os.path.join(HERE, f"launch_r11_{stage}.sh")
    with open(sh, "w") as f:
        f.write(f"""#!/usr/bin/env bash
# round-11 {stage}: capacity ladder. PREREQ: ConfigMap kai-bn11-code built from code/hgq2
# INCLUDING the r11 configs (see make_code_configmap.sh; the jobs hard-fail on a stale one).
# Staged in waves of 5, matching the round-7 matrix precedent (5 concurrent Zenodo pulls).
# Usage: ./launch_r11_{stage}.sh   |   ./launch_r11_{stage}.sh delete
set -euo pipefail
CTX="nautilus"; NS="cms-ml"; HERE="$(cd "$(dirname "$0")" && pwd)"
ALL=({' '.join(names)})
if [ "${{1:-apply}}" = "delete" ]; then
  for j in "${{ALL[@]}}"; do kubectl --context "$CTX" -n "$NS" delete job "$j" --ignore-not-found; done
  exit 0
fi
wait_for_wave() {{
  local jobs=("$@")
  while true; do
    local pending=0
    for j in "${{jobs[@]}}"; do
      local done_n
      done_n=$(kubectl --context "$CTX" -n "$NS" get job "$j" \\
        -o jsonpath='{{.status.succeeded}}{{.status.failed}}' 2>/dev/null || true)
      [ -n "$done_n" ] || pending=$((pending+1))
    done
    [ "$pending" -eq 0 ] && break
    echo "  ... $pending/${{#jobs[@]}} still running"; sleep 60
  done
}}
wave=()
for j in "${{ALL[@]}}"; do
  kubectl --context "$CTX" -n "$NS" delete job "$j" --ignore-not-found >/dev/null 2>&1 || true
  kubectl --context "$CTX" -n "$NS" apply -f "$HERE/$j.yaml"
  wave+=("$j")
  if [ "${{#wave[@]}}" -ge 5 ]; then
    echo "=== wave of ${{#wave[@]}} launched; waiting ==="
    wait_for_wave "${{wave[@]}}"; wave=()
  fi
done
[ "${{#wave[@]}}" -gt 0 ] && {{ echo "=== final wave of ${{#wave[@]}} ==="; wait_for_wave "${{wave[@]}}"; }}
echo; echo "=== round-11 {stage} complete ({len(names)} jobs) ==="
kubectl --context "$CTX" -n "$NS" get jobs -l app=bnjet-r11
""")
    os.chmod(sh, 0o755)
    print(f"wrote {os.path.basename(sh)}  ({len(names)} jobs)")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "stage0"
    emit({"stage0": STAGE0, "stage1": STAGE1}[stage], stage)
