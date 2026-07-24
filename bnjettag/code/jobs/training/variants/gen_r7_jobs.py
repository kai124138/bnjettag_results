#!/usr/bin/env python3
"""
ROUND-7 (deployable-scale) QAT-stack matrix job generator (decisions.md 2026-07-13).

Adapts gen_final_jobs.py (the FINAL / `large` QAT generator — DO NOT edit that file) to the
deployable-scale round-7 restart. Two sizes x five variants x three seeds = 30 GPU training
Jobs, trained natively in HGQ2 so the trained model IS the hardware model (binary {-1,+1}
STE weights + trainable-scale static activation grids, code in bnjettag/code/hgq2/bnhgq2/
{qat,train}.py). New W&B run names r7-<size>-<variant>-s<seed> in project bnjettag-final;
job names kai-bn7f-<size>-<variant>-s<n>  ("bn7f" = round-7 QAT FINAL stack; the QKeras-path
probes were kai-bn7p/d-*).

Two sizes (arch verified locally 2026-07-14 on the QAT stack, `.venv-hgq2`):
  small  D32/H4/L2/FFN64   ->  QAT-stack params 19,201 (fp32 skeleton 19,075)  [PRIMARY]
  tiny   D16/H2/L2/FFN32   ->  QAT-stack params  5,345 (fp32 skeleton  5,219)  [FALLBACK]

Recipe (stage-0 probe: lower-LR-is-better at this scale, 0.734 @ 2e-5 vs 0.707 @ 5e-5, and
the winning probe used the long schedule): peak LR 2e-5, warmup 1 + linear decay 100, ES
patience 15, batch 256, Adam beta2=0.98, wd 0.01, clipvalue 1.0. All carried by the config's
own `train` block (configs/r7-<size>-<variant>.json) — this file only emits the Jobs.

RIGHT-SIZED k8s ask (decisions.md 2026-07-13): deployable-scale jobs are data-bound, not
FLOP-bound (n_part=10 slice on read -> resident ~0.6 GB). 1 GPU / 2 cpu / 8Gi mem / 16Gi eph,
requests==limits; GPU allowlist WIDENED and PREFERRING the abundant mid-tier (A10, L4,
RTX-2080-Ti, Tesla-V100 both form-factors) so we stop demanding A100/H100 and stop pending on
affinity. Pascal stays excluded; BAD_NODES (suncave-*, ry-gpu-08, chase-ci-07) stay excluded.

PVC-FREE (same proven pattern as the final campaign): code from ConfigMap kai-bnf-code (the
EXISTING make_code_configmap.sh packs code/hgq2, which now also carries the r7-*.json configs
+ preflight_r7.sh — so no new configmap script is needed), data from Zenodo, outputs to an
emptyDir, durability via train.py's wandb.save. Checkpoint identity = the W&B run
r7-<size>-<variant>-s<seed>; local out_dir = /work/outputs/<size>-<variant>-s<seed>
(run_stage.py --out-dir, additive).

Everything is GENERATED — never hand-edit a kai-bn7f-*.yaml (regenerate here). Writes:
  * kai-bn7f-<size>-<variant>-s<n>.yaml   (30) — ConfigMap code + Zenodo data, PVC-free
  * LOOSE ENDS (experiment-log 2026-07-15; launched by hand, NOT via launch_r7_staged.sh —
    the 30-Job matrix is untouched and stays byte-identical):
      - kai-bn7f-small-w1a4-s{4,5,6}.yaml  (3) — W1A4-small seed extension (the unstable cell,
        0.6561 +- 0.0412; extend the seed axis 3 -> 6). Same config r7-small-w1a4.json.
      - kai-bn7lr-tiny-{lr1e5,lr5e5}-s1.yaml (2) + derived configs r7-tiny-w1a8-{lr1e5,lr5e5}.json
        (tiny's own LR probe: confirm the ~0.58 binary collapse is fundamental, not recipe;
        control = the existing kai-bn7f-tiny-w1a8-s1 @ peak LR 2e-5).
  * preflight_r7.sh    (build all 12 r7 configs + hard-assert param counts + Zenodo HEAD)
                       -> written into BOTH variants/ AND code/hgq2/ (must ship in the CM)
  * launch_r7_staged.sh  (6 waves of 5 = one full precision sweep per (size, seed) wave)

Run:  python3 gen_r7_jobs.py
"""
import copy
import json
import os
import stat

HERE = os.path.dirname(os.path.abspath(__file__))

# Container image + pins: mirror .venv-hgq2 EXACTLY (py3.12 / TF 2.21 / hgq2 0.1.9). Kept
# byte-identical to gen_final_jobs.py so the r7 stack == the locally-verified stack.
IMAGE = "python:3.12"
PIP_PINS = ('"tensorflow[and-cuda]==2.21.0" "keras==3.15.0" "hgq2==0.1.9" "quantizers==1.2.2" '
            '"scikit-learn==1.9.0" "h5py==3.14.0" "wandb==0.28.0" "hls4ml==1.3.0" "numpy==2.5.0"')
DATA_URL = "https://zenodo.org/records/3602260/files/hls4ml_LHCjet_150p_train.tar.gz?download=1"
CODE_CM = "kai-bnf-code"                   # REUSE the final campaign's ConfigMap (make_code_configmap.sh is unchanged)
OUT_ROOT = "/work/outputs"                 # emptyDir; the durable copy is on W&B (wandb.save)

SIZES = ["small", "tiny"]
VARIANTS = ["fp32", "w8a8", "w1a8", "w1a6", "w1a4"]
SEEDS = [1, 2, 3]

# --- Round-7 loose ends (experiment-log 2026-07-15) -----------------------------------------
# Two follow-up sets APPENDED to the 30-Job matrix. The matrix itself is untouched (the 30
# kai-bn7f-<size>-<variant>-s{1,2,3}.yaml stay byte-identical); these are launched by hand,
# NOT wired into launch_r7_staged.sh.
#
# SET 1 — W1A4-small seed extension (instability characterization). Verified W1A4-small is
# seed-unstable: 0.6561 +- 0.0412 over 3 seeds — ~5x the spread of every other variant.
# Extend the seed axis 3 -> 6 (add 4,5,6), SAME config r7-small-w1a4.json; W&B runs
# r7-small-w1a4-s{4,5,6}, Job stems kai-bn7f-small-w1a4-s{4,5,6} (matrix naming, matrix config).
W1A4_SMALL_EXTRA_SEEDS = [4, 5, 6]
#
# SET 2 — tiny's own LR probe. tiny inherited small's peak LR 2e-5 without its own sweep; its
# binary collapse (~0.58 AUC) must be confirmed FUNDAMENTAL (bits-for-parameters), not a recipe
# artifact. Two configs derived from r7-tiny-w1a8.json (only train.lr + name changed), seed 1
# each; the existing kai-bn7f-tiny-w1a8-s1 (peak LR 2e-5) is the control. Job stems kai-bn7lr-*,
# W&B runs r7lr-tiny-w1a8-{lr1e5,lr5e5}-s1 (distinct "bn7lr"/"r7lr" prefixes flag these as probes).
# Round 2 of the probe (2026-07-15, after seed-1 verdicts): lr5e5 hit val 0.7176 vs the
# 2e-5 control's ~0.58 — the collapse was RECIPE, not fundamental. Extend: lr5e5 gets a
# full seed triple (does it replicate?) and lr1e4 probes whether even higher is better
# (small collapsed above 2e-4; tiny's optimum is clearly not small's). lr1e5 (0.5438,
# worse) stays a single dead probe.
# Round 3 (2026-07-15 evening): lr5e5 replicated (0.7176/0.7150/0.6998) and lr1e4 s1 beat it
# at 0.7309 — the optimum is still uphill. lr1e4 gets its seed triple; lr2e4 probes the top
# (small collapsed above 2e-4; find tiny's cliff).
# Round 4 (2026-07-16): ladder STILL climbing — lr1e4 triple 0.7309/0.7251/0.7358, lr2e4 s1
# 0.7570 (matches FP32-tiny, itself trained at the inherited LR). Bracket the cliff: 4e-4 and
# 8e-4, single seeds. NOTE the implication: the whole tiny matrix column (incl. baselines)
# was trained at a bad LR — once the optimum is bracketed, the 5k curve point needs a
# retrained column at tiny's own recipe, not a footnote.
TINY_LR_PROBE = [("lr1e5", 1e-05, [1]),
                 ("lr5e5", 5e-05, [1, 2, 3]),
                 ("lr1e4", 1e-04, [1, 2, 3]),
                 ("lr2e4", 2e-04, [1, 2, 3]),   # tripled 07-16: 0.7570 s1 needs replication (4e-4 s1 = 0.6873, 8e-4 s1 = 0.7322 — noisy region)
                 ("lr4e4", 4e-04, [1]),
                 ("lr8e4", 8e-04, [1])]   # (tag, peak_lr, seeds)
#
# SET 4 — tiny column RETRAIN at tiny's own recipe (decisions.md 2026-07-16). The LR ladder
# (1e-5 .. 8e-4, 12 probe runs) settled at peak LR 1e-4: 0.7306 ± 0.0054 vs 2e-4's
# 0.7376 ± 0.0216 — indistinguishable mean, 4x tighter, farther from the 4e-4 cliff (0.687).
# The matrix tiny column (INCLUDING baselines) inherited small's 2e-5, so the curve's 5k
# point is invalid as trained; retrain all five variants x 3 seeds at 1e-4 (one recipe per
# column, baselines identical — the comparability rule). Derived configs r7b-tiny-<v>.json
# (only train.lr + name changed from r7-tiny-<v>.json); Job stems kai-bn7b-*, W&B runs
# r7b-tiny-<variant>-s<seed> (prefix r7b -> roc_final eval-all --run-prefix r7b works as-is).
TINY_RECAL_LR = 1e-04
TINY_RECAL_SEEDS = [1, 2, 3]

# SET 5 — round-8 FIT experiment (research-log 2026-07-17: the reference implementation
# keeps ALL normalization out of fabric via offline per-feature standardization). Two
# standalone configs, both small W1A8 @ the small recipe: r8-small-w1a8-std (input_std
# only — isolates standardization's own effect) and r8-small-w1a8-stdnn (input_std +
# arch.norm='none' — the fit candidate; hypothesis: recovers most of the −9.3-pt raw
# ablation cost). Stats travel as input_std.json next to the checkpoint (train.py knob,
# guarded). Job stems kai-bn8-*, W&B runs r8-small-w1a8-{std,stdnn}-s{1,2,3}.
R8_FIT_SEEDS = [1, 2, 3]
R8_FIT_TAGS = ["std", "stdnn"]
# Baseline completion (2026-07-18, after std-s2 hit val 0.8719 vs the raw control's 0.75):
# any std-W1A8 number needs std-trained baselines for a fair gap — identical treatment rule.
R8_BASELINES = ["fp32", "w8a8"]

# SET 3 — norm ablation (the round-6 experiment, promoted 2026-07-15). The whole-model census
# puts SubLN at 19,584/33,696 DSP AND 2.10M/5.21M LUT (RF=8) — the single biggest consumer of
# both. arch.norm='none' (bnhgq2 build.py/qat.py, guarded) removes every norm; same params
# (19,201 — PSubLN is parameter-free), same recipe. Standalone config r7-small-w1a8-nonorm.json
# (NOT generator-derived); Job stems kai-bn7n-*, W&B runs r7n-small-w1a8-nonorm-s{1,2,3}.
NORM_ABLATION_SEEDS = [1, 2, 3]

# QAT-stack count_params, hard-asserted by preflight_r7.sh (measured 2026-07-14, .venv-hgq2).
# Quantized variants carry the weight/act quantizer bit-configs (+126 params over fp32).
PARAMS_QAT = {"small": 19201, "tiny": 5345}    # w8a8 / w1a8 / w1a6 / w1a4
PARAMS_FP32 = {"small": 19075, "tiny": 5219}   # fp32 skeleton (dummy quantizers, no bit-config vars)

NOTES = {
    "fp32": "BASELINE FP32 (float weights + acts, same skeleton) — the abstract's comparison point.",
    "w8a8": "BASELINE W8A8 (8-bit static weights + acts) — 1-bit-vs-ordinary-quantization reference.",
    "w1a8": "THESIS binary {-1,+1} weights + trainable-scale static 8-bit acts (the target model).",
    "w1a6": "QUANT AXIS binary weights + trainable-scale static 6-bit acts.",
    "w1a4": "QUANT AXIS binary weights + trainable-scale static 4-bit acts (clipvalue-clipped).",
}

# Nodes that eat pods (carried from the final campaign, all measured):
#  * k8s-chase-ci-07 : UnexpectedAdmissionError attractor (advertises free GPUs BECAUSE
#    nothing starts there).
#  * ry-gpu-08 : goes bad mid-campaign (CUDA_ERROR_NOT_INITIALIZED -> admission errors).
#  * suncave-* : UCSD CAVE display wall — 3090s show "free" but cuDNN init fails.
BAD_NODES = ["k8s-chase-ci-07.calit2.optiputer.net",
             "ry-gpu-08.sdsc.optiputer.net"] + [f"suncave-{i}" for i in range(1, 21)]

# GPU allowlist (nvidia.com/gpu.product labels). WIDENED for round-7: the deployable-scale
# jobs are tiny, so we ADD the abundant mid-tier and PREFER it (leave the A100/H100 flagships
# to people who need them — decisions.md 2026-07-13). Pascal (P100/P40/GTX-10xx/Titan-Xp) is
# NOT in the allowlist -> excluded by construction.
GPU_FLAGSHIP = [                         # allowed, but LAST-resort preference
    "NVIDIA-H200-NVL", "NVIDIA-H100-80GB-HBM3",
    "NVIDIA-RTX-PRO-6000-Blackwell-Max-Q-Workstation-Edition"]
GPU_A100 = [                             # allowed, de-preferred (leave for others)
    "NVIDIA-A100-SXM4-80GB", "NVIDIA-A100-80GB-PCIe", "NVIDIA-A100-PCIE-40GB"]
GPU_HIGH = [                             # fine, mid preference
    "NVIDIA-GeForce-RTX-4090", "NVIDIA-L40S", "NVIDIA-L40", "NVIDIA-RTX-A6000"]
GPU_ABUNDANT = [                         # the TARGET tier (top preference) — plentiful, non-flagship
    "NVIDIA-A10", "NVIDIA-L4", "NVIDIA-GeForce-RTX-2080-Ti",
    "Tesla-V100-SXM2-32GB", "Tesla-V100-PCIE-32GB",
    "NVIDIA-A40", "NVIDIA-RTX-A5000", "NVIDIA-GeForce-RTX-3090"]
GPU_PRODUCTS = GPU_FLAGSHIP + GPU_A100 + GPU_HIGH + GPU_ABUNDANT   # the required allowlist (18)


def _vals(xs):
    return "\n".join(f"                - {x}" for x in xs)


def _pref_vals(xs):
    return "[" + ", ".join(xs) + "]"


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
{_vals(GPU_PRODUCTS)}
          preferredDuringSchedulingIgnoredDuringExecution:
          # PREFER the abundant mid-tier (A10/L4/2080-Ti/V100/A40/A5000/3090) — plentiful and
          # more than enough for a <=19k-param model. De-prefer A100s, flagship = last resort.
          - weight: 100
            preference:
              matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values: {_pref_vals(GPU_ABUNDANT)}
          - weight: 60
            preference:
              matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values: {_pref_vals(GPU_HIGH)}
          - weight: 20
            preference:
              matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values: {_pref_vals(GPU_A100)}
          - weight: 10
            preference:
              matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values: {_pref_vals(GPU_FLAGSHIP)}"""

# PVC-FREE job template (@-token substitution — avoids brace-escaping the embedded shell).
JOB_TMPL = r'''apiVersion: batch/v1
kind: Job
metadata:
  name: @JOB@
  labels:
    app: bnjet-r7f
    bn7f: "@LABEL@"
spec:
  # backoffLimit 2: an UnexpectedAdmissionError (node GPU device-plugin race, zero side
  # effects) must not kill the Job. Duplicate-W&B-run risk from a mid-train retry is
  # accepted (the ROC fetcher prefers the finished run if duplicates appear).
  backoffLimit: 2
  activeDeadlineSeconds: @DEADLINE@
  template:
    metadata:
      labels:
        app: bnjet-r7f
        bn7f: "@LABEL@"
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
          # TF 2.21 pip wheel's RPATH discovery of the nvidia-*-cu12 wheel libs is broken;
          # explicit LD_LIBRARY_PATH over the wheel lib dirs fixes it (verified 2026-07-07).
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
          echo "[train] $(date) @RUN@  size=@SIZE@ config=@CONFIG@ seed=@SEED@"
          cd "$CODE"
          python -u "$CODE/run_stage.py" train \
            --config "$CODE/configs/@CONFIG@" --seed @SEED@ --out-dir "$OUT" \
            2>&1 | tee "$OUT/train.log"
          echo "[done] $(date) @RUN@"
        env:
        - name: WANDB_API_KEY
          valueFrom:
            secretKeyRef:
              name: kai-wandb
              key: WANDB_API_KEY
        resources:
          # RIGHT-SIZED (decisions.md 2026-07-13): deployable-scale jobs are data-bound, not
          # FLOP-bound. 2 cpu / 8Gi mem / 16Gi eph (tarball 2.7 GB deleted post-extract +
          # extracted h5 + pip wheels). requests==limits. The 24Gi/4cpu final ask was
          # inherited large-model sizing and pended r7 on Insufficient cpu/memory.
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
          name: @CODE_CM@
      - name: work
        emptyDir:
          sizeLimit: 16Gi
'''


def render_job(*, job, label, run, out_key, config, size, seed, note, deadline="14400"):
    """Low-level @-token substitution into JOB_TMPL. Every emitted Job routes through here.
    `job`      = k8s Job name == YAML file stem (kai-bn7f-* matrix / kai-bn7lr-* probes),
    `run`      = W&B run name (WANDB_RUN_NAME), `out_key` = emptyDir path /work/outputs/<out_key>,
    `config`   = configs/<config> filename passed to run_stage.py train --config.
    None of the substitution VALUES contain an @-token, so replacement order is irrelevant and
    the output is deterministic (the 30 matrix YAMLs stay byte-identical across regenerations)."""
    subs = {
        "@JOB@": job, "@LABEL@": label,
        "@RUN@": run, "@OUT@": f"{OUT_ROOT}/{out_key}", "@OUT_ROOT@": OUT_ROOT,
        "@AFFINITY@": NODE_AFFINITY, "@IMAGE@": IMAGE, "@PINS@": PIP_PINS,
        "@DATA_URL@": DATA_URL, "@CODE_CM@": CODE_CM,
        "@CONFIG@": config,
        "@SIZE@": size, "@SEED@": str(seed), "@NOTE@": note,
        "@DEADLINE@": deadline,
    }
    y = JOB_TMPL
    for k, v in subs.items():
        y = y.replace(k, v)
    return y


def job_yaml(size, variant, seed):
    """The canonical size x variant x seed matrix Job (kai-bn7f-<size>-<variant>-s<seed>)."""
    key = f"{size}-{variant}-s{seed}"
    return render_job(job=f"kai-bn7f-{key}", label=key, run=f"r7-{key}", out_key=key,
                      config=f"r7-{size}-{variant}.json", size=size, seed=seed,
                      note=NOTES[variant])


PREFLIGHT = r'''#!/usr/bin/env bash
# ROUND-7 preflight — PVC-FREE, CPU-only (no GPU burned). Builds all @NCONFIGS@ r7 configs from the
# SAME ConfigMap the training jobs mount and hard-gates on the exact QAT-stack param counts +
# the binary {-1,+1} gate + Zenodo data-source reachability. Run in a CPU pod on image
# @IMAGE@ with the code ConfigMap @CODE_CM@ mounted at /cmcode (apply it first with the
# EXISTING ./make_code_configmap.sh, which now also packs the r7-*.json configs + this script):
#   kubectl -n cms-ml run kai-r7f-preflight --restart=Never -it --image=@IMAGE@ \
#     --overrides='{"spec":{"containers":[{"name":"p","image":"@IMAGE@","command":["bash","-lc",
#       "tar xzf /cmcode/hgq2.tar.gz -C /tmp && bash /tmp/hgq2/preflight_r7.sh"],
#       "volumeMounts":[{"name":"cm","mountPath":"/cmcode"}]}],
#       "volumes":[{"name":"cm","configMap":{"name":"@CODE_CM@"}}]}}'
# Require the literal PREFLIGHT_ALL_PASS in the output before launching any GPU job.
set -uo pipefail
export KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES=-1
CODE=${BNF_CODE:-/work/code}
mkdir -p "$CODE"
if [ -f /cmcode/hgq2.tar.gz ]; then tar -xzf /cmcode/hgq2.tar.gz -C "$CODE" --strip-components=1; fi
fail=0
echo "=== BNJetTag ROUND-7 preflight (PVC-free) $(date) ==="

echo "[deps] pinned install (mirror .venv-hgq2)"
pip install -q --no-cache-dir @PINS@ || { echo "  FAIL deps"; fail=1; }

echo "--- data source reachability: Zenodo train tarball (HEAD, no download) ---"
if python - <<'PYEOF'
import urllib.request
req = urllib.request.Request("@DATA_URL@", method="HEAD")
r = urllib.request.urlopen(req, timeout=90)
cl = int(r.headers.get("Content-Length", "0"))
assert r.status == 200 and cl > 2_500_000_000, f"status={r.status} content-length={cl}"  # real tarball = 2,725,115,104 B
print(f"  OK Zenodo train tarball reachable: {cl} bytes")
PYEOF
then echo "  PASS data-source"; else echo "  FAIL data-source"; fail=1; fi

echo "--- build all @NCONFIGS@ r7 configs (CPU) + HARD-ASSERT param counts + binary {-1,+1} gate ---"
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
if len(configs) != @NCONFIGS@:
    print(f"  FAIL expected @NCONFIGS@ r7 configs, found {len(configs)}"); ok = False
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
'''


def preflight_sh():
    # config count the preflight gates on: the 10-cell matrix + the tiny LR-probe configs
    # + the standalone norm-ablation config (r7-small-w1a8-nonorm.json, SET 3).
    nconfigs = len(SIZES) * len(VARIANTS) + len(TINY_LR_PROBE) + 1 + len(VARIANTS) + len(R8_FIT_TAGS) + len(R8_BASELINES)  # + r7b + r8
    return (PREFLIGHT.replace("@IMAGE@", IMAGE).replace("@CODE_CM@", CODE_CM)
            .replace("@DATA_URL@", DATA_URL).replace("@PINS@", PIP_PINS)
            .replace("@NCONFIGS@", str(nconfigs)))


def launch_sh(waves, key2job):
    ls = ["#!/usr/bin/env bash",
          "# Launch the ROUND-7 QAT-stack matrix on NRP, STAGED (one full precision sweep per wave).",
          "# Usage:  ./launch_r7_staged.sh            # wave-by-wave (needs laptop alive)",
          "#         ./launch_r7_staged.sh delete     # tear everything down",
          "# PVC-FREE. PRE-REQS (in order):",
          "#   1. ./make_code_configmap.sh    # code source ConfigMap kai-bnf-code (now packs r7 configs + preflight_r7.sh)",
          "#   2. preflight_r7.sh -> PREFLIGHT_ALL_PASS   (CPU pod with the ConfigMap mounted)",
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
          '      sleep 60',
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
    ls.append('echo "=== ROUND-7 matrix complete ==="')
    ls.append('kubectl --context "$CTX" -n "$NS" get jobs -l app=bnjet-r7f')
    return "\n".join(ls) + "\n"


def main():
    written = []
    key2job = {}
    for size in SIZES:
        for v in VARIANTS:
            for s in SEEDS:
                key = f"{size}-{v}-s{s}"
                path = os.path.join(HERE, f"kai-bn7f-{key}.yaml")
                with open(path, "w") as f:
                    f.write(job_yaml(size, v, s))
                written.append(path)
                key2job[key] = f"kai-bn7f-{key}"

    # --- Round-7 loose ends (experiment-log 2026-07-15) — APPENDED, matrix untouched ----------
    # These are NOT added to key2job/waves, so launch_r7_staged.sh stays byte-identical. They are
    # launched by hand (report gives the exact kubectl apply commands).
    configs_dir = os.path.abspath(os.path.join(HERE, "..", "..", "..", "hgq2", "configs"))
    extra_jobs = []                              # (job_name, run_name) — for the summary/report

    # SET 1: W1A4-small seeds 4-6 (extend the unstable cell's seed axis; matrix config + naming).
    for s in W1A4_SMALL_EXTRA_SEEDS:
        key = f"small-w1a4-s{s}"
        path = os.path.join(HERE, f"kai-bn7f-{key}.yaml")
        with open(path, "w") as f:
            f.write(render_job(job=f"kai-bn7f-{key}", label=key, run=f"r7-{key}", out_key=key,
                               config="r7-small-w1a4.json", size="small", seed=s,
                               note="SEED-EXTENSION (W1A4-small instability, s4-6). " + NOTES["w1a4"]))
        written.append(path)
        extra_jobs.append((f"kai-bn7f-{key}", f"r7-{key}"))

    # SET 2: tiny LR probe — derive the two LR-variant configs from r7-tiny-w1a8.json (only
    # train.lr + name changed) and emit one seed-1 Job each. The configs SHIP in the ConfigMap,
    # so make_code_configmap.sh MUST be re-run before launch (flagged in the summary below).
    with open(os.path.join(configs_dir, "r7-tiny-w1a8.json")) as f:
        base_cfg = json.load(f)
    for tag, lr, probe_seeds in TINY_LR_PROBE:
        cfg = copy.deepcopy(base_cfg)
        cfg["name"] = f"r7-tiny-w1a8-{tag}"
        cfg["train"]["lr"] = lr
        cfg_name = f"r7-tiny-w1a8-{tag}.json"
        cfg_path = os.path.join(configs_dir, cfg_name)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")
        written.append(cfg_path)
        for s in probe_seeds:
            key = f"tiny-{tag}-s{s}"              # out_key + label; Job stem kai-bn7lr-<key>
            job = f"kai-bn7lr-{key}"
            run = f"r7lr-tiny-w1a8-{tag}-s{s}"
            path = os.path.join(HERE, f"{job}.yaml")
            with open(path, "w") as f:
                f.write(render_job(job=job, label=key, run=run, out_key=key, config=cfg_name,
                                   size="tiny", seed=s,
                                   note=f"LR PROBE peak_lr={lr:g} (tiny inherited small's 2e-5 with "
                                        f"no probe; round 2: lr5e5 s1 hit 0.7176 val — collapse was "
                                        f"recipe, not fundamental). " + NOTES["w1a8"]))
            written.append(path)
            extra_jobs.append((job, run))

    # SET 3: norm ablation — 3 seeds of the standalone r7-small-w1a8-nonorm.json (authored with
    # the bnhgq2 arch.norm='none' option, NOT derived here; assert it exists so a stale checkout
    # fails loudly instead of emitting Jobs for a config the ConfigMap won't contain).
    nonorm_cfg = os.path.join(configs_dir, "r7-small-w1a8-nonorm.json")
    if not os.path.isfile(nonorm_cfg):
        raise SystemExit(f"SET 3 config missing: {nonorm_cfg}")
    for s in NORM_ABLATION_SEEDS:
        key = f"small-nonorm-s{s}"
        job = f"kai-bn7n-{key}"
        run = f"r7n-small-w1a8-nonorm-s{s}"
        path = os.path.join(HERE, f"{job}.yaml")
        with open(path, "w") as f:
            f.write(render_job(job=job, label=key, run=run, out_key=key,
                               config="r7-small-w1a8-nonorm.json", size="small", seed=s,
                               note="NORM ABLATION (arch.norm='none'; SubLN = top DSP+LUT "
                                    "consumer in the whole-model census). " + NOTES["w1a8"]))
        written.append(path)
        extra_jobs.append((job, run))

    # SET 4: tiny column retrain at 1e-4 — derive r7b-tiny-<variant>.json from each matrix
    # tiny config (only name + train.lr change) and emit 15 Jobs (5 variants x 3 seeds).
    for v in VARIANTS:
        with open(os.path.join(configs_dir, f"r7-tiny-{v}.json")) as f:
            tcfg = json.load(f)
        tcfg = copy.deepcopy(tcfg)
        tcfg["name"] = f"r7b-tiny-{v}"
        tcfg["train"]["lr"] = TINY_RECAL_LR
        cfg_name = f"r7b-tiny-{v}.json"
        with open(os.path.join(configs_dir, cfg_name), "w") as f:
            json.dump(tcfg, f, indent=2)
            f.write("\n")
        written.append(os.path.join(configs_dir, cfg_name))
        for s in TINY_RECAL_SEEDS:
            key = f"tiny-{v}-r7b-s{s}"             # out_key + label; Job stem kai-bn7b-<...>
            job = f"kai-bn7b-tiny-{v}-s{s}"
            run = f"r7b-tiny-{v}-s{s}"
            path = os.path.join(HERE, f"{job}.yaml")
            with open(path, "w") as f:
                f.write(render_job(job=job, label=key, run=run, out_key=key, config=cfg_name,
                                   size="tiny", seed=s,
                                   note=f"TINY COLUMN RETRAIN @ peak_lr={TINY_RECAL_LR:g} "
                                        f"(ladder verdict 2026-07-16; the matrix tiny column "
                                        f"inherited small's 2e-5 and is invalid as trained). "
                                        + NOTES[v]))
            written.append(path)
            extra_jobs.append((job, run))

    # SET 5: round-8 fit experiment — standalone configs (authored, not derived); assert
    # they exist so a stale checkout fails loudly.
    for tag in R8_FIT_TAGS:
        cfgf = os.path.join(configs_dir, f"r8-small-w1a8-{tag}.json")
        if not os.path.isfile(cfgf):
            raise SystemExit(f"SET 5 config missing: {cfgf}")
        for sd in R8_FIT_SEEDS:
            key = f"small-{tag}-s{sd}"
            job = f"kai-bn8-{key}"
            run = f"r8-small-w1a8-{tag}-s{sd}"
            path = os.path.join(HERE, f"{job}.yaml")
            with open(path, "w") as f:
                f.write(render_job(job=job, label=key, run=run, out_key=key,
                                   config=f"r8-small-w1a8-{tag}.json", size="small", seed=sd,
                                   note=("ROUND-8 FIT: offline per-feature standardization"
                                         + (" + arch.norm='none' (the fit candidate)" if tag == "stdnn"
                                            else " (norms kept — isolates the std effect)")
                                         + ". " + NOTES["w1a8"])))
            written.append(path)
            extra_jobs.append((job, run))

    # SET 5b: round-8 std baselines (fp32/w8a8 with input_std, norms kept).
    for v in R8_BASELINES:
        cfgf = os.path.join(configs_dir, f"r8-small-{v}-std.json")
        if not os.path.isfile(cfgf):
            raise SystemExit(f"SET 5b config missing: {cfgf}")
        for sd in R8_FIT_SEEDS:
            key = f"small-{v}std-s{sd}"
            job = f"kai-bn8-{key}"
            run = f"r8-small-{v}-std-s{sd}"
            path = os.path.join(HERE, f"{job}.yaml")
            with open(path, "w") as f:
                f.write(render_job(job=job, label=key, run=run, out_key=key,
                                   config=f"r8-small-{v}-std.json", size="small", seed=sd,
                                   note="ROUND-8 STD BASELINE (identical treatment for the "
                                        "fair std-vs-std gap). " + NOTES[v]))
            written.append(path)
            extra_jobs.append((job, run))

    # Staged waves: one full precision sweep (5 variants) per (size, seed) — each wave is a
    # scientifically complete slice, and 5 tiny right-sized jobs schedule comfortably together.
    waves = [[f"{size}-{v}-s{s}" for v in VARIANTS] for size in SIZES for s in SEEDS]

    # preflight_r7.sh must ALSO live inside code/hgq2/ so it ships in the ConfigMap and can be
    # run as /work/code/preflight_r7.sh inside a pod (PVC-free). Mirrors gen_final_jobs.py.
    hgq2_dir = os.path.abspath(os.path.join(HERE, "..", "..", "..", "hgq2"))
    for d in (HERE, hgq2_dir):
        p = os.path.join(d, "preflight_r7.sh")
        with open(p, "w") as f:
            f.write(preflight_sh())
        os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR)
        written.append(p)

    p = os.path.join(HERE, "launch_r7_staged.sh")
    with open(p, "w") as f:
        f.write(launch_sh(waves, {k: key2job[k] for k in sum(waves, [])}))
    os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR)
    written.append(p)

    for p in written:
        print("wrote", p)
    n_matrix = len(SIZES) * len(VARIANTS) * len(SEEDS)
    print(f"\n{n_matrix} matrix YAMLs + {len(extra_jobs)} loose-end YAMLs "
          f"+ {len(TINY_LR_PROBE)} derived configs + preflight (x2) + launch written.")
    print("loose ends (launch BY HAND — NOT in launch_r7_staged.sh):")
    for job, run in extra_jobs:
        print(f"  {job}.yaml   (W&B run {run})")
    print("configmap: RE-RUN ./make_code_configmap.sh  ->  ConfigMap kai-bnf-code")
    print("           (the new r7-tiny-w1a8-{lr1e5,lr5e5}.json MUST ship before launching the probes)")


if __name__ == "__main__":
    main()
