#!/usr/bin/env python3
"""
Round-7 — the DEPLOYABLE-SCALE restart.

Why this round exists (2026-07-13). Every era-2 number we have is on `large`
(D256/H8/L8/FFN1024 = 6,375,173 params -> 63.4M MACs). That model cannot be an L1
trigger tagger and cannot be synthesized as one hls4ml project: the monolith attempt
died weight-INLINING-dominated (~101 MB of inlined ±1 constant arrays, 27M Vitis IR).
Every workaround we built since — block-by-block probes, the BRAM-ROM hybrid strategy —
exists only to route around a model that was too big to begin with.

Round-7 fixes the cause instead of the symptom: retrain at deployable scale, then run the
ordinary FULL continuous hls4ml conversion (no block splitting, no hybrid ROM trick).
At 18k params the inlined-weight blob drops ~350x, to a few hundred kB.

Two sizes, chosen against the literature (research-log 2026-07-13). Every published
param count for this problem class lands in 3,347-33,625:
  small  D32/H4/L2/FFN64  ->  18,405 params, 182k MACs   [PRIMARY / headline]
         mid-range; nearest synthesized precedent is the Odagiu et al. MLP
         (20k-27k params, 95-105 ns, 2.1% DSP, 7-9% LUT on a VU13P, arXiv:2402.01876)
  tiny   D16/H2/L2/FFN32  ->   4,853 params,  50k MACs   [FALLBACK + size axis]
         sits in the Deep Sets / Interaction-Network band (3,347-7,400), which has
         multiple fully resource-tabulated FPGA precedents.
hls4ml's own FAQ says it has "successfully used ... O(10k) parameters" and warns that
ATTENTION is harder than dense/conv at equal param count — so `small` is defensible but
not conservative, and `tiny` is the pre-trained landing zone if it does not fit.

STAGE 0 (probe) — the LR is not inherited blindly.
r5's peak LR 5e-5 was tuned for the 6.4M `large` model. Carrying it to a model 346x
smaller risks the exact tuned-vs-untuned trap round-4 already caught: the small model
underfits and we wrongly conclude "deployable scale cannot tag". So stage 0 sweeps the
peak LR on the thesis config (binary W1A8, small) with EVERYTHING else at the r5 recipe.
The schedule is the real one (warmup 1 + linear decay over 40, ES patience 10 on val_auc),
not a truncated one, so the winning probe run IS a valid final W1A8 model - no rerun.

STAGE 1 (matrix) — the remaining 9 runs at the winning LR:
  small: W1A6, W1A4, FP32, W8A8      (W1A8 already produced by the probe winner)
  tiny : W1A8, W1A6, W1A4, FP32, W8A8
Caveat logged deliberately: `tiny` adopts `small`'s winning LR rather than getting its
own probe. Flag if tiny's val_auc looks unstable.

PVC-FREE (pattern proven by kai-roc-r5-pvcfree, 2026-07-02; PVC mountability still not
trusted after the rook-cephfs outage):
  * train data : hls4ml_LHCjet_150p_train.tar.gz, Zenodo record 3602260 (2.7 GB)
  * code       : ConfigMap kai-qkerasmodel-r5 (md5 4d0a3fed... = verified == local code)
  * checkpoint : wandb.save from inside qkerasModel.py -> W&B runs r7*
  * no volume  : emptyDir only

Run:
  python3 gen_round7_jobs.py probe                # 4 probe yamls + launch_r7_probe.sh
  python3 gen_round7_jobs.py matrix --lr 0.0002   # 9 matrix yamls + launch_r7_matrix.sh
  (both stages always (re)write preflight_r7.sh)
"""
import argparse
import os
import stat

HERE = os.path.dirname(os.path.abspath(__file__))

# ── architectures (param counts verified locally 2026-07-13 by building the real model;
#    invariant across BN_VARIANT and BN_ACT_BITS — same shapes, different quantization) ──
SIZES = {
    "small": dict(D=32, H=4, L=2, FFN=64, params=18405),
    "tiny":  dict(D=16, H=2, L=2, FFN=32, params=4853),
}

# ── the r5 recipe, verbatim, MINUS the peak LR (which stage 0 determines) ───────────────
BASE_RECIPE = dict(
    BN_WARMUP_EPOCHS="1", BN_DECAY_EPOCHS="40", BN_DECAY_POWER="1.0",
    BN_BETA2="0.98", BN_WEIGHT_DECAY="0.01", BN_CLIPNORM="1.0",
    BN_L1_REG="0", BN_BATCH="256",
    BN_ES_MONITOR="val_auc", BN_ES_MODE="max", BN_ES_PATIENCE="10",
)

# ── stage 0: peak-LR probe on the thesis config (binary W1A8, small) ───────────────────
#    5e-5 is the inherited r5 value and is included as the control point.
PROBE_LRS = [
    ("lr05",  "0.00005", "control: the inherited r5 peak LR, tuned for 6.4M `large`"),
    ("lr20",  "0.0002",  "4x the r5 LR"),
    ("lr50",  "0.0005",  "10x the r5 LR"),
    ("lr100", "0.001",   "20x the r5 LR (~the BitNet paper's peak)"),
]

# ── stage 1: the variant matrix ────────────────────────────────────────────────────────
VARIANTS = [
    ("a8",   {"BN_VARIANT": "bitnet",  "BN_ACT_BITS": "8"}, "THESIS: binary {-1,+1} W1A8"),
    ("a6",   {"BN_VARIANT": "bitnet",  "BN_ACT_BITS": "6"}, "QUANT AXIS: binary W1A6"),
    ("a4",   {"BN_VARIANT": "bitnet",  "BN_ACT_BITS": "4"}, "QUANT AXIS: binary W1A4"),
    ("fp32", {"BN_VARIANT": "vanilla", "BN_ACT_BITS": "8"}, "BASELINE: FP32 vanilla"),
    ("w8a8", {"BN_VARIANT": "w8a8",    "BN_ACT_BITS": "8"}, "BASELINE: W8A8"),
]

# Deployable-scale jobs are DATA-bound, not FLOP-bound: the model is 18k/4.9k params and
# an epoch is dominated by streaming the jet dataset, not by matmul. So round-7 deliberately
# does NOT demand A100/H100-class cards the way the 6.4M `large` rounds did — that allowlist
# was rejecting 202/524 nodes and leaving these jobs Pending behind big-model queues.
# We now accept (and PREFER) the abundant mid-tier cards: A10 / L4 / RTX-2080-Ti / V100 /
# A4000. All are fine for TF 2.11. Pascal (GTX-1080/1080-Ti, TITAN-Xp) stays excluded —
# slow, and not worth the cuDNN risk when there is plenty of Turing+ capacity.
# Leaving the A100/H100s to the people who actually need them is also good cluster manners.
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
                - NVIDIA-A10
                - NVIDIA-L4
                - NVIDIA-GeForce-RTX-2080-Ti
                - Tesla-V100-SXM2-32GB
                - Tesla-V100-SXM2-16GB
                - Tesla-V100-PCIE-16GB
                - NVIDIA-RTX-A4000
                - NVIDIA-TITAN-RTX
                - Quadro-RTX-6000
                - NVIDIA-RTX-5000-Ada-Generation
                - NVIDIA-GeForce-RTX-3090
                - NVIDIA-RTX-A5000
                - NVIDIA-A40
                - NVIDIA-RTX-A6000
                - NVIDIA-L40
                - NVIDIA-L40S
                - NVIDIA-GeForce-RTX-4090
                - NVIDIA-A100-SXM4-80GB
                - NVIDIA-A100-80GB-PCIe
                - NVIDIA-A100-PCIE-40GB
          preferredDuringSchedulingIgnoredDuringExecution:
          # prefer the plentiful mid-tier first — these jobs do not need more.
          - weight: 100
            preference:
              matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values: [NVIDIA-A10, NVIDIA-L4, NVIDIA-GeForce-RTX-2080-Ti, NVIDIA-RTX-A4000]
          - weight: 80
            preference:
              matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values: [Tesla-V100-SXM2-32GB, Tesla-V100-SXM2-16GB, Tesla-V100-PCIE-16GB, NVIDIA-GeForce-RTX-3090]
          - weight: 40
            preference:
              matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values: [NVIDIA-RTX-A5000, NVIDIA-A40, NVIDIA-RTX-A6000, NVIDIA-L40, NVIDIA-TITAN-RTX]"""

JOB_TMPL = """apiVersion: batch/v1
kind: Job
metadata:
  name: {job}
  labels:
    app: {app}
    bnv: "{key}"
spec:
  backoffLimit: 0
  activeDeadlineSeconds: 86400
  template:
    metadata:
      labels:
        app: {app}
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
          # deployable scale `{size}` -> {params} params (round-7, 2026-07-13)
          export BN_D_MODEL={D} BN_N_HEADS={H} BN_N_LAYERS={L} BN_FFN_DIM={FFN}
          export BN_N_PART=10
{recipe_exports}{env_exports}
          echo "[preflight] $(date) --sanity build + param-count gate"
          env CUDA_VISIBLE_DEVICES=-1 python -u /work/code/qkerasModel.py --sanity 2>&1 | tail -5
          env CUDA_VISIBLE_DEVICES=-1 python -u - <<'PYGATE'
          import qkerasModel as M
          n = M.build_bitnet_jet_tagger().count_params()
          print(f"[params] built={{n}} expected={params}")
          assert n == {params}, f"PARAM GATE FAILED: {{n}} != {params}"
          print("[params] OK")
          PYGATE
          echo "[train] $(date) {run}  size={size} D={D} H={H} L={L} FFN={FFN} params={params}  {knobdesc}"
          cd /work/out
          python -u /work/code/qkerasModel.py "$DATA" 2>&1 | tee /work/out/train_{key}.log
          echo "[done] $(date) {run}"
        env:
        - name: WANDB_API_KEY
          valueFrom:
            secretKeyRef:
              name: kai-wandb
              key: WANDB_API_KEY
        # Right-sized for deployable scale. The `large` rounds asked for 24Gi/4cpu; that was
        # blocking us on 55 more nodes for no reason. load_hls4ml_jets slices to n_part=10 as
        # it reads (qkerasModel.py:783), so the resident dataset is (~880k,10,16) f32 ~= 0.6 GB;
        # peak with the concatenate + one file's full 150-particle array is ~2 GB. 8Gi is 4x
        # headroom. ephemeral-storage stays 20/24Gi: the 2.7 GB tarball + extract + pip need it,
        # and it was never a binding constraint.
        resources:
          limits:
            nvidia.com/gpu: "1"
            memory: 8Gi
            cpu: "2"
            ephemeral-storage: 24Gi
          requests:
            nvidia.com/gpu: "1"
            memory: 8Gi
            cpu: "2"
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


# ── stage 0b: the DIAGNOSTIC wave (2026-07-13, after reading the stage-0 curves) ────────
# What stage 0 actually showed at `small`, binary W1A8:
#   peak LR 5e-5 -> best val_auc 0.7666 | 2e-4 -> 0.7461 | 5e-4 -> 0.7256 | 1e-3 -> 0.7041
#   i.e. MONOTONIC — lower LR is better, and 5e-5 (the r5 control) wins. High LR collapses
#   the binary model in 2-3 epochs (straight-through estimator instability).
# But the lr05 curve says the interesting thing: train_auc goes FLAT at ~0.72 from epoch 18
# (while lr was still 3e-5, so it did NOT run out of LR), train_acc plateaus at ~0.42 on a
# 5-class problem, and val_auc oscillates +-0.05 epoch-to-epoch (0.656 ... 0.766). So:
#   * train ~= val  -> not overfitting, not schedule-starved: it PLATEAUED.
#   * ES-on-max-val_auc is selecting a lucky draw out of that noise; 0.7666 is optimistic.
#   * +-0.05 swings at 18k binary weights smell like weight-flip instability that averages
#     out at 6.4M weights but not here.
# Two hypotheses, and guessing between them would be malpractice:
#   H1 capacity  — 18k params simply cannot do better on this task.
#   H2 binarize  — binary weights are what breaks at this scale, not the size.
# FP32-vs-W1A8 AT THE SAME SIZE settles it in one run — and it is the number the thesis is
# actually about (the binary-vs-FP32 GAP at fixed scale, not absolute AUC).
DIAG = [
    # decisive: same size, same recipe, full precision. H1 vs H2.
    dict(key="diag-small-fp32", job="kai-bn7d-small-fp32", run="r7d-small-fp32-lr05",
         size="small", lr="0.00005", env={"BN_VARIANT": "vanilla", "BN_ACT_BITS": "8"},
         over={}, note="DECISIVE: FP32 @ small, identical recipe. If this also plateaus ~0.72 -> H1 "
                       "(scale, not binarization). If it reaches ~0.85 -> H2 (binarization at small scale)."),
    dict(key="diag-small-w8a8", job="kai-bn7d-small-w8a8", run="r7d-small-w8a8-lr05",
         size="small", lr="0.00005", env={"BN_VARIANT": "w8a8", "BN_ACT_BITS": "8"},
         over={}, note="BASELINE: W8A8 @ small, identical recipe — the second rung of the quant ladder."),
    # does the binary model just need a LONGER schedule? (isolates schedule from LR)
    dict(key="diag-small-a8-long", job="kai-bn7d-small-a8-long", run="r7d-small-a8-lr05-long",
         size="small", lr="0.00005", env={"BN_VARIANT": "bitnet", "BN_ACT_BITS": "8"},
         over={"BN_DECAY_EPOCHS": "120", "BN_ES_PATIENCE": "25", "BN_WARMUP_EPOCHS": "3"},
         note="SCHEDULE TEST: same 5e-5 peak, decay stretched 40 -> 120. vs kai-bn7p-lr05 this "
              "isolates whether the 40-epoch decay (tuned for the 6.4M model) was the binding constraint."),
    # the stage-0 optimum was at the EDGE of the sweep — extend LR downward, with room to converge
    dict(key="diag-small-a8-lr02", job="kai-bn7d-small-a8-lr02", run="r7d-small-a8-lr02-long",
         size="small", lr="0.00002", env={"BN_VARIANT": "bitnet", "BN_ACT_BITS": "8"},
         over={"BN_DECAY_EPOCHS": "120", "BN_ES_PATIENCE": "25", "BN_WARMUP_EPOCHS": "3"},
         note="BRACKET THE OPTIMUM: stage 0's winner (5e-5) sat at the LOW edge of the sweep, so we "
              "found a boundary, not a peak. 2e-5 + long schedule probes below it."),
]


def exports_block(d):
    if not d:
        return ""
    return "          export " + " ".join(f"{k}={v}" for k, v in d.items()) + "\n"


def render(job, key, run, note, size, lr, env, app, over=None):
    a = SIZES[size]
    recipe = dict(BASE_RECIPE)
    recipe["BN_LR"] = lr
    if over:                       # schedule overrides (decay / patience / warmup / epochs)
        recipe.update(over)
    full_env = dict(env)
    full_env.setdefault("BN_SEED", "1")
    return JOB_TMPL.format(
        job=job, key=key, run=run, note=note, app=app, size=size,
        D=a["D"], H=a["H"], L=a["L"], FFN=a["FFN"], params=a["params"],
        affinity=NODE_AFFINITY,
        recipe_exports=exports_block(recipe),
        env_exports=exports_block(full_env),
        knobdesc=",".join(f"{k}={v}" for k, v in full_env.items()) + f",BN_LR={lr}",
    )


LAUNCHER_HEAD = """#!/usr/bin/env bash
# {title}
# Usage:  ./{fname}            # apply all (cluster scheduler self-limits; survives laptop close)
#         ./{fname} delete     # tear down
set -euo pipefail
CTX="nautilus"; NS="cms-ml"
HERE="$(cd "$(dirname "$0")" && pwd)"

ALL=({jobs})
if [ "${{1:-apply}}" = "delete" ]; then
  for j in "${{ALL[@]}}"; do kubectl --context "$CTX" -n "$NS" delete job "$j" --ignore-not-found; done
  exit 0
fi

for j in "${{ALL[@]}}"; do
  kubectl --context "$CTX" -n "$NS" delete job "$j" --ignore-not-found >/dev/null 2>&1 || true
  kubectl --context "$CTX" -n "$NS" apply -f "$HERE/$j.yaml"
done

echo
echo "=== {title} launched ==="
kubectl --context "$CTX" -n "$NS" get jobs -l app={app}
"""


def write_launcher(fname, title, app, jobs):
    path = os.path.join(HERE, fname)
    with open(path, "w") as f:
        f.write(LAUNCHER_HEAD.format(title=title, fname=fname, app=app, jobs=" ".join(jobs)))
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
    print(f"wrote {path}")


# ── preflight: CPU-only pod, builds EVERY round-7 config and gates on the param count ───
PREFLIGHT = """#!/usr/bin/env bash
# Round-7 preflight — CPU-only, no GPU burned. Builds every round-7 config from the
# SAME ConfigMap the training jobs mount, and gates on the exact param counts.
# Requires the literal PREFLIGHT_ALL_PASS in the output before any GPU job is launched.
set -euo pipefail
CTX="nautilus"; NS="cms-ml"; POD="kai-r7-preflight"

kubectl --context "$CTX" -n "$NS" delete pod "$POD" --ignore-not-found >/dev/null 2>&1 || true
cat <<'YAML' | kubectl --context "$CTX" -n "$NS" apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: kai-r7-preflight
  labels: {app: bnjet-r7-preflight}
spec:
  restartPolicy: Never
  containers:
  - name: pf
    image: tensorflow/tensorflow:2.11.1-gpu
    command: ["bash","-c"]
    args:
    - |
      set -eo pipefail
      mkdir -p /work/code && cp /qkcode/qkerasModel.py /work/code/
      grep -q "from __future__ import annotations" /work/code/qkerasModel.py || \\
        sed -i '1i from __future__ import annotations' /work/code/qkerasModel.py
      export PYTHONPATH=/work/code CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3
      export MPLBACKEND=Agg
      # qkerasModel.py imports matplotlib at module scope; qkeras pulls in tensorflow_model_optimization
      pip install -q qkeras==0.9.0 "tensorflow==2.11.1" "matplotlib<3.8" scikit-learn h5py
      python -u - <<'PY'
      import importlib, os, sys
      SIZES = {"small": (32,4,2,64,18405), "tiny": (16,2,2,32,4853)}
      VARIANTS = [("bitnet","8"),("bitnet","6"),("bitnet","4"),("vanilla","8"),("w8a8","8")]
      fails = []
      for size,(D,H,L,FFN,expect) in SIZES.items():
          for var,bits in VARIANTS:
              os.environ.update(BN_D_MODEL=str(D), BN_N_HEADS=str(H), BN_N_LAYERS=str(L),
                                BN_FFN_DIM=str(FFN), BN_N_PART="10",
                                BN_VARIANT=var, BN_ACT_BITS=bits)
              for m in [m for m in list(sys.modules) if m.startswith("qkerasModel")]:
                  del sys.modules[m]
              M = importlib.import_module("qkerasModel")
              n = M.build_bitnet_jet_tagger().count_params()
              ok = (n == expect)
              print(f"[cfg] {size:<6} {var:<7} A{bits}  params={n:>7,}  expect={expect:>7,}  {'OK' if ok else 'FAIL'}")
              if not ok:
                  fails.append((size,var,bits,n,expect))
      if fails:
          print("PREFLIGHT_FAILED", fails); sys.exit(1)
      print("PREFLIGHT_ALL_PASS")
      PY
    volumeMounts:
    - {name: qkcode, mountPath: /qkcode}
    - {name: work,   mountPath: /work}
    resources:
      limits:   {memory: 8Gi, cpu: "2"}
      requests: {memory: 8Gi, cpu: "2"}
  volumes:
  - name: qkcode
    configMap: {name: kai-qkerasmodel-r5}
  - name: work
    emptyDir: {sizeLimit: 4Gi}
YAML

echo "[preflight] waiting for pod to finish (CPU-only, ~3-5 min incl. pip)..."
kubectl --context "$CTX" -n "$NS" wait --for=condition=Ready pod/"$POD" --timeout=300s >/dev/null 2>&1 || true
kubectl --context "$CTX" -n "$NS" logs -f "$POD" 2>/dev/null | tee /tmp/r7_preflight.log || true

echo
if grep -q PREFLIGHT_ALL_PASS /tmp/r7_preflight.log; then
  echo "=== PREFLIGHT_ALL_PASS — cleared to launch round-7 ==="
  kubectl --context "$CTX" -n "$NS" delete pod "$POD" --ignore-not-found >/dev/null 2>&1 || true
  exit 0
fi
echo "=== PREFLIGHT FAILED — do NOT launch. Pod kept for inspection: $POD ==="
exit 1
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["probe", "diag", "matrix"])
    ap.add_argument("--lr", help="winning peak LR from stage 0 (required for `matrix`)")
    args = ap.parse_args()

    # preflight is stage-independent; always refresh it
    pf = os.path.join(HERE, "preflight_r7.sh")
    with open(pf, "w") as f:
        f.write(PREFLIGHT)
    os.chmod(pf, os.stat(pf).st_mode | stat.S_IXUSR)
    print(f"wrote {pf}")

    if args.stage == "probe":
        jobs = []
        for tag, lr, why in PROBE_LRS:
            job = f"kai-bn7p-{tag}"
            y = render(job=job, key=f"probe-{tag}", run=f"r7p-small-a8-{tag}",
                       note=f"LR PROBE ({why}) — binary W1A8 @ small, r5 recipe otherwise.",
                       size="small", lr=lr,
                       env={"BN_VARIANT": "bitnet", "BN_ACT_BITS": "8"},
                       app="bnjet-r7-probe")
            with open(os.path.join(HERE, f"{job}.yaml"), "w") as f:
                f.write(y)
            print(f"wrote {job}.yaml   (peak LR {lr} — {why})")
            jobs.append(job)
        write_launcher("launch_r7_probe.sh",
                       "round-7 stage-0: peak-LR probe (binary W1A8 @ small)",
                       "bnjet-r7-probe", jobs)
        print(f"\n{len(jobs)} probe jobs. Run preflight_r7.sh, then launch_r7_probe.sh.")
        return

    if args.stage == "diag":
        jobs = []
        for d in DIAG:
            y = render(job=d["job"], key=d["key"], run=d["run"], note=d["note"],
                       size=d["size"], lr=d["lr"], env=d["env"], over=d["over"],
                       app="bnjet-r7-diag")
            with open(os.path.join(HERE, f"{d['job']}.yaml"), "w") as f:
                f.write(y)
            sched = d["over"].get("BN_DECAY_EPOCHS", BASE_RECIPE["BN_DECAY_EPOCHS"])
            print(f"wrote {d['job']}.yaml   (LR {d['lr']}, decay {sched})")
            jobs.append(d["job"])
        write_launcher("launch_r7_diag.sh",
                       "round-7 stage-0b: diagnostic wave (FP32-vs-binary @ small, schedule, LR floor)",
                       "bnjet-r7-diag", jobs)
        print(f"\n{len(jobs)} diagnostic jobs.")
        return

    if not args.lr:
        ap.error("`matrix` needs --lr (the peak LR that won stage 0)")

    jobs = []
    for size in ("small", "tiny"):
        for vtag, env, note in VARIANTS:
            # the probe winner already IS small/a8 — don't retrain it
            if size == "small" and vtag == "a8":
                print("skip  kai-bn7-small-a8-s1  (produced by the stage-0 probe winner)")
                continue
            job = f"kai-bn7-{size}-{vtag}-s1"
            y = render(job=job, key=f"{size}-{vtag}-s1", run=f"r7-{size}-{vtag}-s1",
                       note=f"{note} @ deployable scale `{size}` ({SIZES[size]['params']:,} params).",
                       size=size, lr=args.lr, env=dict(env), app="bnjet-r7")
            with open(os.path.join(HERE, f"{job}.yaml"), "w") as f:
                f.write(y)
            print(f"wrote {job}.yaml")
            jobs.append(job)
    write_launcher("launch_r7_matrix.sh",
                   f"round-7 stage-1: variant matrix @ peak LR {args.lr}",
                   "bnjet-r7", jobs)
    print(f"\n{len(jobs)} matrix jobs at peak LR {args.lr}.")


if __name__ == "__main__":
    main()
