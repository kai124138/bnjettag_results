#!/usr/bin/env python3
"""
Round-8 — the INPUT-STANDARDIZATION 2x2. The pivotal experiment.

## The bug (measured 2026-07-13 from data/val/jetImage_*.h5, top-10-by-pT, 16 features)

`qkerasModel.py` used to claim "the dataset ships already preprocessed". It does not.
Raw per-feature std spans **0.0394 .. 117.49 — a 2,979x ratio**: GeV-scale momenta
(px/py/pz/e/pt, std ~78-117) sit next to the O(1) relative/substructure features
(ptrel/etarel/phirel/deltaR/costhetarel, std ~0.04-0.075).

The model's first op is BitLinear(input_proj) with norm_inside=True, i.e. LayerNorm ACROSS
the 16 features of a particle. Its mean/std are dominated by the momenta, so the post-LN
per-feature std lands at **1.40 for px/py/pz vs ~0.185 for every relative feature — 8.9x**.

Why that is fatal for BINARY specifically, and only mildly annoying for FP32:
  BitLinear weights are {-beta,+beta} with a SINGLE per-tensor beta. There is no per-feature
  gain to learn. The layer computes beta * sum(+-x_i), so a feature arriving with 7.5x less
  amplitude contributes 7.5x less — permanently. FP32 fixes this in one step by learning
  w ~ 1/std. Binary structurally cannot. The A8 absmax activation quantizer then sets its
  range from the dominant momenta, costing the small features ~3 more bits on top.

Standardizing per-feature BEFORE the LN (they do not commute) drops the post-LN imbalance
**8.9x -> 2.1x** and lifts the relative features from std ~0.185 to ~0.7-1.2 (verified).

## Why this is the pivotal experiment

era-2 @ 6.4M `large`, VERIFIED ROC-test: FP32 0.8561 · W8A8 0.8561 · **W1A8 0.8503 (-0.58 pts)**.
That -0.58 is the thesis: binary is nearly free, and it buys 0 DSP.
era-2 @ 18k `small`, round-7 training-time val_auc: FP32 train_auc 0.7901 · W1A8 0.7447 (best
binary recipe) — a **~4.5-pt** gap. So binarization costs ~0.6 pts at 6.4M but ~4.5+ pts at 18k.
If that is intrinsic, the thesis survives on paper and dies on the trigger board.
The input-scale bug PREDICTS exactly this scale-dependence: a fixed handicap that 6.4M params
absorb and 18k params cannot.

## The design — a 2x2 that tests the MECHANISM, not just "did it get better"

              BN_STANDARDIZE=0        BN_STANDARDIZE=1
  bitnet A8   r8-a8-nostd  (control)  r8-a8-std   (treatment)
  vanilla     r8-fp32-nostd(control)  r8-fp32-std (treatment)

  Mechanism CONFIRMED  if  delta(binary) >> delta(fp32).
  Mechanism REFUTED    if  delta(binary) ~= delta(fp32)  -> it is generic preprocessing, and
                           my causal story is wrong. Say so.

All four cells share ONE recipe (peak LR 2e-5, warmup 3, decay 120, ES patience 25 on val_auc)
— the best binary recipe found in round-7 — so the ONLY differences are BN_VARIANT and
BN_STANDARDIZE. FP32 is given the same long/low schedule; with decay 120 + patience 25 it has
ample room to converge (it converged in 42 epochs at 5e-5/decay-40).

Code: ConfigMap **kai-qkerasmodel-r8** (md5 8609137e... = qkerasModel.py WITH the standardizer).
`kai-qkerasmodel-r5` (md5 4d0a3fed...) is left FROZEN — it is the documented r5 provenance.
Each job hard-fails if the mounted code lacks the knob, because a stale ConfigMap would make
BN_STANDARDIZE=1 a SILENT NO-OP and the "treatment" would secretly be a second control.

Run:  python3 gen_round8_jobs.py     # writes kai-bn8-*.yaml + launch_r8.sh
"""
import os
import stat

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGMAP = "kai-qkerasmodel-r8"

# small = the deployable scale (verified 18,405 params; invariant across variant/act-bits)
D, H, L, FFN, PARAMS = 32, 4, 2, 64, 18405

# one shared recipe for all four cells — round-7's best binary config
RECIPE = dict(
    BN_LR="0.00002", BN_WARMUP_EPOCHS="3", BN_DECAY_EPOCHS="120", BN_DECAY_POWER="1.0",
    BN_BETA2="0.98", BN_WEIGHT_DECAY="0.01", BN_CLIPNORM="1.0",
    BN_L1_REG="0", BN_BATCH="256",
    BN_ES_MONITOR="val_auc", BN_ES_MODE="max", BN_ES_PATIENCE="25",
)

CELLS = [
    dict(key="a8-nostd",   job="kai-bn8-a8-nostd",   run="r8-small-a8-nostd",
         variant="bitnet",  bits="8", std="0",
         note="CONTROL (binary): raw inputs — the 8.9x post-LN imbalance is present."),
    dict(key="a8-std",     job="kai-bn8-a8-std",     run="r8-small-a8-std",
         variant="bitnet",  bits="8", std="1",
         note="TREATMENT (binary): per-feature standardized inputs — imbalance 8.9x -> 2.1x. "
              "Binary cannot learn a per-feature gain, so this is where the big gain should land."),
    dict(key="fp32-nostd", job="kai-bn8-fp32-nostd", run="r8-small-fp32-nostd",
         variant="vanilla", bits="8", std="0",
         note="CONTROL (fp32): raw inputs."),
    dict(key="fp32-std",   job="kai-bn8-fp32-std",   run="r8-small-fp32-std",
         variant="vanilla", bits="8", std="1",
         note="TREATMENT (fp32) — THE FALSIFIER: FP32 can already learn w ~ 1/std, so if the "
              "mechanism is right this should gain LITTLE. If it gains as much as binary, my "
              "causal story is wrong and this is just generic preprocessing."),
]

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
                values: [Tesla-V100-SXM2-32GB, Tesla-V100-SXM2-16GB, Tesla-V100-PCIE-16GB, NVIDIA-GeForce-RTX-3090]"""

JOB_TMPL = """apiVersion: batch/v1
kind: Job
metadata:
  name: {job}
  labels:
    app: bnjet-r8
    bnv: "{key}"
spec:
  backoffLimit: 0
  activeDeadlineSeconds: 86400
  template:
    metadata:
      labels:
        app: bnjet-r8
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
          grep -q "from __future__ import annotations" /work/code/qkerasModel.py || \\
            sed -i '1i from __future__ import annotations' /work/code/qkerasModel.py
          export PYTHONPATH="/work/code:$PYTHONPATH"

          # ---- STALE-CONFIGMAP GATE ------------------------------------------------------
          # If the mounted code lacks the knob, BN_STANDARDIZE=1 is a SILENT NO-OP and this
          # "treatment" run is secretly a second control. Fail loudly instead of lying.
          grep -q "BN_STANDARDIZE" /work/code/qkerasModel.py || {{
            echo "[fatal] mounted qkerasModel.py has no BN_STANDARDIZE knob -> ConfigMap {cm} is STALE."
            exit 1
          }}
          echo "[code] md5=$(md5sum /work/code/qkerasModel.py | cut -d' ' -f1)  (expect the r8 code)"

          echo "[setup] $(date) deps"
          apt-get update -qq && apt-get install -y -qq graphviz
          pip install -q qkeras==0.9.0 "tensorflow==2.11.1" "matplotlib<3.8" pandas seaborn mplhep pydot scikit-learn h5py wandb
          nvidia-smi --query-gpu=name --format=csv,noheader || true
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

          # ---- {note} ----
          export BN_D_MODEL={D} BN_N_HEADS={H} BN_N_LAYERS={L} BN_FFN_DIM={FFN}
          export BN_N_PART=10
{recipe_exports}          export BN_VARIANT={variant} BN_ACT_BITS={bits} BN_SEED=1
          export BN_STANDARDIZE={std}

          echo "[preflight] $(date) --sanity (includes the standardizer self-test) + param gate"
          env CUDA_VISIBLE_DEVICES=-1 python -u /work/code/qkerasModel.py --sanity 2>&1 | tail -8
          env CUDA_VISIBLE_DEVICES=-1 python -u - <<'PYGATE'
          import qkerasModel as M
          n = M.build_bitnet_jet_tagger().count_params()
          assert n == {PARAMS}, f"PARAM GATE FAILED: {{n}} != {PARAMS}"
          assert M.STANDARDIZE == ({std} == 1), "BN_STANDARDIZE did not reach the module"
          print(f"[gate] params={{n}} standardize={{M.STANDARDIZE}} OK")
          PYGATE
          echo "[train] $(date) {run}  variant={variant} A{bits} STANDARDIZE={std}  D={D} L={L} params={PARAMS}"
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
          limits:   {{nvidia.com/gpu: "1", memory: 8Gi, cpu: "2", ephemeral-storage: 24Gi}}
          requests: {{nvidia.com/gpu: "1", memory: 8Gi, cpu: "2", ephemeral-storage: 20Gi}}
        volumeMounts:
        - name: qkcode
          mountPath: /qkcode
        - name: work
          mountPath: /work
      volumes:
      - name: qkcode
        configMap:
          name: {cm}
      - name: work
        emptyDir:
          sizeLimit: 24Gi
"""


# ── stage `lrsweep` (2026-07-14) — re-tune the peak LR for the STANDARDIZED binary model ──
# Why this is mandatory before quoting any binary penalty at deployable scale:
# the 2x2 ran every cell at peak LR 2e-5, which was tuned in round-7 on RAW binary. Standardizing
# the inputs changes the loss landscape (post-LN feature imbalance 8.9x -> 2.1x, i.e. much better
# conditioned), so the optimal LR almost certainly moved. Quoting the standardized binary-vs-FP32
# gap (-4.80 pts) at an LR tuned for the raw model is the round-4 tuned-vs-untuned trap IN REVERSE.
# Round-7 found RAW binary collapses above 2e-4 (best epoch 2-3 = straight-through-estimator
# blow-up). Better conditioning should raise that ceiling — so this probes UPWARD. 2e-5 is the
# control (it reproduces the a8-std cell) and must land near 0.8486 or something is wrong.
LRSWEEP = [
    dict(key="std-lr02", job="kai-bn9-a8-std-lr02", run="r9-small-a8-std-lr02",
         variant="bitnet", bits="8", std="1", lr="0.00002",
         note="CONTROL: reproduces the r8 a8-std cell (expect ~0.8486). If it does not, the run-to-run "
              "variance is too large to read any of this off one seed."),
    dict(key="std-lr05", job="kai-bn9-a8-std-lr05", run="r9-small-a8-std-lr05",
         variant="bitnet", bits="8", std="1", lr="0.00005"),
    dict(key="std-lr10", job="kai-bn9-a8-std-lr10", run="r9-small-a8-std-lr10",
         variant="bitnet", bits="8", std="1", lr="0.0001"),
    dict(key="std-lr20", job="kai-bn9-a8-std-lr20", run="r9-small-a8-std-lr20",
         variant="bitnet", bits="8", std="1", lr="0.0002",
         note="This LR COLLAPSED the raw binary model (best epoch 3). If standardization has fixed the "
              "conditioning, it should now be trainable — that is itself evidence for the mechanism."),
]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="ab", choices=["ab", "lrsweep"])
    args = ap.parse_args()

    if args.stage == "lrsweep":
        jobs = []
        for c in LRSWEEP:
            rec = dict(RECIPE); rec["BN_LR"] = c["lr"]
            recx = "".join(f"          export {k}={v}\n" for k, v in rec.items())
            cc = {k: v for k, v in c.items() if k != "lr"}
            cc.setdefault("note", f"standardized binary @ peak LR {c['lr']}")
            y = JOB_TMPL.format(D=D, H=H, L=L, FFN=FFN, PARAMS=PARAMS, cm=CONFIGMAP,
                                affinity=NODE_AFFINITY, recipe_exports=recx, **cc)
            with open(os.path.join(HERE, f"{c['job']}.yaml"), "w") as f:
                f.write(y)
            print(f"wrote {c['job']}.yaml   STANDARDIZED binary @ peak LR {c['lr']}")
            jobs.append(c["job"])
        p = os.path.join(HERE, "launch_r9_lrsweep.sh")
        with open(p, "w") as f:
            f.write(f"""#!/usr/bin/env bash
# round-9: re-tune peak LR for the STANDARDIZED binary model.
set -euo pipefail
CTX="nautilus"; NS="cms-ml"; HERE="$(cd "$(dirname "$0")" && pwd)"
ALL=({' '.join(jobs)})
if [ "${{1:-apply}}" = "delete" ]; then
  for j in "${{ALL[@]}}"; do kubectl --context "$CTX" -n "$NS" delete job "$j" --ignore-not-found; done
  exit 0
fi
for j in "${{ALL[@]}}"; do
  kubectl --context "$CTX" -n "$NS" delete job "$j" --ignore-not-found >/dev/null 2>&1 || true
  kubectl --context "$CTX" -n "$NS" apply -f "$HERE/$j.yaml"
done
echo; kubectl --context "$CTX" -n "$NS" get jobs -l app=bnjet-r8
""")
        os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR)
        print(f"wrote {p}\n\n{len(jobs)} LR-sweep jobs (standardized binary).")
        return

    jobs = []
    rec = "".join(f"          export {k}={v}\n" for k, v in RECIPE.items())
    for c in CELLS:
        y = JOB_TMPL.format(D=D, H=H, L=L, FFN=FFN, PARAMS=PARAMS, cm=CONFIGMAP,
                            affinity=NODE_AFFINITY, recipe_exports=rec, **c)
        with open(os.path.join(HERE, f"{c['job']}.yaml"), "w") as f:
            f.write(y)
        print(f"wrote {c['job']}.yaml   variant={c['variant']:<8} STANDARDIZE={c['std']}")
        jobs.append(c["job"])

    ls = f"""#!/usr/bin/env bash
# round-8: the input-standardization 2x2 (mechanism test).
# Usage: ./launch_r8.sh   |   ./launch_r8.sh delete
set -euo pipefail
CTX="nautilus"; NS="cms-ml"; HERE="$(cd "$(dirname "$0")" && pwd)"
ALL=({' '.join(jobs)})
if [ "${{1:-apply}}" = "delete" ]; then
  for j in "${{ALL[@]}}"; do kubectl --context "$CTX" -n "$NS" delete job "$j" --ignore-not-found; done
  exit 0
fi
for j in "${{ALL[@]}}"; do
  kubectl --context "$CTX" -n "$NS" delete job "$j" --ignore-not-found >/dev/null 2>&1 || true
  kubectl --context "$CTX" -n "$NS" apply -f "$HERE/$j.yaml"
done
echo; echo "=== round-8 standardization 2x2 launched ==="
kubectl --context "$CTX" -n "$NS" get jobs -l app=bnjet-r8
"""
    p = os.path.join(HERE, "launch_r8.sh")
    with open(p, "w") as f:
        f.write(ls)
    os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR)
    print(f"wrote {p}\n\n{len(jobs)} jobs (2x2). ConfigMap={CONFIGMAP}.")


if __name__ == "__main__":
    main()
