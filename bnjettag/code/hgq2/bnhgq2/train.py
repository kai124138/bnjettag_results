"""FINAL-campaign training stage: native HGQ2 QAT (decisions.md 2026-07-07).

`python run_stage.py train --config configs/final-<variant>.json --seed <n> [--smoke]`

Trains one variant/seed from scratch. The trained model IS (essentially) the hardware
model — binary {-1,+1} STE weights + static per-tensor activation grids (qat.py). Emits
the INTERFACE CONTRACT artifacts (the parallel ROC job depends on them, do not deviate):
  * model_best.keras  — best epoch by val macro-OvR AUC, Keras-3 format; reload in a fresh
    process via: import hgq; bnhgq2.compat.apply_keras_compat(); bnhgq2.subln.register_subln();
    import bnhgq2.qat; keras.models.load_model("model_best.keras")
  * train_meta.json   — variant, seed, config hash, params, best epoch/AUC, lr, timestamps
Both are written to PVC /data/outputs/final/<variant>-s<seed>/ AND uploaded via wandb.save.

Recipe carried from r5 (Adam beta2=0.98, peak 5e-5, warmup 1ep + poly decay, wd 0.01,
batch 256, early-stop patience 10 on val macro-OvR AUC). Gradient clipping: clipvalue=1.0
(NOT r5's global_clipnorm — that overflows float32 on this deep-SubLN stack and NaNs A4
regardless of LR; clipvalue is the overflow-safe unit-scale clip, verified — LEDGER 2026-07-07).
"""
from __future__ import annotations

import glob
import json
import os
import time

import numpy as np


# --------------------------------------------------------------------------- #
# data (era-2, top-N by pT — byte-identical logic to data.load_eval_set)       #
# --------------------------------------------------------------------------- #
def load_train_data(data_dir: str, n_part: int, max_files: int | None = None):
    import h5py
    from .data import CLASS_LABELS

    files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
    if not files:
        raise FileNotFoundError(f"no .h5 files in {data_dir}")
    if max_files:
        files = files[:max_files]

    def _names(ds):
        return [n.decode() if isinstance(n, bytes) else str(n) for n in ds]

    Xs, Ys = [], []
    for fp in files:
        with h5py.File(fp, "r") as hf:
            const = hf["jetConstituentList"][:]
            jets = hf["jets"][:]
            jnames = _names(hf["jetFeatureNames"][:])
            pnames = _names(hf["particleFeatureNames"][:])
        miss = [l for l in CLASS_LABELS if l not in jnames]
        if miss:
            raise KeyError(f"{fp}: labels {miss} not in jetFeatureNames")
        lab_idx = [jnames.index(l) for l in CLASS_LABELS]
        pt_col = next((i for i, n in enumerate(pnames) if n.endswith("_pt")), None)
        if pt_col is not None:
            order = np.argsort(-const[:, :, pt_col], axis=1, kind="stable")
            const = np.take_along_axis(const, order[:, :, None], axis=1)
        Xs.append(const[:, :n_part, :].astype("float32"))
        Ys.append(jets[:, lab_idx].astype("float32"))
    return np.concatenate(Xs), np.concatenate(Ys), len(files)


# --------------------------------------------------------------------------- #
# metric + callbacks                                                           #
# --------------------------------------------------------------------------- #
def macro_ovr_auc(y_onehot, scores):
    from sklearn.metrics import roc_auc_score
    per = []
    for c in range(y_onehot.shape[1]):
        yc = y_onehot[:, c]
        per.append(roc_auc_score(yc, scores[:, c]) if 0 < yc.sum() < len(yc) else float("nan"))
    return float(np.nanmean(per)), [float(p) for p in per]


def _softmax(z):
    z = z.astype(np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _recalib_callback(qat, model, taps, X, act_bits, epochs):
    """Callback: re-run MSE activation calibration on the drifted activations at the start
    of each epoch in `epochs`, then re-freeze (act_calib='recalib', mechanism 2)."""
    import keras

    class Recalibrate(keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            if epoch in epochs:
                si = qat.calibrate_activations(self.model, taps, X, act_bits)
                if si:
                    v = list(si.values())
                    print(f"  [recalib epoch {epoch}] {len(si)} act sites -> i in "
                          f"[{min(v)},{max(v)}] (re-froze on drifted activations)", flush=True)
    return Recalibrate()


def make_callbacks(Xval, Yval, best_path, es_patience, lr, warmup_epochs,
                   decay_epochs, decay_power, use_wandb, state):
    import keras

    class MacroAUC(keras.callbacks.Callback):
        """Compute val macro-OvR AUC, write logs['val_macro_auc'], save best checkpoint."""

        def on_epoch_end(self, epoch, logs=None):
            logs = logs if logs is not None else {}
            scores = _softmax(np.asarray(self.model.predict(Xval, batch_size=1024,
                                                            verbose=0)))
            auc, per = macro_ovr_auc(Yval, scores)
            logs["val_macro_auc"] = auc
            print(f"  [epoch {epoch}] val_macro_auc={auc:.5f} per_class={[round(p,4) for p in per]}",
                  flush=True)
            if auc > state["best_auc"]:
                state["best_auc"], state["best_epoch"] = auc, epoch
                self.model.save(best_path)
                print(f"  [epoch {epoch}] new best -> saved {best_path}", flush=True)
            if use_wandb:
                import wandb
                wandb.log({**logs, "epoch": epoch, "val_macro_auc": auc,
                           **{f"val_auc_{i}": p for i, p in enumerate(per)}})

    cbs = [MacroAUC(),
           keras.callbacks.EarlyStopping(monitor="val_macro_auc", mode="max",
                                         patience=es_patience, verbose=1,
                                         restore_best_weights=True)]

    if warmup_epochs > 0 or decay_epochs > 0:
        def sched(epoch, _lr):
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return lr * float(epoch + 1) / float(warmup_epochs)
            if decay_epochs > 0:
                prog = min(1.0, float(epoch - warmup_epochs) / float(decay_epochs))
                return lr * (1.0 - prog) ** decay_power
            return lr
        cbs.append(keras.callbacks.LearningRateScheduler(sched, verbose=0))
    return cbs


# --------------------------------------------------------------------------- #
# the stage                                                                    #
# --------------------------------------------------------------------------- #
def train(cfg: dict, seed: int, out_dir: str, smoke: bool = False,
          cfg_hash: str = "") -> dict:
    import keras
    import tensorflow as tf
    from .compat import apply_keras_compat
    from .subln import register_subln
    from . import qat

    apply_keras_compat()
    register_subln()
    keras.utils.set_random_seed(int(seed))

    tr = cfg["train"]
    A = cfg["arch"]
    n_part = A["n_part"]
    act_bits = int(cfg["quant"]["act_bits"])
    os.makedirs(out_dir, exist_ok=True)

    epochs = 3 if smoke else int(tr["epochs"])
    warmup = (0 if smoke else int(tr["warmup_epochs"]))
    decay = (2 if smoke else int(tr["decay_epochs"]))
    max_files = 2 if smoke else tr.get("max_files")

    # ---- data (BNHGQ2_TRAIN_DATA overrides the config path for local/portable runs) ----
    data_dir = os.environ.get("BNHGQ2_TRAIN_DATA", tr["data"])
    X, Y, nfiles = load_train_data(data_dir, n_part, max_files=max_files)
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(len(X))
    X, Y = X[idx], Y[idx]
    vsplit = float(tr.get("validation_split", 0.20))
    nval = int(len(X) * vsplit)
    Xval, Yval = X[:nval], Y[:nval]
    Xtr, Ytr = X[nval:], Y[nval:]
    print(f"[train] data: {nfiles} files, {len(X)} jets ({len(Xtr)} train / {len(Xval)} val), "
          f"class counts {Y.sum(0).astype(int).tolist()}", flush=True)

    # Round-8 knob (guarded; absent key => byte-identical behavior): offline per-feature
    # standardization from the TRAIN split only. Stats travel with the checkpoint as
    # input_std.json — eval/convert must apply the SAME stats via their --input-std flag.
    input_std = bool(cfg["arch"].get("input_std", False))
    std_path = None
    input_mu = input_sigma = None          # passed to the builder for the pair_bias branch
    if input_std:
        from .data import input_std_stats, apply_input_std
        mu, sigma = input_std_stats(Xtr)
        input_mu, input_sigma = mu, sigma
        Xtr = apply_input_std(Xtr, mu, sigma)
        Xval = apply_input_std(Xval, mu, sigma)
        std_path = os.path.join(out_dir, "input_std.json")
        with open(std_path, "w") as f:
            json.dump({"mu": mu.tolist(), "sigma": sigma.tolist(),
                       "computed_from": "train split only",
                       "contract": "hardware receives standardized inputs (offline preprocessing)"},
                      f, indent=1)
        print(f"[train] input_std ON: per-feature z-score from train split "
              f"(|mu|max={float(np.abs(mu).max()):.4g}, sigma_max={float(sigma.max()):.4g}) "
              f"-> input_std.json", flush=True)

    # ---- model + activation calibration ----
    model, taps = qat.build_qat_model(cfg, seed=seed,
                                      input_mu=input_mu, input_sigma=input_sigma)
    site_i = qat.calibrate_activations(model, taps, Xtr, act_bits)
    act_calib = str(cfg["quant"].get("act_calib", "frozen")).lower()
    grid_before = qat.act_grid_params(model)
    nparams = int(model.count_params())
    ntrain = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    print(f"[train] {cfg['name']} params={nparams:,} trainable={ntrain:,} "
          f"act_sites={len(site_i)} act_calib={act_calib}", flush=True)
    if cfg["quant"]["weight"] == "binary_absmean":
        effs = qat.effective_weight_values(model)
        ok = all(len(v) == 2 and not (v == 0).any() for v in effs.values())
        assert ok, "BINARY GATE FAILED at build: effective weights not exactly two-valued"
        print(f"[train] binary gate OK ({len(effs)} bit-layers exactly +/-beta)", flush=True)

    # ---- optimizer (clipvalue: overflow-safe unit-scale clip, see module docstring) ----
    lr = float(tr["lr"])
    clip_mode = tr.get("clip_mode", "value")
    opt_kw = dict(learning_rate=lr, beta_1=0.9, beta_2=float(tr.get("beta2", 0.98)),
                  weight_decay=float(tr.get("weight_decay", 0.01)))
    if clip_mode == "norm":
        opt_kw["global_clipnorm"] = float(tr.get("clipnorm", 1.0))
    else:
        opt_kw["clipvalue"] = float(tr.get("clipvalue", 1.0))
    optimizer = keras.optimizers.Adam(**opt_kw)
    model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                  optimizer=optimizer, metrics=["categorical_accuracy"])

    # ---- wandb ----
    use_wandb = bool(os.environ.get("WANDB_API_KEY")) and bool(tr.get("wandb_project"))
    if use_wandb:
        import wandb
        wandb.init(project=tr["wandb_project"],
                   name=os.environ.get("WANDB_RUN_NAME") or f"final-{_variant(cfg)}-s{seed}",
                   config={"variant": _variant(cfg), "seed": int(seed),
                           "act_bits": act_bits, "weight": cfg["quant"]["weight"],
                           "config_hash": cfg_hash, "params": nparams,
                           "lr": lr, "clip_mode": clip_mode, "smoke": smoke,
                           **{f"arch_{k}": v for k, v in A.items()}})

    # ---- train ----
    best_path = os.path.join(out_dir, "model_best.keras")
    state = {"best_auc": -1.0, "best_epoch": -1}
    cbs = make_callbacks(Xval, Yval, best_path, int(tr.get("es_patience", 10)),
                         lr, warmup, decay, float(tr.get("decay_power", 1.0)),
                         use_wandb, state)
    # act_calib="recalib": re-run MSE calibration on the DRIFTED activations at the given
    # epoch boundaries, then re-freeze (cheap; no quantizer surgery). Distributions have
    # stabilized by then, so the frozen grid re-aligns. (Mechanism 2 / fallback.)
    recal_epochs = [int(e) for e in cfg["quant"].get("act_recalib_epochs", [])]
    if act_calib == "recalib" and recal_epochs and not smoke:
        cbs.append(_recalib_callback(qat, model, taps, Xtr, act_bits, recal_epochs))
    elif act_calib == "recalib" and recal_epochs:  # smoke: recalibrate on epoch 1 to exercise it
        cbs.append(_recalib_callback(qat, model, taps, Xtr, act_bits, [1]))
    t0 = time.time()
    model.fit(Xtr, Ytr, epochs=epochs, batch_size=int(tr.get("batch", 256)),
              verbose=2, callbacks=cbs)
    dt = time.time() - t0
    grid_after = qat.act_grid_params(model)
    if act_calib == "trainable":
        moved = sum(1 for k in grid_before if grid_before.get(k) != grid_after.get(k))
        print(f"[train] trainable-scale grids moved at {moved}/{len(grid_before)} sites "
              f"during training (drift tracking)", flush=True)

    if state["best_epoch"] < 0:  # no epoch improved (e.g. degenerate smoke) -> save final
        model.save(best_path)
        scores = _softmax(np.asarray(model.predict(Xval, batch_size=1024, verbose=0)))
        state["best_auc"], _ = macro_ovr_auc(Yval, scores)
        state["best_epoch"] = epochs - 1

    meta = {
        "variant": _variant(cfg), "seed": int(seed), "config": cfg["name"],
        "config_hash": cfg_hash, "params": nparams, "trainable_params": ntrain,
        "best_epoch": state["best_epoch"], "best_val_macro_auc": state["best_auc"],
        "act_bits": act_bits, "weight": cfg["quant"]["weight"],
        "lr": lr, "clip_mode": clip_mode, "epochs_run": epochs, "smoke": smoke,
        "n_train": int(len(Xtr)), "n_val": int(len(Xval)), "n_files": nfiles,
        "train_seconds": round(dt, 1), "act_calib_i": site_i,
        "act_calib": act_calib, "act_recalib_epochs": recal_epochs,
        "act_grid_before": grid_before, "act_grid_after": grid_after,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t0)),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
    }
    meta_path = os.path.join(out_dir, "train_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[train] DONE best_epoch={state['best_epoch']} "
          f"best_val_macro_auc={state['best_auc']:.5f} in {dt:.0f}s", flush=True)
    print(f"[train] wrote {best_path} + {meta_path}", flush=True)

    if use_wandb:
        import wandb
        run_ref = None
        try:
            run_ref = (wandb.run.entity, wandb.run.project, wandb.run.id)
            # base_path keeps the <leaf>/model_best.keras name; policy="now" starts the
            # upload immediately instead of racing the end-of-run sync.
            base = os.path.dirname(os.path.abspath(out_dir))
            wandb.save(best_path, base_path=base, policy="now")
            wandb.save(meta_path, base_path=base, policy="now")
            if std_path:
                wandb.save(std_path, base_path=base, policy="now")
            wandb.summary["best_val_macro_auc"] = state["best_auc"]
            wandb.summary["best_epoch"] = state["best_epoch"]
        except Exception as e:
            print(f"[train] wandb.save warn: {e}", flush=True)
        wandb.finish()
        # Durability gate (2026-07-15): 4 of 8 loose-end runs finished with the checkpoint
        # silently ABSENT from W&B (end-of-run sync race; pods are emptyDir-only, so the
        # artifact was simply lost). The run is the checkpoint's only durable home — verify
        # via the public API (size-exact) and re-upload onto the finished run until it is
        # really there; exit nonzero if it cannot be made durable.
        offline = os.environ.get("WANDB_MODE", "").lower().startswith(("off", "dis"))
        if run_ref and not smoke and not offline:
            leaf = os.path.basename(os.path.abspath(out_dir))
            fname = f"{leaf}/{os.path.basename(best_path)}"
            want = os.path.getsize(best_path)
            ok = False
            for attempt in range(1, 6):
                try:
                    api_run = wandb.Api().run("/".join(run_ref))
                    got = next((f.size for f in api_run.files() if f.name == fname), 0)
                    if got == want:
                        print(f"[train] durability OK: {fname} on W&B ({got} bytes)", flush=True)
                        ok = True
                        break
                    print(f"[train] durability: {fname} size={got} != {want} "
                          f"(attempt {attempt}) — re-uploading", flush=True)
                    api_run.upload_file(best_path, root=base)
                    api_run.upload_file(meta_path, root=base)
                    if std_path:
                        api_run.upload_file(std_path, root=base)
                    time.sleep(10)
                except Exception as e:
                    print(f"[train] durability attempt {attempt} error: {e}", flush=True)
                    time.sleep(20)
            if not ok:
                raise SystemExit(f"[train] FATAL: {fname} not durable on W&B after retries")
    return meta


def _variant(cfg: dict) -> str:
    w = cfg["quant"]["weight"]
    if w == "none":
        return "fp32"
    if w == "int8_absmax":
        return "w8a8"
    return f"w1a{cfg['quant']['act_bits']}"
