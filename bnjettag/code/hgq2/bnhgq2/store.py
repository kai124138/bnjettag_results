"""Results store: one JSON per (config, stage) + a manifest, under results/hgq2/.

Everything a report quotes must trace back to a file written here (or to a
pre-existing verified artifact it references by path).
"""
from __future__ import annotations

import datetime
import json
import os

from .config import PROJECT_ROOT

# BNHGQ2_STORE overrides the store root (the PVC code layout resolves PROJECT_ROOT to /,
# so cluster runs point this at a writable path, e.g. /data/outputs/final/_store).
STORE = os.environ.get("BNHGQ2_STORE",
                       os.path.join(PROJECT_ROOT, "bnjettag", "results", "hgq2"))


def _jsonable(x):
    import numpy as np
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist() if x.size <= 64 else f"<array shape={x.shape} dtype={x.dtype}>"
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def write_stage(cfg: dict, cfg_hash: str, stage: str, payload: dict) -> str:
    run_dir = os.path.join(STORE, "runs", cfg_hash)
    os.makedirs(run_dir, exist_ok=True)
    rec = {
        "config_name": cfg["name"],
        "config_hash": cfg_hash,
        "stage": stage,
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
        "payload": _jsonable(payload),
    }
    path = os.path.join(run_dir, f"{stage}.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    _update_manifest(cfg, cfg_hash, stage, os.path.relpath(path, STORE))
    return path


def read_stage(cfg_hash: str, stage: str) -> dict | None:
    path = os.path.join(STORE, "runs", cfg_hash, f"{stage}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _update_manifest(cfg: dict, cfg_hash: str, stage: str, relpath: str):
    mpath = os.path.join(STORE, "manifest.json")
    manifest = {}
    if os.path.exists(mpath):
        with open(mpath) as f:
            manifest = json.load(f)
    ent = manifest.setdefault(cfg_hash, {
        "name": cfg["name"],
        "config_path": os.path.relpath(cfg.get("_path", "?"), PROJECT_ROOT),
        "stages": {},
    })
    ent["stages"][stage] = {
        "file": relpath,
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
