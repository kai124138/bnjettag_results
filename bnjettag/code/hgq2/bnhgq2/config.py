"""Config load / validate / hash.

A config fully describes one model+quantization point. The pipeline never reads
architecture or precision from anywhere else. Paths are relative to the PROJECT
ROOT (the directory that contains bnjettag/), resolved here.
"""
from __future__ import annotations

import hashlib
import json
import os

# code/hgq2/bnhgq2/config.py -> project root is 4 levels up
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)

REQUIRED_ARCH = {
    "n_part", "n_feat", "d_model", "n_heads", "n_layers", "ffn_dim", "n_classes",
    "pool", "ffn_act", "norm", "norm_placement", "pos_enc", "softmax_free",
}
# Rebuild/port configs describe a trained QKeras checkpoint to re-emit; FINAL-campaign
# TRAIN configs (with a "train" block) train from scratch and have no checkpoint yet.
REQUIRED_TOP_REBUILD = {"name", "era", "arch", "quant", "checkpoint", "reference_npz", "hls"}
REQUIRED_TOP_TRAIN = {"name", "era", "arch", "quant", "train", "hls"}
REQUIRED_TRAIN = {"data", "epochs", "lr"}


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    is_train = "train" in cfg
    required = REQUIRED_TOP_TRAIN if is_train else REQUIRED_TOP_REBUILD
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"{path}: missing top-level keys {sorted(missing)}")
    amiss = REQUIRED_ARCH - set(cfg["arch"])
    if amiss:
        raise ValueError(f"{path}: missing arch keys {sorted(amiss)}")
    if cfg["quant"]["weight"] not in ("binary_absmean", "none", "int8_absmax"):
        raise ValueError(f"unknown weight quant {cfg['quant']['weight']}")
    if is_train:
        tmiss = REQUIRED_TRAIN - set(cfg["train"])
        if tmiss:
            raise ValueError(f"{path}: missing train keys {sorted(tmiss)}")
    cfg["_path"] = os.path.abspath(path)
    return cfg


def cfg_hash(cfg: dict) -> str:
    """Stable 8-hex-char key of the scientific content (private keys excluded)."""
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    blob = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def resolve(cfg: dict, key: str) -> str:
    """Resolve a config path field against the project root."""
    return os.path.join(PROJECT_ROOT, cfg[key])
