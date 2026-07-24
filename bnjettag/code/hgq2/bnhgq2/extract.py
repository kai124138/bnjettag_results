"""h5py-only extraction of a trained BitNet QKeras checkpoint.

Pulls (a) every latent FP32 kernel/bias and (b) the folded positional-encoding
constant, straight from the Keras-2 HDF5 — no TensorFlow 2.11 environment needed.
Verified 2026-07-04: the PE constant is serialized inline in model_config as the
`y` call-kwarg of the TFOpLambda add node.
"""
from __future__ import annotations

import json

import h5py
import numpy as np


def _walk_weights(group, prefix=""):
    out = {}
    for k in group:
        item = group[k]
        if isinstance(item, h5py.Dataset):
            out[prefix + k] = np.array(item)
        else:
            out.update(_walk_weights(item, prefix + k + "/"))
    return out


def extract_checkpoint(h5_path: str) -> dict:
    """Returns {'layers': {name: {'kernel': arr, 'bias': arr|None}},
    'pos_encoding': (n_part, d_model) array, 'layer_order': [...]}."""
    with h5py.File(h5_path, "r") as f:
        cfg = json.loads(f.attrs["model_config"])
        raw = _walk_weights(f["model_weights"])

    # ---- weights: last two path components are '<layer>/<kernel|bias>:0' ----
    layers: dict[str, dict] = {}
    for path, arr in raw.items():
        parts = path.split("/")
        wname = parts[-1].split(":")[0]  # kernel | bias
        lname = parts[-2]
        layers.setdefault(lname, {"kernel": None, "bias": None})[wname] = arr

    # ---- folded PE constant from the TFOpLambda add node ----
    def _find_y(x):
        """Recursively locate a dict holding a 'y' call-kwarg in inbound_nodes."""
        if isinstance(x, dict):
            if "y" in x and x["y"] is not None:
                return x["y"]
            for v in x.values():
                r = _find_y(v)
                if r is not None:
                    return r
        elif isinstance(x, list):
            for v in x:
                r = _find_y(v)
                if r is not None:
                    return r
        return None

    pe = None
    for l in cfg["config"]["layers"]:
        if l["class_name"] == "TFOpLambda" and "add" in l["config"].get("name", ""):
            y = _find_y(l.get("inbound_nodes", []))
            if y is not None:
                pe = np.asarray(y, dtype=np.float32)
    if pe is None:
        raise ValueError(f"{h5_path}: folded PE constant not found in model_config")

    order = [l["config"]["name"] for l in cfg["config"]["layers"]]
    return {"layers": layers, "pos_encoding": pe, "layer_order": order}


def layer_names(n_layers: int) -> list[str]:
    """Canonical BitLinear names in forward order for an L-layer model."""
    names = ["input_proj"]
    for i in range(n_layers):
        b = f"bit_block_{i}"
        names += [f"{b}_attn_Wq", f"{b}_attn_Wk", f"{b}_attn_Wv", f"{b}_attn_Wo",
                  f"{b}_ffn_fc1", f"{b}_ffn_fc2"]
    names += ["head_fc1", "head_fc2"]
    return names
