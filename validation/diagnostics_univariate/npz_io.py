"""Dataset npz round-trip for S11 (pure numpy + json; importable from every env)."""
from __future__ import annotations

import json
import os

import numpy as np

BLOCKS = ("Y", "X", "Z_cont", "Z_disc")


def save_npz(path: str, d: dict) -> None:
    """Arrays Y, X, Z_cont, Z_disc (absent blocks omitted) + ``meta`` as a JSON string."""
    arrays = {k: np.asarray(d[k]) for k in BLOCKS if d.get(k) is not None}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, meta_json=np.array(json.dumps(d["meta"])), **arrays)
    os.replace(tmp, path)


def load_npz(path: str) -> dict:
    """Inverse of save_npz; absent blocks come back as None."""
    with np.load(path, allow_pickle=False) as f:
        out = {k: (f[k] if k in f.files else None) for k in BLOCKS}
        out["meta"] = json.loads(str(f["meta_json"]))
    return out
