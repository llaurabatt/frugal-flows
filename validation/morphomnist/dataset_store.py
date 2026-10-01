"""Datasets are rebuilt, not copied into every run folder (2026-10-01).

Until 2026-09-30 every run saved the dataset's images ``Y`` and true per-unit effects ``ITE``
in its ``arrays.npz`` -- identical across the fit seeds of a dataset and across the flow,
frengression and the baselines, and ~75 % of everything stored. From 2026-10-01 the fitting
scripts no longer save them. The generator is deterministic given the settings each run
records in ``config.json``, so a run's data are rebuilt on demand and checked against the
``dataset_id`` and ``data_hash`` the run recorded (md5 of Y, X and ATE).

  load_dataset(run_dir)   -> the generator's dict for the run's dataset (verified)
  run_arrays(run_dir)     -> the run's arrays.npz as a dict, with "Y" and "ITE" added from the
                             rebuilt dataset when the file does not hold them (old runs do)

A rebuilt dataset is cached once per dataset_id in runs/datasets/<dataset_id>.npz
(Y and ITE only), so repeated analyses do not rebuild it.
"""
from __future__ import annotations

import json
import os
from dataclasses import fields

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "runs", "datasets")
DROPPED_KEYS = ("Y", "ITE")      # what the fitting scripts no longer save


def _config_record(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "config.json")) as f:
        return json.load(f)


def build_for_run(run_dir: str) -> dict:
    """Rebuild the dataset a run was fitted on and check it is the same one."""
    import exp_ate_recovery as E

    rec = _config_record(run_dir)
    stored = rec.get("config", {})
    known = {f.name for f in fields(E.Config)}
    if "arm" not in stored:
        # a Frengression run's config (no "arm"): its own same-named fit fields (y_scaling, ...)
        # mean something else; only the data settings are needed to rebuild the dataset
        known -= set(E.FLOW_FIT_ONLY_FIELDS)
    cfg = E.Config(**{k: v for k, v in stored.items() if k in known})
    data = E.build_data(cfg)
    for key in ("dataset_id", "data_hash"):
        if rec.get(key) is not None and data.get(key) != rec[key]:
            raise ValueError(f"{os.path.basename(run_dir)}: rebuilt {key} {data.get(key)} != recorded {rec[key]}")
    return data


def load_dataset(run_dir: str, use_cache: bool = True) -> dict:
    """Y and ITE of the run's dataset (plus the rest of the generator's dict when rebuilt)."""
    rec = _config_record(run_dir)
    did = rec.get("dataset_id")
    path = os.path.join(CACHE_DIR, f"{did}.npz") if did else None
    if use_cache and path and os.path.exists(path):
        with np.load(path) as z:
            cached = {k: z[k] for k in z.files}
        if str(cached.get("data_hash")) == str(rec.get("data_hash")):
            return cached
    data = build_for_run(run_dir)
    if use_cache and path:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = path + ".tmp.npz"
        np.savez(tmp, Y=np.asarray(data["Y"]), ITE=np.asarray(data["ITE"]),
                 data_hash=np.asarray(data["data_hash"]), dataset_id=np.asarray(data["dataset_id"]))
        os.replace(tmp, path)
    return data


class RunArrays(dict):
    """A dict with ``.files``, so code written for ``np.load(...)`` keeps working."""
    @property
    def files(self):
        return list(self.keys())


def run_arrays(run_dir: str, need=DROPPED_KEYS) -> RunArrays:
    """The run's saved arrays, with the dropped dataset arrays filled in when needed."""
    with np.load(os.path.join(run_dir, "arrays.npz")) as z:
        out = RunArrays({k: z[k] for k in z.files})
    missing = [k for k in need if k not in out]
    if missing:
        data = load_dataset(run_dir)
        for k in missing:
            out[k] = np.asarray(data[k])
    return out
