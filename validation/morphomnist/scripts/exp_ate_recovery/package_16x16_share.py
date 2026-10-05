"""Build a shareable archive of 16x16 run folders (weights + config + metrics + W&B link), 2026-10-05.

    python scripts/exp_ate_recovery/package_16x16_share.py part1 --out /mnt/disk-geoff/ff-project/share
    python scripts/exp_ate_recovery/package_16x16_share.py part2 --out /mnt/disk-geoff/ff-project/share

part1: the Gaussian-scale flexible flow (E1-E6 x datasets 1-5 x fit seeds {k, 1001-1004}) and frengression
       with seed k (E1-E6 x datasets 1-5).
part2: frengression fit seeds 1001-1004 (E1-E6 x datasets 1-5).
Each run folder keeps config.json, metrics.json, wandb.json, the weights (model.eqx / model.pt) and, for
frengression runs read out at 50000 draws, readout_mc5000.npz. The archive unpacks into
validation/morphomnist/ (runs/exp_ate_recovery/..., runs/frengression/...) and carries index.csv and README.md.
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import zipfile

import numpy as np

MM = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, MM)
import dataset_store as DS  # noqa: E402

KEEP = ("config.json", "metrics.json", "wandb.json", "model.eqx", "model.pt", "readout_mc5000.npz")
PRESETS = ("e1", "e2", "e3", "e4", "e5", "e6")


def latest(pattern, model_file):
    c = sorted(d for d in glob.glob(os.path.join(MM, "runs", pattern))
               if os.path.exists(os.path.join(d, "metrics.json")) and os.path.exists(os.path.join(d, model_file)))
    return c[-1] if c else None


def runs_for(part):
    out, missing = [], []
    for p in PRESETS:
        for k in range(1, 6):
            cells = []
            if part == "part1":
                cells += [("gaussian_flow", "exp_ate_recovery",
                           f"*_ff_{p}_flexgauss_sa{k}_lr0.001_copw16_ystd_k256_s{s}_d0-9_*", "model.eqx", s)
                          for s in (k, 1001, 1002, 1003, 1004)]
                cells.append(("frengression", "frengression", f"*_frengression_{p}_sa{k}_k256_s{k}_d0-9_*", "model.pt", k))
            else:
                cells += [("frengression", "frengression", f"*_frengression_{p}_sa{k}_k256_s{s}_d0-9_*", "model.pt", s)
                          for s in (1001, 1002, 1003, 1004)]
            for model, root, pat, mf, s in cells:
                d = latest(os.path.join(root, pat), mf)
                (out if d else missing).append((model, p.upper(), k, s, d or pat))
    return out, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("part1", "part2"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs, missing = runs_for(a.part)
    if missing:
        print(f"{len(missing)} runs not finished yet, e.g.", missing[:3]); sys.exit(1)
    os.makedirs(a.out, exist_ok=True)
    commit = subprocess.run(["git", "-C", MM, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    rows = []
    for model, preset, k, s, d in runs:
        ate = np.asarray(DS.run_arrays(d, need=())["ATE"])
        err = float(np.abs(DS.effect_map(d, 5000) - ate).mean())
        cfg = json.load(open(os.path.join(d, "config.json")))
        rows.append(dict(model=model, preset=preset, dataset=k, fit_seed=s, effect_map_mae=round(err, 6),
                         run_path=os.path.relpath(d, MM), dataset_id=cfg.get("dataset_id"),
                         code_commit=(cfg.get("git") or {}).get("commit", "")[:7],
                         wandb_url=json.load(open(os.path.join(d, "wandb.json"))).get("url", "")))
    name = f"ff_morphomnist_16x16_{a.part}"
    idx = os.path.join(a.out, f"{name}_index.csv")
    with open(idx, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    what = ("Gaussian-scale flexible frugal flow (5 fit seeds per dataset) and frengression (fit seed = dataset)"
            if a.part == "part1" else "frengression, fit seeds 1001-1004")
    readme = f"""# MorphoMNIST 16x16 run folders, {a.part}

Contents: {what}; 16x16, all ten MNIST digits (n = 60000), presets E1-E6, datasets (assignment seeds) 1-5.
{len(rows)} run folders. Built from frugal-flows branch multi-y-gaussian at commit {commit}.

Unpack inside frugal-flows/validation/morphomnist/: the folders land in runs/exp_ate_recovery/ (flows) and
runs/frengression/. Each keeps config.json (every setting and seed), metrics.json, wandb.json (W&B link) and
the weights (model.eqx for flows, model.pt for frengression). arrays.npz and plots are not included; they can
be recomputed from the weights.

index.csv: one row per run; effect_map_mae = mean over the 256 pixels of |estimated - true ATE|, with the
effect read out from 5000 paired draws (for frengression runs made at 50000 draws, the 5000-draw re-read in
readout_mc5000.npz). Average the effect maps of the 5 flow fits of a dataset for the 5-fit estimate.

Reload (from validation/morphomnist/, environment frugal-flows; frengression needs frugal-flows-frengression):

    import exp_ate_recovery as E, dataset_store as DS
    tau = E.reload_effect_map("runs/exp_ate_recovery/<run>")   # equals DS.effect_map(run) exactly
    flow = E.load_model("runs/exp_ate_recovery/<run>")          # the fitted flow
    import exp_frengression_recovery as F
    model, inputs = F.load_model("runs/frengression/<run>")      # frengression

The dataset is rebuilt from config.json (MNIST files are in the repo's data/) and checked against its
recorded hash. See validation/morphomnist/README.md ("Current state", "Shared run folders").
"""
    zpath = os.path.join(a.out, f"{name}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{name}_README.md", readme)
        z.write(idx, f"{name}_index.csv")
        for r in rows:
            d = os.path.join(MM, r["run_path"])
            for f in KEEP:
                if os.path.exists(os.path.join(d, f)):
                    z.write(os.path.join(d, f), os.path.join(r["run_path"], f))
    print(f"{zpath}: {len(rows)} runs, {os.path.getsize(zpath) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
