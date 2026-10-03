"""Package the n = 5000 Gaussian-spline fits (G-flex-std) for sharing (2026-10-03).

Per experiment, one zip with every G-flex-std fit: the phase-1 fit (fit seed = k, runs/sub5k) and the
averaging fits (fit seeds 1001-1004, runs/gausshp, read from the avg5 launcher logs). Each fit folder
keeps model.eqx, config.json, metrics.json, wandb.json, arrays.npz and plots/. Also writes index.csv
(one row per fit) and the code as a git bundle of the current branch. Re-run to add experiments that
finished later; existing zips are kept unless --force.

  python package_fits.py OUT_DIR [--force]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
MM = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.abspath(os.path.join(MM, "..", ".."))
EXP = {"exp1_rct_homogeneous": "E1", "exp2_confounded_homogeneous": "E2", "exp3_confounded_heterogeneous": "E3",
       "exp4_covariate_cate": "E4", "exp5_quantile_effect": "E5", "exp6_spatial_cate": "E6"}
KEEP = ("model.eqx", "config.json", "metrics.json", "wandb.json", "arrays.npz", "realism.json")


def fits():
    out = []
    for r in json.load(open(os.path.expanduser(os.environ.get("FF_RUNS_LOG", "~/work/halo-runs") + "/sub5k/results.json"))):
        if r["code"] == "G-flex-std":
            out.append((os.path.join(MM, "runs", "sub5k", r["run"]), "phase1 (fit seed = k)"))
    pat = re.compile(r"END\s+hp-base:(E\d):k(\d+):s(\d+) rc=0 done (\S+)")
    seen = set()
    for lf in sorted(glob.glob(os.path.expanduser(os.environ.get("FF_RUNS_LOG", "~/work/halo-runs") + "/avg5_gauss*/launcher.log"))):
        for line in open(lf):
            m = pat.search(line)
            if m and m[4] not in seen:
                seen.add(m[4])
                out.append((os.path.join(MM, "runs", "gausshp", m[4]), "averaging seed"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for d, role in fits():
        if not os.path.exists(os.path.join(d, "model.eqx")):
            continue
        c = json.load(open(os.path.join(d, "config.json")))["config"]
        m = json.load(open(os.path.join(d, "metrics.json")))
        w = json.load(open(os.path.join(d, "wandb.json"))) if os.path.exists(os.path.join(d, "wandb.json")) else {}
        rows.append({"exp": EXP[c["preset"]], "dataset_k": c["seed_assign"], "fit_seed": c["seed_fit"], "role": role,
                     "ate_mae": round(m["ate_mae"], 5), "best_epoch": m.get("best_epoch"),
                     "folder": f"{EXP[c['preset']]}/{os.path.basename(d)}", "wandb_url": w.get("url", ""), "_src": d})
    rows.sort(key=lambda r: (r["exp"], r["dataset_k"], r["fit_seed"]))
    for exp in sorted({r["exp"] for r in rows}):
        R = [r for r in rows if r["exp"] == exp]
        zp = os.path.join(a.out, f"fits_{exp}.zip")
        complete = len(R) == 50
        tag = "" if complete else f" (partial: {len(R)} of 50 fits)"
        if os.path.exists(zp) and not a.force:
            print(f"keep {zp}")
            continue
        with zipfile.ZipFile(zp + ".tmp", "w", zipfile.ZIP_DEFLATED) as z:
            for r in R:
                for f in KEEP:
                    p = os.path.join(r["_src"], f)
                    if os.path.exists(p):
                        z.write(p, os.path.join(r["folder"], f))
                for p in glob.glob(os.path.join(r["_src"], "plots", "*")):
                    z.write(p, os.path.join(r["folder"], "plots", os.path.basename(p)))
        os.replace(zp + ".tmp", zp)
        print(f"wrote {zp}: {len(R)} fits{tag}, {os.path.getsize(zp) / 1e6:.0f} MB")
    with open(os.path.join(a.out, "index.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "_src"])
        wr.writeheader()
        wr.writerows([{k: v for k, v in r.items() if k != "_src"} for r in rows])
    head = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "-C", REPO, "branch", "--show-current"], capture_output=True, text=True).stdout.strip()
    subprocess.run(["git", "-C", REPO, "bundle", "create", os.path.join(a.out, "code.bundle"), branch], check=True,
                   capture_output=True)
    print(f"index.csv: {len(rows)} fits; code.bundle: {branch} @ {head}")


if __name__ == "__main__":
    main()
