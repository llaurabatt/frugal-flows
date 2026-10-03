"""Provenance map: every n = 5000 fit behind the sub5k / gausshp / averaging results -> its run folder and W&B run.

One row per finished run (metrics.json present): experiment, dataset k, method/setting, fit seed, ATE MAE, W&B group,
W&B URL, run folder. Writes ~/work/halo-runs/WANDB_MAP.csv and WANDB_MAP.md (the md also lists, per reported table,
which rows produce each number and how to filter them in W&B).
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
MM = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.expanduser(os.environ.get("FF_RUNS_LOG", "~/work/halo-runs"))
EXP = {"exp1_rct_homogeneous": "E1", "exp2_confounded_homogeneous": "E2", "exp3_confounded_heterogeneous": "E3",
       "exp4_covariate_cate": "E4", "exp5_quantile_effect": "E5", "exp6_spatial_cate": "E6"}
ARM = {("flexible_continuous", "none", "zero"): "U-flex-raw", ("location_translation", "none", "zero"): "U-LT-raw",
       ("flexible_continuous", "standardize", "zero"): "U-flex-std",
       ("flexible_continuous_gaussian", "standardize", "zero"): "G-flex-std",
       ("location_translation_gaussian", "standardize", "scalar"): "G-LT-std",
       ("location_translation_gaussian", "standardize", "naive"): "G-LT-head"}
TAGS = re.compile(r"_(lr[0-9.e-]+|copw\d+|pat\d+|mw\d+|mkn\d+|batch\d+|fl\d+|md\d+|cfl\d+|cmd\d+|ckn\d+|ecdf)(?=_)")


def rows():
    out = []
    for root, kind in (("sub5k", "flow"), ("gausshp", "flow"), ("sub5k_frengression", "freng")):
        for d in sorted(glob.glob(os.path.join(MM, "runs", root, "*"))):
            mp, cp, wp = (os.path.join(d, f) for f in ("metrics.json", "config.json", "wandb.json"))
            if not os.path.exists(mp):
                continue
            m, c = json.load(open(mp)), json.load(open(cp))
            c = c.get("config", c)
            if c.get("n") != 5000:
                continue
            w = json.load(open(wp)) if os.path.exists(wp) else {}
            name = os.path.basename(d)
            if kind == "freng":
                method = "frengression"
            else:
                method = ARM.get((c.get("arm"), c.get("y_scaling", "none"), c.get("shift_init", "zero")), c.get("arm"))
                extra = [t for t in TAGS.findall(name) if t not in ("lr0.001", "copw16")]
                if extra:
                    method += " [" + ",".join(extra) + "]"
            k, s = int(c["seed_assign"]), int(c["seed_fit"])
            out.append({"exp": EXP[c["preset"]], "k": k, "method": method, "fit_seed": s,
                        "role": "single (seed k)" if s == k else "extra seed (averaging)",
                        "ate_mae": round(m["ate_mae"], 5), "wandb_group": w.get("group", ""),
                        "wandb_url": w.get("url", ""), "run_folder": os.path.relpath(d, MM)})
    return out


def main():
    R = rows()
    keys = list(R[0])
    with open(os.path.join(OUT, "WANDB_MAP.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(sorted(R, key=lambda r: (r["exp"], r["method"], r["k"], r["fit_seed"])))
    groups = {}
    for r in R:
        groups.setdefault((r["wandb_group"], r["method"], r["role"]), 0)
        groups[(r["wandb_group"], r["method"], r["role"])] += 1
    lines = ["# Where every n = 5000 number comes from (W&B proj-lb / Frugal Images)", "",
             f"{len(R)} finished runs. Full row-level map: `~/work/halo-runs/WANDB_MAP.csv`.",
             "Columns: exp, k (dataset = assignment seed), method, fit_seed, role, ate_mae, wandb_group, wandb_url, run_folder.",
             "", "## How each reported table is built", "",
             "| Table | Rows used | W&B filter |", "|---|---|---|",
             "| Phase 1 grid (E1–E6, one fit per dataset) | role = single (seed k), methods without [..] | group `sub5k_e2e4` |",
             "| Sweep round 1 | k = 11–13 (+ k = 1–10 confirmation), G-flex-std [setting] | group `gausshp_sub5k`, tag = setting |",
             "| Sweep round 2 | k = 11–15, G-flex-std [copula settings / ecdf] (+ ckn4 on k = 1–10) | group `gausshp_sub5k` |",
             "| Gaussian 5-fit average | G-flex-std, fit seeds k and 1001–1004 (k from sub5k, the rest from gausshp) | groups `sub5k_e2e4` + `avg5_gauss_sub5k` |",
             "| Frengression 5-fit average | frengression, fit seeds k and 1001–1004, k = 1–5 | group `sub5k_e2e4`, tag `freng` |",
             "", "Each ATE MAE in the map is the run's own `metrics.json` `ate_mae`, the same value logged to W&B. "
             "An averaged-map number is NOT a single run: it is the MAE of the mean of the 5 runs' tau_hat maps "
             "(recompute with `score_avg5.py`).", "",
             "## Run counts", "", "| W&B group | method | role | runs |", "|---|---|---|---|"]
    lines += [f"| {g} | {m} | {r} | {n} |" for (g, m, r), n in sorted(groups.items())]
    open(os.path.join(OUT, "WANDB_MAP.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:20]))
    print(f"... {len(R)} rows -> {OUT}/WANDB_MAP.csv")


if __name__ == "__main__":
    main()
