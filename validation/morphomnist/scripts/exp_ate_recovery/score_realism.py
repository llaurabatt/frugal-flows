"""Aggregate realism.json files (realism.py) for the sub5k fits: arm x metric per experiment, mean over
datasets k, paired ratio Gaussian / uniform with wins. Writes ~/work/halo-runs/realism/results.md|json."""
from __future__ import annotations

import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MM = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.expanduser(os.environ.get("FF_RUNS_LOG", "~/work/halo-runs") + "/realism")
EXP = {"exp1_rct_homogeneous": "E1", "exp2_confounded_homogeneous": "E2", "exp3_confounded_heterogeneous": "E3",
       "exp4_covariate_cate": "E4", "exp5_quantile_effect": "E5", "exp6_spatial_cate": "E6"}
CODE = {("flexible_continuous", "none"): "U-flex-raw", ("flexible_continuous", "standardize"): "U-flex-std",
        ("flexible_continuous_gaussian", "standardize"): "G-flex-std"}
METRICS = [("fid_cf", "FID CF (paired; 0 = perfect)"), ("fid_int0", "FID do(0)"), ("fid_int1", "FID do(1)"),
           ("cf_mae_logit", "CF MAE logit"), ("cf_mae_pixel", "CF MAE pixel"), ("cf_auc", "CF real-vs-gen AUC")]
REF = [("fid_factual_vs_cf_true", "FID do-nothing (factual vs true CF)"), ("fid_floor_fulln_est", "FID floor est. (n=5000)"),
       ("cf_mae_logit_donothing", "CF MAE logit do-nothing")]


def main():
    recs = []
    for p in glob.glob(os.path.join(MM, "runs", "sub5k", "*", "realism.json")):
        r = json.load(open(p))
        r["exp"], r["code"] = EXP[r["preset"]], CODE.get((r["arm"], r["y_scaling"]))
        recs.append(r)
    os.makedirs(OUT, exist_ok=True)
    json.dump(recs, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    L = ["# Counterfactual realism, n = 5000, 8x8 (benchmark-style FID + unit-level CF error)", "",
         "Mean over datasets k. Ratio = G-flex-std / U-flex-raw (paired by k), wins = G better. "
         "Lower is better for all metrics; AUC 0.5 = indistinguishable.", ""]
    for exp in sorted({r["exp"] for r in recs}):
        R = [r for r in recs if r["exp"] == exp]
        codes = [c for c in ("G-flex-std", "U-flex-raw", "U-flex-std") if any(r["code"] == c for r in R)]
        by = {c: {r["k"]: r for r in R if r["code"] == c} for c in codes}
        L += [f"## {exp}", "", "| metric | " + " | ".join(f"{c} (n={len(by[c])})" for c in codes) + " | G/U-raw ratio (wins) |",
              "|---|" + "---|" * (len(codes) + 1)]
        for key, lab in METRICS:
            cells = [f"{np.mean([v[key] for v in by[c].values()]):.4f}" for c in codes]
            ks = sorted(set(by.get("G-flex-std", {})) & set(by.get("U-flex-raw", {})))
            rat = "—"
            if ks:
                g = np.array([by["G-flex-std"][k][key] for k in ks]); u = np.array([by["U-flex-raw"][k][key] for k in ks])
                rat = f"{g.mean() / u.mean():.2f} ({int((g < u).sum())}/{len(ks)})"
            L.append(f"| {lab} | " + " | ".join(cells) + f" | {rat} |")
        L += ["", "Reference scales: " + "; ".join(f"{lab} {np.mean([r[key] for r in R]):.4f}" for key, lab in REF), ""]
    md = "\n".join(L)
    open(os.path.join(OUT, "results.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
