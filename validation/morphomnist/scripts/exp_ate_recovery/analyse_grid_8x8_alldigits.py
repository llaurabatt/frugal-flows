"""Paper grid, 8x8, all ten digits (grid_8x8_alldigits.sh; 2026-10-01).

For each preset E1-E6 and each dataset (assignment seed 1..10), compared on the same data:
  FF avg5      the frugal flow's estimate = average of its 5 fits' effect maps (the method)
  FF single    mean over the 5 single fits (to show what averaging buys)
  freng avg5   frengression, average of its 5 fits;  freng single: mean over its single fits
  OLS          per-pixel regression (runs/baselines)
Per dataset and method: error over all pixels (mean |estimate - true ATE|), signed error on the disc /
ring / background regions, and the leftover slope (slope of the error map on the dataset's confounding
map; meaningful only with confounding, i.e. not E1).
Table: mean over datasets (standard error over datasets).
Criterion (agreed 2026-09-30, "performs as frengression on the ATE"): FF avg5 passes on a preset if its
error is not larger than frengression's (ONE fit, seed = dataset seed, from 2026-10-01) beyond noise: paired mean difference (FF - freng) over
datasets < 2 x its standard error. The ratio of the two errors is reported too.
Writes analysis/grid_8x8_alldigits.md and .csv next to this script.
"""
import collections
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

MM = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, MM)
import dataset_store as DS  # noqa: E402
from exp_ate_recovery import region_masks  # noqa: E402

SHORT = {"exp1_rct_homogeneous": "E1", "exp2_confounded_homogeneous": "E2", "exp3_confounded_heterogeneous": "E3",
         "exp4_covariate_cate": "E4", "exp5_quantile_effect": "E5", "exp6_spatial_cate": "E6"}
OUT = os.path.join(MM, "runs", "exp_ate_recovery", "analysis")   # outputs stay under runs/


def cells(root, model_file):
    """{(preset, dataset): [run_dir, ...]} for finished all-digits 8x8 runs with saved weights."""
    out = collections.defaultdict(list)
    for d in glob.glob(os.path.join(MM, "runs", root, "*_k64_s*_d0-9_*/")):
        need_model = os.environ.get("GRID_TEST_NO_MODEL") != "1"     # test switch: accept weightless runs
        if not (os.path.exists(d + "metrics.json") and (os.path.exists(d + model_file) or not need_model)):
            continue
        c = json.load(open(d + "config.json"))["config"]
        if root == "exp_ate_recovery" and (c.get("model") != "ff" or c.get("arm") != "flexible_continuous"):
            continue
        out[(SHORT[c["preset"]], c["seed_assign"])].append(d)
    return out


def scores(tau, ate, imb, masks):
    err = tau - ate
    disc, ring, far = masks
    return {"mae": np.abs(err).mean(), "disc": err[disc].mean(), "ring": err[ring].mean(), "far": err[far].mean(),
            "slope": np.polyfit(imb, err, 1)[0]}


ff, fr = cells("exp_ate_recovery", "model.eqx"), cells("frengression", "model.pt")
bidx = pd.read_csv(os.path.join(MM, "runs", "baselines", "index.csv"))
rows = []
for key in sorted(set(ff) | set(fr)):
    preset, k = key
    ref = (ff.get(key) or fr.get(key))[0]
    a = DS.run_arrays(ref)
    T = a["X"][:, 0].astype(bool)
    ate = a["ATE"]
    imb = a["Y"][T].mean(0) - a["Y"][~T].mean(0) - ate
    did = json.load(open(ref + "config.json")).get("dataset_id")
    masks = region_masks(8, json.load(open(ref + "config.json")).get("effective_radius") or 2)
    # frengression's estimate is ONE fit, the one with the dataset's own seed (user, 2026-10-01: its fits
    # vary 2-4x less than the flow's and its 5-fit average was only 2-7 % better); extra seeds that
    # finished before that decision are reported separately as "freng avgN"
    own = [r for r in fr.get(key, []) if json.load(open(r + "config.json"))["config"]["seed_fit"] == k]
    if own:
        tau = np.asarray(DS.run_arrays(own[0], need=())["tau_hat"])
        rows.append({"preset": preset, "dataset": k, "method": "freng (seed k)", **scores(tau, ate, imb, masks)})
    for name, runs in (("FF", ff.get(key, [])), ("freng", fr.get(key, []))):
        taus = [np.asarray(DS.run_arrays(r, need=())["tau_hat"]) for r in runs]
        if not taus:
            continue
        rows.append({"preset": preset, "dataset": k, "method": f"{name} avg{len(taus)}", **scores(np.mean(taus, 0), ate, imb, masks)})
        single = pd.DataFrame([scores(t, ate, imb, masks) for t in taus]).mean()
        rows.append({"preset": preset, "dataset": k, "method": f"{name} single", **single.to_dict()})
    b = bidx[(bidx.dataset_id == did) & (bidx.method == "ols")]
    if len(b):
        bd = os.path.join(MM, "runs", "baselines", b.iloc[0].run_id)
        tau = np.load(os.path.join(bd, "arrays.npz"))["tau_hat_ols"]
        rows.append({"preset": preset, "dataset": k, "method": "OLS", **scores(tau, ate, imb, masks)})

x = pd.DataFrame(rows)
if x.empty:
    print("no finished all-digits 8x8 runs with saved weights yet"); sys.exit(0)
x["method"] = x.method.str.replace(r"avg[1-4]$", "avg<5", regex=True)
se = lambda s: s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else np.nan  # noqa: E731
tab = []
for (p, m), s in x.groupby(["preset", "method"]):
    tab.append({"preset": p, "method": m, "datasets": len(s),
                **{c: f"{s[c].mean():+.4f} ({se(s[c]):.4f})" for c in ("mae", "disc", "ring", "far", "slope")}})
tab = pd.DataFrame(tab)
crit = []
for p in sorted(x.preset.unique()):
    a_ = x[(x.preset == p) & (x.method == "FF avg5")].set_index("dataset").mae
    b_ = x[(x.preset == p) & (x.method == "freng (seed k)")].set_index("dataset").mae
    both = a_.index.intersection(b_.index)
    if len(both) < 2:
        crit.append({"preset": p, "paired datasets": len(both), "verdict": "not enough data"}); continue
    diff = a_[both] - b_[both]
    ok = diff.mean() < 2 * se(diff)
    crit.append({"preset": p, "paired datasets": len(both), "FF avg5 error": f"{a_[both].mean():.4f}",
                 "freng (seed k) error": f"{b_[both].mean():.4f}", "ratio": f"{a_[both].mean() / b_[both].mean():.2f}",
                 "diff (se)": f"{diff.mean():+.4f} ({se(diff):.4f})", "verdict": "PASS" if ok else "FAIL"})
crit = pd.DataFrame(crit)
os.makedirs(OUT, exist_ok=True)
x.to_csv(os.path.join(OUT, "grid_8x8_alldigits.csv"), index=False)
with open(os.path.join(OUT, "grid_8x8_alldigits.md"), "w") as f:
    f.write("# Paper grid, 8x8, all ten digits\n\nGenerated by analyse_grid_8x8_alldigits.py. See its docstring for the measures.\n\n")
    f.write("## Criterion: FF avg5 vs frengression, one fit with the dataset's seed (paired over datasets)\n\n" + crit.to_markdown(index=False) + "\n\n")
    f.write("## All methods (mean over datasets, standard error in brackets)\n\n" + tab.to_markdown(index=False) + "\n")
pd.set_option("display.width", 250)
print(crit.to_string(index=False)); print(); print(tab.to_string(index=False))
print(f"\nwritten: {os.path.join(OUT, 'grid_8x8_alldigits.md')}")
