"""Leftover-confounding fixes, 2026-09-28 (docs/leftover_confounding/README.md).

Versions, all at fit seed 2001 on datasets (assignment seeds) 1..5, plain setting otherwise
(joint ff, lr 0.001, copula width 16, margin 48/8, batch 100, no averaging, patience 30):
  plain    umarg_check_8x8.sh
  ecdf     covariate ranks from the empirical CDF            (ecdf_8x8.sh)
  umw<w>   copula u-marginal penalty with weight w          (umpen_8x8.sh)
Same data and keys across versions, so each dataset's fits are paired.

Per version and preset, means over the 5 single fits (standard error over them):
  slope   slope of the error map on the dataset's confounding map (naive difference minus
          truth): the share of the confounding left in the estimate (OLS ~0.005 on E2);
  disc    mean signed error over the disc pixels;  mae  mean absolute error over all pixels;
  KS u    two-sample KS between the copula's own u-marginal (image ranks uniform from the
          base) and the observed covariate ranks; ~0.02 is the noise level here;
  val NLL held-out negative log likelihood per image (validation loss at the kept epoch);
  epoch   the kept epoch.
Plus OLS and frengression on the same datasets (single fits in the baselines index).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

MM = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, MM)
from exp_ate_recovery import region_masks  # noqa: E402
import dataset_store as DS  # noqa: E402  (Y / ITE rebuilt when a run did not save them)

RUNS = os.path.join(MM, "runs", "exp_ate_recovery")
disc, ring, far = region_masks(8, 2)
d = pd.read_csv(os.path.join(RUNS, "index.csv"))
num = lambda c, default: (pd.to_numeric(d[c], errors="coerce").fillna(default) if c in d  # noqa: E731
                          else pd.Series(default, index=d.index))
if "u_z_method" not in d:
    d["u_z_method"] = "flow"
g = d[(d.model == "ff") & (d.hidden_ranks_rule == "spread_all") & d.preset.isin(["E1", "E2"])
      & (d.seed_fit == 2001) & d.seed_assign.isin([1, 2, 3, 4, 5]) & (d.lr.astype(float) == 0.001)
      & (d.copula_nn_width == 16) & (d.nn_width == 48) & (d.batch_size == 100) & (num("ema_epochs", 0) == 0)
      & (num("copula_lr_mult", 1) == 1) & (num("max_patience", 30) == 30) & d.base_shift.isna()].copy()
g["version"] = np.where(d.loc[g.index, "u_z_method"].fillna("flow") == "ecdf", "ecdf", "plain")
w = num("copula_umarg_weight", 0).loc[g.index]
for i in w[w > 0].index:
    g.loc[i, "version"] = f"umw{w[i]:g}"

rows = []
for r in g.itertuples():
    rd = os.path.join(RUNS, r.run_id)
    if not os.path.exists(os.path.join(rd, "arrays.npz")):
        continue
    a = DS.run_arrays(rd)
    m = json.load(open(os.path.join(rd, "metrics.json")))
    T = a["X"][:, 0].astype(bool)
    imb = a["Y"][T].mean(0) - a["Y"][~T].mean(0) - a["ATE"]
    err = a["tau_hat"] - a["ATE"]
    ks = stats.ks_2samp(a["cop_u_marg"][:, 0], a["u_z"][:, 0]).statistic if "cop_u_marg" in a.files else np.nan
    rows.append({"preset": r.preset, "version": r.version, "dataset": r.seed_assign,
                 "slope": np.polyfit(imb, err, 1)[0], "disc": err[disc].mean(), "mae": np.abs(err).mean(),
                 "KS u": ks, "val NLL": m.get("val_loss_at_best", m.get("best_val_loss")),
                 "epoch": m.get("best_epoch")})
x = pd.DataFrame(rows)
se = lambda s: s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else np.nan  # noqa: E731
order = {"plain": 0, "ecdf": 1}
out = []
for (p, v), s in x.groupby(["preset", "version"]):
    out.append({"preset": p, "version": v, "fits": len(s),
                **{c: f"{s[c].mean():+.4f} ({se(s[c]):.4f})" for c in ("slope", "disc")},
                "mae": f"{s.mae.mean():.4f}", "KS u": f"{s['KS u'].mean():.4f}",
                "val NLL": f"{s['val NLL'].mean():.3f}", "epoch": f"{s.epoch.mean():.0f}",
                "_o": order.get(v, 2 + float(v[3:]) if v.startswith("umw") else 9)})
out = pd.DataFrame(out).sort_values(["preset", "_o"]).drop(columns="_o")
b = pd.read_csv(os.path.join(MM, "runs", "baselines", "index.csv"))
for p in ("E1", "E2"):
    ds = g[g.preset == p].dataset_id.unique()
    for meth in ("ols", "frengression"):
        bb = b[(b.method == meth) & b.dataset_id.isin(ds)]
        out = pd.concat([out, pd.DataFrame([{"preset": p, "version": meth, "fits": len(bb),
                                             "disc": f"{bb.signed_disc.mean():+.4f}", "mae": f"{bb.mae_all.mean():.4f}"}])])
pd.set_option("display.width", 220)
print(out.fillna("").to_string(index=False))
print("\nE2 slope per dataset:")
print(x[x.preset == "E2"].pivot(index="dataset", columns="version", values="slope").round(3).to_string())
