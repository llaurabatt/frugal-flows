"""Leftover-confounding fixes (plan step A): copula width 8, copula learning rate x3 / x10,
against the plain version (copula width 16, one learning rate).

Joint fits, lr 0.001, margin 48/8, batch 100, no weight averaging, hidden_ranks_rule
spread_all, E1 and E2, datasets = assignment seeds 1..5, fit seeds {k, 1001..1004}. The FF
estimate for a dataset is the average of its 5 fits. Reported per version and preset, as means
over the 5 datasets (standard error over datasets in brackets):
  * slope of the averaged estimate's error map on the dataset's imbalance map (the share of
    the dataset's confounding left in the estimate; OLS ~0.005 on E2);
  * disc / ring / far signed error of the averaged estimate;
  * whole-image MAE of the averaged estimate, and of a typical single fit;
  * held-out copula loss (median over fits).
Plus OLS, naive and frengression on the same datasets.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

MM = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, MM)
from exp_ate_recovery import region_masks  # noqa: E402
import dataset_store as DS  # noqa: E402  (Y / ITE rebuilt when a run did not save them)

RUNS = os.path.join(MM, "runs", "exp_ate_recovery")
disc, ring, far = region_masks(8, 2)
d = pd.read_csv(os.path.join(RUNS, "index.csv"))
b = pd.read_csv(os.path.join(MM, "runs", "baselines", "index.csv"))
d["rule"] = d["hidden_ranks_rule"].fillna("legacy")
for col, default in (("ema_epochs", 0), ("copula_lr_mult", 1.0)):
    d[col] = pd.to_numeric(d[col], errors="coerce").fillna(default) if col in d else default
g = d[(d.model == "ff") & (d.rule == "spread_all") & d.preset.isin(["E1", "E2"]) & d.base_shift.isna()
      & (d.lr.astype(float) == 0.001) & (d.nn_width == 48) & (d.rqs_knots == 8) & (d.batch_size == 100)
      & (d.ema_epochs == 0) & (d.selected_on.fillna("val_loss") == "val_loss")
      & d.seed_assign.isin([1, 2, 3, 4, 5])]
g = g[(g.seed_fit == g.seed_assign) | g.seed_fit.isin([1001, 1002, 1003, 1004])]
g["version"] = "copw" + g.copula_nn_width.astype(int).astype(str) + np.where(
    g.copula_lr_mult.astype(float) != 1.0, " coplr" + g.copula_lr_mult.astype(float).map(lambda v: f"{v:g}"), "")
se = lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan  # noqa: E731

rows = []
for (p, v), s in g.groupby(["preset", "version"]):
    per = []
    for k, sk in s.groupby("seed_assign"):
        T, singles, vc = [], [], []
        for r in sk.itertuples():
            a = DS.run_arrays(os.path.join(RUNS, r.run_id))
            T.append(a["tau_hat"])
            singles.append(np.abs(a["tau_hat"] - a["ATE"]).mean())
            vc.append(json.load(open(os.path.join(RUNS, r.run_id, "metrics.json"))).get("val_copula_nll"))
        X = a["X"][:, 0].astype(bool)
        Y = a["Y"]
        imb = Y[X].mean(0) - Y[~X].mean(0) - a["ATE"]
        err = np.mean(T, axis=0) - a["ATE"]
        per.append({"n": len(T), "slope": np.polyfit(imb, err, 1)[0], "disc": err[disc].mean(),
                    "ring": err[ring].mean(), "far": err[far].mean(), "mae_avg": np.abs(err).mean(),
                    "mae_single": np.median(singles), "val_cop": np.median([x for x in vc if x is not None])})
    t = pd.DataFrame(per)
    rows.append({"preset": p, "version": v, "datasets": len(t), "fits per dataset": f"{t.n.min()}-{t.n.max()}",
                 **{c: f"{t[c].mean():+.4f} ({se(t[c]):.4f})" for c in ("slope", "disc", "ring", "far")},
                 "MAE of 5-fit average": f"{t.mae_avg.mean():.4f}", "single-fit MAE": f"{t.mae_single.mean():.4f}",
                 "held-out copula loss": f"{t.val_cop.mean():+.3f}"})
out = pd.DataFrame(rows)
for p in ("E1", "E2"):
    ds = g[g.preset == p].dataset_id.unique()
    for m in ("ols", "frengression", "naive"):
        bb = b[(b.method == m) & b.dataset_id.isin(ds)]
        out = pd.concat([out, pd.DataFrame([{"preset": p, "version": m, "datasets": len(bb),
                                             "disc": f"{bb.signed_disc.mean():+.4f}", "ring": f"{bb.signed_ring.mean():+.4f}",
                                             "MAE of 5-fit average": f"{bb.mae_all.mean():.4f}"}])])
pd.set_option("display.width", 260)
print(out.fillna("").to_string(index=False))
