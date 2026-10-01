"""Noise fixes: does weight averaging (ema20) and / or batch 500 make a single fit less variable?

Runs of noise_fixes_8x8.sh: joint fits (ff), lr 0.001, copula width 16, margin 48/8,
hidden_ranks_rule spread_all, E1 and E2, datasets = assignment seeds 1..5, fit seeds
{k, 1001..1004}, versions plain / ema20 / batch500 / batch500+ema20.

For every version and preset, over the 5 datasets:
  * spread between fit seeds: per dataset, the mean over pixels of the standard deviation of
    the error across its 5 fits; then the mean over datasets;
  * median single-fit MAE, and MAE of the 5-fit average (per dataset, then mean);
  * slope of each fit's error map on its dataset's imbalance map (E2: leftover confounding);
  * failures: fits whose MAE exceeds 3x the version's median.
Also the naive, OLS and frengression MAE on the same datasets.
"""
import os

import numpy as np
import pandas as pd

import sys  # noqa: E402
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
import dataset_store as DS  # noqa: E402  (Y / ITE rebuilt when a run did not save them)

MM = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RUNS = os.path.join(MM, "runs", "exp_ate_recovery")
d = pd.read_csv(os.path.join(RUNS, "index.csv"))
b = pd.read_csv(os.path.join(MM, "runs", "baselines", "index.csv"))
d["rule"] = d["hidden_ranks_rule"].fillna("legacy")
# index built before 2026-09-27 has no ema_epochs column; every run in it had no averaging
d["ema_epochs"] = d["ema_epochs"].fillna(0).astype(int) if "ema_epochs" in d else 0
g = d[(d.model == "ff") & (d.rule == "spread_all") & d.preset.isin(["E1", "E2"]) & d.base_shift.isna()
      & (d.lr.astype(float) == 0.001) & (d.copula_nn_width == 16) & (d.nn_width == 48) & (d.rqs_knots == 8)
      & d.seed_assign.isin([1, 2, 3, 4, 5])]
g = g[(g.seed_fit == g.seed_assign) | g.seed_fit.isin([1001, 1002, 1003, 1004])]
g["version"] = (np.where(g.batch_size.astype(int) == 500, "batch500", "batch100")
                + np.where(g.ema_epochs > 0, "+ema20", ""))

rows = []
for (p, v), s in g.groupby(["preset", "version"]):
    spreads, singles, avgs, slopes, maes = [], [], [], [], []
    for k, sk in s.groupby("seed_assign"):
        E, T = [], []
        for r in sk.itertuples():
            a = DS.run_arrays(os.path.join(RUNS, r.run_id))
            X = a["X"][:, 0].astype(bool)
            Y = a["Y"]
            err = a["tau_hat"] - a["ATE"]
            imb = Y[X].mean(0) - Y[~X].mean(0) - a["ATE"]
            E.append(err)
            T.append(a["tau_hat"])
            slopes.append(np.polyfit(imb, err, 1)[0])
            maes.append(np.abs(err).mean())
        E = np.array(E)
        spreads.append(E.std(axis=0, ddof=1).mean() if len(E) > 1 else np.nan)
        singles.append(np.median(np.abs(E).mean(axis=1)))
        avgs.append(np.abs(E.mean(axis=0)).mean())
    med = np.median(maes)
    rows.append({"preset": p, "version": v, "fits": len(s), "datasets": s.seed_assign.nunique(),
                 "spread between fit seeds": np.nanmean(spreads), "median single-fit MAE": np.mean(singles),
                 "MAE of 5-fit average": np.mean(avgs), "slope on imbalance": np.mean(slopes),
                 "failed fits (MAE > 3x median)": int(np.sum(np.array(maes) > 3 * med))})
t = pd.DataFrame(rows)
for p in ("E1", "E2"):
    ds = g[g.preset == p].dataset_id.unique()
    for m in ("naive", "ols", "frengression"):
        t = pd.concat([t, pd.DataFrame([{"preset": p, "version": m,
                                          "median single-fit MAE": b[(b.method == m) & b.dataset_id.isin(ds)].mae_all.mean()}])])
pd.set_option("display.width", 250)
print(t.round(4).to_string(index=False))
