"""Fit-seed control: is the margin's own error tied to the dataset or to training randomness?

For E1 datasets (assignment seeds 1..5), 5 margin-only fits each that differ only in the fit
seed (k and 1001..1004), all at lr 0.001, width 48, 8 knots, hidden_ranks_rule spread_all.
For every fit: the error map (estimate minus truth), the dataset's imbalance (observed
treated-minus-untreated mean image minus truth), and the flow's own part (error minus
imbalance = its two sampled arm means minus the two arms' actual mean images, differenced).

Reported:
  * per dataset: MAE of each fit, and MAE of the average of its fits' estimates;
  * correlation of own-error maps between two fits on the SAME dataset (different fit seeds)
    against the correlation between two fits on DIFFERENT datasets;
  * a variance split of the own-error map per pixel: between datasets vs between fit seeds
    within a dataset.
If fits on the same dataset agree (high same-dataset correlation, averaging barely helps),
the own error is tied to the data. If they disagree, it is training randomness.
"""
import itertools
import os

import numpy as np
import pandas as pd

import sys  # noqa: E402
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
import dataset_store as DS  # noqa: E402  (Y / ITE rebuilt when a run did not save them)

MM = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RUNS = os.path.join(MM, "runs", "exp_ate_recovery")
d = pd.read_csv(os.path.join(RUNS, "index.csv"))
d["rule"] = d["hidden_ranks_rule"].fillna("legacy")
g = d[(d.preset == "E1") & (d.model == "margin") & (d.rule == "spread_all") & d.base_shift.isna()
      & (d.lr.astype(float) == 0.001) & d.seed_assign.isin([1, 2, 3, 4, 5])]
g = g[(g.seed_fit == g.seed_assign) | g.seed_fit.isin([1001, 1002, 1003, 1004])]

fits = []
for r in g.itertuples():
    a = DS.run_arrays(os.path.join(RUNS, r.run_id))
    X = a["X"][:, 0].astype(bool)
    Y = a["Y"]
    err = a["tau_hat"] - a["ATE"]
    imb = Y[X].mean(0) - Y[~X].mean(0) - a["ATE"]
    fits.append({"k": int(r.seed_assign), "fs": int(r.seed_fit), "err": err, "own": err - imb,
                 "tau": a["tau_hat"], "ate": a["ATE"], "imb": imb})

print(f"fits: {len(fits)}; per dataset: {pd.Series([f['k'] for f in fits]).value_counts().sort_index().to_dict()}")
rows = []
for k in sorted({f["k"] for f in fits}):
    F = sorted([f for f in fits if f["k"] == k], key=lambda f: f["fs"])
    ens = np.mean([f["tau"] for f in F], axis=0) - F[0]["ate"]
    rows.append({"dataset (assignment seed)": k, "n fits": len(F),
                 "MAE per fit (fit seeds k, 1001-1004)": " ".join(f"{np.abs(f['err']).mean():.4f}" for f in F),
                 "median single-fit MAE": np.median([np.abs(f["err"]).mean() for f in F]),
                 "MAE of the average of the fits": np.abs(ens).mean(),
                 "imbalance MAE (naive estimator)": np.abs(F[0]["imb"]).mean(),
                 "own-error MAE, median": np.median([np.abs(f["own"]).mean() for f in F])})
t = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(t.round(4).to_string(index=False))

same = [np.corrcoef(a["own"], b["own"])[0, 1] for a, b in itertools.combinations(fits, 2) if a["k"] == b["k"]]
diff = [np.corrcoef(a["own"], b["own"])[0, 1] for a, b in itertools.combinations(fits, 2) if a["k"] != b["k"]]
print(f"\ncorrelation of own-error maps: same dataset, different fit seed: mean {np.mean(same):+.2f} "
      f"(n pairs {len(same)}); different datasets: mean {np.mean(diff):+.2f} (n pairs {len(diff)})")

# variance split per pixel of the own error: between datasets vs within (between fit seeds)
O = {k: np.array([f["own"] for f in fits if f["k"] == k]) for k in sorted({f["k"] for f in fits})}
within = np.mean([v.var(axis=0, ddof=1).mean() for v in O.values()])
means = np.array([v.mean(axis=0) for v in O.values()])
between = means.var(axis=0, ddof=1).mean()
n = np.mean([len(v) for v in O.values()])
between_true = max(between - within / n, 0.0)          # remove the within-part's contribution to dataset means
print(f"own-error variance per pixel: between fit seeds on one dataset {within:.6f}; "
      f"between datasets (corrected) {between_true:.6f}; share tied to the dataset "
      f"{between_true / (between_true + within):.2f}")
