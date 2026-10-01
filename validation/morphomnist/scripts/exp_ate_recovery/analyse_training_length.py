"""Does the E2 leftover confounding shrink with longer training? (training_length_8x8.sh)

For each ep600_pat600 fit, from metrics["track"] (effect read-out every 20 epochs) and
arrays["loss_val"] (validation loss every epoch):
  * slope = slope of the estimate's error map on the dataset's confounding map (naive
    difference minus truth), i.e. the share of the confounding still in the estimate;
  * disc = mean signed error over the disc pixels; mae = mean absolute error over all pixels;
reported at three points: the epoch patience-30 early stopping would have kept (the lowest
validation loss before 30 epochs pass without a new one), the epoch with the lowest
validation loss over all 600, and epoch 600. The read-out nearest each epoch is used (the
read-outs are 20 epochs apart). For comparison, the plain fit of the same dataset and fit
seed (patience 30, otherwise identical) with its final estimate.
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
d = d[(d.model == "ff") & (d.hidden_ranks_rule == "spread_all") & d.preset.isin(["E1", "E2"])
      & (d.lr.astype(float) == 0.001) & (d.copula_nn_width == 16) & (d.nn_width == 48) & (d.rqs_knots == 8)
      & (d.batch_size == 100) & (pd.to_numeric(d.ema_epochs, errors="coerce").fillna(0) == 0)
      & (pd.to_numeric(d.copula_lr_mult, errors="coerce").fillna(1) == 1) & d.base_shift.isna()
      & (d.selected_on.fillna("val_loss") == "val_loss") & d.seed_assign.isin([1, 2, 3, 4, 5])
      & (d.seed_fit == d.seed_assign)]
long = d[d.max_patience == 600]
plain = d[d.max_patience == 30]


def final(run_id):
    a = DS.run_arrays(os.path.join(RUNS, run_id))
    T = a["X"][:, 0].astype(bool)
    imb = a["Y"][T].mean(0) - a["Y"][~T].mean(0) - a["ATE"]
    err = a["tau_hat"] - a["ATE"]
    return {"slope": np.polyfit(imb, err, 1)[0], "disc": err[disc].mean(), "mae": np.abs(err).mean()}


rows, curves = [], {}
for r in long.itertuples():
    m = json.load(open(os.path.join(RUNS, r.run_id, "metrics.json")))
    val = DS.run_arrays(os.path.join(RUNS, r.run_id))["loss_val"]
    tr = {t["epoch"]: t for t in m["track"]}
    eps = np.array(sorted(tr))
    best, stop = 0, len(val) - 1
    for e in range(len(val)):
        if val[e] <= val[best]:
            best = e
        if e - best > 30:
            stop = e
            break
    at = lambda e: tr[int(eps[np.argmin(np.abs(eps - e))])]  # noqa: E731
    points = {"early stop keeps": best + 1, "lowest val loss": int(np.argmin(val)) + 1, "epoch 600": 600}
    p = plain[(plain.preset == r.preset) & (plain.seed_assign == r.seed_assign)]
    ref = final(p.run_id.iloc[0]) if len(p) else None
    row = {"preset": r.preset, "dataset": r.seed_assign}
    for lab, e in points.items():
        row[f"{lab}: epoch"] = e
        row[f"{lab}: slope"] = at(e)["slope"]
        row[f"{lab}: disc"] = at(e)["signed_disc"]
        row[f"{lab}: mae"] = at(e)["ate_mae"]
    if ref:
        row.update({"plain fit: slope": ref["slope"], "plain fit: disc": ref["disc"], "plain fit: mae": ref["mae"]})
    rows.append(row)
    curves[(r.preset, r.seed_assign)] = [(e, tr[e]["slope"], tr[e]["signed_disc"]) for e in eps]

out = pd.DataFrame(rows).sort_values(["preset", "dataset"])
pd.set_option("display.width", 250)
for p in ("E2", "E1"):
    o = out[out.preset == p]
    print(f"\n{p}: per dataset")
    print(o.drop(columns="preset").round(4).to_string(index=False))
    print(f"{p}: mean over datasets")
    print(o.drop(columns=["preset", "dataset"]).mean().round(4).to_string())
print("\nslope over training (every 100 epochs), E2:")
for (p, k), c in sorted(curves.items()):
    if p == "E2":
        print(f"  dataset {k}: " + " ".join(f"{e}:{s:+.3f}" for e, s, _ in c if e in (20, 100, 200, 300, 400, 500, 600)))
