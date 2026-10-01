"""Compare the versions of the same grid cell that differ only in how the copula was built
or stopped: legacy numbering + joint stopping (the 2026-09-21 grid), spread numbering +
joint stopping (ranks fix, 2026-09-25), spread numbering + copula stopping (copsel).
Pairs on dataset_id and seed_fit. Prints per-fit rows and per-preset means, with the
held-out and training copula loss and the kept epoch.

    python scripts/exp_ate_recovery/compare_copula_versions.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

MM = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RUNS = os.path.join(MM, "runs", "exp_ate_recovery")
sys.path.insert(0, MM)
from exp_ate_recovery import region_masks  # noqa: E402

disc, ring, far = region_masks(8, 2)
d = pd.read_csv(os.path.join(RUNS, "index.csv"))
d["rule"] = d["hidden_ranks_rule"].fillna("legacy") if "hidden_ranks_rule" in d else "legacy"
d["sel"] = d["selected_on"].fillna("val_loss") if "selected_on" in d else "val_loss"
d = d[d.rule != "spread_all"]   # the 2026-09-26 margin fix is compared in its own script
d["version"] = np.where(d.rule == "legacy", "legacy+joint",
                        np.where(d.sel == "select_fn", "spread+copula", "spread+joint"))
d["version"] = np.where((d.lr.astype(float) != 0.01) & (d.rule != "legacy"),
                        d["version"] + "+lr" + d.lr.astype(float).map(lambda v: f"{v:g}"), d["version"])
g = d[d.seed_assign.notna() & d.base_shift.isna() & (d.model == "ff")]
cells = g[g.version != "legacy+joint"][["dataset_id", "seed_fit"]].drop_duplicates()
g = g.merge(cells, on=["dataset_id", "seed_fit"])

rows = []
for r in g.itertuples():
    m = json.load(open(os.path.join(RUNS, r.run_id, "metrics.json")))
    vc = m.get("val_copula_nll")
    tc = m.get("train_copula_nll")
    if tc is None and vc is not None:
        tc = (m["all_copula_nll"] - 0.1 * vc) / 0.9
    rows.append({"preset": r.preset, "seed": int(r.seed_fit), "version": r.version,
                 "best_epoch": r.best_epoch, "epochs": r.epochs_run, "mae_all": r.mae_all,
                 "disc": r.signed_disc, "ring": r.signed_ring, "far": r.signed_far,
                 "train_cop": tc, "val_cop": vc, "ks_v": m.get("cop_ks_v_thickness")})
t = pd.DataFrame(rows).sort_values(["preset", "seed", "version"])
pd.set_option("display.width", 250)
print(t.round(4).to_string(index=False))
print()
order = ["legacy+joint", "spread+joint", "spread+copula", "spread+joint+lr0.001"]
s = t.groupby(["preset", "version"])[["best_epoch", "mae_all", "disc", "ring", "train_cop", "val_cop", "ks_v"]]
print(s.mean().reindex(order, level=1).round(4).to_string())
print("\nn fits per cell:", t.groupby(["preset", "version"]).size().to_dict())
