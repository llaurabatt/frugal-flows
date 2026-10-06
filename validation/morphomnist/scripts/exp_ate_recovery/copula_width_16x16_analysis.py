"""Compare copula widths 16 / 32 / 64 at 16x16 (copula_width_16x16*.sh), 2026-10-05.

    python scripts/exp_ate_recovery/copula_width_16x16_analysis.py [--n 5000]

One row per fit (E2 / E4, dataset 1, fit seeds 1 and 1001): error of the effect map over all pixels, the
signed error averaged over the disc, the ring of pixels bordering it and the rest of the background (same
regions as the paper tables), the leftover slope (error map regressed on the naive estimate's error), how
training ended, the best epoch and the held-out loss there. At n = 60000 the width-16 fits are the grid's.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

MM = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, MM)
import dataset_store as DS  # noqa: E402
import exp_ate_recovery as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()
    tag = f"_n{a.n}" if a.n else ""
    disc, ring, far = E.region_masks(16, E.Config(size=16).effective_radius)
    rows = []
    for p in ("e2", "e4"):
        for w in (16, 32, 64):
            for s in (1, 1001):
                c = sorted(d for d in glob.glob(os.path.join(
                    MM, "runs", "exp_ate_recovery", f"*_ff_{p}_flexgauss_sa1_lr0.001_copw{w}{tag}_ystd_k256_s{s}_d0-9_*"))
                    if os.path.exists(os.path.join(d, "metrics.json")))
                if not c:
                    continue
                d = c[-1]
                data = DS.run_arrays(d, need=("Y",))
                ate = np.asarray(data["ATE"]); T = np.asarray(data["X"])[:, 0].astype(bool); Y = np.asarray(data["Y"])
                imb = Y[T].mean(0) - Y[~T].mean(0) - ate
                err = DS.effect_map(d) - ate
                m = json.load(open(os.path.join(d, "metrics.json")))
                rows.append(dict(preset=p.upper(), width=w, seed=s, n=len(Y), mae=np.abs(err).mean() * 1e3,
                                 disc=err[disc].mean() * 1e3, ring=err[ring].mean() * 1e3, far=err[far].mean() * 1e3,
                                 slope=np.polyfit(imb, err, 1)[0], ended=m.get("termination"),
                                 best_epoch=m.get("best_epoch"), val_loss=m.get("val_loss_at_best")))
    x = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(x.round(3).to_string(index=False))
    print("\nmean over the two fit seeds:")
    print(x.groupby(["preset", "width"])[["mae", "disc", "ring", "far", "slope", "val_loss"]].mean().round(3).to_string())
    out = os.path.join(MM, "runs", "exp_ate_recovery", "analysis", f"copula_width_16x16{tag}.csv")
    x.to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
