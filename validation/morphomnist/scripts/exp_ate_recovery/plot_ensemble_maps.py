"""Pixel maps: single fits vs averaged maps, E2 all digits, datasets 1-3 (companion to plot_ensemble_vs_mean.py)."""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_ensemble_vs_mean import collect, OUT

ks = [1, 2, 3]
data = {k: collect(k) for k in ks}
rows = [("Frengression, 1 fit", "FR", "single"), ("Laura's flow, fit seed k", "U-raw", "single"),
        ("Laura's flow, average of 5 fits", "U-raw", "avg"), ("Gaussian spline, fit seed k", "G-std", "single"),
        ("Gaussian spline, average of 3 fits", "G-std", "avg"), ("Gaussian loc. translation, 1 fit", "G-LT", "single")]
fig, axes = plt.subplots(len(rows), len(ks), figsize=(3.0 * len(ks) + 1.5, 2.6 * len(rows)), constrained_layout=True)
for i, (label, code, kind) in enumerate(rows):
    for j, k in enumerate(ks):
        truth, arms = data[k]; taus = arms[code]
        est = taus[k] if kind == "single" else np.mean(list(taus.values()), 0)
        a = axes[i, j]
        im = a.imshow((est - truth).reshape(8, 8), cmap="RdBu_r", vmin=-0.05, vmax=0.05)
        a.set_xticks([]); a.set_yticks([])
        a.set_title((f"dataset {k}\n" if i == 0 else "") + f"MAE {np.abs(est - truth).mean():.4f}", fontsize=10)
        if j == 0:
            a.set_ylabel(label.replace(", ", "\n"), fontsize=10, rotation=0, ha="right", va="center", labelpad=10)
fig.colorbar(im, ax=axes, shrink=0.4, label="estimated ATE − true ATE (logit)")
fig.suptitle("E2, all digits 8×8: per-pixel error of the ATE estimate\nsingle fits vs averaged maps (true effect = +1 on the disc, 0 elsewhere)")
p = os.path.join(OUT, "gs_ensemble_maps.png"); fig.savefig(p, dpi=120, bbox_inches="tight"); print(p)
if "--no-wandb" not in sys.argv:
    import wandb
    rid = open(os.path.join(OUT, "wandb_run_id.txt")).read().strip()
    run = wandb.init(entity="proj-lb", project="Frugal Images", id=rid, resume="must")
    run.log({"plots/gs_ensemble_maps": wandb.Image(p)}); run.finish(); print("logged to", rid)
