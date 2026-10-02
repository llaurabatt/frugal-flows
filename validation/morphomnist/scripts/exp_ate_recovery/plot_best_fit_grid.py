"""Best fit per model (lowest ATE MAE among its fits) against the truth, E2 datasets 1-3.
Fits available per dataset: frengression 1, current FF 5, Gaussian spline 3, Gaussian loc. translation 1."""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_ensemble_vs_mean import collect, OUT

ks = [1, 2, 3]
models = [("FR", "Frengression"), ("U-raw", "Frugal flow (current)"), ("G-std", "Gaussian spline"),
          ("G-LT", "Gaussian + loc. translation")]
fig, ax = plt.subplots(2 * len(ks), 1 + len(models), figsize=(3.0 * (1 + len(models)), 5.6 * len(ks)),
                       constrained_layout=True)
for i, k in enumerate(ks):
    truth, arms = collect(k)
    r0, r1 = 2 * i, 2 * i + 1
    im0 = ax[r0, 0].imshow(truth.reshape(8, 8), cmap="viridis", vmin=-0.1, vmax=1.1)
    ax[r0, 0].set_title(f"dataset {k}\ntrue ATE", fontsize=10); ax[r1, 0].axis("off")
    for j, (code, name) in enumerate(models, start=1):
        fits = arms[code]
        maes = {s: np.abs(t - truth).mean() for s, t in fits.items()}
        s = min(maes, key=maes.get); t = fits[s]
        ax[r0, j].imshow(t.reshape(8, 8), cmap="viridis", vmin=-0.1, vmax=1.1)
        ax[r0, j].set_title(f"{name}\nbest of {len(fits)} fit{'s' if len(fits) > 1 else ''} (seed {s})", fontsize=9)
        im1 = ax[r1, j].imshow((t - truth).reshape(8, 8), cmap="RdBu_r", vmin=-0.05, vmax=0.05)
        ax[r1, j].set_title(f"error · ATE MAE {maes[s]:.4f}", fontsize=9)
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
fig.colorbar(im0, ax=ax[0::2, :].ravel().tolist(), shrink=0.3, label="ATE (logit)")
fig.colorbar(im1, ax=ax[1::2, 1:].ravel().tolist(), shrink=0.3, label="estimate − truth")
fig.suptitle("E2 (confounded), all digits 8×8: best fit per model against the truth", fontsize=12)
p = os.path.join(OUT, "gs_best_fit_grid.png"); fig.savefig(p, dpi=115); print(p)
if "--no-wandb" not in sys.argv:
    import wandb
    rid = open(os.path.join(OUT, "wandb_run_id.txt")).read().strip()
    run = wandb.init(entity="proj-lb", project="Frugal Images", id=rid, resume="must")
    run.log({"plots/gs_best_fit_grid": wandb.Image(p)}); run.finish()
