"""One dataset, one fit per model: frengression, current frugal flow, Gaussian-scale location translation.
Optional 4th column (--anchor): the same Gaussian location-translation model with shuffled covariates."""
import os, sys, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_ensemble_vs_mean import collect, OUT, GS, seed_of

k = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--k=")), 1))
anchor = "--anchor" in sys.argv
truth, arms = collect(k)
cols = [("Frengression", arms["FR"][k]), ("Frugal flow (current)", arms["U-raw"][k]),
        ("Frugal flow, Gaussian scale\n+ location translation", arms["G-LT"][k])]
if anchor:
    d = glob.glob(os.path.join(GS, f"*_ff_e2_loctransgauss_sa{k}_*zshuf*"))[0]
    cols.append(("same, covariates shuffled\n(anchor: should fail)", np.load(os.path.join(d, "arrays.npz"))["tau_hat"]))
n = len(cols)
fig, ax = plt.subplots(2, n + 1, figsize=(3.1 * (n + 1), 6.4), constrained_layout=True)
im0 = ax[0, 0].imshow(truth.reshape(8, 8), cmap="viridis", vmin=-0.1, vmax=1.1); ax[0, 0].set_title("true ATE", fontsize=11)
ax[1, 0].axis("off")
for j, (lab, t) in enumerate(cols, start=1):
    ax[0, j].imshow(t.reshape(8, 8), cmap="viridis", vmin=-0.1, vmax=1.1)
    ax[0, j].set_title(f"{lab}\nestimated ATE", fontsize=10)
    im1 = ax[1, j].imshow((t - truth).reshape(8, 8), cmap="RdBu_r", vmin=-0.05, vmax=0.05)
    ax[1, j].set_title(f"error  ·  ATE MAE {np.abs(t - truth).mean():.4f}", fontsize=10)
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
fig.colorbar(im0, ax=ax[0, :], shrink=0.75, label="ATE (logit)")
fig.colorbar(im1, ax=ax[1, 1:], shrink=0.75, label="estimate − truth")
fig.suptitle(f"Experiment 2 (confounded), all digits 8×8, dataset {k}: one fit per model (fit seed {k})", fontsize=12)
p = os.path.join(OUT, f"gs_three_models_k{k}{'_anchor' if anchor else ''}.png"); fig.savefig(p, dpi=130); print(p)
if "--no-wandb" not in sys.argv:
    import wandb
    rid = open(os.path.join(OUT, "wandb_run_id.txt")).read().strip()
    run = wandb.init(entity="proj-lb", project="Frugal Images", id=rid, resume="must")
    run.log({f"plots/{os.path.basename(p)[:-4]}": wandb.Image(p)}); run.finish()
