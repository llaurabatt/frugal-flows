"""Analysis of the assignment-bootstrap grid at 8x8 (see assignment_grid_8x8.sh).

Reads the two indexes and each run's arrays.npz; writes into runs/exp_ate_recovery/analysis/:
  mean_sd_error_maps.png      per configuration (preset x effect x model): the signed error
                              map averaged over the N replicates, and its standard deviation
  paired_effect_minus_zero.png per preset x model: error map with the effect minus error map
                              without it, same replicate (same images, same assignment),
                              averaged over the N pairs   (N = replicates found in the index)
  ff_vs_ols_mae.csv / .md      per replicate: FF MAE, OLS MAE on the same dataset, difference
"""
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MM = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, MM)
from exp_ate_recovery import region_masks  # noqa: E402
import dataset_store as DS  # noqa: E402  (Y / ITE rebuilt when a run did not save them)

RUNS = os.path.join(MM, "runs", "exp_ate_recovery")
OUT = os.path.join(MM, "runs", "exp_ate_recovery", "analysis")   # outputs stay under runs/
os.makedirs(OUT, exist_ok=True)
S = 8
SQUARE = "--square" in sys.argv
if SQUARE:
    # inner centre: rows 3-4, cols 3-4 (4 px); rim: rows 2-5, cols 2-5 minus the centre (12 px);
    # outside: everything else (48 px)
    rr, cc = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
    centre = (rr >= 3) & (rr <= 4) & (cc >= 3) & (cc <= 4)
    square = (rr >= 2) & (rr <= 5) & (cc >= 2) & (cc <= 5)
    REG = {"centre": centre, "rim": square & ~centre, "outside": ~square}
    OUTLINES = [centre, square]
    SUFFIX = "_square"
else:
    d, r, f = region_masks(S, 2)
    REG = {"disc": d.reshape(S, S), "ring": r.reshape(S, S), "far": f.reshape(S, S)}
    OUTLINES = [REG["disc"]]
    SUFFIX = ""
NAMES = list(REG)
disc2d = REG[NAMES[0]]


def region_line(m):
    return "  ".join(f"{k} {m[v].mean():+.3f}" for k, v in REG.items())

ff = pd.read_csv(os.path.join(RUNS, "index.csv"))
bl = pd.read_csv(os.path.join(MM, "runs", "baselines", "index.csv"))
# the 2026-09-21 grid used the legacy hidden-rank rule; the 2026-09-25 rank-fix refits of
# some of its cells share their names apart from the uid, so select on the rule
RULE = "spread" if "--spread" in sys.argv else "legacy"
g = ff[ff.seed_assign.notna() & (ff.get("hidden_ranks_rule", "legacy") == RULE)].copy()
g["effect"] = g.base_shift.fillna(1.0)
g["k"] = g.seed_assign.astype(int)


def err_map(run_id):
    a = DS.run_arrays(os.path.join(RUNS, run_id))
    return (np.asarray(a["tau_hat"]) - np.asarray(a["ATE"])).reshape(S, S)


def panel(ax, m, lim, title, cmap="RdBu_r"):
    im = ax.imshow(m, cmap=cmap, vmin=-lim if cmap == "RdBu_r" else 0, vmax=lim, interpolation="nearest")
    for o in OUTLINES:
        ax.contour(o, levels=[0.5], colors="k", linewidths=0.9)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


# With the effect switched off, E2/E3/E5 build the same dataset (same images, assignment and
# Z = thickness) and E4/E6 the same dataset with Z = thickness + brightness. The E3/E5/E6
# zero-effect fits were byte-identical to E2's / E4's and were deleted on 2026-09-20, so the
# zero-effect reference of a preset is the run of the preset that still holds it.
ZERO_SOURCE = {"E1": "E1", "E2": "E2", "E3": "E2", "E4": "E4", "E5": "E2", "E6": "E4"}

# ---------------------------------------------------------------- 1. mean and sd maps
configs = [(p, e, m) for p in ["E1", "E2", "E3", "E4", "E5", "E6"] for e in (1.0, 0.0)
           for m in (["ff", "margin"] if p == "E1" else ["ff"])
           if e == 1.0 or ZERO_SOURCE[p] == p]
stacks = {}
N = int(g.groupby(["preset", "effect", "model"]).size().max())   # replicates per cell
for p, e, m in configs:
    rows = g[(g.preset == p) & (g.effect == e) & (g.model == m)].sort_values("k")
    assert len(rows) == N, (p, e, m, len(rows), N)
    stacks[(p, e, m)] = np.stack([err_map(r) for r in rows.run_id])   # (N, 8, 8)
means = {c: v.mean(0) for c, v in stacks.items()}
sds = {c: v.std(0, ddof=1) for c, v in stacks.items()}
lim_mean = max(np.abs(v).max() for v in means.values())
lim_sd = max(v.max() for v in sds.values())
n = len(configs)
fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 6.2))
for j, c in enumerate(configs):
    p, e, m = c
    lab = f"{p} {m}\neffect {e:g}"
    im0 = panel(axes[0, j], means[c], lim_mean, lab)
    axes[0, j].text(0.0, -0.06, region_line(means[c]).replace("  ", "\n"), transform=axes[0, j].transAxes,
                    va="top", fontsize=7.5, family="monospace")
    im1 = panel(axes[1, j], sds[c], lim_sd, "", cmap="viridis")
axes[0, 0].set_ylabel(f"mean signed error\nover {N} replicates", fontsize=9)
axes[1, 0].set_ylabel(f"sd of signed error\nover {N} replicates", fontsize=9)
fig.colorbar(im0, ax=axes[0, :].tolist(), shrink=0.8, pad=0.01)
fig.colorbar(im1, ax=axes[1, :].tolist(), shrink=0.8, pad=0.01)
fig.suptitle(f"Assignment grid 8x8: estimated minus true effect, per pixel. Row 1: mean over the {N} replicates "
             f"(one colour scale). Row 2: standard deviation over the {N} replicates (one scale). Black: "
             + ("2x2 centre and 4x4 square" if SQUARE else "disc") + ".", fontsize=10)
fig.savefig(os.path.join(OUT, f"mean_sd_error_maps{SUFFIX}.png"), dpi=130, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 2. paired effect minus zero
pairs = [(p, m) for p, e, m in configs if e == 1.0]
diffs = {}
for p, m in pairs:
    a = stacks[(p, 1.0, m)]
    b = stacks[(ZERO_SOURCE[p], 0.0, m)]      # both ordered by k
    diffs[(p, m)] = (a - b).mean(0)
lim_d = max(np.abs(v).max() for v in diffs.values())
fig, axes = plt.subplots(1, len(pairs), figsize=(2.8 * len(pairs), 3.6))
for ax, (p, m) in zip(axes, pairs):
    d = diffs[(p, m)]
    src = "" if ZERO_SOURCE[p] == p else f"\n(zero-effect run of {ZERO_SOURCE[p]})"
    im = panel(ax, d, lim_d, f"{p} {m}{src}")
    ax.text(0.0, -0.06, region_line(d).replace("  ", "\n"), transform=ax.transAxes, va="top", fontsize=7.5, family="monospace")
fig.colorbar(im, ax=axes.tolist(), shrink=0.8, pad=0.01)
fig.suptitle("Error map WITH the effect minus error map WITHOUT it, same replicate (same images, same assignment), "
             f"averaged over the {N} pairs. Zero everywhere = the error does not depend on the effect being there.", fontsize=10)
fig.savefig(os.path.join(OUT, f"paired_effect_minus_zero{SUFFIX}.png"), dpi=130, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 3. FF vs OLS per replicate
BROOT = os.path.join(MM, "runs", "baselines")


def ols_err_map(dataset_id):
    row = bl[(bl.method == "ols") & (bl.dataset_id == dataset_id)].iloc[0]
    a = DS.run_arrays(os.path.join(BROOT, row.run_id))
    return (np.asarray(a["tau_hat_ols"]) - np.asarray(a["ATE"])).reshape(S, S)


rows = []
for _, r in g[g.model == "ff"].sort_values(["preset", "effect", "k"]).iterrows():
    e_ff, e_ols = err_map(r.run_id), ols_err_map(r.dataset_id)
    row = {"preset": r.preset, "effect": r.effect, "k": r.k,
           "ff_mae": np.abs(e_ff).mean(), "ols_mae": np.abs(e_ols).mean()}
    row["diff"] = row["ff_mae"] - row["ols_mae"]
    for nm, m in REG.items():
        row[f"ff_{nm}"] = e_ff[m].mean()
        row[f"ols_{nm}"] = e_ols[m].mean()
    rows.append(row)
tab = pd.DataFrame(rows)
tab.to_csv(os.path.join(OUT, f"ff_vs_ols_mae{SUFFIX}.csv"), index=False)
summ = tab.groupby(["preset", "effect"]).agg(ff_mae=("ff_mae", "mean"), ols_mae=("ols_mae", "mean"),
                                             diff_mean=("diff", "mean"), diff_sd=("diff", "std"),
                                             diff_min=("diff", "min"), diff_max=("diff", "max")).round(4)
with open(os.path.join(OUT, f"ff_vs_ols_mae{SUFFIX}.md"), "w") as f:
    f.write("# FF vs OLS, MAE over the 64 pixels, per replicate (same dataset: same images, same assignment)\n\n")
    f.write(tab.round(4).to_markdown(index=False))
    f.write(f"\n\n# Paired difference ff_mae - ols_mae, summarised over the {N} replicates\n\n")
    f.write(summ.to_markdown())
print(tab.round(4).to_string(index=False))
print()
print(summ.to_string())
print(f"\nwritten to {OUT}")
