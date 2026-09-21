"""Quality of the generated outcomes, per treatment arm, computed at the end of an
``exp_ate_recovery.py`` run from the interventional draws the effect read-out makes.

Reference: the true potential outcomes ``Y_i(t)`` of the held-out units (flowjax's own
validation rows, reconstructed by ``exp_ate_recovery._fit_val_indices``; every row when
that fails).  The generator builds every image's untreated outcome first and adds the
individual effect on top, so ``Y(0) = Y - T * ITE`` and ``Y(1) = Y(0) + ITE`` are exact
for every unit, whichever arm it was observed in.
Generated: the finite draws of ``Y | do(T=t)`` from the fitted margin (``n_mc`` per arm,
minus the non-finite ones the read-out dropped).  Reference and generated are two samples
of the same population distribution, not paired individuals.

Everything numeric is on the modelling scale (logit pixels); only the gallery is shown in
pixel intensity.  For each arm t in {0, 1}:

* gallery: a few random reference and generated images on one intensity scale;
* moments: reference and generated per-pixel mean and SD maps, their differences, MAE and
  RMSE between corresponding maps, and the Monte Carlo standard error of the generated
  mean (SD of the draws / sqrt(number of draws), averaged over pixels);
* distributions: two-sample KS distance per pixel (largest and mean over pixels, and the
  value at the three pixels the copula figure uses: largest |signed error| in the disc,
  the ring and the far region), with CDF overlays at those pixels;
* dependence: Pearson correlation between every pair of edge-adjacent pixels, reference
  against generated, and their mean absolute discrepancy;
* overall: a logistic-regression classifier on the pixel values, reference against an
  equal-sized subsample of the draws, five-fold stratified cross-validation, out-of-fold
  ROC AUC (0.5 = indistinguishable by a linear rule).

Scalars go to ``metrics.json`` under ``gen_``; the arrays needed to redraw the figures go
to ``arrays.npz`` under the same prefix.
"""
from __future__ import annotations

import matplotlib
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

N_GALLERY = 8
REGIONS = ("disc", "ring", "far")


def inverse_logit(x):
    return 1.0 / (1.0 + np.exp(-x))


def neighbour_pairs(size: int) -> np.ndarray:
    """Flat indices of every edge-adjacent pixel pair of a size x size image, (P, 2)."""
    idx = np.arange(size * size).reshape(size, size)
    horiz = np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], axis=1)
    vert = np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], axis=1)
    return np.vstack([horiz, vert])


def _pair_corr(A: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Pearson correlation between the two columns of each pair, over the rows of A."""
    Z = (A - A.mean(0)) / A.std(0)
    return (Z[:, pairs[:, 0]] * Z[:, pairs[:, 1]]).mean(0)


def _classifier_auc(ref: np.ndarray, gen: np.ndarray, seed: int):
    """Out-of-fold AUC of a logistic regression telling reference from generated rows."""
    X = np.vstack([ref, gen])
    y = np.r_[np.zeros(len(ref)), np.ones(len(gen))]
    oof = np.zeros(len(y))
    aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], oof[te]))
    fpr, tpr, _ = roc_curve(y, oof)
    return float(roc_auc_score(y, oof)), float(np.std(aucs, ddof=1)), fpr, tpr


# --------------------------------------------------------------------------- compute
def compute(data: dict, y0: np.ndarray, y1: np.ndarray, heldout_idx, size: int,
            err: np.ndarray, masks, seed: int = 0):
    """Returns ``(metrics, arrays)``; see the module docstring."""
    Y = np.asarray(data["Y"], dtype=np.float64)
    T = np.asarray(data["X"])[:, 0].astype(int)
    ITE = np.asarray(data["ITE"], dtype=np.float64)
    n, K = Y.shape
    Y0 = Y - T[:, None] * ITE
    Y1 = Y0 + ITE
    if heldout_idx is None:
        idx, rows = np.arange(n), "all"
    else:
        idx, rows = np.asarray(heldout_idx), "heldout"
    m = len(idx)
    rng = np.random.default_rng(seed)
    pairs = neighbour_pairs(size)
    pix = {}
    for rn, mk in zip(REGIONS, masks):
        where = np.flatnonzero(mk)
        pix[rn] = int(where[np.argmax(np.abs(err[where]))])

    met: dict = {"gen_rows": rows, "gen_n_ref": int(m), "gen_n_pairs": int(len(pairs))}
    arr: dict = {"gen_idx": idx, "gen_nb_pairs": pairs}
    for rn in REGIONS:
        met[f"gen_pix_{rn}"] = pix[rn]
    for t, (ref_all, gen) in enumerate(((Y0, y0), (Y1, y1))):
        ref = ref_all[idx]
        gen = np.asarray(gen, dtype=np.float64)
        n_mc = gen.shape[0]
        met[f"gen_n_mc_t{t}"] = int(n_mc)
        # moments
        rm, gm = ref.mean(0), gen.mean(0)
        rs, gs = ref.std(0, ddof=1), gen.std(0, ddof=1)
        met[f"gen_mean_mae_t{t}"] = float(np.abs(gm - rm).mean())
        met[f"gen_mean_rmse_t{t}"] = float(np.sqrt(((gm - rm) ** 2).mean()))
        met[f"gen_mean_mcse_t{t}"] = float((gs / np.sqrt(n_mc)).mean())
        met[f"gen_sd_mae_t{t}"] = float(np.abs(gs - rs).mean())
        met[f"gen_sd_rmse_t{t}"] = float(np.sqrt(((gs - rs) ** 2).mean()))
        arr.update({f"gen_ref_mean_t{t}": rm, f"gen_gen_mean_t{t}": gm,
                    f"gen_ref_sd_t{t}": rs, f"gen_gen_sd_t{t}": gs})
        # per-pixel KS
        ks = np.array([stats.ks_2samp(ref[:, k], gen[:, k]).statistic for k in range(K)])
        met[f"gen_ks_max_t{t}"] = float(ks.max())
        met[f"gen_ks_mean_t{t}"] = float(ks.mean())
        for rn in REGIONS:
            met[f"gen_ks_{rn}_t{t}"] = float(ks[pix[rn]])
        arr[f"gen_ks_t{t}"] = ks
        # neighbouring-pixel correlations
        cr, cg = _pair_corr(ref, pairs), _pair_corr(gen, pairs)
        met[f"gen_nbcorr_mad_t{t}"] = float(np.abs(cg - cr).mean())
        met[f"gen_nbcorr_maxabs_t{t}"] = float(np.abs(cg - cr).max())
        arr.update({f"gen_nb_ref_t{t}": cr, f"gen_nb_gen_t{t}": cg})
        # classifier on an equal-sized subsample of the draws
        sub = gen[rng.choice(n_mc, size=min(m, n_mc), replace=False)]
        auc, auc_sd, fpr, tpr = _classifier_auc(ref, sub, seed)
        met[f"gen_auc_t{t}"] = auc
        met[f"gen_auc_fold_sd_t{t}"] = auc_sd
        arr.update({f"gen_sub_t{t}": sub, f"gen_roc_fpr_t{t}": fpr, f"gen_roc_tpr_t{t}": tpr})
        # gallery rows
        arr[f"gen_gallery_ref_t{t}"] = ref[rng.choice(m, size=min(N_GALLERY, m), replace=False)]
        arr[f"gen_gallery_gen_t{t}"] = sub[:N_GALLERY]
    return met, arr


# --------------------------------------------------------------------------- plots
def _rows_label(met: dict) -> str:
    return "the held-out units" if met["gen_rows"] == "heldout" else "all units (held-out split not recovered)"


def plot_gallery(arr: dict, met: dict, size: int, path: str, title: str):
    fig, axes = plt.subplots(4, N_GALLERY, figsize=(1.6 * N_GALLERY, 7.2))
    labels = ["reference Y(0)", "generated do(T=0)", "reference Y(1)", "generated do(T=1)"]
    rows = [arr["gen_gallery_ref_t0"], arr["gen_gallery_gen_t0"], arr["gen_gallery_ref_t1"], arr["gen_gallery_gen_t1"]]
    for r, (lab, imgs) in enumerate(zip(labels, rows)):
        for c in range(N_GALLERY):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c < len(imgs):
                ax.imshow(inverse_logit(imgs[c]).reshape(size, size), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            else:
                ax.set_axis_off()
        axes[r, 0].set_ylabel(lab, fontsize=9)
    fig.suptitle(f"{title}\nRandom images, pixel intensity on one scale (0 black, 1 white). Reference rows: "
                 f"true potential outcomes of {_rows_label(met)}. Generated rows: draws from the fitted margin. "
                 "Not paired individuals.", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_moments(arr: dict, met: dict, size: int, disc_mask: np.ndarray, path: str, title: str):
    disc2 = disc_mask.reshape(size, size)
    fig, axes = plt.subplots(2, 6, figsize=(21, 7.2))
    for t in (0, 1):
        rm, gm, rs, gs = (arr[f"gen_ref_mean_t{t}"], arr[f"gen_gen_mean_t{t}"],
                          arr[f"gen_ref_sd_t{t}"], arr[f"gen_gen_sd_t{t}"])
        lim_m = (float(min(rm.min(), gm.min())), float(max(rm.max(), gm.max())))
        lim_s = (0.0, float(max(rs.max(), gs.max())))
        dm, ds = gm - rm, gs - rs
        lim_dm, lim_ds = float(max(np.abs(dm).max(), 1e-6)), float(max(np.abs(ds).max(), 1e-6))
        panels = [(rm, f"reference mean, Y({t})", "viridis", lim_m),
                  (gm, f"generated mean, do(T={t})", "viridis", lim_m),
                  (dm, "generated − reference mean", "RdBu_r", (-lim_dm, lim_dm)),
                  (rs, f"reference SD, Y({t})", "viridis", lim_s),
                  (gs, f"generated SD, do(T={t})", "viridis", lim_s),
                  (ds, "generated − reference SD", "RdBu_r", (-lim_ds, lim_ds))]
        for c, (mp, lab, cmap, (lo, hi)) in enumerate(panels):
            ax = axes[t, c]
            im = ax.imshow(mp.reshape(size, size), cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
            ax.contour(disc2, levels=[0.5], colors="k", linewidths=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(lab, fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        axes[t, 2].text(0.0, -0.05, f"MAE {met[f'gen_mean_mae_t{t}']:.4f}  RMSE {met[f'gen_mean_rmse_t{t}']:.4f}\n"
                        f"MC s.e. of the generated mean,\nmean over pixels {met[f'gen_mean_mcse_t{t}']:.4f}",
                        transform=axes[t, 2].transAxes, va="top", fontsize=7.5, family="monospace")
        axes[t, 5].text(0.0, -0.05, f"MAE {met[f'gen_sd_mae_t{t}']:.4f}  RMSE {met[f'gen_sd_rmse_t{t}']:.4f}",
                        transform=axes[t, 5].transAxes, va="top", fontsize=7.5, family="monospace")
    fig.suptitle(f"{title}\nPer-pixel mean and SD on the logit scale. Reference: true potential outcomes of "
                 f"{_rows_label(met)}, n = {met['gen_n_ref']}. Generated: {met['gen_n_mc_t0']} / {met['gen_n_mc_t1']} "
                 "finite draws (do(T=0) / do(T=1)). Mean pair and SD pair each share a colour scale; the "
                 "differences have their own. Black: the effect disc.", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=3.0)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_distributions(arr: dict, met: dict, data_ref: tuple, path: str, title: str):
    """data_ref = (Y0_heldout, Y1_heldout) on the logit scale, for the CDF overlays."""
    fig, axes = plt.subplots(2, 5, figsize=(20, 7.6))
    for t in (0, 1):
        ref, sub = data_ref[t], arr[f"gen_sub_t{t}"]
        for c, rn in enumerate(REGIONS):
            k = met[f"gen_pix_{rn}"]
            ax = axes[t, c]
            for a, lab in ((ref[:, k], "reference"), (sub[:, k], "generated")):
                s = np.sort(a)
                ax.plot(s, (np.arange(len(s)) + 1) / len(s), lw=1.3, label=lab)
            ax.set_title(f"pixel {k} ({rn}), KS {met[f'gen_ks_{rn}_t{t}']:.3f}", fontsize=9)
            ax.set_xlabel("logit pixel value")
            ax.set_ylabel("CDF")
            ax.legend(fontsize=8)
        ax = axes[t, 3]
        cr, cg = arr[f"gen_nb_ref_t{t}"], arr[f"gen_nb_gen_t{t}"]
        ax.scatter(cr, cg, s=8, alpha=0.6)
        ax.plot([-1, 1], [-1, 1], "k--", lw=0.8)
        lo, hi = float(min(cr.min(), cg.min())) - 0.05, float(max(cr.max(), cg.max())) + 0.05
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("reference correlation")
        ax.set_ylabel("generated correlation")
        ax.set_title(f"{met['gen_n_pairs']} adjacent-pixel pairs\nmean |gap| {met[f'gen_nbcorr_mad_t{t}']:.3f}, "
                     f"max {met[f'gen_nbcorr_maxabs_t{t}']:.3f}", fontsize=9)
        ax = axes[t, 4]
        ax.plot(arr[f"gen_roc_fpr_t{t}"], arr[f"gen_roc_tpr_t{t}"], lw=1.3)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.set_title(f"reference vs generated, logistic regression\nout-of-fold AUC {met[f'gen_auc_t{t}']:.3f} "
                     f"(± {met[f'gen_auc_fold_sd_t{t}']:.3f} over 5 folds), chance 0.5", fontsize=9)
        axes[t, 0].set_ylabel(f"do(T={t})\nCDF", fontsize=9)
    fig.suptitle(f"{title}\nRow per arm. Columns 1-3: empirical CDFs at the pixel with the largest |signed error| in "
                 "each region, reference against an equal-sized subsample of the draws (KS on the full draws). "
                 "Column 4: Pearson correlation of each edge-adjacent pixel pair. Column 5: ROC of a linear "
                 f"classifier on the {arr['gen_sub_t0'].shape[1]} logit pixels, reference "
                 f"(n = {met['gen_n_ref']}) against the same number of draws, five-fold cross-validated.", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=3.0)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_all(arr: dict, met: dict, data: dict, size: int, disc_mask: np.ndarray, plots_dir: str, title: str):
    import os
    Y = np.asarray(data["Y"], dtype=np.float64)
    T = np.asarray(data["X"])[:, 0].astype(int)
    ITE = np.asarray(data["ITE"], dtype=np.float64)
    Y0 = Y - T[:, None] * ITE
    idx = np.asarray(arr["gen_idx"])
    plot_gallery(arr, met, size, os.path.join(plots_dir, "samples_gallery.png"), title)
    plot_moments(arr, met, size, disc_mask, os.path.join(plots_dir, "samples_moments.png"), title)
    plot_distributions(arr, met, (Y0[idx], (Y0 + ITE)[idx]),
                       os.path.join(plots_dir, "samples_distributions.png"), title)


GEN_ARRAY_KEYS = tuple(["gen_idx", "gen_nb_pairs"] + [
    f"gen_{k}_t{t}" for t in (0, 1)
    for k in ("ref_mean", "gen_mean", "ref_sd", "gen_sd", "ks", "nb_ref", "nb_gen", "sub",
              "roc_fpr", "roc_tpr", "gallery_ref", "gallery_gen")])
GEN_PLOTS = ("samples_gallery", "samples_moments", "samples_distributions")
