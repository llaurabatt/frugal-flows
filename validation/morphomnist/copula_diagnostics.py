"""Copula diagnostics for a fitted frugal flow, computed at the end of an
``exp_ate_recovery.py`` run (model ``ff`` only) while the flow is in memory.

Notation, for held-out observation i (flowjax's own validation rows, reconstructed by
``exp_ate_recovery._fit_val_indices``; every row when that reconstruction fails):

* ``U_Z,i``  the stage-1 covariate ranks (``u_z`` in the run's arrays);
* ``R_i``    the outcome ranks: ``Y_i`` pushed through the inverse of the fitted causal
             margin given ``T_i``.  One rank per pixel, so ``R_i`` has K entries;
* ``v_i``    the covariate-side base coordinates: ``U_Z,i`` pushed through the inverse of
             the copula block given the whole ``R_i``.  No sampling involved.

The copula block does not see ``T`` (the treatment enters it through a fully masked
conditioning input), so nothing here conditions on treatment.

What is computed (``compute``) and drawn (``plot_all``):

1. margins of ``U_Z``: KS distance from uniform per covariate, histogram, uniform Q-Q;
2. observed dependence vs the copula's prediction: with every ``R_i`` held fixed, draw
   ``U~_Z,i`` from the fitted copula ``n_mc`` times; compare Spearman correlations between
   covariate ranks, and between each pixel's ``R_k`` and each covariate rank, observed vs
   predicted (mean and sd over the ``n_mc`` draws);
3. ``v`` = inverse of the copula on the actual held-out ranks;
4. margins of ``v``: KS distance, histogram, Q-Q (expected uniform);
5. remaining dependence in ``v``: Spearman between coordinates of ``v`` and between each
   ``v_j`` and each pixel's ``R_k`` (expected zero);
6. calibration of each ``v_j`` by treatment group and by quartile of a prespecified
   outcome summary (mean logit intensity over the disc): observed fraction ``v_j <= q``
   against nominal ``q`` for ``q = 0.1 .. 0.9``, with a binomial band, max absolute gap
   and the group's size.

Every scalar goes into ``metrics.json`` under the prefix ``cop_``; the arrays needed to
redraw the figures without the flow go into ``arrays.npz`` under the same prefix.
Discrete covariates: the stage-1 ranks of a discrete covariate are already randomised
inside their probability bins (``causal_flows.univariate_discrete_cdf``), so they enter
these checks as they are.  The digit-0 runs have none.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib
import numpy as np
import paramax
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

Z_NAMES = ("thickness", "brightness")   # column order of Z in prepare_morphomnist_exps
N_COPULA_BLOCKS = 3                      # [rescale, copula flow, affine] at the base end
Q_GRID = np.round(np.arange(0.1, 0.91, 0.1), 2)
GROUPS = ("t0", "t1", "q1", "q2", "q3", "q4")


def z_names(d: int) -> list[str]:
    return [Z_NAMES[j] if j < len(Z_NAMES) else f"z{j}" for j in range(d)]


def _rows_label(met: dict) -> str:
    return "the held-out rows" if met["cop_rows"] == "heldout" else "all rows (held-out split not recovered)"


def _spearman_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Spearman correlation between every column of A and every column of B."""
    ra = stats.rankdata(A, axis=0)
    rb = stats.rankdata(B, axis=0)
    ra = (ra - ra.mean(0)) / ra.std(0)
    rb = (rb - rb.mean(0)) / rb.std(0)
    return ra.T @ rb / A.shape[0]


def _groups(T: np.ndarray, ysum: np.ndarray) -> dict[str, np.ndarray]:
    """Row masks for the six calibration groups on the rows given."""
    q = np.quantile(ysum, [0.25, 0.5, 0.75])
    return {
        "t0": T == 0, "t1": T == 1,
        "q1": ysum <= q[0], "q2": (ysum > q[0]) & (ysum <= q[1]),
        "q3": (ysum > q[1]) & (ysum <= q[2]), "q4": ysum > q[2],
    }


# --------------------------------------------------------------------------- compute
def compute(flow, data: dict, u_z: np.ndarray, heldout_idx, disc_mask: np.ndarray,
            err: np.ndarray, masks, n_mc: int = 20, seed: int = 0, tol: float = 1e-5):
    """Returns ``(metrics, arrays)``; see the module docstring for what each is."""
    B = flow.bijection.bijections
    Y = np.asarray(data["Y"], dtype=np.float64)
    T = np.asarray(data["X"])[:, 0].astype(int)
    n, K = Y.shape
    u_z = np.asarray(u_z, dtype=np.float64)
    d = u_z.shape[1]
    x = jnp.hstack([jnp.asarray(Y), jnp.asarray(u_z)])
    cond = jnp.asarray(data["X"])

    def apply(b, xs, inverse: bool):
        b = paramax.unwrap(b)
        f = b.inverse if inverse else b.transform
        if b.cond_shape is not None:
            return jax.vmap(f)(xs, cond)
        return jax.vmap(f)(xs)

    # data -> margin base: the u_z columns must pass the margin blocks untouched
    xs = x
    for b in reversed(B[N_COPULA_BLOCKS:]):
        xs = apply(b, xs, True)
    assert bool(jnp.allclose(xs[:, K:], x[:, K:], atol=tol)), "u_z altered by the margin blocks"
    r_pm = xs[:, :K]                                   # outcome ranks on [-1, 1]
    # margin base -> copula base.  Blocks 2 and 1 (the u_z affine and the copula MAF) leave
    # the R columns untouched; block 0 is one Affine over every column that maps the
    # [-1, 1] flow space onto the [0, 1] base.  So the base holds R and v both on [0, 1].
    vs = xs
    for b in reversed(B[1:N_COPULA_BLOCKS]):
        vs = apply(b, vs, True)
    assert bool(jnp.allclose(vs[:, :K], r_pm, atol=tol)), "R altered by the copula blocks"
    base = apply(B[0], vs, True)
    R = np.asarray(base[:, :K])
    v = np.asarray(base[:, K:])
    assert R.min() >= -tol and R.max() <= 1 + tol and v.min() >= -tol and v.max() <= 1 + tol, \
        (R.min(), R.max(), v.min(), v.max())
    R, v = np.clip(R, 0.0, 1.0), np.clip(v, 0.0, 1.0)
    r_base = base[:, :K]

    if heldout_idx is None:
        idx, rows = np.arange(n), "all"
    else:
        idx, rows = np.asarray(heldout_idx), "heldout"
    m = len(idx)

    # sampled covariate ranks given the held-out R (copula forward on [R, noise])
    key = jr.PRNGKey(seed)
    cond_idx = cond[idx]
    r_idx = r_base[idx]                                # base-space R of the rows used
    preds = np.empty((n_mc, m, d))

    def fwd(b, ys):
        b = paramax.unwrap(b)
        if b.cond_shape is not None:
            return jax.vmap(b.transform)(ys, cond_idx)
        return jax.vmap(b.transform)(ys)

    for s in range(n_mc):
        key, sub = jr.split(key)
        eps = jr.uniform(sub, (m, d))                  # fresh base noise on [0, 1]
        ys = jnp.hstack([r_idx, eps])
        for b in B[:N_COPULA_BLOCKS]:                  # base -> [R on [-1,1], u_z on [0,1]]
            ys = fwd(b, ys)
        assert bool(jnp.allclose(ys[:, :K], r_pm[idx], atol=tol)), "R altered by the copula forward pass"
        preds[s] = np.clip(np.asarray(ys[:, K:]), 0.0, 1.0)

    U, Rh, V, Th = u_z[idx], R[idx], v[idx], T[idx]
    ysum = Y[idx][:, disc_mask].mean(axis=1)
    names = z_names(d)
    met: dict = {"cop_rows": rows, "cop_n": int(m), "cop_n_mc": int(n_mc)}

    # 1 + 4: margins
    for j, nm in enumerate(names):
        met[f"cop_ks_u_{nm}"] = float(stats.kstest(U[:, j], "uniform").statistic)
        met[f"cop_ks_v_{nm}"] = float(stats.kstest(V[:, j], "uniform").statistic)

    # 2 + 5: dependence between covariates
    for a in range(d):
        for b in range(a + 1, d):
            pa, pb = names[a], names[b]
            rp = np.array([_spearman_matrix(preds[s][:, [a]], preds[s][:, [b]])[0, 0] for s in range(n_mc)])
            met[f"cop_rho_u_{pa}_{pb}_obs"] = float(_spearman_matrix(U[:, [a]], U[:, [b]])[0, 0])
            met[f"cop_rho_u_{pa}_{pb}_pred"] = float(rp.mean())
            met[f"cop_rho_u_{pa}_{pb}_pred_sd"] = float(rp.std(ddof=1))
            met[f"cop_rho_v_{pa}_{pb}"] = float(_spearman_matrix(V[:, [a]], V[:, [b]])[0, 0])

    # 2 + 5: dependence between each pixel's R and each covariate, one map per covariate
    rho_obs = _spearman_matrix(Rh, U)                           # (K, d)
    rho_pred_s = np.stack([_spearman_matrix(Rh, preds[s]) for s in range(n_mc)])   # (n_mc, K, d)
    rho_pred, rho_pred_sd = rho_pred_s.mean(0), rho_pred_s.std(0, ddof=1)
    rho_v = _spearman_matrix(Rh, V)                             # (K, d)
    gap = rho_obs - rho_pred
    reg_names = ("disc", "ring", "far")
    for j, nm in enumerate(names):
        met[f"cop_rho_ru_{nm}_obs_maxabs"] = float(np.abs(rho_obs[:, j]).max())
        met[f"cop_rho_ru_{nm}_gap_maxabs"] = float(np.abs(gap[:, j]).max())
        met[f"cop_rho_ru_{nm}_gap_meanabs"] = float(np.abs(gap[:, j]).mean())
        met[f"cop_rho_ru_{nm}_pred_sd_mean"] = float(rho_pred_sd[:, j].mean())
        met[f"cop_rho_rv_{nm}_maxabs"] = float(np.abs(rho_v[:, j]).max())
        met[f"cop_rho_rv_{nm}_meanabs"] = float(np.abs(rho_v[:, j]).mean())
        for rn, mk in zip(reg_names, masks):
            met[f"cop_rho_ru_{nm}_gap_{rn}"] = float(gap[mk, j].mean())
            met[f"cop_rho_rv_{nm}_{rn}"] = float(rho_v[mk, j].mean())

    # 6: calibration of v by group
    grp = _groups(Th, ysum)
    cal = np.full((d, len(GROUPS), len(Q_GRID)), np.nan)
    for gi, g in enumerate(GROUPS):
        rows_g = grp[g]
        met[f"cop_cal_n_{g}"] = int(rows_g.sum())
        for j, nm in enumerate(names):
            frac = np.array([(V[rows_g, j] <= q).mean() for q in Q_GRID])
            cal[j, gi] = frac
            met[f"cop_cal_{nm}_{g}"] = float(np.abs(frac - Q_GRID).max())
    for nm in names:   # the worst of the four outcome quartiles, one number for the index
        met[f"cop_cal_{nm}_qmax"] = max(met[f"cop_cal_{nm}_{g}"] for g in ("q1", "q2", "q3", "q4"))

    # the pixels shown in the scatter panels: largest |signed error| in each region
    pix = {}
    for rn, mk in zip(reg_names, masks):
        where = np.flatnonzero(mk)
        pix[rn] = int(where[np.argmax(np.abs(err[where]))])
        met[f"cop_pix_{rn}"] = pix[rn]

    arrays = {
        "cop_idx": idx, "cop_R": Rh, "cop_u": U, "cop_v": V, "cop_u_pred": preds,
        "cop_T": Th, "cop_ysum": ysum,
        "cop_rho_obs": rho_obs, "cop_rho_pred": rho_pred, "cop_rho_pred_sd": rho_pred_sd,
        "cop_rho_v": rho_v, "cop_cal": cal,
    }
    return met, arrays


# --------------------------------------------------------------------------- plots
def _qq(ax, u, label):
    u = np.sort(u)
    ax.plot((np.arange(len(u)) + 0.5) / len(u), u, lw=1.2)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("uniform quantile")
    ax.set_ylabel(label)


def _hist(ax, u, label):
    ax.hist(u, bins=20, range=(0, 1), density=True, color="C0", alpha=0.8)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel(label)
    ax.set_ylabel("density")


def plot_margins(arr: dict, met: dict, path: str, title: str):
    U, V = arr["cop_u"], arr["cop_v"]
    d = U.shape[1]
    names = z_names(d)
    fig, axes = plt.subplots(d, 4, figsize=(16, 3.6 * d), squeeze=False)
    for j, nm in enumerate(names):
        _hist(axes[j, 0], U[:, j], f"$U_Z$ {nm}")
        _qq(axes[j, 1], U[:, j], f"$U_Z$ {nm}")
        _hist(axes[j, 2], V[:, j], f"$v$ {nm}")
        _qq(axes[j, 3], V[:, j], f"$v$ {nm}")
        axes[j, 0].set_title(f"stage-1 rank, KS from uniform {met[f'cop_ks_u_{nm}']:.3f}", fontsize=9)
        axes[j, 1].set_title("stage-1 rank, Q-Q", fontsize=9)
        axes[j, 2].set_title(f"copula base coordinate, KS from uniform {met[f'cop_ks_v_{nm}']:.3f}", fontsize=9)
        axes[j, 3].set_title("copula base coordinate, Q-Q", fontsize=9)
    fig.suptitle(f"{title}\nMargins on {_rows_label(met)}, n = {met['cop_n']}. Left pair: the "
                 "stage-1 covariate ranks going INTO the copula. Right pair: the base coordinates "
                 "coming OUT of the copula's inverse given R, which should be uniform.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_dependence(arr: dict, met: dict, path: str, title: str):
    U, V, P, R = arr["cop_u"], arr["cop_v"], arr["cop_u_pred"], arr["cop_R"]
    d = U.shape[1]
    names = z_names(d)
    pix = [(rn, met[f"cop_pix_{rn}"]) for rn in ("disc", "ring", "far")]
    n_pair_rows = 1 if d >= 2 else 0
    n_rows = n_pair_rows + len(pix)
    n_cols = 3 * d
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.3 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_visible(False)
    kw = dict(s=4, alpha=0.35, lw=0)
    if d >= 2:
        a, b = names[0], names[1]
        cols = [("observed", U[:, 0], U[:, 1], f"Spearman {met[f'cop_rho_u_{a}_{b}_obs']:+.3f}"),
                ("copula prediction (draw 1 of {})".format(P.shape[0]), P[0][:, 0], P[0][:, 1],
                 f"Spearman {met[f'cop_rho_u_{a}_{b}_pred']:+.3f} ± {met[f'cop_rho_u_{a}_{b}_pred_sd']:.3f} over draws"),
                ("copula base v", V[:, 0], V[:, 1], f"Spearman {met[f'cop_rho_v_{a}_{b}']:+.3f}")]
        for c, (lab, xx, yy, sub) in enumerate(cols):
            ax = axes[0, c]
            ax.set_visible(True)
            ax.scatter(xx, yy, **kw)
            ax.set_xlabel(f"{a}")
            ax.set_ylabel(f"{b}")
            ax.set_title(f"{lab}\n{sub}", fontsize=9)
    rho_obs, rho_pred, rho_pred_sd, rho_v = arr["cop_rho_obs"], arr["cop_rho_pred"], arr["cop_rho_pred_sd"], arr["cop_rho_v"]
    for i, (rn, k) in enumerate(pix):
        r = n_pair_rows + i
        for j, nm in enumerate(names):
            c0 = 3 * j
            trip = [("observed $U_Z$", U[:, j], f"Spearman {rho_obs[k, j]:+.3f}"),
                    ("copula prediction $\\tilde U_Z$ (draw 1)", P[0][:, j],
                     f"Spearman {rho_pred[k, j]:+.3f} ± {rho_pred_sd[k, j]:.3f} over draws"),
                    ("copula base $v$", V[:, j], f"Spearman {rho_v[k, j]:+.3f}")]
            for c, (lab, yy, sub) in enumerate(trip):
                ax = axes[r, c0 + c]
                ax.set_visible(True)
                ax.scatter(R[:, k], yy, **kw)
                ax.set_xlabel(f"$R$ pixel {k} ({rn})")
                ax.set_ylabel(f"{nm}")
                ax.set_title(f"{lab}\n{sub}", fontsize=9)
    fig.suptitle(f"{title}\nEach dot: one observation of {_rows_label(met)}, n = {met['cop_n']}. "
                 "Row 1 (two covariates only): covariate rank against covariate rank.\n"
                 "Other rows: the outcome rank R of one pixel (the pixel with the largest |signed error| "
                 "in the disc, the ring and the far region) against a covariate rank.\n"
                 "Three columns per covariate: the actual ranks; the ranks the copula draws given that "
                 "observation's whole R; the base coordinate after the copula's inverse.",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_dependence_maps(arr: dict, met: dict, size: int, disc_mask: np.ndarray, err: np.ndarray,
                         path: str, title: str):
    rho_obs, rho_pred, rho_v = arr["cop_rho_obs"], arr["cop_rho_pred"], arr["cop_rho_v"]
    d = rho_obs.shape[1]
    names = z_names(d)
    lim = float(max(np.abs(rho_obs).max(), np.abs(rho_pred).max(), np.abs(rho_v).max(), 1e-3))
    lim_e = float(max(np.abs(err).max(), 1e-3))
    fig, axes = plt.subplots(d, 5, figsize=(18, 4.4 * d), squeeze=False)
    disc2 = disc_mask.reshape(size, size)
    for j, nm in enumerate(names):
        panels = [(rho_obs[:, j], f"observed\nSpearman(R_k, U_Z {nm})", lim),
                  (rho_pred[:, j], f"copula prediction\nSpearman(R_k, Ũ_Z {nm}), mean of {met['cop_n_mc']} draws", lim),
                  (rho_obs[:, j] - rho_pred[:, j], "observed minus predicted\n", lim),
                  (rho_v[:, j], f"after the copula's inverse\nSpearman(R_k, v {nm})", lim),
                  (err, "signed error of the effect map\n(estimate − truth)", lim_e)]
        for c, (m, lab, l) in enumerate(panels):
            ax = axes[j, c]
            im = ax.imshow(m.reshape(size, size), cmap="RdBu_r", vmin=-l, vmax=l, interpolation="nearest")
            ax.contour(disc2, levels=[0.5], colors="k", linewidths=0.9)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(lab, fontsize=8.5)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            if c == 2:
                ax.text(0.0, -0.04, f"disc {met[f'cop_rho_ru_{nm}_gap_disc']:+.3f}\nring {met[f'cop_rho_ru_{nm}_gap_ring']:+.3f}"
                        f"\nfar  {met[f'cop_rho_ru_{nm}_gap_far']:+.3f}\nmax |gap| {met[f'cop_rho_ru_{nm}_gap_maxabs']:.3f}"
                        f"\nMC sd of predicted,\nmean over pixels {met[f'cop_rho_ru_{nm}_pred_sd_mean']:.3f}",
                        transform=ax.transAxes, va="top", fontsize=7.5, family="monospace")
            if c == 3:
                ax.text(0.0, -0.04, f"disc {met[f'cop_rho_rv_{nm}_disc']:+.3f}\nring {met[f'cop_rho_rv_{nm}_ring']:+.3f}"
                        f"\nfar  {met[f'cop_rho_rv_{nm}_far']:+.3f}\nmax |rho| {met[f'cop_rho_rv_{nm}_maxabs']:.3f}",
                        transform=ax.transAxes, va="top", fontsize=7.5, family="monospace")
    fig.suptitle(f"{title}\nOne value per pixel, computed on {_rows_label(met)}, n = {met['cop_n']}. "
                 "Columns 1-4 share one colour scale; column 5 has its own. Black: the effect disc.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=4.0)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(arr: dict, met: dict, path: str, title: str):
    cal = arr["cop_cal"]
    d = cal.shape[0]
    names = z_names(d)
    labels = {"t0": "T = 0", "t1": "T = 1", "q1": "disc-mean Y, quartile 1 (lowest)",
              "q2": "disc-mean Y, quartile 2", "q3": "disc-mean Y, quartile 3", "q4": "disc-mean Y, quartile 4 (highest)"}
    fig, axes = plt.subplots(d, len(GROUPS), figsize=(3.0 * len(GROUPS), 3.8 * d), squeeze=False)
    for j, nm in enumerate(names):
        for gi, g in enumerate(GROUPS):
            ax = axes[j, gi]
            n_g = met[f"cop_cal_n_{g}"]
            band = 1.96 * np.sqrt(Q_GRID * (1 - Q_GRID) / max(n_g, 1))
            ax.fill_between(Q_GRID, Q_GRID - band, Q_GRID + band, color="0.85")
            ax.plot([0, 1], [0, 1], "k--", lw=0.8)
            ax.plot(Q_GRID, cal[j, gi], marker="o", ms=3, lw=1.2)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.set_title(f"{labels[g]}\nn = {n_g}, max |gap| {met[f'cop_cal_{nm}_{g}']:.3f}", fontsize=8.5)
            if gi == 0:
                ax.set_ylabel(f"fraction of v {nm} ≤ q")
            if j == d - 1:
                ax.set_xlabel("nominal q")
    fig.suptitle(f"{title}\nCalibration of the copula base coordinate v within groups of {_rows_label(met)}. "
                 "Grey: ±1.96 binomial standard errors at each q for that group's size.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=3.0)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_all(arr: dict, met: dict, size: int, disc_mask: np.ndarray, err: np.ndarray,
             plots_dir: str, title: str):
    import os
    plot_margins(arr, met, os.path.join(plots_dir, "copula_margins.png"), title)
    plot_dependence(arr, met, os.path.join(plots_dir, "copula_dependence.png"), title)
    plot_dependence_maps(arr, met, size, disc_mask, err, os.path.join(plots_dir, "copula_dependence_maps.png"), title)
    plot_calibration(arr, met, os.path.join(plots_dir, "copula_calibration.png"), title)


COP_ARRAY_KEYS = ("cop_idx", "cop_R", "cop_u", "cop_v", "cop_u_pred", "cop_T", "cop_ysum",
                  "cop_rho_obs", "cop_rho_pred", "cop_rho_pred_sd", "cop_rho_v", "cop_cal")
COP_PLOTS = ("copula_margins", "copula_dependence", "copula_dependence_maps", "copula_calibration")
