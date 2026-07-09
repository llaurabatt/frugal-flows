"""Fitted causal-margin densities vs true interventional densities (Gamma DGP).

Fits frugal flows on the confounded Gamma outcome family (`FAMILIES["gamma"]`,
causal_params=[const=1.0, ate=0.5], data seed 0) and compares the SAMPLED
interventional outcome distributions Y|do(0) / Y|do(1) against the analytic
TRUE interventional densities:

    Y | do(T=t) ~ Gamma(shape k=2, scale theta_t),  theta_t = 0.5*exp(1.0+0.5*t)

(k = 1/GAMMA_PHI with GAMMA_PHI=0.5, per `diagnostics/outcome_families.py`'s
log-link parametrisation: mean_do(t) = exp(const+ate*t), theta = mean*phi.)

Panel A (n=20,000): one spline (`flexible_continuous`) fit and one additive
(`gaussian`) fit, each fit ONCE, sampled via `ate_extraction_suite.intervene`
at n_mc=50,000, KDE'd against the true densities. Demonstrates that the spline
margin tracks the true right-skewed shape while the additive (location-shift)
margin is misspecified for a multiplicative/log-link DGP and can even put mass
at Y<0 (impossible under the true support).

Panel B (n=2,000): the spline arm re-fit from THREE random restarts on the
SAME dataset (spline_stability.py convention: keys 70000/71000/72000), showing
that the density SHAPE is stable across restarts even though the restart-level
ATE wobbles (see SPLINE_BIAS_FINDINGS.md).

Key convention notes (see diagnostics/h1_matrix.py `robust_moments`): the
spline margin composes an RQS on [-1,1] with atanh (Invert(Tanh)), so a base
draw landing at the tanh boundary maps to +/-inf. We finite-filter each sample
array before KDE'ing it and print the drop count -- never silent.

Usage (from validation/, in the frugal-flows-flowjax env, on a shared 2-core box):
  env OMP_NUM_THREADS=2 ~/.local/bin/micromamba run -n frugal-flows-flowjax \\
      python diagnostics/plot_margin_densities.py
"""

from __future__ import annotations

import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)  # must precede any jnp array creation

import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless: write PNGs, no display
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

_VALIDATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VALIDATION_DIR not in sys.path:
    sys.path.insert(0, _VALIDATION_DIR)

import data_processing_and_simulations.causl_sim_data_generation as causl_py  # noqa: E402
from diagnostics.ate_extraction_suite import intervene  # noqa: E402
from diagnostics.outcome_families import FAMILIES, GAMMA_PHI  # noqa: E402
from diagnostics.quick_sense_check import base_hyperparams, fit_model  # noqa: E402

# ----------------------------------------------------------------------------
# Style: replicated from diagnostics/plot_spline_findings.py so this figure
# matches figs 1-3 (Okabe-Ito palette, despine, panel letters, 200 dpi).
# ----------------------------------------------------------------------------
ADDITIVE_COLOR = "#0072B2"  # blue
SPLINE_COLOR = "#D55E00"  # vermillion
TRUTH_COLOR = "black"
DPI = 200

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "figure.titlesize": 15,
        "figure.titleweight": "bold",
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.25, linewidth=0.6)


def panel_letter(ax, letter: str) -> None:
    ax.text(
        -0.08, 1.05, letter, transform=ax.transAxes,
        fontsize=14, fontweight="bold", va="bottom", ha="right",
    )


# ----------------------------------------------------------------------------
# DGP / true density
# ----------------------------------------------------------------------------
CAUSAL_PARAMS = [1.0, 0.5]  # [const, ate] for FAMILIES["gamma"] (log link)
K_SHAPE = 1.0 / GAMMA_PHI  # = 2.0
THETA0 = 0.5 * np.exp(CAUSAL_PARAMS[0] + CAUSAL_PARAMS[1] * 0)
THETA1 = 0.5 * np.exp(CAUSAL_PARAMS[0] + CAUSAL_PARAMS[1] * 1)
TRUE_ATE = K_SHAPE * (THETA1 - THETA0)

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "flexible_te")
os.makedirs(OUTDIR, exist_ok=True)


def true_pdf(x: np.ndarray, theta: float) -> np.ndarray:
    return stats.gamma.pdf(x, a=K_SHAPE, scale=theta)


def finite_filter(arr, label: str) -> np.ndarray:
    """Drop non-finite entries (spline RQS/atanh boundary artefacts). Never silent."""
    arr = np.asarray(arr)
    finite = np.isfinite(arr)
    n_drop = int((~finite).sum())
    print(f"    [{label}] dropped {n_drop}/{arr.size} non-finite samples "
          f"({n_drop / arr.size:.3%})")
    return arr[finite]


def finite_ate(y0, y1) -> tuple[float, int, int]:
    """Paired finite-filtered ATE (mirrors h1_matrix.robust_moments)."""
    y0 = np.asarray(y0)
    y1 = np.asarray(y1)
    finite = np.isfinite(y0) & np.isfinite(y1)
    n_drop = int((~finite).sum())
    ate = float(np.mean((y1 - y0)[finite]))
    return ate, n_drop, int(finite.sum())


def kde_on_grid(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    kde = stats.gaussian_kde(samples)
    return kde(grid)


def main():
    t_start = time.time()
    print(f"True DGP: Y|do(t) ~ Gamma(k={K_SHAPE}, scale=theta_t), "
          f"theta_0={THETA0:.4f}, theta_1={THETA1:.4f}")
    print(f"True ATE = k*(theta_1-theta_0) = {TRUE_ATE:.4f}\n")

    fam = FAMILIES["gamma"]

    # ============================================================ n = 20,000
    print("=== generating n=20000 data (gamma family, causal_params=[1.0,0.5], seed=0) ===")
    data20k = fam.generate(20000, causal_params=CAUSAL_PARAMS, seed=0)
    X20, Y20 = data20k["X"], data20k["Y"]
    Zd20, Zc20 = data20k["Z_disc"], data20k["Z_cont"]
    uz20 = causl_py.generate_uz_samples(Zd20, Zc20, False, 0, base_hyperparams(1500))["uz_samples"]
    print(f"data20k: X{tuple(X20.shape)} Y{tuple(Y20.shape)} u_z{tuple(uz20.shape)}\n")

    # h1_matrix convention: key = jr.key(1000*seed+7); fold_in(1) for fit, fold_in(2)
    # for intervene. We reuse the SAME base key for both arms (as h1_matrix does --
    # the key is redefined per-arm from the same seed-derived value), noted here.
    key20 = jr.key(1000 * 0 + 7)

    print("--- fitting SPLINE (flexible_continuous), n=20000, epochs=1500 ---")
    t0 = time.time()
    ff_spline20, vloss_spline20 = fit_model(
        jr.fold_in(key20, 1), Y20, uz20, X20, "flexible_continuous", 1500
    )
    print(f"    val_loss={vloss_spline20:.4f}  ({time.time() - t0:.1f}s)")
    m_spline20 = intervene(jr.fold_in(key20, 2), ff_spline20, X20.shape[1], 50000)

    print("\n--- fitting ADDITIVE (gaussian), n=20000, epochs=1500 ---")
    t0 = time.time()
    ff_add20, vloss_add20 = fit_model(
        jr.fold_in(key20, 1), Y20, uz20, X20, "gaussian", 1500
    )
    print(f"    val_loss={vloss_add20:.4f}  ({time.time() - t0:.1f}s)")
    m_add20 = intervene(jr.fold_in(key20, 2), ff_add20, X20.shape[1], 50000)

    print("\n--- finite-filtering n=20000 samples ---")
    spline_y0 = finite_filter(m_spline20["y0"], "spline y0")
    spline_y1 = finite_filter(m_spline20["y1"], "spline y1")
    add_y0 = finite_filter(m_add20["y0"], "additive y0")
    add_y1 = finite_filter(m_add20["y1"], "additive y1")

    spline_ate20, spline_drop, spline_keep = finite_ate(m_spline20["y0"], m_spline20["y1"])
    add_ate20, add_drop, add_keep = finite_ate(m_add20["y0"], m_add20["y1"])
    print(f"\nn=20000 SPLINE   ATE (paired, finite) = {spline_ate20:.4f}  "
          f"(dropped {spline_drop}/{spline_drop + spline_keep} pairs)  "
          f"[val_loss={vloss_spline20:.4f}]")
    print(f"n=20000 ADDITIVE ATE (paired, finite) = {add_ate20:.4f}  "
          f"(dropped {add_drop}/{add_drop + add_keep} pairs)  "
          f"[val_loss={vloss_add20:.4f}]")
    print(f"true ATE = {TRUE_ATE:.4f}")

    frac_neg_add0 = float(np.mean(add_y0 < 0))
    frac_neg_add1 = float(np.mean(add_y1 < 0))
    print(f"\nadditive frac(Y<0): do(0)={frac_neg_add0:.3%}  do(1)={frac_neg_add1:.3%}")

    # ============================================================= n = 2,000
    print("\n=== generating n=2000 data (gamma family, causal_params=[1.0,0.5], seed=0) ===")
    data2k = fam.generate(2000, causal_params=CAUSAL_PARAMS, seed=0)
    X2, Y2 = data2k["X"], data2k["Y"]
    Zd2, Zc2 = data2k["Z_disc"], data2k["Z_cont"]
    uz2 = causl_py.generate_uz_samples(Zd2, Zc2, False, 0, base_hyperparams(600))["uz_samples"]
    print(f"data2k: X{tuple(X2.shape)} Y{tuple(Y2.shape)} u_z{tuple(uz2.shape)}\n")

    restart_seeds = [70000, 71000, 72000]  # spline_stability.py convention
    restart_y0, restart_y1, restart_ates = [], [], []
    for i, rseed in enumerate(restart_seeds):
        print(f"--- spline restart {i} (key=jr.key({rseed})), n=2000, epochs=600 ---")
        rkey = jr.key(rseed)
        t0 = time.time()
        ff_r, vloss_r = fit_model(jr.fold_in(rkey, 1), Y2, uz2, X2, "flexible_continuous", 600)
        print(f"    val_loss={vloss_r:.4f}  ({time.time() - t0:.1f}s)")
        m_r = intervene(jr.fold_in(rkey, 2), ff_r, X2.shape[1], 20000)
        r_y0 = finite_filter(m_r["y0"], f"restart{i} y0")
        r_y1 = finite_filter(m_r["y1"], f"restart{i} y1")
        r_ate, r_drop, r_keep = finite_ate(m_r["y0"], m_r["y1"])
        print(f"    restart{i} ATE (paired, finite) = {r_ate:.4f}  "
              f"(dropped {r_drop}/{r_drop + r_keep} pairs)\n")
        restart_y0.append(r_y0)
        restart_y1.append(r_y1)
        restart_ates.append(r_ate)

    # ============================================================ figure
    print("=== building figure ===")
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    x_grid = np.linspace(-3, 16, 400)
    true0 = true_pdf(x_grid, THETA0)
    true1 = true_pdf(x_grid, THETA1)

    # --- Panel A: n=20,000 fitted margins vs truth ---
    axA.plot(x_grid, true0, color=TRUTH_COLOR, lw=2.5, ls="-", label="true p(Y|do(0))")
    axA.plot(x_grid, true1, color=TRUTH_COLOR, lw=2.5, ls="--", label="true p(Y|do(1))")

    spline_kde0 = kde_on_grid(spline_y0, x_grid)
    spline_kde1 = kde_on_grid(spline_y1, x_grid)
    axA.plot(x_grid, spline_kde0, color=SPLINE_COLOR, lw=2, ls="-", label="spline p̂(Y|do(0))")
    axA.plot(x_grid, spline_kde1, color=SPLINE_COLOR, lw=2, ls="--", label="spline p̂(Y|do(1))")

    add_kde0 = kde_on_grid(add_y0, x_grid)
    add_kde1 = kde_on_grid(add_y1, x_grid)
    axA.plot(x_grid, add_kde0, color=ADDITIVE_COLOR, lw=2, ls="-", label="additive p̂(Y|do(0))")
    axA.plot(x_grid, add_kde1, color=ADDITIVE_COLOR, lw=2, ls="--", label="additive p̂(Y|do(1))")

    axA.axvline(0.0, color="gray", lw=1, alpha=0.6, zorder=0)

    if frac_neg_add1 > 0.01:
        axA.annotate(
            f"additive puts {frac_neg_add1:.1%} of do(1) mass at Y<0",
            xy=(0.0, add_kde1[np.searchsorted(x_grid, 0.0)]),
            xytext=(2.2, axA.get_ylim()[1] * 0.55) if axA.get_ylim()[1] > 0 else (2.2, 0.1),
            fontsize=9, color=ADDITIVE_COLOR,
            arrowprops=dict(arrowstyle="-", color=ADDITIVE_COLOR, lw=0.8),
        )
    elif frac_neg_add0 > 0.01:
        axA.annotate(
            f"additive puts {frac_neg_add0:.1%} of do(0) mass at Y<0",
            xy=(0.0, add_kde0[np.searchsorted(x_grid, 0.0)]),
            xytext=(2.2, axA.get_ylim()[1] * 0.55) if axA.get_ylim()[1] > 0 else (2.2, 0.1),
            fontsize=9, color=ADDITIVE_COLOR,
            arrowprops=dict(arrowstyle="-", color=ADDITIVE_COLOR, lw=0.8),
        )

    axA.set_title(f"n = 20,000: fitted margins vs truth (β=1)")
    axA.set_xlabel("Y")
    axA.set_ylabel("density")
    axA.set_xlim(x_grid[0], x_grid[-1])
    style_axis(axA)
    panel_letter(axA, "A")

    # Two-part legend: colour = arm, linestyle = do(0)/do(1).
    colour_handles = [
        Line2D([0], [0], color=TRUTH_COLOR, lw=2.5, label="truth"),
        Line2D([0], [0], color=SPLINE_COLOR, lw=2, label="spline (flexible_continuous)"),
        Line2D([0], [0], color=ADDITIVE_COLOR, lw=2, label="additive (gaussian)"),
    ]
    style_handles = [
        Line2D([0], [0], color="gray", lw=2, ls="-", label="do(0)"),
        Line2D([0], [0], color="gray", lw=2, ls="--", label="do(1)"),
    ]
    leg1 = axA.legend(handles=colour_handles, loc="upper right", fontsize=8.5,
                       frameon=True, framealpha=0.9, edgecolor="none", title="arm")
    axA.add_artist(leg1)
    axA.legend(handles=style_handles, loc="center right", fontsize=8.5,
               frameon=True, framealpha=0.9, edgecolor="none", title="intervention")

    # --- Panel B: n=2,000 spline restart stability ---
    axB.plot(x_grid, true0, color=TRUTH_COLOR, lw=2.5, ls="-", label="true p(Y|do(0))")
    axB.plot(x_grid, true1, color=TRUTH_COLOR, lw=2.5, ls="--", label="true p(Y|do(1))")

    for i in range(3):
        k0 = kde_on_grid(restart_y0[i], x_grid)
        k1 = kde_on_grid(restart_y1[i], x_grid)
        axB.plot(x_grid, k0, color=SPLINE_COLOR, lw=1.5, ls="-", alpha=0.55,
                  label="restarts p̂(Y|do(0))" if i == 0 else None)
        axB.plot(x_grid, k1, color=SPLINE_COLOR, lw=1.5, ls="--", alpha=0.55,
                  label="restarts p̂(Y|do(1))" if i == 0 else None)

    axB.set_title("n = 2,000: spline margin across 3 restarts (same data)")
    axB.set_xlabel("Y")
    axB.set_xlim(x_grid[0], x_grid[-1])
    style_axis(axB)
    panel_letter(axB, "B")
    axB.legend(loc="upper right", fontsize=8.5, frameon=True, framealpha=0.9, edgecolor="none")

    ate_str = " / ".join(f"{a:.2f}" for a in restart_ates)
    axB.text(
        0.97, 0.55, f"restart ATEs: {ate_str}",
        transform=axB.transAxes, fontsize=9.5, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc"),
    )

    fig.suptitle("Fitted causal-margin densities vs true interventional densities (Gamma DGP)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_path = os.path.join(OUTDIR, "fig4_margin_densities.png")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    size = os.path.getsize(out_path)
    ok = size > 20000
    print(f"\nPNG: {out_path}  ({size} bytes)  {'OK' if ok else 'TOO SMALL -- CHECK'}")
    assert os.path.exists(out_path), f"missing file: {out_path}"
    assert ok, f"file too small, likely broken: {out_path}"

    print(f"\ntotal runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
