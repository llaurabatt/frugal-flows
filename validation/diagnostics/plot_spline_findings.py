"""Three publication-quality figures summarising the spline-vs-additive ATE findings.

Headline story (see SPLINE_BIAS_FINDINGS.md for the fuller writeup):
  1. At n=20,000, an additive-shift causal margin (`gaussian`) develops large,
     confounding-beta-dependent bias, while the flexible spline margin
     (`flexible_continuous`) stays close to zero across the same beta sweep.
     The same qualitative pattern already shows up at n=2,000.
  2. At n=2,000 the spline arm is unstable: different random restarts on the
     SAME dataset land on visibly different tau(u) curves. Decomposing each
     restart curve into a level (its own mean) and a de-meaned shape shows the
     restart-to-restart spread is almost entirely a LEVEL shift -- the shapes
     collapse onto (and correlate strongly with) the truth once re-centred.
  3. That level instability is a real restart-noise phenomenon (not a single
     bad seed): 8 restarts on fixed data swing the ATE readout by ~0.13-0.33
     depending on (dgp, optimizer config), and restart noise is minimised
     (not maximised) at an intermediate spline capacity (RQS_knots=8), even
     as mean bias keeps improving with more knots.

Data sources (see docstring inline comments at each loader for exact provenance):
  - /home/danielmanela_gmail_com/round2_outputs/*.csv
  - .../validation/outputs/flexible_te/h1_shard*.csv  (n=20k, beta=1, both arms)
  - /home/danielmanela_gmail_com/round2_outputs/spline_stability_v2_curves.npz

Usage (from validation/, in the frugal-flows-flowjax env, on a shared 2-core box):
  env OMP_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false" \\
      ~/.local/bin/micromamba run -n frugal-flows-flowjax \\
      python diagnostics/plot_spline_findings.py
"""

from __future__ import annotations

import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROUND2_DIR = "/home/danielmanela_gmail_com/round2_outputs"
FLEXTE_DIR = (
    "/home/danielmanela_gmail_com/gdrive/work/Oxford/frugal_flows_project/"
    "frugal-flows/validation/outputs/flexible_te"
)
OUT_DIR = os.path.join(FLEXTE_DIR)  # save PNGs alongside the source CSVs

TRUE_ATE = 1.7634072418790194

# Consistent colour scheme across ALL figures: additive-shift arm vs spline arm.
# Okabe-Ito colorblind-safe palette.
ADDITIVE_COLOR = "#0072B2"  # blue
SPLINE_COLOR = "#D55E00"  # vermillion
TRUTH_COLOR = "black"
NEUTRAL_GRAY = "#555555"

DPI = 200

# ----------------------------------------------------------------------------
# Shared publication style
# ----------------------------------------------------------------------------
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


def style_axis(ax, keep_right_spine: bool = False) -> None:
    """Apply shared despine + light-grid styling to a single axis."""
    ax.spines["top"].set_visible(False)
    if not keep_right_spine:
        ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.25, linewidth=0.6)


def panel_letter(ax, letter: str) -> None:
    """Bold panel label ('A'/'B') just outside the top-left corner of an axis."""
    ax.text(
        -0.08, 1.05, letter, transform=ax.transAxes,
        fontsize=14, fontweight="bold", va="bottom", ha="right",
    )


# ----------------------------------------------------------------------------
# Generic CSV helpers
# ----------------------------------------------------------------------------


def read_csv_rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def drop_error_rows(rows: list[dict]) -> list[dict]:
    """Skip any row with a non-empty 'error' field (per task spec)."""
    return [r for r in rows if not (r.get("error") or "").strip()]


def col_floats(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows], dtype=float)


def mean_std(rows: list[dict], key: str) -> tuple[float, float]:
    """Population mean/std (ddof=0) of a numeric column over the given rows."""
    vals = col_floats(rows, key)
    return float(vals.mean()), float(vals.std(ddof=0))


def filt(rows: list[dict], **kwargs) -> list[dict]:
    out = rows
    for k, v in kwargs.items():
        out = [r for r in out if r.get(k) == v]
    return out


# ----------------------------------------------------------------------------
# FIGURE 1 data
# ----------------------------------------------------------------------------


def load_fig1_data():
    # --- Panel A: n = 20,000 ---
    # additive (gaussian) beta=0
    add_b0 = drop_error_rows(
        read_csv_rows(os.path.join(ROUND2_DIR, "h1_beta20k_additive_b0.csv"))
    )
    add_b0 = filt(add_b0, dgp="gamma_b0", arm="gaussian", n="20000")

    # spline (flexible_continuous) beta=0
    spl_b0 = drop_error_rows(
        read_csv_rows(os.path.join(ROUND2_DIR, "h1_beta20k_spline_b0.csv"))
    )
    spl_b0 = filt(spl_b0, dgp="gamma_b0", arm="flexible_continuous", n="20000")

    # spline (flexible_continuous) beta=1.5
    spl_b15 = drop_error_rows(
        read_csv_rows(os.path.join(ROUND2_DIR, "h1_beta20k_spline_b1.5.csv"))
    )
    spl_b15 = filt(spl_b15, dgp="gamma_b1.5", arm="flexible_continuous", n="20000")

    # additive (gaussian) beta=1.5
    add_b15 = drop_error_rows(
        read_csv_rows(os.path.join(ROUND2_DIR, "h1_beta20k_additive_b1.5.csv"))
    )
    add_b15 = filt(add_b15, dgp="gamma_b1.5", arm="gaussian", n="20000")

    # beta=1, BOTH arms -- pulled from the flexible_te h1_shard*.csv shards.
    # These shards use dgp=='gamma' (no confound_beta column; beta=1 by
    # construction of that sweep) and have no consistent per-seed coverage
    # across shards -- we take whatever non-error rows exist.
    shard_rows: list[dict] = []
    for path in sorted(glob.glob(os.path.join(FLEXTE_DIR, "h1_shard*.csv"))):
        shard_rows.extend(read_csv_rows(path))
    shard_rows = drop_error_rows(shard_rows)
    gamma_20k = filt(shard_rows, dgp="gamma", n="20000")
    add_b1 = filt(gamma_20k, arm="gaussian")
    spl_b1 = filt(gamma_20k, arm="flexible_continuous")

    panel_a = {
        "additive": {
            "beta": [0.0, 1.0, 1.5],
            "mean": [
                mean_std(add_b0, "bias")[0],
                mean_std(add_b1, "bias")[0],
                mean_std(add_b15, "bias")[0],
            ],
            "std": [
                mean_std(add_b0, "bias")[1],
                mean_std(add_b1, "bias")[1],
                mean_std(add_b15, "bias")[1],
            ],
            "n": [len(add_b0), len(add_b1), len(add_b15)],
        },
        "spline": {
            "beta": [0.0, 1.0, 1.5],
            "mean": [
                mean_std(spl_b0, "bias")[0],
                mean_std(spl_b1, "bias")[0],
                mean_std(spl_b15, "bias")[0],
            ],
            "std": [
                mean_std(spl_b0, "bias")[1],
                mean_std(spl_b1, "bias")[1],
                mean_std(spl_b15, "bias")[1],
            ],
            "n": [len(spl_b0), len(spl_b1), len(spl_b15)],
        },
    }

    # --- Panel B: n = 2,000, beta-sweep, both arms, 10 seeds each ---
    beta_rows: list[dict] = []
    for shard in ("h1_beta_shard0.csv", "h1_beta_shard1.csv"):
        beta_rows.extend(read_csv_rows(os.path.join(ROUND2_DIR, shard)))
    beta_rows = drop_error_rows(beta_rows)
    beta_rows = filt(beta_rows, n="2000")

    dgp_by_beta = {"gamma_b0": 0.0, "gamma_b0.5": 0.5, "gamma_b1": 1.0, "gamma_b1.5": 1.5}
    panel_b = {"additive": {"beta": [], "mean": [], "std": [], "n": []},
               "spline": {"beta": [], "mean": [], "std": [], "n": []}}
    for dgp, beta in dgp_by_beta.items():
        add_rows = filt(beta_rows, dgp=dgp, arm="gaussian")
        spl_rows = filt(beta_rows, dgp=dgp, arm="flexible_continuous")
        m, s = mean_std(add_rows, "bias")
        panel_b["additive"]["beta"].append(beta)
        panel_b["additive"]["mean"].append(m)
        panel_b["additive"]["std"].append(s)
        panel_b["additive"]["n"].append(len(add_rows))
        m, s = mean_std(spl_rows, "bias")
        panel_b["spline"]["beta"].append(beta)
        panel_b["spline"]["mean"].append(m)
        panel_b["spline"]["std"].append(s)
        panel_b["spline"]["n"].append(len(spl_rows))

    return panel_a, panel_b


def make_fig1():
    panel_a, panel_b = load_fig1_data()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)

    marker_kw = dict(markeredgecolor="white", markeredgewidth=0.8)

    # --- Panel A ---
    a = panel_a["additive"]
    s = panel_a["spline"]
    axA.errorbar(
        a["beta"], a["mean"], yerr=a["std"], marker="o", markersize=7, lw=2,
        capsize=4, elinewidth=1.5, color=ADDITIVE_COLOR,
        label="additive shift (gaussian margin)", **marker_kw,
    )
    axA.errorbar(
        s["beta"], s["mean"], yerr=s["std"], marker="s", markersize=7, lw=2,
        capsize=4, elinewidth=1.5, color=SPLINE_COLOR,
        label="flexible spline margin", **marker_kw,
    )
    axA.axhline(0.0, ls="--", color="gray", lw=1)
    axA.annotate(
        "erratic: 9/10 seeds overshoot,\none at −1.74 (sd 0.67)",
        xy=(1.5, a["mean"][2]), xytext=(0.98, a["mean"][2] + 0.42),
        fontsize=9, color=ADDITIVE_COLOR,
        arrowprops=dict(arrowstyle="-", color=ADDITIVE_COLOR, lw=0.8),
    )
    axA.set_title("n = 20,000")
    axA.set_xlabel("confounding β")
    axA.set_ylabel("ATE bias (est − truth)")
    axA.set_xticks([0.0, 1.0, 1.5])
    axA.legend(loc="lower left", frameon=True, framealpha=0.9, edgecolor="none")
    style_axis(axA)
    panel_letter(axA, "A")

    # --- Panel B ---
    a = panel_b["additive"]
    s = panel_b["spline"]
    axB.errorbar(
        a["beta"], a["mean"], yerr=a["std"], marker="o", markersize=7, lw=2,
        capsize=4, elinewidth=1.5, color=ADDITIVE_COLOR,
        label="additive shift (gaussian margin)", **marker_kw,
    )
    axB.errorbar(
        s["beta"], s["mean"], yerr=s["std"], marker="s", markersize=7, lw=2,
        capsize=4, elinewidth=1.5, color=SPLINE_COLOR,
        label="flexible spline margin", **marker_kw,
    )
    axB.axhline(0.0, ls="--", color="gray", lw=1)
    axB.set_title("n = 2,000 (β-sweep)")
    axB.set_xlabel("confounding β")
    axB.set_xticks([0.0, 0.5, 1.0, 1.5])
    axB.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="none")
    style_axis(axB)
    panel_letter(axB, "B")

    fig.suptitle(
        "Spline causal margin resists confounding-strength bias; additive shift does not",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = os.path.join(OUT_DIR, "fig1_spline_vs_additive_bias.png")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path, panel_a, panel_b


# ----------------------------------------------------------------------------
# FIGURE 2 data
# ----------------------------------------------------------------------------


def make_fig2():
    npz = np.load(os.path.join(ROUND2_DIR, "spline_stability_v2_curves.npz"))
    u_grid = npz["u_grid"]
    truth = npz["gamma_b1|true"]
    restarts = [npz[f"gamma_b1|baseline|r{i}"] for i in range(8)]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True)

    # --- Panel A: raw tau_hat(u) per restart ---
    for i, r in enumerate(restarts):
        axA.plot(
            u_grid, r, color=SPLINE_COLOR, alpha=0.5, lw=1,
            label="restarts (n=8)" if i == 0 else None,
        )
    axA.plot(u_grid, truth, color=TRUTH_COLOR, lw=3, label="truth")
    axA.set_title("τ̂(u) per restart (n=2000, fixed data)")
    axA.set_xlabel("control-outcome rank u")
    axA.set_ylabel("τ̂(u)")
    axA.legend(loc="best", frameon=True, framealpha=0.9, edgecolor="none")
    style_axis(axA)
    panel_letter(axA, "A")

    # --- Panel B: de-leveled (own-mean-subtracted) curves ---
    truth_deleveled = truth - truth.mean()
    corrs = []
    for i, r in enumerate(restarts):
        r_deleveled = r - r.mean()
        corr = float(np.corrcoef(r_deleveled, truth_deleveled)[0, 1])
        corrs.append(corr)
        axB.plot(
            u_grid, r_deleveled, color=SPLINE_COLOR, alpha=0.5, lw=1,
            label="restarts, de-leveled (n=8)" if i == 0 else None,
        )
    axB.plot(u_grid, truth_deleveled, color=TRUTH_COLOR, lw=3, label="truth, de-meaned")
    axB.set_title("shape after removing each curve's level")
    axB.set_xlabel("control-outcome rank u")
    axB.legend(loc="best", frameon=True, framealpha=0.9, edgecolor="none")
    style_axis(axB)
    panel_letter(axB, "B")

    corr_min, corr_max = min(corrs), max(corrs)
    axB.text(
        0.03, 0.03,
        f"shape–truth corr: {corr_min:.2f}–{corr_max:.2f}",
        transform=axB.transAxes, fontsize=10, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc"),
    )

    fig.suptitle(
        "Restart spread in the spline margin is a LEVEL shift, not a shape change",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = os.path.join(OUT_DIR, "fig2_level_vs_shape.png")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path, corrs


# ----------------------------------------------------------------------------
# FIGURE 3 data
# ----------------------------------------------------------------------------


def make_fig3():
    stab_rows = drop_error_rows(
        read_csv_rows(os.path.join(ROUND2_DIR, "spline_stability_v2.csv"))
    )
    cap_rows = drop_error_rows(
        read_csv_rows(os.path.join(ROUND2_DIR, "spline_capacity_gamma.csv"))
    )

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # --- Panel A: restart spread on fixed data ---
    groups = [
        ("gamma_b0", "baseline"),
        ("gamma_b0", "lowlr"),
        ("gamma_b1", "baseline"),
        ("gamma_b1", "lowlr"),
    ]
    group_labels = [f"{dgp}\n/{cfg}" for dgp, cfg in groups]
    rng = np.random.default_rng(0)
    group_stats = []
    all_ates = []
    for i, (dgp, cfg) in enumerate(groups):
        rows = filt(stab_rows, dgp=dgp, config=cfg)
        ates = col_floats(rows, "ate")
        all_ates.append(ates)
        m, sd = float(ates.mean()), float(ates.std(ddof=0))
        group_stats.append((m, sd, len(ates)))
        jitter = rng.uniform(-0.12, 0.12, size=len(ates))
        axA.scatter(
            np.full(len(ates), i) + jitter, ates,
            color=SPLINE_COLOR, alpha=0.7, s=45, zorder=3,
            edgecolors="white", linewidths=0.8,
            label="per-restart ATE" if i == 0 else None,
        )
        axA.scatter(
            [i], [m], color="black", marker="D", s=70, zorder=4,
            edgecolors="white", linewidths=0.8,
            label="group mean" if i == 0 else None,
        )

    # Headroom so the sd-annotations and the true-ATE line don't collide with
    # the axis edge or each other.
    ymax = max(TRUE_ATE, max(a.max() for a in all_ates))
    ymin = min(a.min() for a in all_ates)
    span = ymax - ymin
    axA.set_ylim(ymin - 0.08 * span, ymax + 0.22 * span)
    for i, ates in enumerate(all_ates):
        sd = group_stats[i][1]
        axA.annotate(
            f"sd={sd:.2f}", xy=(i, ates.max() + 0.05 * span), ha="center", fontsize=9,
        )

    axA.axhline(TRUE_ATE, ls="--", color="gray", lw=1.2, label=f"true ATE ({TRUE_ATE:.3f})")
    axA.set_xticks(range(len(groups)))
    axA.set_xticklabels(group_labels, fontsize=9)
    axA.set_ylabel("ATE")
    axA.set_title("restart spread on FIXED data (n=2000)")
    axA.legend(loc="lower center", ncol=3, frameon=True, framealpha=0.9, edgecolor="none")
    style_axis(axA)
    panel_letter(axA, "A")

    # --- Panel B: capacity vs restart noise ---
    knots = [4, 8, 16]
    bias_means, bias_stds, ate_sds = [], [], []
    for k in knots:
        rows = filt(cap_rows, dgp="gamma_b1", RQS_knots=str(k))
        bm, bs = mean_std(rows, "bias")
        ates = col_floats(rows, "ate")
        bias_means.append(bm)
        bias_stds.append(bs)
        ate_sds.append(float(ates.std(ddof=0)))

    l1 = axB.errorbar(
        knots, bias_means, yerr=bias_stds, marker="o", markersize=7, lw=2,
        capsize=4, elinewidth=1.5, color=SPLINE_COLOR,
        label="mean ATE bias (± sd, left axis)",
        markeredgecolor="white", markeredgewidth=0.8,
    )
    axB.axhline(0.0, ls=":", color="gray", lw=1)
    axB.set_xlabel("RQS knots")
    axB.set_ylabel("mean ATE bias", color=SPLINE_COLOR)
    axB.tick_params(axis="y", labelcolor=SPLINE_COLOR)
    axB.set_xticks(knots)
    style_axis(axB, keep_right_spine=True)
    panel_letter(axB, "B")

    axB2 = axB.twinx()
    l2 = axB2.plot(
        knots, ate_sds, marker="s", markersize=7, lw=2, ls="--",
        color=NEUTRAL_GRAY, label="restart sd(ate) (right axis)",
        markeredgecolor="white", markeredgewidth=0.8,
    )
    axB2.set_ylabel("restart sd(ate)", color=NEUTRAL_GRAY)
    axB2.tick_params(axis="y", labelcolor=NEUTRAL_GRAY)
    axB2.spines["right"].set_visible(True)
    axB2.spines["right"].set_color(NEUTRAL_GRAY)
    axB2.spines["top"].set_visible(False)
    axB2.grid(False)

    min_idx = int(np.argmin(ate_sds))
    axB2.annotate(
        "sweet spot",
        xy=(knots[min_idx], ate_sds[min_idx]),
        xytext=(knots[min_idx] + 1.8, ate_sds[min_idx] + 0.06),
        fontsize=10, color=NEUTRAL_GRAY, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=NEUTRAL_GRAY, lw=1.2),
        ha="left", va="center",
    )

    lines = [l1] + l2
    labels = [ln.get_label() for ln in lines]
    axB.legend(lines, labels, loc="upper left", frameon=True, framealpha=0.9, edgecolor="none")
    axB.set_title("capacity: mean bias vs restart-noise (gamma_b1, n=2000)")

    fig.suptitle(
        "Restart instability is real (not one bad seed) and non-monotone in spline capacity",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = os.path.join(OUT_DIR, "fig3_stability_and_capacity.png")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path, group_stats, knots, bias_means, ate_sds


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    paths = []

    p1, panel_a, panel_b = make_fig1()
    paths.append(p1)

    p2, corrs = make_fig2()
    paths.append(p2)

    p3, group_stats, knots, bias_means, ate_sds = make_fig3()
    paths.append(p3)

    print("=== Saved figures ===")
    for p in paths:
        size = os.path.getsize(p)
        ok = size > 20000
        print(f"{p}  ({size} bytes)  {'OK' if ok else 'TOO SMALL -- CHECK'}")
        assert os.path.exists(p), f"missing file: {p}"
        assert ok, f"file too small, likely broken: {p}"

    print("\n=== Fig 1 sanity: n=20k bias mean (± std, n_rows) per arm/beta ===")
    print("additive (gaussian):")
    for beta, m, s, n in zip(
        panel_a["additive"]["beta"], panel_a["additive"]["mean"],
        panel_a["additive"]["std"], panel_a["additive"]["n"],
    ):
        print(f"  beta={beta}: mean={m:.4f}  std={s:.4f}  n={n}")
    print("spline (flexible_continuous):")
    for beta, m, s, n in zip(
        panel_a["spline"]["beta"], panel_a["spline"]["mean"],
        panel_a["spline"]["std"], panel_a["spline"]["n"],
    ):
        print(f"  beta={beta}: mean={m:.4f}  std={s:.4f}  n={n}")

    print("\n=== Fig 1 sanity: n=2000 beta-sweep bias mean (± std, n_rows) per arm/beta ===")
    print("additive (gaussian):")
    for beta, m, s, n in zip(
        panel_b["additive"]["beta"], panel_b["additive"]["mean"],
        panel_b["additive"]["std"], panel_b["additive"]["n"],
    ):
        print(f"  beta={beta}: mean={m:.4f}  std={s:.4f}  n={n}")
    print("spline (flexible_continuous):")
    for beta, m, s, n in zip(
        panel_b["spline"]["beta"], panel_b["spline"]["mean"],
        panel_b["spline"]["std"], panel_b["spline"]["n"],
    ):
        print(f"  beta={beta}: mean={m:.4f}  std={s:.4f}  n={n}")

    print("\n=== Fig 2 sanity: per-restart shape-vs-truth Pearson corr (de-meaned) ===")
    print("  " + ", ".join(f"{c:.3f}" for c in corrs))
    print(f"  range: {min(corrs):.3f}-{max(corrs):.3f}")

    print("\n=== Fig 3 sanity: restart-spread groups (mean, sd, n) ===")
    labels = ["gamma_b0/baseline", "gamma_b0/lowlr", "gamma_b1/baseline", "gamma_b1/lowlr"]
    for label, (m, sd, n) in zip(labels, group_stats):
        print(f"  {label}: mean_ate={m:.4f}  sd_ate={sd:.4f}  n={n}")

    print("\n=== Fig 3 sanity: capacity sweep (gamma_b1, n=2000) ===")
    for k, bm, asd in zip(knots, bias_means, ate_sds):
        print(f"  knots={k}: mean_bias={bm:.4f}  restart_sd(ate)={asd:.4f}")


if __name__ == "__main__":
    main()
