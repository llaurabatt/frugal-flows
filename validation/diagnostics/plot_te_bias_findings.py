"""Figures for the small-n ATE-bias study (log-cold spline causal margin, Gamma DGP).

Pure plotting: reads the committed study CSVs in
``validation/outputs/flexible_te/te_bias_study/`` (no training) and writes two
figures next to the other flexible-te figures. Regenerate with:

    cd validation && micromamba run -n frugal-flows-flowjax python -m diagnostics.plot_te_bias_findings

Data (all Gamma ``gamma_b1`` unless noted, ``flexible_continuous`` spline, cold start,
15 seeds where applicable, true ATE = 1.7634):
  nsweep_logcold.csv  -- log-cold ATE vs n (n in {200,500,1000,2000,4000}, 5 seeds)
  std_decompose.csv   -- 4 arms log/center/scale/standardize x {500,2000} x 15 seeds
  confound_test.csv   -- gamma_b0 (unconfounded) vs gamma_b1 x {500,2000} x 15 seeds,
                         with fitted log-scale interventional means mu0_hat/mu1_hat

Findings the figures show:
  fig5 -- the log-cold ATE bias is ~0 at n>=2000 but grows (upward) as n shrinks;
          standardizing the fitting scale (center/scale/both) does NOT change it.
  fig6 -- that small-n bias is CONFOUNDING (collapses on the unconfounded DGP) and a
          LOCATION effect: under confounding the fitted control mean is pulled down.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TRUE_ATE = 1.7634
MU0_TRUE, MU1_TRUE = 0.7296, 1.2296
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "outputs", "flexible_te", "te_bias_study")
OUT = os.path.join(HERE, "..", "outputs", "flexible_te")
plt.rcParams.update({"figure.dpi": 120, "font.size": 11, "axes.grid": True, "grid.alpha": 0.3})


def _by_n(df):
    """mean/std/sem of ate and mean bias per n."""
    g = df.groupby("n")["ate"]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["sem"] = out["std"] / np.sqrt(out["count"])
    out["bias"] = out["mean"] - TRUE_ATE
    return out.sort_values("n")


# ------------------------------------------------------------------ fig5
def fig5():
    sweep = pd.read_csv(os.path.join(DATA, "nsweep_logcold.csv"))
    dec = pd.read_csv(os.path.join(DATA, "std_decompose.csv"))
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))

    # Panel A: log-cold mean ATE +/- SD vs n
    s = _by_n(sweep)
    axA.axhline(TRUE_ATE, ls="--", color="k", lw=1.2, label=f"true ATE = {TRUE_ATE}")
    axA.fill_between(s["n"], s["mean"] - s["std"], s["mean"] + s["std"], color="C0", alpha=0.2)
    axA.plot(s["n"], s["mean"], "o-", color="C0", label="log-cold mean ATE (±1 SD)")
    axA.scatter(sweep["n"], sweep["ate"], s=16, color="C0", alpha=0.35, zorder=1)
    axA.set_xscale("log")
    axA.set_xlabel("n (log scale)")
    axA.set_ylabel("estimated ATE")
    axA.set_title("Log-cold ATE recovery vs n\n(unbiased at n≥2000; upward bias grows as n shrinks)")
    axA.legend(fontsize=9)

    # Panel B: |mean bias| vs n for the 4 standardize-decomposition arms
    arms = ["log", "center", "scale", "std"]
    labels = {"log": "log", "center": "log+center", "scale": "log+scale", "std": "log+standardize"}
    for i, arm in enumerate(arms):
        d = dec[dec["arm"] == arm]
        g = d.groupby("n")["ate"].agg(["mean", "std", "count"]).reset_index().sort_values("n")
        g["sem"] = g["std"] / np.sqrt(g["count"])
        axB.errorbar(g["n"] * (1 + 0.03 * (i - 1.5)), (g["mean"] - TRUE_ATE).abs(), yerr=g["sem"],
                     fmt="o-", capsize=3, label=labels[arm], color=f"C{i}")
    axB.set_xscale("log")
    axB.set_xlabel("n (log scale)")
    axB.set_ylabel("|mean bias|")
    axB.set_title("Standardization is a no-op\n(log / center / scale / standardize coincide within SEM)")
    axB.legend(fontsize=9)
    axB.set_ylim(bottom=0)

    fig.tight_layout()
    p = os.path.join(OUT, "fig5_small_n_ate_bias.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


# ------------------------------------------------------------------ fig6
def fig6():
    c = pd.read_csv(os.path.join(DATA, "confound_test.csv"))
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))
    names = {"gamma_b0": "unconfounded (gamma_b0)", "gamma_b1": "confounded (gamma_b1)"}
    colors = {"gamma_b0": "C2", "gamma_b1": "C3"}

    # Panel A: mean bias +/- sem vs n, confounded vs unconfounded
    axA.axhline(0, ls="--", color="k", lw=1.2)
    for dgp in ("gamma_b0", "gamma_b1"):
        d = c[c["dgp"] == dgp]
        g = d.groupby("n")["bias"].agg(["mean", "std", "count"]).reset_index().sort_values("n")
        g["sem"] = g["std"] / np.sqrt(g["count"])
        axA.errorbar(g["n"], g["mean"], yerr=g["sem"], fmt="o-", capsize=4,
                     color=colors[dgp], label=names[dgp])
    axA.set_xscale("log")
    axA.set_xlabel("n (log scale)")
    axA.set_ylabel("mean ATE bias")
    axA.set_title("The small-n bias is CONFOUNDING\n(collapses on the unconfounded DGP)")
    axA.legend(fontsize=9)

    # Panel B: fitted interventional log-means at n=500 (location effect)
    sub = c[c["n"] == 500]
    xs = np.arange(2)  # do(0), do(1)
    w = 0.35
    for j, dgp in enumerate(("gamma_b0", "gamma_b1")):
        d = sub[sub["dgp"] == dgp]
        means = [d["mu0_hat"].mean(), d["mu1_hat"].mean()]
        sems = [d["mu0_hat"].std() / np.sqrt(len(d)), d["mu1_hat"].std() / np.sqrt(len(d))]
        axB.bar(xs + (j - 0.5) * w, means, w, yerr=sems, capsize=4, color=colors[dgp], label=names[dgp])
    axB.hlines([MU0_TRUE, MU1_TRUE], [-0.5, 0.5], [0.5, 1.5], color="k", ls="--", lw=1.4,
               label="true log-mean")
    axB.set_xticks(xs)
    axB.set_xticklabels(["E[log Y | do(0)]", "E[log Y | do(1)]"])
    axB.set_ylabel("fitted log-scale interventional mean")
    axB.set_title("...and it is a LOCATION effect (n=500)\n(confounding pulls the control mean down)")
    axB.legend(fontsize=9)
    axB.set_ylim(0.5, 1.35)

    fig.tight_layout()
    p = os.path.join(OUT, "fig6_confounding_location.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


# ------------------------------------------------------------------ fig7
def fig7():
    """Forest plot: mean ATE estimate (+/- SEM thick, +/- SD thin) for every study
    condition, against the true-ATE line, with faint per-seed dots."""
    sweep = pd.read_csv(os.path.join(DATA, "nsweep_logcold.csv"))
    dec = pd.read_csv(os.path.join(DATA, "std_decompose.csv"))
    conf = pd.read_csv(os.path.join(DATA, "confound_test.csv"))

    rows = []  # (label, section, ate_values)
    for n in sorted(sweep["n"].unique()):
        rows.append((f"n={n}", "n-sweep · log · cold", sweep.loc[sweep["n"] == n, "ate"].values))
    arm_lbl = {"log": "log", "center": "log+center", "scale": "log+scale", "std": "log+standardize"}
    for n in (500, 2000):
        for arm in ("log", "center", "scale", "std"):
            v = dec[(dec["arm"] == arm) & (dec["n"] == n)]["ate"].values
            rows.append((f"{arm_lbl[arm]} (n={n})", "standardize decompose", v))
    dgp_lbl = {"gamma_b0": "unconfounded", "gamma_b1": "confounded"}
    for n in (500, 2000):
        for dgp in ("gamma_b0", "gamma_b1"):
            v = conf[(conf["dgp"] == dgp) & (conf["n"] == n)]["ate"].values
            rows.append((f"{dgp_lbl[dgp]} (n={n})", "confounding test", v))

    sections = ["n-sweep · log · cold", "standardize decompose", "confounding test"]
    sec_color = {s: f"C{i}" for i, s in enumerate(sections)}
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.6))
    ax.axvline(TRUE_ATE, ls="--", color="k", lw=1.3, label=f"true ATE = {TRUE_ATE}")
    y = len(rows)
    ylabels, yticks = [], []
    for label, section, v in rows:
        v = np.asarray(v, dtype=float)
        m, sd, k = v.mean(), v.std(ddof=1), len(v)
        sem = sd / np.sqrt(k)
        c = sec_color[section]
        ax.scatter(v, np.full_like(v, y) + np.linspace(-0.12, 0.12, k), s=12, color=c, alpha=0.3, zorder=1)
        ax.plot([m - sd, m + sd], [y, y], color=c, lw=1.2, alpha=0.6, zorder=2)  # +/- SD (thin)
        ax.errorbar(m, y, xerr=sem, fmt="o", color=c, capsize=4, lw=2.4, ms=6, zorder=3)  # +/- SEM
        ylabels.append(label)
        yticks.append(y)
        y -= 1
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel("estimated ATE  (dot = mean, thick = ±SEM, thin = ±SD, faint = per-seed)")
    ax.set_title("Mean ATE estimates across study conditions\n(Gamma DGP, flexible_continuous spline; true ATE = 1.7634)")
    # section legend
    handles = [plt.Line2D([], [], color="k", ls="--", label=f"true ATE = {TRUE_ATE}")]
    handles += [plt.Line2D([], [], color=sec_color[s], marker="o", ls="", label=s) for s in sections]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    ax.margins(y=0.02)
    fig.tight_layout()
    p = os.path.join(OUT, "fig7_mean_ate_forest.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


if __name__ == "__main__":
    print("wrote:", fig5())
    print("wrote:", fig6())
    print("wrote:", fig7())
