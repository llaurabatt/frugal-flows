"""Aggregate the spline-arm diagnostics and render the diagnostic figure.

Reads:
  outputs/spline_shard*.csv    (spline_ablation.py: conditions x n x seeds)
  outputs/splineep_shard*.csv  (bias_epochs.py --arm flexible_continuous: epochs grid)

Renders a 2x2 panel (phone-friendly):
  A) ATE vs n by condition {baseline, unconfounded, standardized} + true ATE.
  B) tau_sd (effect heterogeneity) vs n by condition. TRUE value = 0 (the gaussian
     outcome has a constant location-shift effect), so any tau_sd>0 is SPURIOUS
     heterogeneity; it should decay to 0 with n.
  C) ATE vs epoch budget (early-stop OFF) at fixed n — does the spline drift?
  D) tau_sd vs epoch budget (early-stop OFF) — does longer training INFLATE
     spurious heterogeneity (overfitting)?

Usage (from validation/):
  micromamba run -n frugal-flows-flowjax python -m diagnostics.spline_plot
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))

COND_LABEL = {"baseline": "baseline (confounded, raw Y)",
              "unconfounded": "unconfounded (X⊥Z, raw Y)",
              "standardized": "standardized Y"}
COND_COLOR = {"baseline": "#d62728", "unconfounded": "#ff7f0e", "standardized": "#9467bd"}


def _load(pattern):
    paths = sorted(glob.glob(os.path.join(_HERE, "..", "outputs", pattern)))
    if not paths:
        paths = sorted(glob.glob(os.path.join(_HERE, "outputs", pattern)))
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True) if paths else None


def _agg(df, group, val):
    df[val] = pd.to_numeric(df[val], errors="coerce")
    g = df.dropna(subset=[val]).groupby(group)
    return g[val].agg(["mean", "std", "size"]).reset_index()


def _line_by_cond(ax, abl, conditions, val, ns_ticks):
    for cond in conditions:
        s = _agg(abl[abl["condition"] == cond], "n", val).sort_values("n")
        c = COND_COLOR[cond]
        ax.plot(s["n"], s["mean"], "-o", color=c, lw=2, ms=7, label=COND_LABEL[cond])
        ax.fill_between(s["n"], s["mean"] - s["std"], s["mean"] + s["std"], color=c, alpha=0.16)
    ax.set_xscale("log"); ax.set_xticks(ns_ticks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("training sample size n (log)", fontsize=12)
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="best")


def _line_by_n(ax, eps, val, eps_ticks):
    for n_val, mk in zip(sorted(eps["n"].unique()), ["-o", "-s", "-^"]):
        s = _agg(eps[eps["n"] == n_val], "epochs", val).sort_values("epochs")
        ax.plot(s["epochs"], s["mean"], mk, lw=2, ms=7, label=f"n={n_val}")
        ax.fill_between(s["epochs"], s["mean"] - s["std"], s["mean"] + s["std"], alpha=0.16)
    ax.set_xscale("log"); ax.set_xticks(eps_ticks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("epoch budget (early-stop OFF, log)", fontsize=12)
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="best")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default=os.path.join(_HERE, "outputs"))
    args = p.parse_args()

    abl = _load("spline_shard*.csv")
    eps = _load("splineep_shard*.csv")
    if abl is None:
        raise SystemExit("no spline_shard*.csv found")
    true_ate = float(pd.to_numeric(abl["true_ate"]).iloc[0])
    conditions = [c for c in ["baseline", "unconfounded", "standardized"] if c in set(abl["condition"])]
    ns_ticks = sorted(abl["n"].unique())

    fig, axes2 = plt.subplots(2, 2, figsize=(13, 10.2))
    axes = axes2.ravel()

    # A: ATE vs n
    axes[0].axhline(true_ate, color="black", ls="--", lw=1.8, label=f"true ATE = {true_ate:+.2f}")
    _line_by_cond(axes[0], abl, conditions, "ate_crn", ns_ticks)
    axes[0].set_ylabel("estimated ATE", fontsize=12)
    axes[0].set_title("A. spline ATE vs n (recovers early)\nby condition", fontsize=12)

    # B: tau_sd vs n
    axes[1].axhline(0.0, color="black", ls="--", lw=1.8, label="true heterogeneity = 0")
    _line_by_cond(axes[1], abl, conditions, "tau_sd", ns_ticks)
    axes[1].set_ylabel("tau_sd (effect heterogeneity)", fontsize=12)
    axes[1].set_title("B. SPURIOUS heterogeneity vs n\n(true=0; should decay)", fontsize=12)

    # C: ATE vs epochs
    axes[2].axhline(true_ate, color="black", ls="--", lw=1.8, label=f"true ATE = {true_ate:+.2f}")
    if eps is not None:
        _line_by_n(axes[2], eps, "ate_crn", sorted(eps["epochs"].unique()))
    else:
        axes[2].text(0.5, 0.5, "no splineep_shard*.csv yet", ha="center", transform=axes[2].transAxes)
    axes[2].set_ylabel("estimated ATE", fontsize=12)
    axes[2].set_title("C. spline ATE vs epochs (early-stop off)", fontsize=12)

    # D: tau_sd vs epochs
    axes[3].axhline(0.0, color="black", ls="--", lw=1.8, label="true heterogeneity = 0")
    if eps is not None:
        _line_by_n(axes[3], eps, "tau_sd", sorted(eps["epochs"].unique()))
    else:
        axes[3].text(0.5, 0.5, "no splineep_shard*.csv yet", ha="center", transform=axes[3].transAxes)
    axes[3].set_ylabel("tau_sd (effect heterogeneity)", fontsize=12)
    axes[3].set_title("D. does longer training inflate\nspurious heterogeneity?", fontsize=12)

    fig.suptitle("Diagnostics for the SPLINE (flexible_continuous) arm — gaussian outcome", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(args.outdir, "spline_diagnosis.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)

    # printed summary
    for val, name in [("ate_crn", "ATE"), ("tau_sd", "tau_sd (spurious heterogeneity)")]:
        abl[val] = pd.to_numeric(abl[val], errors="coerce")
        print(f"\n{name} vs n (mean over seeds):")
        print(abl.pivot_table(index="condition", columns="n", values=val, aggfunc="mean")
              .to_string(float_format=lambda v: f"{v:+.3f}"))
    print(f"\nfigure: {out}")


if __name__ == "__main__":
    main()
