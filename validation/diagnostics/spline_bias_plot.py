"""Aggregate the spline-bias battery into phone-friendly figures.

Reads (from validation/outputs/, fallback diagnostics/outputs/):
  taucurve_shard*.csv  (E3: per-seed tau(u) curves)   -> spline_tau_decomp.png  [centrepiece]
  capacity_shard*.csv  (E1: capacity sweep)           -> spline_capacity.png
  confound_shard*.csv  (E2: confounding sweep)        -> spline_confounding.png
  profile_shard*.csv   (E5: additive profile likelihood) -> additive_profile.png

Each figure is self-contained; missing inputs are skipped with a note.

Usage (from validation/):
  micromamba run -n frugal-flows-flowjax python -m diagnostics.spline_bias_plot
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUTDIR = os.path.join(_HERE, "outputs")

_VALIDATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VALIDATION_DIR not in sys.path:
    sys.path.insert(0, _VALIDATION_DIR)
from diagnostics.ate_extraction_suite import TAU_CURVE_BINS  # noqa: E402
from diagnostics.outcome_families import FAMILIES  # noqa: E402

U_CENTERS = (np.arange(TAU_CURVE_BINS) + 0.5) / TAU_CURVE_BINS

# condition -> (family key for analytic truth, nice label, colour)
COND = {
    "baseline": ("gaussian", "baseline (confounded)", "#d62728"),
    "unconfounded": ("gaussian_unconfounded", "unconfounded (X⊥Z)", "#ff7f0e"),
    "gamma": ("gamma", "gamma (genuine heterogeneity)", "#1f77b4"),
}


def _load(pattern):
    paths = sorted(glob.glob(os.path.join(_HERE, "..", "outputs", pattern)))
    if not paths:
        paths = sorted(glob.glob(os.path.join(_OUTDIR, pattern)))
    if not paths:
        return None
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def _num(df, *cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ------------------------------------------------------------------ E3 centrepiece
def fig_tau_decomp(cp, outdir):
    df = _load("taucurve_shard*.csv")
    if df is None:
        print("skip tau_decomp: no taucurve_shard*.csv")
        return None
    df = _num(df, "n", "seed", "true_ate", "tau_sd")
    conds = [c for c in COND if c in set(df["condition"])]
    ns = sorted(df["n"].dropna().unique())
    fig, axes = plt.subplots(len(conds), len(ns), figsize=(6.2 * len(ns), 3.6 * len(conds)),
                             squeeze=False)
    for i, cond in enumerate(conds):
        fam_key, label, color = COND[cond]
        fam = FAMILIES[fam_key]
        true_curve = fam.true_tau_curve(cp, U_CENTERS)
        for j, n in enumerate(ns):
            ax = axes[i][j]
            sub = df[(df["condition"] == cond) & (df["n"] == n)]
            curves = []
            for _, r in sub.iterrows():
                if not isinstance(r["tau_curve"], str):
                    continue
                tc = np.array([float(x) for x in r["tau_curve"].split(";")])
                curves.append(tc)
                ax.plot(U_CENTERS, tc, color="0.7", lw=0.7, alpha=0.6, zorder=1)
            if curves:
                A = np.vstack(curves)
                mean, sd = np.nanmean(A, 0), np.nanstd(A, 0)
                ax.fill_between(U_CENTERS, mean - sd, mean + sd, color=color, alpha=0.22,
                                zorder=2, label="seed ±1 SD (variance)")
                ax.plot(U_CENTERS, mean, color=color, lw=2.4, zorder=4,
                        label="seed-mean τ(u) (bias)")
            ax.plot(U_CENTERS, true_curve, "--", color="black", lw=1.8, zorder=5,
                    label="analytic truth")
            ts = sub["tau_sd"].mean()
            ax.set_title(f"{label}  |  n={int(n)}   (mean τ_sd={ts:.3f})", fontsize=11)
            ax.set_xlabel("outcome quantile u", fontsize=10)
            ax.set_ylabel("τ(u) = Q₁(u) − Q₀(u)", fontsize=10)
            ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc="best")
    fig.suptitle("Spline effect curve τ(u):  seed-MEAN = bias,  seed-SD envelope = variance\n"
                 "unconfounded: flat mean + wide envelope ⇒ VARIANCE (averages away).  "
                 "baseline: mean bowed off truth ⇒ confounding BIAS.  gamma: mean tracks rising truth ⇒ real heterogeneity",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(outdir, "spline_tau_decomp.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    # printed summary
    print("\n[E3] mean τ_sd by condition × n:")
    print(df.pivot_table(index="condition", columns="n", values="tau_sd", aggfunc="mean")
          .to_string(float_format=lambda v: f"{v:.3f}"))
    return out


# ------------------------------------------------------------------ E1 capacity
def fig_capacity(outdir):
    df = _load("capacity_shard*.csv")
    if df is None:
        print("skip capacity: no capacity_shard*.csv")
        return None
    df = _num(df, "n", "level", "tau_sd", "ate", "true_ate")
    ns = sorted(df["n"].dropna().unique())
    axes_names = ["knots", "depth", "layers"]
    axes_names = [a for a in axes_names if a in set(df["axis"])]
    fig, axes = plt.subplots(len(ns), len(axes_names),
                             figsize=(4.6 * len(axes_names), 3.6 * len(ns)), squeeze=False)
    for i, n in enumerate(ns):
        for j, axis in enumerate(axes_names):
            ax = axes[i][j]
            sub = df[(df["n"] == n) & (df["axis"] == axis)]
            g = sub.groupby("level")["tau_sd"].agg(["mean", "std"]).reset_index().sort_values("level")
            ax.errorbar(g["level"], g["mean"], yerr=g["std"], fmt="-o", lw=2, ms=7,
                        color="#1f77b4", capsize=4)
            ax.axhline(0.0, color="black", ls="--", lw=1.4, label="true τ_sd = 0")
            ax.set_title(f"n={int(n)} — {axis}", fontsize=11)
            ax.set_xlabel(f"{axis} level", fontsize=10)
            ax.set_ylabel("spurious τ_sd", fontsize=10)
            ax.grid(alpha=0.3)
            ax.set_ylim(bottom=0)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle("E1: spurious effect heterogeneity vs spline CAPACITY (homogeneous DGP, true τ_sd=0)\n"
                 "rising with capacity ⇒ over-flexibility (H1); flat ⇒ capacity is not the driver",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(outdir, "spline_capacity.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print("\n[E1] mean τ_sd by axis × level (n pooled):")
    print(df.pivot_table(index=["axis", "level"], columns="n", values="tau_sd", aggfunc="mean")
          .to_string(float_format=lambda v: f"{v:.3f}"))
    return out


# ------------------------------------------------------------------ E2 confounding
def fig_confounding(outdir):
    df = _load("confound_shard*.csv")
    if df is None:
        print("skip confounding: no confound_shard*.csv")
        return None
    df = _num(df, "beta", "n", "tau_sd", "ate", "obs_contrast", "true_ate")
    ns = sorted(df["n"].dropna().unique())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    # A: tau_sd vs beta
    for n in ns:
        g = df[df["n"] == n].groupby("beta")["tau_sd"].agg(["mean", "std"]).reset_index()
        axes[0].errorbar(g["beta"], g["mean"], yerr=g["std"], fmt="-o", lw=2, ms=7, capsize=4, label=f"n={int(n)}")
    axes[0].axhline(0.0, color="black", ls="--", lw=1.4, label="true τ_sd = 0")
    axes[0].set_title("A. spurious τ_sd vs confounding β\nrising ⇒ residual confounding (H2)", fontsize=11)
    axes[0].set_xlabel("confounding strength β"); axes[0].set_ylabel("spurious τ_sd")
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=9); axes[0].set_ylim(bottom=0)
    # B: ate vs beta
    for n in ns:
        g = df[df["n"] == n].groupby("beta")["ate"].agg(["mean", "std"]).reset_index()
        axes[1].errorbar(g["beta"], g["mean"], yerr=g["std"], fmt="-o", lw=2, ms=7, capsize=4, label=f"n={int(n)}")
    ta = float(df["true_ate"].dropna().iloc[0])
    axes[1].axhline(ta, color="black", ls="--", lw=1.4, label=f"true ATE = {ta:+.2f}")
    axes[1].set_title("B. ATE vs β (recovery stays intact?)", fontsize=11)
    axes[1].set_xlabel("confounding strength β"); axes[1].set_ylabel("estimated ATE")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=9)
    # C: obs contrast sanity
    g = df.groupby("beta")["obs_contrast"].mean().reset_index()
    axes[2].plot(g["beta"], g["obs_contrast"], "-s", lw=2, ms=7, color="#7f7f7f")
    axes[2].axhline(ta, color="black", ls="--", lw=1.4, label=f"true ATE = {ta:+.2f}")
    axes[2].set_title("C. observational contrast E[Y|1]−E[Y|0]\n(generator sanity: grows with β)", fontsize=11)
    axes[2].set_xlabel("confounding strength β"); axes[2].set_ylabel("obs contrast")
    axes[2].grid(alpha=0.3); axes[2].legend(fontsize=9)
    fig.suptitle("E2: spline spurious heterogeneity & ATE vs confounding strength", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(outdir, "spline_confounding.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print("\n[E2] mean τ_sd by β × n:")
    print(df.pivot_table(index="beta", columns="n", values="tau_sd", aggfunc="mean")
          .to_string(float_format=lambda v: f"{v:.3f}"))
    return out


# ------------------------------------------------------------------ E5 profile
def fig_profile(outdir):
    df = _load("profile_shard*.csv")
    if df is None:
        print("skip profile: no profile_shard*.csv")
        return None
    df = _num(df, "n", "seed", "grid_ate", "val_loss", "ate_after", "true_ate")
    prof = df[df["kind"] == "profile"]
    free = df[df["kind"] == "free"]
    ns = sorted(prof["n"].dropna().unique())
    ta = float(df["true_ate"].dropna().iloc[0])
    fig, axes = plt.subplots(1, len(ns), figsize=(5.2 * len(ns), 4.6), squeeze=False)
    axes = axes[0]
    for j, n in enumerate(ns):
        ax = axes[j]
        g = prof[prof["n"] == n].groupby("grid_ate")["val_loss"].agg(["mean", "std"]).reset_index().sort_values("grid_ate")
        ax.errorbar(g["grid_ate"], g["mean"], yerr=g["std"], fmt="-o", lw=2, ms=6, color="#1f77b4",
                    capsize=3, label="profiled NLL(ate)")
        # minimum of the profile
        gi = g["mean"].values.argmin()
        ax.scatter([g["grid_ate"].values[gi]], [g["mean"].values[gi]], s=140, color="#1f77b4",
                   edgecolor="white", zorder=5, label=f"profile min @ {g['grid_ate'].values[gi]:.2f}")
        ax.axvline(ta, color="black", ls="--", lw=1.6, label=f"true ATE = {ta:+.2f}")
        # where the FREE optimiser lands (vertical marker): comparing its ATE to the
        # profile minimum is scale-free (unlike NLL, whose split differs across fits).
        fn = free[free["n"] == n]
        if len(fn):
            fa = fn["ate_after"].mean()
            ax.axvline(fa, color="#d62728", ls=":", lw=2.0,
                       label=f"free-fit lands @ {fa:.2f}")
        ax.set_title(f"n={int(n)}", fontsize=12)
        ax.set_xlabel("frozen causal-margin ate", fontsize=10)
        ax.set_ylabel("profiled validation NLL (lower=better)", fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best")
    fig.suptitle("E5 (additive contrast): profile likelihood of the gaussian-arm ATE\n"
                 "profile min below truth at small n ⇒ finite-sample non-identifiability; "
                 "free optimiser lands (red) short of the profile min ⇒ optimisation stops low; both ease as n grows",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out = os.path.join(outdir, "additive_profile.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print("\n[E5] profiled NLL by ate × n (mean over seeds):")
    print(prof.pivot_table(index="grid_ate", columns="n", values="val_loss", aggfunc="mean")
          .to_string(float_format=lambda v: f"{v:.4f}"))
    print("free-fit ate by n:", free.groupby("n")["ate_after"].mean().to_dict())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default=_OUTDIR)
    p.add_argument("--const", type=float, default=0.0)
    p.add_argument("--ate", type=float, default=1.0)
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    cp = [args.const, args.ate]

    made = []
    for f in (fig_tau_decomp(cp, args.outdir), fig_capacity(args.outdir),
              fig_confounding(args.outdir), fig_profile(args.outdir)):
        if f:
            made.append(f)
    print("\nfigures:")
    for f in made:
        print(" ", f)


if __name__ == "__main__":
    main()
