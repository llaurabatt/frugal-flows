"""Aggregate the sharded ATE sweep and plot mean ATE +/- uncertainty vs n.

Reads every `outputs/sweep_*.csv` (the shards from `ate_sweep.py`), pools the
per-fit rows, and for each (family, arm, n) computes the mean ATE and its spread
over seeds. Produces:

  * ate_sweep_recovery.png  — ATE vs n (log x), one panel per outcome family;
    per arm: mean line + shaded +/-1 SD band + individual-run dots; true ATE dashed.
  * ate_sweep_relerr.png    — |relative error| vs n per arm/family (convergence),
    with the 15% reference line.

Also prints a numerical summary table (mean ATE, SD, SEM, mean rel-err, n_ok).

Usage (from validation/):
  micromamba run -n frugal-flows-flowjax python -m diagnostics.ate_sweep_plot
  micromamba run -n frugal-flows-flowjax python -m diagnostics.ate_sweep_plot --glob 'outputs/sweep_*.csv'
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

ARM_LABEL = {"gaussian": "additive shift", "flexible_continuous": "spline",
             "location_translation": "loc-translation"}
ARM_COLOR = {"gaussian": "#d62728", "flexible_continuous": "#1f77b4",
             "location_translation": "#2ca02c"}


def load(pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern if os.path.isabs(pattern) else os.path.join(_HERE, "..", pattern)))
    if not paths:
        paths = sorted(glob.glob(os.path.join(_HERE, "outputs", "sweep_*.csv")))
    if not paths:
        raise SystemExit(f"no sweep CSVs found for pattern {pattern!r}")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    print(f"loaded {len(df)} rows from {len(paths)} shard(s): {[os.path.basename(p) for p in paths]}")
    n_err = df["error"].notna().sum() if "error" in df else 0
    if n_err:
        print(f"  {n_err} fits errored (dropped from ATE stats)")
    df["ate"] = pd.to_numeric(df["ate"], errors="coerce")
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    ok = df.dropna(subset=["ate"])
    g = ok.groupby(["family", "arm", "n"])
    out = g.agg(
        n_ok=("ate", "size"),
        ate_mean=("ate", "mean"),
        ate_sd=("ate", "std"),
        true_ate=("true_ate", "first"),
        relerr_mean=("ate_relerr", "mean"),
        tau_sd_mean=("tau_sd", "mean"),
        frac_neg_mean=("frac_neg", "mean"),
    ).reset_index()
    out["ate_sem"] = out["ate_sd"] / np.sqrt(out["n_ok"].clip(lower=1))
    out["bias"] = out["ate_mean"] - out["true_ate"]
    return out.sort_values(["family", "arm", "n"]).reset_index(drop=True)


def fig_recovery(df: pd.DataFrame, summ: pd.DataFrame, outpath: str):
    families = list(dict.fromkeys(summ["family"]))
    arms = list(dict.fromkeys(summ["arm"]))
    fig, axes = plt.subplots(1, len(families), figsize=(6.6 * len(families), 5.6), squeeze=False)
    for j, fam in enumerate(families):
        ax = axes[0][j]
        true_ate = summ.loc[summ["family"] == fam, "true_ate"].iloc[0]
        ax.axhline(true_ate, color="black", ls="--", lw=1.8, zorder=1,
                   label=f"true ATE = {true_ate:+.2f}")
        for arm in arms:
            s = summ[(summ["family"] == fam) & (summ["arm"] == arm)].sort_values("n")
            if s.empty:
                continue
            c = ARM_COLOR.get(arm, "gray")
            ns = s["n"].to_numpy()
            ax.plot(ns, s["ate_mean"], "-o", color=c, lw=2, ms=7, zorder=3,
                    label=f"{ARM_LABEL.get(arm, arm)} (mean +/- SD)")
            ax.fill_between(ns, s["ate_mean"] - s["ate_sd"], s["ate_mean"] + s["ate_sd"],
                            color=c, alpha=0.18, zorder=2)
            # individual runs, jittered horizontally on log scale
            raw = df[(df["family"] == fam) & (df["arm"] == arm)].dropna(subset=["ate"])
            jit = raw["n"] * np.random.default_rng(0).uniform(0.93, 1.07, len(raw))
            ax.scatter(jit, raw["ate"], s=14, color=c, alpha=0.30, zorder=2, edgecolor="none")
        ax.set_xscale("log")
        ax.set_xticks(sorted(summ["n"].unique()))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("training sample size n (log)", fontsize=12)
        ax.set_ylabel("estimated ATE", fontsize=12)
        ax.set_title(f"{fam} outcome", fontsize=14)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10, loc="best")
    fig.suptitle("ATE recovery vs sample size (10 seeds/point; band = +/-1 SD)", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def fig_relerr(summ: pd.DataFrame, outpath: str):
    families = list(dict.fromkeys(summ["family"]))
    arms = list(dict.fromkeys(summ["arm"]))
    fig, axes = plt.subplots(1, len(families), figsize=(6.6 * len(families), 5.2), squeeze=False)
    for j, fam in enumerate(families):
        ax = axes[0][j]
        ax.axhline(0.15, color="gray", ls=":", lw=1.5, label="15% reference")
        for arm in arms:
            s = summ[(summ["family"] == fam) & (summ["arm"] == arm)].sort_values("n")
            if s.empty:
                continue
            c = ARM_COLOR.get(arm, "gray")
            ax.plot(s["n"], s["relerr_mean"], "-o", color=c, lw=2, ms=7,
                    label=ARM_LABEL.get(arm, arm))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(sorted(summ["n"].unique()))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("training sample size n (log)", fontsize=12)
        ax.set_ylabel("mean |relative ATE error| (log)", fontsize=12)
        ax.set_title(f"{fam} outcome", fontsize=14)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=10, loc="best")
    fig.suptitle("ATE relative error vs sample size", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glob", default="outputs/sweep_*.csv")
    p.add_argument("--outdir", default=os.path.join(_HERE, "outputs"))
    args = p.parse_args()

    df = load(args.glob)
    summ = summarise(df)
    os.makedirs(args.outdir, exist_ok=True)
    summ.to_csv(os.path.join(args.outdir, "ate_sweep_summary.csv"), index=False)

    cols = ["family", "arm", "n", "n_ok", "true_ate", "ate_mean", "ate_sd", "ate_sem",
            "relerr_mean", "tau_sd_mean", "frac_neg_mean"]
    print("\n" + summ[cols].to_string(index=False,
          float_format=lambda v: f"{v:+.3f}"))

    f1 = os.path.join(args.outdir, "ate_sweep_recovery.png")
    f2 = os.path.join(args.outdir, "ate_sweep_relerr.png")
    fig_recovery(df, summ, f1)
    fig_relerr(summ, f2)
    print(f"\nfigures:\n  {f1}\n  {f2}")
    print(f"summary CSV: {os.path.join(args.outdir, 'ate_sweep_summary.csv')}")


if __name__ == "__main__":
    main()
