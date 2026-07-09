"""Aggregate the bias-diagnosis ablations and render the diagnostic figure.

Reads:
  outputs/bias_shard*.csv    (bias_ablation.py: conditions x n x seeds)
  outputs/biasep_shard*.csv  (bias_epochs.py:   epochs grid, early-stop OFF)

Renders one 1x3 panel (phone-friendly):
  A) ATE vs n by condition {baseline, unconfounded, init_truth} + true ATE.
     init_truth flat at truth while baseline starts low => optimisation can't
     reach truth from ate=0 (the bias is an optimisation pathology, not the MLE).
  B) learned `scale` vs n by condition + true scale=1. Scale inflates exactly
     when ate is stuck low (variance-partition symptom).
  C) ATE vs epoch budget at fixed n (early-stop OFF). Flat below truth => more
     epochs do NOT help; only more data => confirms flat-gradient, not undertraining.

Usage (from validation/):
  micromamba run -n frugal-flows-flowjax python -m diagnostics.bias_plot
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

COND_LABEL = {"baseline": "baseline (init 0, confounded)",
              "unconfounded": "unconfounded (X⊥Z, init 0)",
              "init_truth": "init at truth"}
COND_COLOR = {"baseline": "#d62728", "unconfounded": "#ff7f0e", "init_truth": "#2ca02c"}


def _load(pattern):
    # shards are written to validation/outputs/ (cwd-relative in the runners);
    # also check diagnostics/outputs/ as a fallback.
    paths = sorted(glob.glob(os.path.join(_HERE, "..", "outputs", pattern)))
    if not paths:
        paths = sorted(glob.glob(os.path.join(_HERE, "outputs", pattern)))
    if not paths:
        return None
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def _agg(df, group, val):
    df[val] = pd.to_numeric(df[val], errors="coerce")
    g = df.dropna(subset=[val]).groupby(group)
    return g[val].agg(["mean", "std", "size"]).reset_index()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default=os.path.join(_HERE, "outputs"))
    args = p.parse_args()

    abl = _load("bias_shard*.csv")
    eps = _load("biasep_shard*.csv")
    if abl is None:
        raise SystemExit("no bias_shard*.csv found")
    true_ate = float(pd.to_numeric(abl["true_ate"]).iloc[0])
    conditions = [c for c in ["baseline", "unconfounded", "init_truth"] if c in set(abl["condition"])]

    fig, axes2 = plt.subplots(2, 2, figsize=(13, 10.2))
    axes = axes2.ravel()

    # ---- Panel A: ATE vs n by condition ----
    axA = axes[0]
    axA.axhline(true_ate, color="black", ls="--", lw=1.8, label=f"true ATE = {true_ate:+.2f}")
    for cond in conditions:
        s = _agg(abl[abl["condition"] == cond], "n", "ate_crn").sort_values("n")
        c = COND_COLOR[cond]
        axA.plot(s["n"], s["mean"], "-o", color=c, lw=2, ms=7, label=COND_LABEL[cond])
        axA.fill_between(s["n"], s["mean"] - s["std"], s["mean"] + s["std"], color=c, alpha=0.16)
    axA.set_xscale("log"); axA.set_xticks(sorted(abl["n"].unique()))
    axA.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axA.set_xlabel("training sample size n (log)", fontsize=12)
    axA.set_ylabel("estimated ATE", fontsize=12)
    axA.set_title("A. ATE vs n by condition\ninit-truth stays at truth; init-0 can't reach it", fontsize=12)
    axA.grid(alpha=0.3); axA.legend(fontsize=9, loc="best")

    # ---- Panel B: learned scale vs n ----
    axB = axes[1]
    axB.axhline(1.0, color="black", ls="--", lw=1.8, label="true scale = 1.0")
    for cond in conditions:
        s = _agg(abl[abl["condition"] == cond], "n", "scale_param").sort_values("n")
        c = COND_COLOR[cond]
        axB.plot(s["n"], s["mean"], "-o", color=c, lw=2, ms=7, label=COND_LABEL[cond])
        axB.fill_between(s["n"], s["mean"] - s["std"], s["mean"] + s["std"], color=c, alpha=0.16)
    axB.set_xscale("log"); axB.set_xticks(sorted(abl["n"].unique()))
    axB.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axB.set_xlabel("training sample size n (log)", fontsize=12)
    axB.set_ylabel("learned Gaussian scale", fontsize=12)
    axB.set_title("B. scale inflates when ATE is stuck\n(variance-partition symptom)", fontsize=12)
    axB.grid(alpha=0.3); axB.legend(fontsize=9, loc="best")

    # ---- Panel C: ATE vs epochs (early-stop off) ----
    axC = axes[2]
    axC.axhline(true_ate, color="black", ls="--", lw=1.8, label=f"true ATE = {true_ate:+.2f}")
    if eps is not None:
        for n_val, mk in zip(sorted(eps["n"].unique()), ["-o", "-s"]):
            s = _agg(eps[eps["n"] == n_val], "epochs", "ate_crn").sort_values("epochs")
            axC.plot(s["epochs"], s["mean"], mk, lw=2, ms=7, label=f"n={n_val}")
            axC.fill_between(s["epochs"], s["mean"] - s["std"], s["mean"] + s["std"], alpha=0.16)
        axC.set_xscale("log"); axC.set_xticks(sorted(eps["epochs"].unique()))
        axC.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    else:
        axC.text(0.5, 0.5, "no biasep_shard*.csv yet", ha="center", transform=axC.transAxes)
    axC.set_xlabel("epoch budget (early-stop OFF, log)", fontsize=12)
    axC.set_ylabel("estimated ATE", fontsize=12)
    axC.set_title("C. ATE vs epochs (early-stop off)\nflat below truth => not undertraining", fontsize=12)
    axC.grid(alpha=0.3); axC.legend(fontsize=9, loc="best")

    # ---- Panel D: validation loss vs n (baseline vs init_truth) ----
    axD = axes[3]
    for cond in [c for c in ["baseline", "init_truth"] if c in conditions]:
        s = _agg(abl[abl["condition"] == cond], "n", "val_loss").sort_values("n")
        c = COND_COLOR[cond]
        axD.plot(s["n"], s["mean"], "-o", color=c, lw=2, ms=7, label=COND_LABEL[cond])
    axD.set_xscale("log"); axD.set_xticks(sorted(abl["n"].unique()))
    axD.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axD.set_xlabel("training sample size n (log)", fontsize=12)
    axD.set_ylabel("validation NLL (lower = better fit)", fontsize=12)
    axD.set_title("D. which solution fits better?\nn=100: low-ATE wins (non-ident.); n≥200: truth wins (trap)", fontsize=12)
    axD.grid(alpha=0.3); axD.legend(fontsize=9, loc="best")

    fig.suptitle("Diagnosing the small-n ATE bias of the additive (gaussian-margin) arm", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(args.outdir, "bias_diagnosis.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)

    # ---- printed summary ----
    print("ATE vs n (mean over seeds):")
    piv = abl.copy(); piv["ate_crn"] = pd.to_numeric(piv["ate_crn"], errors="coerce")
    print(piv.pivot_table(index="condition", columns="n", values="ate_crn", aggfunc="mean")
          .to_string(float_format=lambda v: f"{v:+.3f}"))
    if eps is not None:
        print("\nATE vs epochs, early-stop OFF (mean over seeds):")
        eps["ate_crn"] = pd.to_numeric(eps["ate_crn"], errors="coerce")
        print(eps.pivot_table(index="n", columns="epochs", values="ate_crn", aggfunc="mean")
              .to_string(float_format=lambda v: f"{v:+.3f}"))
    print(f"\nfigure: {out}")


if __name__ == "__main__":
    main()
