"""Analyse the copula-hp sweep: does more copula capacity / better u_z shrink the
CONFOUNDING-induced bias shape of τ(u)?

Reads outputs/copulahp_shard*.csv. Per config computes, on the confounded baseline
(true τ(u) flat = ATE):
  bias_shape  = std_u( seed-mean τ(u) − truth )   -- systematic curvature that survives averaging
  bias_rms    = rms_u( seed-mean τ(u) − truth )   -- total seed-mean deviation from truth
  variance    = mean_u( seed-SD τ(u) )            -- per-seed scatter
  tau_sd      = mean over seeds of the scalar tau_sd
  ate         = mean over seeds

Renders copula_hp.png: (A) seed-mean τ(u) per config vs flat truth; (B) bias_shape by
config; (C) tau_sd + variance floor by config.

Usage (from validation/):
  micromamba run -n frugal-flows-flowjax python -m diagnostics.spline_copula_hp_plot
"""

from __future__ import annotations

import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUTDIR = os.path.join(_HERE, "outputs")
_VALIDATION_DIR = os.path.dirname(_HERE)
if _VALIDATION_DIR not in sys.path:
    sys.path.insert(0, _VALIDATION_DIR)
from diagnostics.ate_extraction_suite import TAU_CURVE_BINS  # noqa: E402

U = (np.arange(TAU_CURVE_BINS) + 0.5) / TAU_CURVE_BINS
ORDER = ["baseline", "cop_deep", "cop_wide", "cop_layers", "cop_knots", "cop_big",
         "marginal_flow", "mflow_cop_big", "lr_low"]


def _load():
    paths = sorted(glob.glob(os.path.join(_HERE, "..", "outputs", "copulahp_shard*.csv")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(_OUTDIR, "copulahp_shard*.csv")))
    if not paths:
        raise SystemExit("no copulahp_shard*.csv found")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def main():
    df = _load()
    for c in ("tau_sd", "ate", "true_ate", "n"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    truth = float(df["true_ate"].dropna().iloc[0])  # flat truth level
    configs = [c for c in ORDER if c in set(df["config"])]

    rows = []
    curves = {}
    for cfg in configs:
        sub = df[df["config"] == cfg]
        A = np.vstack([np.array([float(x) for x in s.split(";")])
                       for s in sub["tau_curve"] if isinstance(s, str)])
        mean, sd = A.mean(0), A.std(0)
        curves[cfg] = mean
        rows.append({
            "config": cfg,
            "bias_shape": float(np.std(mean - truth)),
            "bias_rms": float(np.sqrt(np.mean((mean - truth) ** 2))),
            "variance": float(sd.mean()),
            "tau_sd": float(sub["tau_sd"].mean()),
            "ate": float(sub["ate"].mean()),
            "nseed": len(sub),
        })
    summ = pd.DataFrame(rows).set_index("config")
    print("copula-hp sweep (confounded baseline, n=200):")
    print(summ.to_string(float_format=lambda v: f"{v:.3f}"))
    base = summ.loc["baseline"]
    print(f"\nbaseline bias_shape={base['bias_shape']:.3f}  tau_sd={base['tau_sd']:.3f}")
    print("Δ vs baseline (negative = reduced the confounding bias shape):")
    print((summ["bias_shape"] - base["bias_shape"]).to_string(float_format=lambda v: f"{v:+.3f}"))

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    cmap = plt.cm.viridis(np.linspace(0, 0.92, len(configs)))
    for cfg, col in zip(configs, cmap):
        axes[0].plot(U, curves[cfg], lw=2, color=col, label=cfg)
    axes[0].axhline(truth, color="black", ls="--", lw=1.8, label="flat truth")
    axes[0].set_title("A. seed-mean τ(u) per config\n(flatter to truth = less confounding bias)", fontsize=11)
    axes[0].set_xlabel("outcome quantile u"); axes[0].set_ylabel("seed-mean τ(u)")
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=7, ncol=2)

    x = np.arange(len(configs))
    axes[1].bar(x, summ.loc[configs, "bias_shape"], color="#d62728", alpha=0.85)
    axes[1].axhline(base["bias_shape"], color="black", ls=":", lw=1.5, label="baseline")
    axes[1].set_xticks(x); axes[1].set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
    axes[1].set_title("B. bias shape  std_u(seed-mean − truth)\n(lower = copula removed more confounding)", fontsize=11)
    axes[1].set_ylabel("bias shape"); axes[1].grid(axis="y", alpha=0.3); axes[1].legend(fontsize=8)

    w = 0.4
    axes[2].bar(x - w / 2, summ.loc[configs, "tau_sd"], width=w, color="#1f77b4", label="tau_sd (total)")
    axes[2].bar(x + w / 2, summ.loc[configs, "variance"], width=w, color="#9edae5", label="variance floor")
    axes[2].set_xticks(x); axes[2].set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
    axes[2].set_title("C. tau_sd vs per-seed variance floor", fontsize=11)
    axes[2].set_ylabel("τ spread"); axes[2].grid(axis="y", alpha=0.3); axes[2].legend(fontsize=8)

    fig.suptitle("Copula capacity / u_z quality vs the spline's confounding bias (confounded baseline, n=200)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(_OUTDIR, "copula_hp.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"\nfigure: {out}")


if __name__ == "__main__":
    main()
