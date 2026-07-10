"""Figure + ranking for the 10-D complex-copula HP check.

Pure analysis: reads the sharded CSVs in
``validation/outputs/flexible_te/copula10d_hp/shard*.csv`` (no training) and answers
"which frugal-flow settings recover the ATE under weak vs strong 10-D confounder
dependence?". Writes one figure (a panel per dependence regime, mean ATE ±SEM per
HP config against the true-ATE line) and prints a per-regime ranking table.

Regenerate (after the shards exist):
    cd validation && micromamba run -n frugal-flows-flowjax python -m diagnostics.plot_copula10d_hp
"""
from __future__ import annotations

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TRUE_ATE = 1.7634
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "outputs", "flexible_te", "copula10d_hp")
OUT = os.path.join(HERE, "..", "outputs", "flexible_te")
REGIMES = ["weak", "mixed", "strong"]
REGIME_TITLE = {"weak": "weak dependence (ρ≈0.12)", "mixed": "mixed / complex (ρ≈0.1–0.7)",
                "strong": "strong dependence (ρ≈0.68)"}
CONFIG_ORDER = ["base", "cop_wide", "cop_wider", "cop_deep", "cop_layers", "cop_knots", "cop_big", "mflow"]
plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def load():
    results = os.path.join(DATA, "results.csv")
    if os.path.exists(results):
        files = [results]                                  # canonical consolidated file
    else:
        files = sorted(glob.glob(os.path.join(DATA, "shard*.csv")))  # raw shards (pre-consolidation)
    if not files:
        raise FileNotFoundError(f"no results.csv or shard CSVs in {DATA}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # A cell may appear twice (an original failed/NaN row + a backfilled valid row);
    # keep the valid one. Sort NaN-last, then drop_duplicates keeps the first (valid).
    df = df.sort_values("ate", na_position="last")
    df = df.drop_duplicates(subset=["regime", "config", "seed"], keep="first")
    n_err = df["ate"].isna().sum()
    if n_err:
        print(f"WARNING: {n_err} cells still failed (NaN ate) after backfill — dropped")
    return df[df["ate"].notna()].copy()


def rankings(df):
    """Print, per regime, configs ranked by |mean bias| (with sd across seeds)."""
    for r in REGIMES:
        d = df[df["regime"] == r]
        if d.empty:
            continue
        g = d.groupby("config")["ate"].agg(["mean", "std", "count"])
        g["bias"] = g["mean"] - TRUE_ATE
        g["sem"] = g["std"] / np.sqrt(g["count"])
        g = g.reindex([c for c in CONFIG_ORDER if c in g.index])
        g = g.sort_values("bias", key=lambda s: s.abs())
        print(f"\n=== {r} ({REGIME_TITLE[r]}) — configs ranked by |mean bias| ===")
        print(f"{'config':<11}{'mean_ate':>10}{'bias':>9}{'sem':>8}{'sd':>8}")
        for cfg, row in g.iterrows():
            print(f"{cfg:<11}{row['mean']:>10.3f}{row['bias']:>+9.3f}{row['sem']:>8.3f}{row['std']:>8.3f}")


def figure(df):
    fig, axes = plt.subplots(1, len(REGIMES), figsize=(5.2 * len(REGIMES), 4.8), sharey=True)
    if len(REGIMES) == 1:
        axes = [axes]
    for ax, r in zip(axes, REGIMES):
        d = df[df["regime"] == r]
        cfgs = [c for c in CONFIG_ORDER if c in set(d["config"])]
        ax.axhline(TRUE_ATE, ls="--", color="k", lw=1.2, label=f"true = {TRUE_ATE}")
        for i, cfg in enumerate(cfgs):
            v = d[d["config"] == cfg]["ate"].values
            m, sd, k = v.mean(), v.std(ddof=1), len(v)
            sem = sd / np.sqrt(k)
            ax.scatter(np.full_like(v, i) + np.linspace(-0.15, 0.15, k), v, s=12, color="C0", alpha=0.28, zorder=1)
            ax.errorbar(i, m, yerr=sem, fmt="o", color="C3", capsize=4, lw=2, ms=6, zorder=3)
        ax.set_xticks(range(len(cfgs)))
        ax.set_xticklabels(cfgs, rotation=45, ha="right", fontsize=8)
        ax.set_title(REGIME_TITLE[r])
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel("estimated ATE")
    fig.suptitle("10-D complex-copula HP check: ATE recovery by config × dependence regime\n"
                 "(Gamma, log+standardize, n=2000; red = mean ±SEM, faint = per-seed; true ATE 1.7634)",
                 y=1.03)
    fig.tight_layout()
    p = os.path.join(OUT, "fig9_copula10d_hp.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


if __name__ == "__main__":
    df = load()
    print(f"loaded {len(df)} fits across {df['regime'].nunique()} regimes × {df['config'].nunique()} configs")
    rankings(df)
    print("\nwrote:", figure(df))
