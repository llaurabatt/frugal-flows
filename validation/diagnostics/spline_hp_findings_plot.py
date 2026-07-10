"""Aggregate the spline HP battery shards into an adjudication table + figures.

Reads the per-fit CSV shards written by ``diagnostics.spline_hp_battery`` and
turns them into:
  1. a per-cell adjudication table (mean_bias +/- 2*sd/sqrt(K), sd_ate, mean_qte,
     tot_n_drop) printed to stdout, plus the restart-averaging (H3b) readout;
  2. ``spline_hp_bias_vs_sd.png``   -- bias interval + twin-axis restart-sd per cell;
  3. ``spline_hp_tau_overlay.png``  -- per-cell tau(u): per-restart (grey) + restart
     mean (colour) + analytic Gamma truth (dashed).

A "cell" is one (warm_start, transform, lr_schedule) combination; ``warm_start`` is
treated as ``cold`` when the column is absent (Phase-1 CSVs, before warm-start exists).

Decision rules (see SPLINE_HP_FINDINGS.md): adjudicate on ORIGINAL-Y-scale metrics
only -- never val_loss across transforms. Bias is "real" iff |mean_bias| > 2*sd/sqrt(K)
(project idiom, h1_matrix.py:18); a cell is unbiased when that interval covers 0.

Usage (from validation/):
  micromamba run -n frugal-flows-flowjax python -m diagnostics.spline_hp_findings_plot
  # custom inputs / output dir:
  python -m diagnostics.spline_hp_findings_plot --csv-glob 'spline_hp_shard*.csv' \
      --outdir outputs/flexible_te
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_VALIDATION_DIR = os.path.dirname(_HERE)
if _VALIDATION_DIR not in sys.path:
    sys.path.insert(0, _VALIDATION_DIR)
from diagnostics.ate_extraction_suite import TAU_CURVE_BINS  # noqa: E402
from diagnostics.outcome_families import FAMILIES  # noqa: E402

# The battery's fixed DGP (spline_hp_battery.py: DGP / CAUSAL_PARAMS).
FAM = FAMILIES["gamma_b1"]
CAUSAL_PARAMS = [1.0, 0.5]
U_CENTERS = (np.arange(TAU_CURVE_BINS) + 0.5) / TAU_CURVE_BINS

# Okabe-Ito, matching plot_spline_findings.py's palette family.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]
DEFAULT_GLOBS = ["spline_hp_shard*.csv", "spline_hp_battery*.csv"]


def load_shards(outdir: str, patterns: list[str]) -> pd.DataFrame | None:
    paths: list[str] = []
    for pat in patterns:
        paths += glob.glob(os.path.join(outdir, pat))
    paths = sorted(set(paths))
    if not paths:
        return None
    print(f"[spline_hp_findings_plot] loading {len(paths)} shard(s):")
    for p in paths:
        print(f"    {os.path.relpath(p, _VALIDATION_DIR)}")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    # Drop rows that errored during the fit (the plotters' contract).
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"].astype(str).str.strip() == "")]
    if "warm_start" not in df.columns:  # Phase-1 CSVs predate the axis
        df["warm_start"] = "cold"
    df["warm_start"] = df["warm_start"].fillna("cold").replace("", "cold")
    for c in ("ate", "bias", "tau_sd", "qte_int_err", "true_ate", "n_drop", "restart"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


def parse_tau_curve(cell: str) -> np.ndarray | None:
    if not isinstance(cell, str) or not cell.strip():
        return None
    try:
        return np.array([float(x) for x in cell.split(";")])
    except ValueError:
        return None


def cell_key(row) -> tuple[str, str, str]:
    return (str(row["warm_start"]), str(row["transform"]), str(row["lr_schedule"]))


def ordered_cells(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Cells ordered cold-before-warm, raw/log/standardize, const/cosine."""
    ws_ord = {"cold": 0, "warm": 1}
    tr_ord = {"raw": 0, "log": 1, "asinh": 2, "standardize": 3}
    lr_ord = {"const": 0, "cosine": 1}
    cells = sorted(
        {cell_key(r) for _, r in df.iterrows()},
        key=lambda c: (ws_ord.get(c[0], 9), tr_ord.get(c[1], 9), lr_ord.get(c[2], 9)),
    )
    return cells


def cell_label(c: tuple[str, str, str]) -> str:
    return f"{c[0]}/{c[1]}/{c[2]}"


def summarize(df: pd.DataFrame) -> dict:
    """Per-cell stats keyed by (warm_start, transform, lr_schedule)."""
    stats = {}
    true_ate = float(df["true_ate"].dropna().iloc[0]) if df["true_ate"].notna().any() else float("nan")
    for c in ordered_cells(df):
        sub = df[[cell_key(r) == c for _, r in df.iterrows()]]
        ate = sub["ate"].to_numpy(dtype=float)
        ate = ate[np.isfinite(ate)]
        k = len(ate)
        if k == 0:
            continue
        bias = ate - true_ate
        sd_ate = float(np.std(ate))
        half = 2.0 * sd_ate / np.sqrt(k) if k > 0 else float("nan")
        # H3b restart-averaging: average the LEVEL across restarts.
        ate_avg = float(np.mean(ate))
        curves = [c_ for c_ in (parse_tau_curve(v) for v in sub.get("tau_curve", [])) if c_ is not None]
        tau_avg = np.nanmean(np.vstack(curves), axis=0) if curves else None
        stats[c] = dict(
            k=k, true_ate=true_ate,
            mean_bias=float(np.mean(bias)), sd_ate=sd_ate, half_ci=half,
            min_ate=float(np.min(ate)), max_ate=float(np.max(ate)),
            mean_qte=float(np.nanmean(sub["qte_int_err"].to_numpy(dtype=float))),
            tot_n_drop=int(np.nansum(sub["n_drop"].to_numpy(dtype=float))),
            ate_avg=ate_avg, avg_abs_bias=abs(ate_avg - true_ate),
            median_abs_bias=float(np.median(np.abs(bias))),
            curves=curves, tau_avg=tau_avg,
        )
    return stats


def print_table(stats: dict):
    print("\n=== spline HP battery -- per-cell adjudication ===")
    print("metrics are on the ORIGINAL Y scale; do NOT compare val_loss across transforms.")
    hdr = (f"{'cell (ws/transform/lr)':<28}{'k':>3}{'mean_bias':>11}{'2sd/sqrtK':>11}"
           f"{'unbiased':>10}{'sd_ate':>9}{'mean_qte':>10}{'n_drop':>8}")
    print(hdr); print("-" * len(hdr))
    for c, s in stats.items():
        unbiased = "yes" if abs(s["mean_bias"]) <= s["half_ci"] else "NO"
        print(f"{cell_label(c):<28}{s['k']:>3}{s['mean_bias']:>+11.4f}{s['half_ci']:>11.4f}"
              f"{unbiased:>10}{s['sd_ate']:>9.4f}{s['mean_qte']:>10.4f}{s['tot_n_drop']:>8}")

    print("\n=== restart-averaging (H3b): does averaging the LEVEL buy sqrt(K)? ===")
    hdr2 = f"{'cell':<28}{'ate_avg':>9}{'|avg-true|':>12}{'median|bias|':>14}{'sd_ate':>9}{'helps':>7}"
    print(hdr2); print("-" * len(hdr2))
    for c, s in stats.items():
        helps = "yes" if (s["avg_abs_bias"] <= s["median_abs_bias"] and s["avg_abs_bias"] <= s["sd_ate"]) else "no"
        print(f"{cell_label(c):<28}{s['ate_avg']:>9.3f}{s['avg_abs_bias']:>12.4f}"
              f"{s['median_abs_bias']:>14.4f}{s['sd_ate']:>9.4f}{helps:>7}")


def fig_bias_vs_sd(stats: dict, outpath: str):
    cells = list(stats.keys())
    x = np.arange(len(cells))
    mean_bias = [stats[c]["mean_bias"] for c in cells]
    half = [stats[c]["half_ci"] for c in cells]
    sd_ate = [stats[c]["sd_ate"] for c in cells]
    ndrop = [stats[c]["tot_n_drop"] for c in cells]

    fig, ax = plt.subplots(figsize=(max(7.0, 1.5 * len(cells)), 5.0))
    ax.axhline(0.0, color="black", lw=1.2, ls="--", zorder=1, label="true ATE (bias=0)")
    ax.errorbar(x, mean_bias, yerr=half, fmt="o", color=PALETTE[0], capsize=4,
                lw=1.8, ms=7, zorder=3, label="mean bias  ± 2·sd/√K")
    for xi, mb, hc, nd in zip(x, mean_bias, half, ndrop):
        if nd:
            ax.annotate(f"n_drop={nd}", (xi, mb + hc), fontsize=7, ha="center",
                        va="bottom", color=PALETTE[1])
    ax.set_xticks(x)
    ax.set_xticklabels([cell_label(c) for c in cells], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("bias  (ATE_hat − true)", fontsize=11, color=PALETTE[0])
    ax.set_title("Spline HP battery: per-cell ATE bias interval + restart noise", fontsize=12)

    ax2 = ax.twinx()
    ax2.plot(x, sd_ate, "s--", color=PALETTE[2], ms=6, lw=1.4, zorder=2,
             label="restart sd(ATE)")
    ax2.set_ylabel("restart sd(ATE)", fontsize=11, color=PALETTE[2])
    ax2.set_ylim(bottom=0)

    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"[fig] {outpath}  ({os.path.getsize(outpath)//1024} kB)")


def fig_tau_overlay(stats: dict, outpath: str):
    cells = [c for c in stats if stats[c]["curves"]]
    if not cells:
        print("skip tau_overlay: no tau_curve data in shards")
        return
    true_curve = FAM.true_tau_curve(CAUSAL_PARAMS, U_CENTERS)
    ncol = min(4, len(cells))
    nrow = int(np.ceil(len(cells) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.2 * nrow), squeeze=False)
    for idx, c in enumerate(cells):
        ax = axes[idx // ncol][idx % ncol]
        s = stats[c]
        for tc in s["curves"]:
            ax.plot(U_CENTERS, tc, color="0.72", lw=0.7, alpha=0.6, zorder=1)
        if s["tau_avg"] is not None:
            ax.plot(U_CENTERS, s["tau_avg"], color=PALETTE[idx % len(PALETTE)], lw=2.4,
                    zorder=3, label="restart-mean τ(u)")
        ax.plot(U_CENTERS, true_curve, "--", color="black", lw=1.6, zorder=4, label="truth")
        ax.set_title(f"{cell_label(c)}  (k={s['k']}, sd={s['sd_ate']:.3f})", fontsize=9)
        ax.set_xlabel("outcome quantile u", fontsize=8)
        ax.set_ylabel("τ(u)", fontsize=8)
        if idx == 0:
            ax.legend(fontsize=7, loc="best")
    for j in range(len(cells), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Spline HP battery: per-restart τ(u) vs analytic Gamma truth", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"[fig] {outpath}  ({os.path.getsize(outpath)//1024} kB)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default="outputs/flexible_te",
                   help="dir holding the shard CSVs AND where figures are written (relative to validation/)")
    p.add_argument("--csv-glob", action="append", default=None,
                   help="shard glob(s) within --outdir; repeatable. Default: spline_hp_shard*, spline_hp_battery*")
    args = p.parse_args()

    outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(_VALIDATION_DIR, args.outdir)
    patterns = args.csv_glob or DEFAULT_GLOBS
    df = load_shards(outdir, patterns)
    if df is None:
        raise SystemExit(f"no shards matched {patterns} in {outdir}")

    stats = summarize(df)
    if not stats:
        raise SystemExit("no successful (non-error) rows to summarize")
    print_table(stats)
    fig_bias_vs_sd(stats, os.path.join(outdir, "spline_hp_bias_vs_sd.png"))
    fig_tau_overlay(stats, os.path.join(outdir, "spline_hp_tau_overlay.png"))


if __name__ == "__main__":
    main()
