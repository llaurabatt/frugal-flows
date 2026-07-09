"""Aggregate the H1-beta shards (confounding-strength sweep on the Gamma DGP).

Reads ~/round2_outputs/h1_beta_shard*.csv (one row per dgp/confound_beta/arm/n/seed,
schema defined by FIELDNAMES in diagnostics/h1_matrix.py), groups by
(confound_beta, arm, n), and prints mean +/- sd across seeds for ATE bias and
tau_sd, plus means of qte_int_err, the OLS/IPW anchors, overlap_frac_clipped and
runtime. Applies the same pre-registered decision rule as h1_aggregate.py:

  bias is "REAL" for a cell iff  |mean bias| > 2 * sd / sqrt(k)   (k = n_seeds)

Then adjudicates, per arm, whether |mean bias| is monotonically non-decreasing
in confound_beta (0, 0.5, 1, 1.5) -- the qualitative prediction that stronger
Z->X confounding should not shrink the additive arm's misspecification bias --
and whether the beta=0 (no confounding) cell's bias is REAL, which it should
not be if the residual bias at beta>0 is confounding-driven rather than a flat
flow/optimisation artefact.

Finally cross-checks the gamma_b1 cell (beta=1.0) against the reference values
from the earlier (unsharded) H1 matrix run on the same DGP:
  gaussian:             mean bias +0.063 (sd 0.103)
  flexible_continuous:  mean bias -0.111 (sd 0.142)

Usage (from validation/):
  micromamba run -n frugal-flows-flowjax python -m diagnostics.beta_aggregate
  (or plain python3 -- only needs numpy)
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from collections import defaultdict

import numpy as np

DEFAULT_GLOB = os.path.expanduser("~/round2_outputs/h1_beta_shard*.csv")

# columns we aggregate per cell (mean/sd computed via nanstat, nan-filtered)
VALUE_COLS = ["bias", "tau_sd", "qte_int_err", "ols_ate", "ipw_ate",
              "ipw_ate_unclipped", "overlap_frac_clipped", "secs"]

# (beta, arm) -> (reference mean bias, reference sd) from the earlier
# (unsharded) H1 matrix run on the same gamma_b1 DGP, for a consistency check.
REFERENCE_BIAS = {
    (1.0, "gaussian"): (0.063, 0.103),
    (1.0, "flexible_continuous"): (-0.111, 0.142),
}


def load(pattern):
    rows = []
    for path in sorted(glob.glob(os.path.expanduser(pattern))):
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
    return rows


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def agg(rows):
    """cell (confound_beta, arm, n) -> dict[col] -> list of floats (incl. NaN)."""
    cells = defaultdict(lambda: defaultdict(list))
    n_err = 0
    for r in rows:
        if r.get("error"):
            n_err += 1
            continue
        beta = round(_f(r.get("confound_beta")), 6)
        key = (beta, r["arm"], int(r["n"]))
        for col in VALUE_COLS:
            cells[key][col].append(_f(r.get(col)))
    return cells, n_err


def nanstat(vals):
    """mean, sd, k (finite count), n_dropped (NaN count) -- nan-filtered."""
    arr = np.array(vals, dtype=float)
    finite = arr[~np.isnan(arr)]
    k = int(finite.size)
    n_drop = int(arr.size - k)
    if k == 0:
        return math.nan, math.nan, 0, n_drop
    mean = float(finite.mean())
    sd = float(finite.std(ddof=1)) if k > 1 else math.nan
    return mean, sd, k, n_drop


def bias_is_real(mean, sd, k):
    if k <= 1 or math.isnan(sd):
        return None
    return abs(mean) > 2.0 * sd / math.sqrt(k)


def fmt_pm(mean, sd):
    if math.isnan(mean):
        return "   nan   "
    if math.isnan(sd):
        return f"{mean:+.3f}±  nan"
    return f"{mean:+.3f}±{sd:.3f}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glob", default=DEFAULT_GLOB, help=f"shard glob (default {DEFAULT_GLOB})")
    args = p.parse_args()

    rows = load(args.glob)
    if not rows:
        print(f"no shards matched {args.glob}")
        return

    cells, n_err = agg(rows)
    print(f"loaded {len(rows)} rows ({n_err} skipped: non-empty 'error' field) from {args.glob}\n")
    if not cells:
        print("no usable (non-error) rows found.")
        return

    # ---- per-column NaN-drop report (missing/unparseable numeric fields) ----
    drop_totals = defaultdict(int)
    total_vals = defaultdict(int)
    for c in cells.values():
        for col in VALUE_COLS:
            vals = c[col]
            total_vals[col] += len(vals)
            drop_totals[col] += sum(1 for v in vals if math.isnan(v))
    any_drops = any(drop_totals[col] for col in VALUE_COLS)
    if any_drops:
        print("NaN/missing-field drops per column (dropped/total):")
        for col in VALUE_COLS:
            if drop_totals[col]:
                print(f"    {col:<20}{drop_totals[col]}/{total_vals[col]}")
        print()

    betas = sorted({k[0] for k in cells})
    arms = sorted({k[1] for k in cells})
    ns = sorted({k[2] for k in cells})

    def cell_stats(beta, arm, n):
        c = cells.get((beta, arm, n))
        if not c:
            return None
        out = {}
        for col in VALUE_COLS:
            out[col] = nanstat(c[col])  # (mean, sd, k, n_drop)
        return out

    # ---- one table per arm, betas (x ns) as rows, sorted by (arm, beta) ----
    hdr = (f"{'beta':<7}{'n':>7}{'k':>4}{'bias (mean±sd)':>18}{'verdict':>9}"
           f"{'tau_sd (mean±sd)':>18}{'qte_err':>9}{'ols_ate':>9}{'ipw_ate':>9}"
           f"{'ipw_unclip':>11}{'clip_frac':>10}{'secs':>8}")

    for arm in arms:
        print("=" * len(hdr))
        print(f"arm = {arm}")
        print("=" * len(hdr))
        print(hdr)
        print("-" * len(hdr))
        for beta in betas:
            for n in ns:
                st = cell_stats(beta, arm, n)
                if not st:
                    continue
                bm, bs, k, _ = st["bias"]
                tm, ts, _, _ = st["tau_sd"]
                qm = st["qte_int_err"][0]
                om = st["ols_ate"][0]
                im = st["ipw_ate"][0]
                iu = st["ipw_ate_unclipped"][0]
                cf = st["overlap_frac_clipped"][0]
                sm = st["secs"][0]
                real = bias_is_real(bm, bs, k)
                rflag = "n/a" if real is None else ("YES" if real else "no")
                print(f"{beta:<7}{n:>7}{k:>4}{fmt_pm(bm, bs):>18}{rflag:>9}"
                      f"{fmt_pm(tm, ts):>18}{qm:>9.3f}{om:>9.3f}{im:>9.3f}"
                      f"{iu:>11.3f}{cf:>10.3f}{sm:>8.1f}")
        print()

    # ---- adjudication -----------------------------------------------------
    print("=" * 72)
    print("ADJUDICATION")
    print("=" * 72)
    target_betas = [0.0, 0.5, 1.0, 1.5]
    for arm in arms:
        seq = []
        for beta in target_betas:
            st = None
            for n in ns:
                s = cell_stats(beta, arm, n)
                if s:
                    st = s
                    break
            if st is None:
                seq.append((beta, None))
            else:
                bm, bs, k, _ = st["bias"]
                seq.append((beta, (bm, bs, k)))

        present = [(b, v) for b, v in seq if v is not None]
        print(f"\narm = {arm}")
        print(f"  mean bias by beta: " +
              ", ".join(f"beta={b}: {v[0]:+.3f}" if v else f"beta={b}: (missing)"
                        for b, v in seq))
        if len(present) >= 2:
            abs_biases = [abs(v[0]) for _, v in present]
            mono = all(abs_biases[i] <= abs_biases[i + 1] + 1e-9 for i in range(len(abs_biases) - 1))
            print(f"  |bias| monotonically non-decreasing in beta: {'YES' if mono else 'NO'}")
        else:
            print("  |bias| monotonicity in beta: n/a (fewer than 2 beta cells present)")

        zero = dict(seq).get(0.0)
        if zero is None:
            print("  beta=0 cell: (missing)")
        else:
            bm, bs, k = zero
            real = bias_is_real(bm, bs, k)
            rflag = "n/a (k<2)" if real is None else ("REAL" if real else "not real")
            print(f"  beta=0 cell bias: {bm:+.3f}±{bs if not math.isnan(bs) else float('nan'):.3f} "
                  f"(k={k})  -> {rflag}")

    # ---- consistency check against the earlier (unsharded) H1 matrix ------
    print("\n" + "=" * 72)
    print("CONSISTENCY CHECK vs earlier H1 matrix (gamma_b1, beta=1.0)")
    print("=" * 72)
    for (ref_beta, ref_arm), (ref_mean, ref_sd) in REFERENCE_BIAS.items():
        st = None
        for n in ns:
            s = cell_stats(ref_beta, ref_arm, n)
            if s:
                st = s
                break
        if st is None:
            print(f"  {ref_arm:<22} beta={ref_beta}: (no matching cell in loaded shards)")
            continue
        bm, bs, k, _ = st["bias"]
        diff = bm - ref_mean
        print(f"  {ref_arm:<22} beta={ref_beta}: computed bias={bm:+.3f}±{bs:.3f} (k={k})  "
              f"vs reference {ref_mean:+.3f}±{ref_sd:.3f}   diff={diff:+.3f}")


if __name__ == "__main__":
    main()
