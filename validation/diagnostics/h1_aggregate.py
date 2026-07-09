"""Aggregate the H1-matrix shards and resolve the pre-registered hypotheses.

Reads outputs/flexible_te/h1_shard*.csv, groups by (dgp, arm, n), and prints
mean +/- sd across seeds for ATE bias, tau_sd, and integrated QTE error, with the
OLS/IPW anchors. Then applies the pre-registered decision rules:

  bias is "real" for a cell iff  |mean bias| > 2 * sd / sqrt(n_seeds).
  H1: on gamma, ADDITIVE arm's ATE-bias is real at n=20000 AND SPLINE's is not.
  H2: spline |mean bias| <= additive |mean bias| at every (dgp, n) cell.
  H3: on gamma, spline QTE error decreases in n; on gaussian, spline tau_sd
      decreases in n.

Usage (from validation/):
  micromamba run -n frugal-flows-flowjax python -m diagnostics.h1_aggregate
  (or plain python3 — only needs numpy)
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from collections import defaultdict

import numpy as np


def load(pattern):
    rows = []
    for path in sorted(glob.glob(pattern)):
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
    """cell (dgp, arm, n) -> dict of arrays."""
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("error"):
            continue
        key = (r["dgp"], r["arm"], int(r["n"]))
        for col in ("bias", "ate", "tau_sd", "qte_int_err", "ols_ate", "ipw_ate", "true_ate", "val_loss"):
            cells[key][col].append(_f(r.get(col)))
    return cells


def stat(vals):
    a = np.array([v for v in vals if not math.isnan(v)])
    if a.size == 0:
        return math.nan, math.nan, 0
    return float(a.mean()), float(a.std(ddof=1) if a.size > 1 else 0.0), a.size


def bias_is_real(mean, sd, k):
    if k <= 1 or math.isnan(sd):
        return None
    return abs(mean) > 2.0 * sd / math.sqrt(k)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glob", default=None, help="shard glob (default outputs/flexible_te/h1_shard*.csv)")
    args = p.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    pattern = args.glob or os.path.join(here, "..", "outputs", "flexible_te", "h1_shard*.csv")

    rows = load(pattern)
    if not rows:
        print(f"no shards matched {pattern}")
        return
    cells = agg(rows)
    n_err = sum(1 for r in rows if r.get("error"))
    print(f"loaded {len(rows)} rows ({n_err} errored) from {pattern}\n")

    dgps = sorted({k[0] for k in cells})
    arms = sorted({k[1] for k in cells})
    ns = sorted({k[2] for k in cells})

    # ---- per-cell table --------------------------------------------------
    hdr = f"{'dgp':<9}{'arm':<22}{'n':>7}{'seeds':>6}{'ATE bias (mean±sd)':>22}{'real?':>7}{'tau_sd':>9}{'QTE err':>9}{'OLS bias':>10}{'IPW bias':>10}"
    print(hdr)
    print("-" * len(hdr))
    for dgp in dgps:
        for n in ns:
            for arm in arms:
                c = cells.get((dgp, arm, n))
                if not c:
                    continue
                bm, bs, k = stat(c["bias"])
                tsd, _, _ = stat(c["tau_sd"])
                qe, _, _ = stat(c["qte_int_err"])
                true_ate, _, _ = stat(c["true_ate"])
                ols_m, _, _ = stat(c["ols_ate"])
                ipw_m, _, _ = stat(c["ipw_ate"])
                real = bias_is_real(bm, bs, k)
                rflag = "YES" if real else ("no" if real is False else "?")
                print(f"{dgp:<9}{arm:<22}{n:>7}{k:>6}{bm:>+13.3f} ± {bs:>5.3f}{rflag:>7}"
                      f"{tsd:>9.3f}{qe:>9.3f}{ols_m - true_ate:>+10.3f}{ipw_m - true_ate:>+10.3f}")
        print()

    # ---- pre-registered verdicts ----------------------------------------
    print("=" * 72)
    print("PRE-REGISTERED VERDICTS")
    print("=" * 72)

    def cell_bias(dgp, arm, n):
        c = cells.get((dgp, arm, n))
        if not c:
            return None
        bm, bs, k = stat(c["bias"])
        return bm, bs, k, bias_is_real(bm, bs, k)

    # H1 (scalar ATE consistency on gamma at n=20000)
    if "gamma" in dgps and 20000 in ns:
        add = cell_bias("gamma", "gaussian", 20000)
        spl = cell_bias("gamma", "flexible_continuous", 20000)
        if add and spl:
            h1 = (add[3] is True) and (spl[3] is False)
            print(f"H1 [additive ATE bias persists @ n=20k on gamma, spline's does not]: "
                  f"{'SUPPORTED' if h1 else 'REJECTED'}")
            print(f"    additive: bias={add[0]:+.3f}±{add[1]:.3f} real={add[3]}   "
                  f"spline: bias={spl[0]:+.3f}±{spl[1]:.3f} real={spl[3]}")
            if not h1:
                print("    (note: if additive bias is NOT real, the additive arm recovers the")
                print("     SCALAR ATE on gamma — its failure is distributional/QTE, see H3.)")
    # H2 (spline |bias| <= additive |bias| everywhere)
    h2_all = True
    h2_lines = []
    for dgp in dgps:
        for n in ns:
            add = cell_bias(dgp, "gaussian", n)
            spl = cell_bias(dgp, "flexible_continuous", n)
            if add and spl:
                ok = abs(spl[0]) <= abs(add[0]) + 1e-9
                h2_all &= ok
                h2_lines.append(f"    {dgp} n={n}: |spline|={abs(spl[0]):.3f} vs |additive|={abs(add[0]):.3f}  {'ok' if ok else 'VIOLATED'}")
    print(f"\nH2 [spline |bias| <= additive |bias| at every cell]: {'SUPPORTED' if h2_all else 'REJECTED'}")
    for ln in h2_lines:
        print(ln)
    # H3 (QTE error decreases in n on gamma; tau_sd decreases in n on gaussian)
    print("\nH3 [heterogeneity readouts move the right way in n]:")
    if "gamma" in dgps:
        qs = [(n, stat(cells[("gamma", "flexible_continuous", n)]["qte_int_err"])[0])
              for n in ns if ("gamma", "flexible_continuous", n) in cells]
        mono = all(qs[i][1] >= qs[i + 1][1] for i in range(len(qs) - 1))
        print(f"    gamma spline QTE err by n: {[(n, round(q, 3)) for n, q in qs]}  "
              f"{'decreasing (SUPPORTED)' if mono else 'NOT monotone'}")
    if "gaussian" in dgps:
        ts = [(n, stat(cells[("gaussian", "flexible_continuous", n)]["tau_sd"])[0])
              for n in ns if ("gaussian", "flexible_continuous", n) in cells]
        mono = all(ts[i][1] >= ts[i + 1][1] for i in range(len(ts) - 1))
        print(f"    gaussian spline tau_sd by n: {[(n, round(t, 3)) for n, t in ts]}  "
              f"{'decreasing (SUPPORTED)' if mono else 'NOT monotone'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
