"""E-vii(a): zero-refit overlap diagnostic for the Gamma DGP.

Dan's hypothesis (H-overlap) for the additive arm's sign-flipping ATE bias on the
Gamma DGP is that the DGP has a severe positivity/overlap problem, and the bias is
outcome-model EXTRAPOLATION into unsupported regions (low variance, high bias — the
extrapolation signature, as opposed to the variance blow-up that would hit a
*weighting* estimator).

This script spends NO training compute. Because the DGP is known and R `set.seed`
makes regeneration deterministic, we recompute the TRUE propensity e(Z) for each
Gamma dataset already used by the H1 matrix and summarise its overlap, then
correlate per-seed overlap against the per-seed ADDITIVE-arm ATE bias read straight
out of the committed `h1_shard*.csv`.

True propensity (verified from outcome_families._build_rscript):
    X ~ Bernoulli(e(Z)),  e(Z) = logistic( 0 + 1*Zc1 + 1*Zc2 + 1*Zc3 )   (beta=1)
    Zc4 is NOT in the propensity — use only the first 3 Z_cont columns.
    lin pred = Zc1+Zc2+Zc3 ~ N(2, 11)  (mean 2, sd ~3.3) => ~40% of units e>0.95.

Caveat (pre-registered): within a fixed (DGP, n) the across-seed overlap variation is
small, so a NULL within-n correlation does NOT kill H-overlap — E-vii(b)'s beta sweep
is the decisive test. A POSITIVE correlation here is strong early evidence.

Run (from validation/, in the frugal-flows-flowjax env):
    micromamba run -n frugal-flows-flowjax python -m diagnostics.overlap_diagnostic
"""

from __future__ import annotations

import glob
import math
import os
import sys

import numpy as np

_VALIDATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VALIDATION_DIR not in sys.path:
    sys.path.insert(0, _VALIDATION_DIR)

from diagnostics.outcome_families import FAMILIES  # noqa: E402

CP = [1.0, 0.5]          # causal_params used by the H1 matrix (const, ate)
NS = [2000, 5000, 20000]
SEEDS = list(range(10))
SHARD_GLOB = os.path.join(_VALIDATION_DIR, "outputs", "flexible_te", "h1_shard*.csv")


def _expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def true_propensity(Z_cont):
    """e(Z) = logistic(Zc1+Zc2+Zc3) from the first 3 continuous confounders."""
    Z = np.asarray(Z_cont, dtype=float)
    lin = Z[:, 0] + Z[:, 1] + Z[:, 2]
    return _expit(lin), lin


def overlap_stats(e, x):
    """Overlap summaries + IPW-weight ESS using the TRUE propensity e and treatment x."""
    x = np.asarray(x, dtype=float).ravel()
    frac_extreme = float(np.mean((e < 0.05) | (e > 0.95)))
    # ATE-style HT weights: treated get 1/e, controls get 1/(1-e).
    w = x / e + (1.0 - x) / (1.0 - e)
    ess = float((w.sum() ** 2) / np.sum(w ** 2))
    return {
        "e_mean": float(e.mean()),
        "e_min": float(e.min()),
        "e_max": float(e.max()),
        "frac_extreme": frac_extreme,          # frac[e<0.05 or e>0.95]
        "ess": ess,
        "ess_frac": ess / x.shape[0],          # ESS / n
        "frac_treated": float(x.mean()),
    }


def load_additive_gamma_bias():
    """Per-(n, seed) additive-arm ATE bias on the Gamma DGP from committed shards."""
    import csv
    rows = {}
    files = sorted(glob.glob(SHARD_GLOB))
    if not files:
        raise FileNotFoundError(f"no shards matched {SHARD_GLOB}")
    for fpath in files:
        with open(fpath, newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("error"):
                    continue
                if r["dgp"] == "gamma" and r["arm"] == "gaussian":
                    try:
                        rows[(int(r["n"]), int(r["seed"]))] = float(r["bias"])
                    except (ValueError, KeyError):
                        pass
    return rows, files


def _corr(xs, ys):
    """Pearson + Spearman; returns (pearson_r, spearman_r) or (nan, nan) if degenerate."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 3 or np.ptp(xs) == 0 or np.ptp(ys) == 0:
        return float("nan"), float("nan")
    from scipy.stats import pearsonr, spearmanr
    return float(pearsonr(xs, ys)[0]), float(spearmanr(xs, ys)[0])


def main():
    fam = FAMILIES["gamma"]
    print(f"[E-vii(a)] Gamma DGP true-propensity overlap, causal_params={CP}")
    print(f"           true_ate = {fam.true_ate(CP):+.4f}\n")

    bias_map, shard_files = load_additive_gamma_bias()
    print(f"loaded additive-arm gamma bias for {len(bias_map)} (n,seed) cells "
          f"from {len(shard_files)} shards\n")

    records = []  # (n, seed, stats..., bias)
    print(f"{'n':>6} {'seed':>4} {'e_mean':>7} {'frac_extreme':>12} {'ESS/n':>7} "
          f"{'add_bias':>9}")
    for n in NS:
        for seed in SEEDS:
            data = fam.generate(n, causal_params=CP, seed=seed)
            e, lin = true_propensity(data["Z_cont"])
            st = overlap_stats(e, data["X"])
            bias = bias_map.get((n, seed), float("nan"))
            st.update(n=n, seed=seed, bias=bias, lin_mean=float(lin.mean()),
                      lin_var=float(lin.var()))
            records.append(st)
            print(f"{n:>6} {seed:>4} {st['e_mean']:>7.3f} {st['frac_extreme']:>12.3f} "
                  f"{st['ess_frac']:>7.3f} {bias:>+9.3f}")

    # sanity: lin pred should be ~N(2, 11) => mean ~2, var ~11 (confirms column order)
    lin_means = np.array([r["lin_mean"] for r in records])
    lin_vars = np.array([r["lin_var"] for r in records])
    print(f"\n[sanity] lin-pred mean={lin_means.mean():.2f} (expect ~2.0), "
          f"var={lin_vars.mean():.2f} (expect ~11) -> confirms Zc order / beta=1")

    # ---- correlations: overlap vs additive bias --------------------------------
    print("\n=== Correlation: overlap severity vs additive-arm ATE bias ===")
    print("(more-negative bias at large n is the extrapolation-under-positivity signature)\n")

    def report(label, recs):
        fe = [r["frac_extreme"] for r in recs]
        essf = [r["ess_frac"] for r in recs]
        bs = [r["bias"] for r in recs]
        keep = [i for i in range(len(bs)) if np.isfinite(bs[i])]
        fe = [fe[i] for i in keep]; essf = [essf[i] for i in keep]; bs = [bs[i] for i in keep]
        p_fe, s_fe = _corr(fe, bs)
        p_es, s_es = _corr(essf, bs)
        print(f"  {label:>22} (k={len(bs):>2}): "
              f"frac_extreme~bias  Pearson={p_fe:+.3f} Spearman={s_fe:+.3f} | "
              f"ESS/n~bias  Pearson={p_es:+.3f} Spearman={s_es:+.3f}")

    for n in NS:
        report(f"within n={n}", [r for r in records if r["n"] == n])
    report("pooled (all n,seed)", records)

    # per-n mean overlap vs per-n mean bias (the systematic, n-confounded trend)
    print("\n=== Per-n means (systematic trend across sample size) ===")
    print(f"{'n':>6} {'frac_extreme':>12} {'ESS/n':>7} {'mean_add_bias':>13}")
    for n in NS:
        recs = [r for r in records if r["n"] == n]
        bs = [r["bias"] for r in recs if np.isfinite(r["bias"])]
        print(f"{n:>6} {np.mean([r['frac_extreme'] for r in recs]):>12.3f} "
              f"{np.mean([r['ess_frac'] for r in recs]):>7.3f} "
              f"{(np.mean(bs) if bs else float('nan')):>+13.3f}")

    print("\nNOTE: within-n across-seed overlap variation is expected to be small "
          "(large n => stable empirical overlap), so within-n nulls are uninformative;\n"
          "the per-n trend + E-vii(b) beta sweep are the real tests.")


if __name__ == "__main__":
    main()
