"""E-vii(b) support: Monte-Carlo "trimmed true ATE" for the Gamma DGP.

Purpose: the additive-arm bias hypothesis (H-overlap, see
`diagnostics/overlap_diagnostic.py`) needs a TRUE trimmed estimand to compare
fitted models against, not just the untrimmed analytic ATE
`exp(const+ate) - exp(const)`. This script computes that trimmed truth
directly by Monte Carlo, using the fact that the Gamma DGP is FRUGAL: the
outcome's copula quantile `u_y` is coupled to `Z` only, independent of `X`
given `Z`. That makes the per-unit counterfactual pair exact, closed-form
functions of the observed data — no model fitting required:

    phi = 0.5 (GAMMA_PHI), k = 1/phi = 2 (gamma shape)
    theta_t = phi * exp(const + ate*t)                 (gamma scale under do(X=t))
    u_i     = gamma.cdf(Y_i, a=k, scale=theta_{X_i})    (recovered copula quantile)
    Y_i(t)  = gamma.ppf(u_i, a=k, scale=theta_t)        (counterfactual outcome)
    tau_i   = Y_i(1) - Y_i(0) = (theta_1 - theta_0) * gamma.ppf(u_i, a=k, scale=1)

Full ATE = mean(tau_i) over all units (must recover the analytic ATE up to MC
error, since u_i-recovery is exact). TRIMMED ATE restricts the mean to units
with TRUE propensity 0.05 < e(Z_i) < 0.95 — valid because u_y ⟂ X | Z, so
conditioning/restricting on Z preserves the law of u_y (and hence of tau).

Run (from validation/, in the frugal-flows-flowjax env):
    micromamba run -n frugal-flows-flowjax python -m diagnostics.trimmed_estimand
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.stats import gamma as _gamma

_VALIDATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VALIDATION_DIR not in sys.path:
    sys.path.insert(0, _VALIDATION_DIR)

from diagnostics.outcome_families import FAMILIES, GAMMA_PHI  # noqa: E402

TRIM_LO, TRIM_HI = 0.05, 0.95


def _expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def true_propensity(Z_cont, beta: float):
    """e(Z) = expit(beta*(Zc1+Zc2+Zc3)), intercept 0, Zc4 excluded (beta-general)."""
    Z = np.asarray(Z_cont, dtype=float)
    lin = beta * (Z[:, 0] + Z[:, 1] + Z[:, 2])
    return _expit(lin)


def per_unit_tau(Y, X, const: float, ate: float):
    """Exact per-unit counterfactual pair via frugal u_y recovery.

    Returns (tau, theta0, theta1) where tau_i = Y_i(1) - Y_i(0).
    """
    Y = np.asarray(Y, dtype=float).ravel()
    X = np.asarray(X, dtype=float).ravel()
    k = 1.0 / GAMMA_PHI
    theta0 = GAMMA_PHI * np.exp(const + ate * 0)
    theta1 = GAMMA_PHI * np.exp(const + ate * 1)
    theta_obs = np.where(X == 0, theta0, theta1)
    u = _gamma.cdf(Y, a=k, scale=theta_obs)
    tau = (theta1 - theta0) * _gamma.ppf(u, a=k, scale=1.0)
    return tau, theta0, theta1


def run_one(beta: float, n: int, seed: int, const: float, ate: float):
    """Generate one (beta, seed) dataset and compute full/trimmed MC ATE stats."""
    fam = FAMILIES[f"gamma_b{beta:g}"]
    data = fam.generate(n, causal_params=[const, ate], seed=seed)
    X = np.asarray(data["X"], dtype=float).ravel()
    e = true_propensity(data["Z_cont"], beta)
    tau, theta0, theta1 = per_unit_tau(data["Y"], X, const, ate)

    keep = (e > TRIM_LO) & (e < TRIM_HI)
    trimmed_away = ~keep
    frac_trimmed = float(trimmed_away.mean())
    full_ate_hat = float(tau.mean())
    trimmed_ate_hat = float(tau[keep].mean()) if keep.any() else float("nan")
    trimmed_away_mean_tau = (
        float(tau[trimmed_away].mean()) if trimmed_away.any() else float("nan")
    )
    return {
        "beta": beta,
        "seed": seed,
        "frac_trimmed": frac_trimmed,
        "full_ate_hat": full_ate_hat,
        "trimmed_ate_hat": trimmed_ate_hat,
        "trimmed_away_mean_tau": trimmed_away_mean_tau,
    }


def _parse_float_list(s: str):
    return [float(x) for x in s.split(",") if x.strip() != ""]


def _parse_int_list(s: str):
    return [int(x) for x in s.split(",") if x.strip() != ""]


def main():
    p = argparse.ArgumentParser(
        description="Monte-Carlo trimmed true ATE for the Gamma DGP, beta sweep."
    )
    p.add_argument("--betas", type=str, default="0,0.5,1,1.5")
    p.add_argument("--n", type=int, default=50000)
    p.add_argument("--seeds", type=str, default="100,101")
    p.add_argument("--const", type=float, default=1.0)
    p.add_argument("--ate", type=float, default=0.5)
    args = p.parse_args()

    betas = _parse_float_list(args.betas)
    seeds = _parse_int_list(args.seeds)
    const, ate = args.const, args.ate
    analytic_ate = np.exp(const + ate) - np.exp(const)

    print(f"[trimmed_estimand] Gamma DGP, causal_params=[{const},{ate}], "
          f"n={args.n}, seeds={seeds}")
    print(f"           analytic full ATE = exp(const+ate)-exp(const) = "
          f"{analytic_ate:.4f}\n")

    per_beta = {}
    for beta in betas:
        rows = [run_one(beta, args.n, seed, const, ate) for seed in seeds]
        per_beta[beta] = rows

    # ---- summary table (averaged across seeds per beta) ------------------------
    header = (f"{'beta':>6} {'frac_trimmed':>12} {'full_ATE_hat':>13} "
              f"{'trimmed_ATE_hat':>16} {'analytic_ATE':>13} {'trim_shift':>11}")
    print(header)
    print("-" * len(header))
    beta_means = {}
    for beta in betas:
        rows = per_beta[beta]
        frac_trimmed = float(np.mean([r["frac_trimmed"] for r in rows]))
        full_ate_hat = float(np.mean([r["full_ate_hat"] for r in rows]))
        trimmed_ate_hat = float(np.mean([r["trimmed_ate_hat"] for r in rows]))
        trim_shift = trimmed_ate_hat - analytic_ate
        beta_means[beta] = {
            "frac_trimmed": frac_trimmed,
            "full_ate_hat": full_ate_hat,
            "trimmed_ate_hat": trimmed_ate_hat,
            "trim_shift": trim_shift,
        }
        print(f"{beta:>6.3g} {frac_trimmed:>12.4f} {full_ate_hat:>13.4f} "
              f"{trimmed_ate_hat:>16.4f} {analytic_ate:>13.4f} {trim_shift:>+11.4f}")

    # ---- mean tau among trimmed-away units, per beta ----------------------------
    print("\n=== Mean tau_i among TRIMMED-AWAY units (e outside (0.05, 0.95)) ===")
    print(f"{'beta':>6} {'mean_tau_trimmed_away':>22}")
    for beta in betas:
        rows = per_beta[beta]
        vals = [r["trimmed_away_mean_tau"] for r in rows if np.isfinite(r["trimmed_away_mean_tau"])]
        mean_val = float(np.mean(vals)) if vals else float("nan")
        print(f"{beta:>6.3g} {mean_val:>22.4f}")

    # ---- self-tests --------------------------------------------------------------
    print("\n=== Self-tests ===")

    # (a) at every beta, full_ATE_hat recovers the analytic ATE up to MC error.
    test_a_pass = True
    for beta in betas:
        diff = abs(beta_means[beta]["full_ate_hat"] - analytic_ate)
        ok = diff < 0.05
        test_a_pass &= ok
        print(f"  (a) beta={beta:g}: |full_ATE_hat - {analytic_ate:.4f}| = "
              f"{diff:.5f}  -> {'PASS' if ok else 'FAIL'}")
    print(f"  (a) overall: {'PASS' if test_a_pass else 'FAIL'}")

    # (b) at beta=0, trimming should be a no-op (e ≡ expit(0) = 0.5 for all units).
    if 0.0 in beta_means:
        m = beta_means[0.0]
        diff_b = abs(m["trimmed_ate_hat"] - m["full_ate_hat"])
        ok_diff = diff_b < 1e-9
        ok_frac = m["frac_trimmed"] == 0.0
        ok_b = ok_diff and ok_frac
        print(f"  (b) beta=0: |trimmed_ATE_hat - full_ATE_hat| = {diff_b:.2e} "
              f"(<1e-9: {ok_diff}), frac_trimmed = {m['frac_trimmed']:.6f} "
              f"(==0: {ok_frac})  -> {'PASS' if ok_b else 'FAIL'}")
    else:
        print("  (b) beta=0 not in --betas sweep -> SKIPPED")


if __name__ == "__main__":
    main()
