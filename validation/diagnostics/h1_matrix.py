"""H1 matrix: does the ADDITIVE arm's ATE bias PERSIST at large n on a
misspecified (Gamma) DGP, while the SPLINE arm's vanishes?

This is the one open piece of the flexible-TE hypothesis test. Prior diagnostics
(SPLINE_BIAS_FINDINGS.md, E1-E6) characterised small-n behaviour (n<=1000): the
additive arm's small-n ATE attenuation is finite-sample non-identifiability, and
the spline's spurious tau_sd is a variance floor + confounding residual. What none
of them ran is the CONSISTENCY question at large n on a DGP where the additive
location-shift margin is genuinely MISSPECIFIED for the mean:

  Gamma outcome, log link  =>  E[Y|do(X=t)] = exp(const + ate*t)  (multiplicative)
  true ATE = exp(const+ate) - exp(const);  true tau(u) = (theta1-theta0)*G^{-1}(u;k)
  is quantile-RISING (genuine heterogeneity). A location-shift margin cannot represent
  this; a treatment-conditioned spline can.

PRE-REGISTERED HYPOTHESES + decision rules (recorded BEFORE running):
  bias is "real" for a cell iff |mean bias across seeds| > 2 * (sd / sqrt(n_seeds)).
  H1: on the Gamma DGP the ADDITIVE arm's ATE-bias interval EXCLUDES 0 at n=20000
      (persistent misspecification bias), while the SPLINE arm's interval COVERS 0.
  H2: the spline arm's |mean bias| <= the additive arm's |mean bias| at every
      (dgp, n) cell.
  H3: on the Gamma DGP the spline's integrated QTE error decreases in n (it is
      tracking the true rising tau(u), not a flat guess); on the Gaussian-null DGP
      the spline's tau_sd decreases in n (spurious heterogeneity shrinks).

Anchors: OLS (Y ~ X + Z) coefficient on X, and IPW (logistic propensity on Z,
Horvitz-Thompson) — cheap, and both are ALSO misspecified on the Gamma DGP, so they
contextualise "how hard is this cell" rather than serving as an oracle.

Sharded by --seeds like the other diagnostics; rerunnable; one row per (dgp, arm, n, seed).

Usage (from validation/, frugal-flows-flowjax env):
  micromamba run -n frugal-flows-flowjax python -m diagnostics.h1_matrix \
      --seeds 0 --ns 2000,5000,20000 --out outputs/flexible_te/h1_shard0.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)  # must precede any jnp array creation

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402

# np.trapz was removed in numpy 2.0 in favour of np.trapezoid; support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz

_VALIDATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VALIDATION_DIR not in sys.path:
    sys.path.insert(0, _VALIDATION_DIR)

import data_processing_and_simulations.causl_sim_data_generation as causl_py  # noqa: E402
from diagnostics.ate_extraction_suite import intervene, tau_curve  # noqa: E402
from diagnostics.outcome_families import FAMILIES  # noqa: E402
from diagnostics.quick_sense_check import base_hyperparams, fit_model  # noqa: E402

FIELDNAMES = ["dgp", "confound_beta", "arm", "n", "seed", "true_ate", "ate", "bias",
              "tau_sd", "qte_int_err", "val_loss", "ols_ate",
              "ipw_ate", "ipw_ate_unclipped",
              "overlap_min", "overlap_max", "overlap_p01", "overlap_p99",
              "overlap_frac_clipped", "secs", "error"]


def _beta_of(dgp):
    """Recover the Z->X confounding strength beta from a family name for the CSV.

    `gamma_b0.5` -> 0.5; the plain `gamma`/`gaussian` families use beta=1.0; the
    `*_unconfounded` families use beta=0.0. beta does not affect the ground-truth
    ATE (log-link margin is beta-independent); it is recorded only to group the sweep.
    """
    if "_b" in dgp:
        try:
            return float(dgp.rsplit("_b", 1)[1])
        except ValueError:
            pass
    if dgp.endswith("unconfounded"):
        return 0.0
    return 1.0

# epochs per sample size (patience scales as epochs//10 inside base_hyperparams)
EPOCHS_BY_N = {2000: 600, 5000: 800, 20000: 1500}


# --- design matrix from causl output ---------------------------------------
def _design(Z_disc, Z_cont):
    """Stack available confounder blocks into an (n, k) float array (no intercept)."""
    blocks = []
    for Z in (Z_cont, Z_disc):
        if Z is not None:
            arr = np.asarray(Z, dtype=float)
            if arr.size:
                blocks.append(arr.reshape(arr.shape[0], -1))
    if not blocks:
        raise ValueError("no confounders found")
    return np.hstack(blocks)


# --- baseline estimators (numpy-only, no sklearn/statsmodels dependency) ----
def ols_ate(Y, X, Zdes):
    """OLS coefficient on X in  Y ~ 1 + X + Z. Misspecified on Gamma (informative)."""
    y = np.asarray(Y, dtype=float).ravel()
    x = np.asarray(X, dtype=float).ravel()
    n = y.shape[0]
    D = np.hstack([np.ones((n, 1)), x[:, None], Zdes])
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    return float(beta[1])  # coefficient on X


def _logistic_irls(D, y, iters=50, ridge=1e-6):
    """IRLS logistic regression; returns coefficients. D includes an intercept col."""
    p = D.shape[1]
    beta = np.zeros(p)
    for _ in range(iters):
        eta = np.clip(D @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1.0 - mu), 1e-6, None)
        H = D.T @ (D * w[:, None]) + ridge * np.eye(p)
        g = D.T @ (y - mu)
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


# overlap/IPW summary keys (also used to build the all-NaN fallback on failure)
IPW_KEYS = ["ipw_ate", "ipw_ate_unclipped",
            "overlap_min", "overlap_max", "overlap_p01", "overlap_p99",
            "overlap_frac_clipped"]


def ipw_ate(Y, X, Zdes, clip=0.025):
    """Horvitz-Thompson IPW ATE with a logistic propensity e(Z)=P(X=1|Z).

    Returns a dict with the clipped HT estimate (`ipw_ate`), the raw/unclipped HT
    estimate (`ipw_ate_unclipped`), and overlap summaries of the ESTIMATED propensity
    e_raw (min/max/1st-99th percentile, and the fraction outside [clip, 1-clip]).
    Under poor overlap the clip converts weight variance into bias, so reporting both
    the clipped and unclipped estimate + the overlap fraction is the E-vii(b) diagnostic.
    All from the single logistic propensity already fit here (no extra model).
    """
    y = np.asarray(Y, dtype=float).ravel()
    x = np.asarray(X, dtype=float).ravel()
    n = y.shape[0]
    D = np.hstack([np.ones((n, 1)), Zdes])
    beta = _logistic_irls(D, x)
    e_raw = 1.0 / (1.0 + np.exp(-np.clip(D @ beta, -30, 30)))
    e_clip = np.clip(e_raw, clip, 1.0 - clip)

    def _ht(e):
        return float(np.mean(x * y / e) - np.mean((1.0 - x) * y / (1.0 - e)))

    return {
        "ipw_ate": _ht(e_clip),
        "ipw_ate_unclipped": _ht(e_raw),
        "overlap_min": float(e_raw.min()),
        "overlap_max": float(e_raw.max()),
        "overlap_p01": float(np.quantile(e_raw, 0.01)),
        "overlap_p99": float(np.quantile(e_raw, 0.99)),
        "overlap_frac_clipped": float(np.mean((e_raw < clip) | (e_raw > 1.0 - clip))),
    }


def qte_integrated_error(fam, cp, y0, y1, lo=0.02, hi=0.98):
    """Trapezoid integral of |tau_hat(u) - tau_true(u)| over u in [lo, hi]."""
    u, tau_hat = tau_curve(y0, y1)
    tau_true = fam.true_tau_curve(cp, u)
    mask = (u >= lo) & (u <= hi)
    return float(_trapz(np.abs(tau_hat[mask] - tau_true[mask]), u[mask]))


def robust_moments(y0, y1):
    """Finite-filtered paired ATE + tau_sd.

    The spline margin is RQS on [-1,1] composed with atanh (Invert(Tanh)), whose
    range is all of R; a base draw landing at the tanh boundary maps to +/-inf.
    On heavy-tailed (Gamma) outcomes at large n a single one of ~n_mc samples can
    overflow, poisoning the mean. These are numerical-boundary artefacts, not valid
    draws, so we drop non-finite PAIRS (keeping the CRN pairing) and report how many
    were dropped (never silent). Returns (ate, tau_sd, n_dropped, n_kept).
    """
    y0 = np.asarray(y0); y1 = np.asarray(y1)
    finite = np.isfinite(y0) & np.isfinite(y1)
    n_drop = int((~finite).sum())
    tau = (y1 - y0)[finite]
    return float(np.mean(tau)), float(np.std(tau)), n_drop, int(finite.sum())


def run(args):
    ns = [int(x) for x in args.ns.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    dgps = [d.strip() for d in args.dgps.split(",") if d.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    cp = [args.const, args.ate]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    total = len(dgps) * len(arms) * len(ns) * len(seeds)
    done = 0
    print(f"[shard {args.out}] dgps={dgps} arms={arms} ns={ns} seeds={seeds} "
          f"=> {total} fits  causal_params={cp}", flush=True)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        w.writeheader(); fh.flush()
        for dgp in dgps:
            fam = FAMILIES[dgp]
            true_ate = fam.true_ate(cp)
            for n in ns:
                epochs = EPOCHS_BY_N.get(n, max(400, n // 15))
                for seed in seeds:
                    # one dataset per (dgp, n, seed), shared across arms + baselines
                    data = fam.generate(n, causal_params=cp, seed=seed)
                    X, Y = data["X"], data["Y"]
                    Z_disc, Z_cont = data["Z_disc"], data["Z_cont"]
                    uz = causl_py.generate_uz_samples(Z_disc, Z_cont, False, seed,
                                                      base_hyperparams(epochs))
                    u_z = uz["uz_samples"]
                    try:
                        Zdes = _design(Z_disc, Z_cont)
                        ols = ols_ate(Y, X, Zdes)
                        ipw = ipw_ate(Y, X, Zdes)  # dict: clipped/unclipped IPW + overlap
                    except Exception as e:  # noqa: BLE001
                        ols = float("nan")
                        ipw = {k: float("nan") for k in IPW_KEYS}
                        print(f"  baseline error {dgp} n={n} seed={seed}: {e!r}", flush=True)

                    for arm in arms:
                        key = jr.key(1000 * seed + 7)
                        t0 = time.time()
                        row = {k: "" for k in FIELDNAMES}
                        row.update(dgp=dgp, confound_beta=_beta_of(dgp), arm=arm, n=n,
                                   seed=seed, true_ate=true_ate, ols_ate=ols, **ipw)
                        try:
                            ff, val_loss = fit_model(jr.fold_in(key, 1), Y, u_z, X, arm, epochs)
                            m = intervene(jr.fold_in(key, 2), ff, X.shape[1], args.n_mc)
                            ate, tau_sd, n_drop, n_keep = robust_moments(m["y0"], m["y1"])
                            if n_drop:
                                print(f"  [drop] {dgp}/{arm} n={n} seed={seed}: "
                                      f"{n_drop}/{n_drop + n_keep} non-finite MC samples filtered", flush=True)
                            row.update(
                                ate=ate, bias=ate - true_ate, tau_sd=tau_sd,
                                qte_int_err=qte_integrated_error(fam, cp, m["y0"], m["y1"]),
                                val_loss=val_loss,
                            )
                        except Exception as e:  # noqa: BLE001
                            row.update(ate=float("nan"), error=repr(e)[:200])
                        row["secs"] = round(time.time() - t0, 1)
                        w.writerow(row); fh.flush()
                        done += 1
                        tag = ("ERR " + row["error"]) if row["error"] else (
                            f"ate={float(row['ate']):+.3f} (true {true_ate:+.3f}) "
                            f"bias={float(row['bias']):+.3f} tau_sd={float(row['tau_sd']):.3f}")
                        print(f"[{args.out}] {done}/{total} {dgp}/{arm} n={n} seed={seed} "
                              f"{tag} ols={ols:+.3f} ipw={ipw['ipw_ate']:+.3f} ({row['secs']}s)", flush=True)
    print(f"[shard {args.out}] DONE {done}/{total}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dgps", default="gamma,gaussian")
    p.add_argument("--arms", default="gaussian,flexible_continuous")
    p.add_argument("--ns", default="2000,5000,20000")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--n-mc", type=int, default=20000)
    p.add_argument("--const", type=float, default=1.0, help="causal_params[0] (b0)")
    p.add_argument("--ate", type=float, default=0.5, help="causal_params[1] (b1); Gamma ATE=e^{b0+b1}-e^{b0}")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
