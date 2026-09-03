"""Classical per-pixel ATE estimators on the MorphoMNIST experiments.

Runs the standard estimators a careful applied statistician would reach for,
on exactly the datasets ``prepare_morphomnist_exps`` produces, and scores them
with exactly the metrics ``exp_ate_recovery`` uses. The point is a like-for-like
comparison: same data, same truth, same scores, so the only thing that differs
is the estimator.

Read the OLS row first. The effect in these experiments is ADDITIVE in logit
space and the outcome is modelled in logit space, so K independent regressions
of ``Y_k`` on ``(T, Z)`` are close to correctly specified and should be strong.
If they beat the flow on ATE recovery that is a fact worth knowing early: it
does not make the flow useless -- a regression cannot produce ``p(Y | do(T))``,
interventional samples, or the quantile-resolved ``tau(u)`` -- but it does
determine what the paper can claim about ATE recovery specifically.

Estimators
----------
    naive        difference in means. No adjustment; the size of its error is
                 the size of the confounding.
    ipw          inverse propensity weighting, propensity ESTIMATED by logistic
                 regression on the covariate basis. Hajek (self-normalised).
    ols          per-pixel OLS of Y_k on [T, basis(Z)]; tau_hat is the
                 coefficient on T. The natural competitor here.
    aipw         augmented IPW (doubly robust), 5-fold cross-fitted so the
                 nuisance models are not evaluated on their own training data.
    oracle_ipw   IPW using the TRUE propensity, which no real estimator has.
                 This is the sampling-noise floor, not a competitor.

Covariate basis (``--basis``)
-----------------------------
The frugal flow sees the covariates through a rank/quantile transform and a
flexible copula, so it can represent nonlinear covariate dependence. Giving the
baselines only a linear term would be an unfair comparison. Each continuous
covariate is therefore mapped to its rank and expanded:

    linear   [1, u]
    poly3    [1, u, u^2, u^3] per covariate, plus pairwise products (default)
    poly5    [1, ..., u^5] per covariate, plus pairwise products

Usage
-----
    python baselines.py --preset exp4_covariate_cate --size 8
    python baselines.py --all --size 8 --seeds 1 2 3 4 5
    python baselines.py --all --size 8 --seeds 1 2 3 --csv results.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from prepare_morphomnist_exps import PRESETS, build_preset

METHODS = ("naive", "ipw", "ols", "aipw", "oracle_ipw")
BASES = ("linear", "poly3", "poly5")
RUNS_ROOT = os.path.join(SCRIPT_DIR, "runs", "baselines")


# --------------------------------------------------------------------------- #
# covariate basis
# --------------------------------------------------------------------------- #
def _ranks(a: np.ndarray) -> np.ndarray:
    """Column-wise rank scores in (0,1). Matches the flow's stage-one view of Z."""
    n = a.shape[0]
    order = np.argsort(a, axis=0, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(n)[:, None].repeat(a.shape[1], 1), axis=0)
    return (ranks + 0.5) / n


def design_matrix(Z: np.ndarray, basis: str = "poly3") -> np.ndarray:
    """``[1, expansion(Z)]``. Discrete (0/1) columns are passed through as-is.

    Rank-transforming first means the expansion is on a bounded, uniformly
    spread variable, so high powers stay conditioned -- a raw polynomial in a
    skewed covariate is numerically much worse behaved.
    """
    Z = np.asarray(Z, dtype=np.float64)
    is_binary = np.array([np.isin(np.unique(Z[:, j]), (0.0, 1.0)).all()
                          for j in range(Z.shape[1])])
    cont, disc = Z[:, ~is_binary], Z[:, is_binary]

    cols = [np.ones((len(Z), 1))]
    if cont.shape[1]:
        u = _ranks(cont)
        deg = {"linear": 1, "poly3": 3, "poly5": 5}[basis]
        for d in range(1, deg + 1):
            cols.append(u**d)
        # pairwise interactions between covariates (not with themselves)
        for j in range(u.shape[1]):
            for k in range(j + 1, u.shape[1]):
                cols.append((u[:, [j]] * u[:, [k]]))
    if disc.shape[1]:
        cols.append(disc[:, 1:] if disc.shape[1] > 1 else disc)  # drop one level
    return np.hstack(cols)


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares with a small ridge for conditioning. y may be (n,) or (n,K)."""
    XtX = X.T @ X
    XtX[np.diag_indices_from(XtX)] += 1e-8 * np.trace(XtX) / len(XtX)
    return np.linalg.solve(XtX, X.T @ y)


# --------------------------------------------------------------------------- #
# estimators -- each returns tau_hat of shape (K,)
# --------------------------------------------------------------------------- #
def est_naive(Y, T, X, _true_p, **kw):
    return Y[T].mean(0) - Y[~T].mean(0)


def _hajek(Y, T, p):
    """Self-normalised IPW. Stabilising the weights keeps a near-0/1 propensity
    from dominating the average, which matters here: propensities reach 0.999."""
    w1, w0 = T / p, (~T) / (1 - p)
    return (Y * w1[:, None]).sum(0) / w1.sum() - (Y * w0[:, None]).sum(0) / w0.sum()


def est_ipw(Y, T, X, _true_p, **kw):
    p = _fit_propensity(X, T)
    return _hajek(Y, T, p)


def est_oracle_ipw(Y, T, X, true_p, **kw):
    return _hajek(Y, T, np.asarray(true_p))


def est_ols(Y, T, X, _true_p, **kw):
    """Per-pixel OLS of Y on [T, basis(Z)]; tau_hat is the coefficient on T.

    Solved for all K pixels at once -- the design matrix does not depend on the
    pixel, so this is one factorisation and a matmul, not K regressions.
    """
    D = np.hstack([T[:, None].astype(np.float64), X])
    return _ols(D, Y)[0]


def _fit_propensity(X, T, folds=None):
    lr = LogisticRegression(max_iter=2000, C=1e3)
    lr.fit(X, T.astype(int))
    return np.clip(lr.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)


def est_aipw(Y, T, X, _true_p, n_folds=5, seed=0, **kw):
    """Cross-fitted AIPW. Nuisances are never evaluated on their own fold."""
    n, K = Y.shape
    mu1 = np.zeros((n, K))
    mu0 = np.zeros((n, K))
    p = np.zeros(n)
    for tr, te in KFold(n_folds, shuffle=True, random_state=seed).split(X):
        lr = LogisticRegression(max_iter=2000, C=1e3).fit(X[tr], T[tr].astype(int))
        p[te] = lr.predict_proba(X[te])[:, 1]
        for t, mu in ((True, mu1), (False, mu0)):
            m = tr[T[tr] == t]
            mu[te] = X[te] @ _ols(X[m], Y[m])
    p = np.clip(p, 1e-6, 1 - 1e-6)
    g1 = (T[:, None] / p[:, None]) * (Y - mu1) + mu1
    g0 = ((~T)[:, None] / (1 - p)[:, None]) * (Y - mu0) + mu0
    return (g1 - g0).mean(0)


ESTIMATORS = {"naive": est_naive, "ipw": est_ipw, "ols": est_ols,
              "aipw": est_aipw, "oracle_ipw": est_oracle_ipw}


# --------------------------------------------------------------------------- #
# scoring -- identical keys to exp_ate_recovery.evaluate
# --------------------------------------------------------------------------- #
def score(tau_hat: np.ndarray, data: dict) -> dict:
    ATE = np.asarray(data["ATE"])
    support = ATE != 0
    err = tau_hat - ATE
    return {
        "ate_mae": float(np.abs(err).mean()),
        "ate_rmse": float(np.sqrt((err**2).mean())),
        "ate_max_abs_err": float(np.abs(err).max()),
        "ate_mae_on_support": float(np.abs(err[support]).mean()),
        "ate_mae_off_support": float(np.abs(err[~support]).mean()),
        "ate_corr": float(np.corrcoef(tau_hat, ATE)[0, 1]),
        "att_mae": float(np.abs(tau_hat - np.asarray(data["ATT"])).mean()),
        "atc_mae": float(np.abs(tau_hat - np.asarray(data["ATC"])).mean()),
    }


def run_one(preset: str, size: int, seed: int, basis: str, n: int | None,
            digit: int | None) -> list[dict]:
    """Every estimator on one dataset. Returns one row per method."""
    data = build_preset(preset, size=size, seed=seed, n=n, digit=digit)
    Y = np.asarray(data["Y"], dtype=np.float64)
    T = np.asarray(data["X"])[:, 0].astype(bool)
    X = design_matrix(np.asarray(data["Z"]), basis)
    true_p = np.asarray(data["PROPENSITY"])

    rows = []
    for name, fn in ESTIMATORS.items():
        tau = np.asarray(fn(Y, T, X, true_p, seed=seed))
        rows.append({"preset": preset, "seed": seed, "method": name,
                     "basis": basis, "n_pixels": Y.shape[1],
                     "n_units": Y.shape[0], **score(tau, data)})
    return rows


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def print_table(rows: list[dict], metric: str = "ate_mae"):
    """preset x method, mean +- sd over seeds."""
    presets = sorted({r["preset"] for r in rows}, key=lambda p: p[:4])
    w = max(len(p) for p in presets) + 2
    print(f"\n{metric}  (mean +- sd over seeds)\n")
    print("preset".ljust(w) + "".join(m.rjust(18) for m in METHODS))
    for p in presets:
        line = p.ljust(w)
        for m in METHODS:
            v = [r[metric] for r in rows if r["preset"] == p and r["method"] == m]
            line += (f"{np.mean(v):.4f}+-{np.std(v):.4f}".rjust(18) if v
                     else "-".rjust(18))
        print(line)
    print(f"\n(oracle_ipw is the sampling-noise floor, not a competitor: it uses "
          f"the TRUE propensity)")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="exp4_covariate_cate", choices=list(PRESETS))
    ap.add_argument("--all", action="store_true", help="every preset")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--digit", type=int, default=0)
    ap.add_argument("--all-digits", action="store_true")
    ap.add_argument("--basis", default="poly3", choices=BASES)
    ap.add_argument("--metric", default="ate_mae")
    ap.add_argument("--csv", default=None, help="also write a tidy CSV here")
    args = ap.parse_args(argv)

    presets = list(PRESETS) if args.all else [args.preset]
    digit = None if args.all_digits else args.digit

    rows = []
    for p in presets:
        for s in args.seeds:
            rows += run_one(p, args.size, s, args.basis, args.n, digit)
            print(f"  done: {p} seed {s}", flush=True)

    print_table(rows, args.metric)

    os.makedirs(RUNS_ROOT, exist_ok=True)
    out = args.csv or os.path.join(
        RUNS_ROOT, f"baselines_k{args.size**2}_{args.basis}.csv")
    keys = list(rows[0])
    with open(out, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"\nwrote {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
