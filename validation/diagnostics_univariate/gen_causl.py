"""S11 (halo prereg Amendment A6): generate one causl dataset and save it as npz.

Runs in the micromamba env ``frugal-flows`` (the only env with rpy2 + R ``causl``). It does NOT
import ``frugal_flows`` or ``validation/data_processing_and_simulations`` (that module imports the
package at top level); the R scripts are copied verbatim from
``validation/data_processing_and_simulations/causl_sim_data_generation.py`` (M1, M2, M3) and from
``validation/diagnostics/outcome_families.py`` at commit 9acb24a (branch history of
``spline-bias-analysis``; the July H1 matrix's gamma DGP).

Model -> generator mapping (see README in the S11 report):
  M1  4 gamma covariates             generate_mixed_samples            (causalSamp, fams c(3,3,3,3))
  M2  2 gamma + 2 binary covariates  generate_discrete_samples         (rfrugalParam, fams c(3,3,5,5))
  M3  5 gamma + 5 binary covariates  generate_many_discrete_samples    (rfrugalParam, 5 x 3 + 5 x 5)
  gamma_margin  July H1 gamma-outcome DGP: 4 Gaussian covariates, Y ~ Gamma(log link, phi 0.5),
                beta = c(1, 0.5), Z->X confounding beta = 1; true ATE = e^1.5 - e^1 = 1.7634.
For M1-M3 the causal margin is Y | do(T) ~ N(1 + ATE*T, 1): causal_params = [1, ATE]
(Y beta = c(intercept, slope) in causl), as written in A6 and the paper's section 4.1.

Usage:
  micromamba run -n frugal-flows python gen_causl.py --model M2 --ate 1 --n 25000 --seed 3 --out x.npz
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage

from npz_io import load_npz, save_npz  # noqa: F401  (re-exported)


def _m1(N, cp, seed):  # = causl_sim_data_generation.generate_mixed_samples
    return f"""
    library(causl)
    pars <- list(Zc1 = list(beta = c(1), phi=1),
                 Zc2 = list(beta = c(1), phi=1),
                 Zc3 = list(beta = c(1), phi=1),
                 Zc4 = list(beta = c(1), phi=1),
                 X = list(beta = c(-2,1,1,1,1)),
                 Y = list(beta = c({cp[0]}, {cp[1]}), phi=1),
                 cop = list(beta=matrix(c(0.5,0.3,0.1,0.8,
                                              0.4,0.1,0.8,
                                                  0.1,0.8,
                                                      0.8), nrow=1)))

    set.seed({seed})  # for consistency
    fams <- list(c(3,3,3,3),5,1,1)
    data_samples <- causalSamp({N}, formulas=list(list(Zc1~1, Zc2~1, Zc3~1, Zc4~1), X~Zc1+Zc2+Zc3+Zc4, Y~X, ~1), family=fams, pars=pars)
    """


def _m2(N, cp, seed):  # = causl_sim_data_generation.generate_discrete_samples
    return f"""
    library(causl)
    forms <- list(list(Zc1 ~ 1, Zc2 ~ 1, Zd3 ~ 1, Zd4 ~ 1), X ~ Zc1*Zc2+Zd3+Zd4, Y ~ X, ~ 1)
    fams <- list(c(3,3,5,5), 5, 1, 1)
    pars <- list(Zc1 = list(beta=2, phi=1),
                Zc2 = list(beta=2, phi=1),
                Zd3 = list(beta=0),
                Zd4 = list(beta=0),
                X = list(beta=c(-0.3,0.1,0.2,0.5,-0.2,1)),
                Y = list(beta=c({cp[0]}, {cp[1]}), phi=1),
                cop = list(beta=matrix(c(0.5,0.3,0.1,0.8,
                                             0.4,0.1,0.8,
                                                 0.1,0.8,
                                                     0.8), nrow=1)))
    set.seed({seed})
    data_samples <- rfrugalParam({N}, formulas = forms, family = fams, pars = pars)
    """


def _m3(N, cp, seed):  # = causl_sim_data_generation.generate_many_discrete_samples (active pars)
    return f"""
    library(causl)
    forms <- list(list(Zc1 ~ 1, Zc2 ~ 1, Zc3 ~ 1, Zc4 ~ 1, Zc5 ~ 1, Zd1 ~ 1, Zd2 ~ 1, Zd3 ~ 1, Zd4 ~ 1, Zd5 ~ 1), X ~ Zc1+Zc2+Zc3+Zc4+Zc5+Zd1+Zd2+Zd3+Zd4+Zd5, Y ~ X, ~ 1)
    fams <- list(c(3,3,3,3,3,5,5,5,5,5), 5, 1, 1)
    pars <- list(Zc1 = list(beta=1.3, phi=1),
                Zc2 = list(beta=1.3, phi=1),
                Zc3 = list(beta=1.3, phi=1),
                Zc4 = list(beta=1.3, phi=1),
                Zc5 = list(beta=1.3, phi=1),
                Zd1 = list(beta=0),
                Zd2 = list(beta=0),
                Zd3 = list(beta=0),
                Zd4 = list(beta=0),
                Zd5 = list(beta=0),
                X = list(beta=c(-0.3,0.1,0.2,0.5,-0.2,1,0.3,-0.4,0.7,-0.1,0.9)),
                Y = list(beta=c({cp[0]}, {cp[1]}), phi=1),
                cop = list(beta=matrix(c(0.3,0.4,0.5,0.1,-0.2,-0.7,0.5,-0.4, 0.5,
                                            -0.3,0.6,-0.3,0.4,-0.4,0.6,0.3,  0.2,
                                                -0.5,0.2,-0.1,-0.1,0.0,-0.4,-0.4,
                                                    -0.2,-0.2,-0.5,0.5,0.3,  0.4,
                                                        -0.1,-0.1,-0.5,-0.6,-0.2,
                                                             -0.0,0.4,0.2,   0.5,
                                                                 -0.5,0.4,  -0.4,
                                                                      0.4,   0.4,
                                                                             0.4), nrow=1)))
    set.seed({seed})
    data_samples <- rfrugalParam({N}, formulas = forms, family = fams, pars = pars)
    """


GAMMA_PHI = 0.5
GAMMA_CP = (1.0, 0.5)  # July H1 matrix defaults (--const 1.0 --ate 0.5)


def _gamma_margin(N, cp, seed):  # = outcome_families._build_rscript(y_family=3, y_phi=0.5, beta=1)
    beta = 1.0
    return f"""
    library(causl)
    pars <- list(Zc1 = list(beta = 0, phi=1),
                 Zc2 = list(beta = c(1,1), phi=1),
                 Zc3 = list(beta = c(1,1), phi=1),
                 Zc4 = list(beta = c(0,1,1,1), phi=0.5),
                 X = list(beta = c(0,{beta},{beta},{beta})),
                 Y = list(beta = c({cp[0]}, {cp[1]}), phi={GAMMA_PHI}),
                 cop = list(beta=matrix(c(2,1,0.5,1,1,1,1,1,1,1), nrow=1)))

    set.seed({seed})
    fams <- list(c(1,1,1,1), 5, 3, 1)
    data_samples <- causalSamp({N}, formulas=list(list(Zc1~1, Zc2~Zc1, Zc3~Zc1, Zc4~Zc3+Zc2+Zc1), X~Zc1+Zc2+Zc3, Y~X, ~1), family=fams, pars=pars)
    """


MODELS = {
    "M1": ("generate_mixed_samples", _m1),
    "M2": ("generate_discrete_samples", _m2),
    "M3": ("generate_many_discrete_samples", _m3),
    "gamma_margin": ("outcome_families.FAMILIES['gamma'] @9acb24a (H1 matrix)", _gamma_margin),
}


def causal_params(model: str, ate: float):
    return list(GAMMA_CP) if model == "gamma_margin" else [1.0, float(ate)]


def true_ate(model: str, cp) -> float:
    if model == "gamma_margin":  # log link: E[Y|do(t)] = exp(b0 + b1 t)
        return math.exp(cp[0] + cp[1]) - math.exp(cp[0])
    return float(cp[1])


def run_r(script: str):
    """Same plumbing as causl_sim_data_generation.generate_data_samples, numpy instead of jnp."""
    with localconverter(ro.default_converter + pandas2ri.converter):
        df = SignatureTranslatedAnonymousPackage(script, "powerpack").data_samples
    zd = [c for c in df.columns if c.startswith("Zd")]
    zc = [c for c in df.columns if c.startswith("Zc")]
    Z_disc = df[zd].values.astype(int) if zd else None
    Z_cont = df[zc].values.astype(np.float64) if zc else None
    X = df["X"].values.astype(np.float64)[:, None]
    Y = df["Y"].values.astype(np.float64)[:, None]
    return Z_disc, Z_cont, X, Y


def _logistic_overlap(X, Zdes, iters=50):
    """Estimated-propensity overlap: fraction of units with e(Z) outside [0.05, 0.95]."""
    n = X.shape[0]
    D = np.hstack([np.ones((n, 1)), Zdes])
    x = X.ravel()
    b = np.zeros(D.shape[1])
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-np.clip(D @ b, -30, 30)))
        w = np.clip(mu * (1 - mu), 1e-6, None)
        step = np.linalg.solve(D.T @ (D * w[:, None]) + 1e-6 * np.eye(D.shape[1]), D.T @ (x - mu))
        b += step
        if np.max(np.abs(step)) < 1e-8:
            break
    e = 1 / (1 + np.exp(-np.clip(D @ b, -30, 30)))
    return float(np.mean((e < 0.05) | (e > 0.95)))


def generate(model: str, ate: float, n: int, seed: int) -> dict:
    gen_name, builder = MODELS[model]
    cp = causal_params(model, ate)
    Z_disc, Z_cont, X, Y = run_r(builder(n, cp, seed))
    x = X.ravel()
    naive = float(Y[x == 1, 0].mean() - Y[x == 0, 0].mean())
    Zdes = np.hstack([b for b in (Z_cont, Z_disc) if b is not None]).astype(float)
    ols = np.linalg.lstsq(np.hstack([np.ones((n, 1)), X, Zdes]), Y[:, 0], rcond=None)[0][1]
    return dict(Y=Y, X=X, Z_cont=Z_cont, Z_disc=Z_disc,
                meta=dict(model=model, generator=gen_name, causal_params=cp, true_ate=true_ate(model, cp),
                          ate_arg=float(ate), n=int(n), seed=int(seed), naive=naive, ols=float(ols),
                          frac_treated=float(x.mean()), overlap_frac_outside_05_95=_logistic_overlap(X, Zdes),
                          causl_version=str(ro.r('as.character(packageVersion("causl"))')[0]),
                          r_version=str(ro.r("R.version.string")[0])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--ate", type=float, default=1.0, help="M1-M3 slope; ignored for gamma_margin")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    d = generate(a.model, a.ate, a.n, a.seed)
    save_npz(a.out, d)
    m = d["meta"]
    print(f"wrote {a.out}: {a.model} n={a.n} seed={a.seed} true_ate={m['true_ate']:.4f} "
          f"naive={m['naive']:.4f} ols={m['ols']:.4f} treated={m['frac_treated']:.3f} "
          f"overlap_out={m['overlap_frac_outside_05_95']:.3f}")


if __name__ == "__main__":
    main()
