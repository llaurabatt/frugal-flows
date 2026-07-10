"""10-D complex-copula HP check: which frugal-flow settings recover the ATE under
weak vs strong high-dimensional confounder dependence?

DGP (verified in copula10d_probe.py): 10 continuous Gaussian confounders + Gamma
outcome (log link, phi=0.5, cp=[1.0,0.5] -> true ATE 1.7634), binary X ~ all Z with
overlap-preserving coefficients (X.mean ~ 0.5). The Y-Z copula is a 55-param
Gaussian copula (C(11,2)) whose strength sets the dependence regime:
  weak   : all beta 0.15   -> Z-Y spearman ~0.12
  mixed  : heterogeneous   -> Z-Y spearman spanning ~0.1..0.7 (the 'complex' one)
  strong : all beta 1.6    -> Z-Y spearman ~0.68

Fit recipe: flexible_continuous spline causal margin, log-then-standardize outcome
(the winning recipe), n=2000. HP configs vary the COPULA flow capacity (the part the
10-D confounders stress); the margin spline is held at default (prior finding: spline
capacity is a non-knob). One CSV row per fit. Shardable: --shard i --nshards N.

Usage: python copula10d_hp.py --shard 0 --nshards 5 --out outputs/.../shard0.csv
"""
import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import data_processing_and_simulations.causl_sim_data_generation as causl  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402
from frugal_flows.causal_flows import train_frugal_flow  # noqa: E402

from diagnostics.quick_sense_check import base_hyperparams, model_args  # noqa: E402

K = 10
CP = (1.0, 0.5)
TRUE_ATE = math.exp(CP[0] + CP[1]) - math.exp(CP[0])  # 1.7634
N = 2000
EPOCHS = 500
N_MC = 20000
SEEDS = list(range(8))

# --- dependence regimes: 55-param (C(11,2)) copula upper-triangle beta vectors ----
_NCOP = (K + 1) * K // 2  # 55 = C(11,2)


def _factor_copula(lo, hi):
    """Heterogeneous BUT positive-definite Gaussian copula over the K+1 variables.

    Build a PD correlation matrix from a single-factor model (R_ij = a_i a_j,
    a_i in (0,1) => R = diag(1-a^2) + a a^T is PD), then map to causl's beta via
    beta = atanh(rho) (causl's Gaussian-copula link is ~tanh, verified empirically).
    Heterogeneous loadings give a genuine MIX: some pairs near-independent, some
    strongly dependent. A flat random beta vector is NOT PD (R rejects it), which
    is why the mix must be constructed in correlation space, not beta space.
    """
    a = np.linspace(lo, hi, K + 1)
    R = np.outer(a, a)
    np.fill_diagonal(R, 1.0)
    beta = np.arctanh(np.clip(R, -0.999, 0.999))
    return np.array([beta[i, j] for i in range(K + 1) for j in range(i + 1, K + 1)])


REGIMES = {
    "weak": np.full(_NCOP, 0.15),        # equicorrelation, Z-Y spearman ~0.12
    "mixed": _factor_copula(0.25, 0.97),  # heterogeneous PD, Z-Y |r| ~0.27..0.56
    "strong": np.full(_NCOP, 1.6),        # equicorrelation, Z-Y spearman ~0.68
}

# --- HP configs: overrides to the COPULA flow (base_hyperparams) -------------------
# (use_marginal_flow is popped out; the rest are train_frugal_flow **hp kwargs)
CONFIGS = {
    "base": {},
    "cop_wide": {"nn_width": 100},
    "cop_wider": {"nn_width": 200},
    "cop_deep": {"nn_depth": 8},
    "cop_layers": {"flow_layers": 8},
    "cop_knots": {"RQS_knots": 12},
    "cop_big": {"nn_depth": 8, "nn_width": 100, "flow_layers": 8, "RQS_knots": 12},
    "mflow": {"__marginal_flow__": True},
}


def build_rscript(cop_vec, seed, n, x_scale=0.25):
    z_names = [f"Zc{i}" for i in range(1, K + 1)]
    z_pars = "\n".join([f"                 {z} = list(beta=0, phi=1)," for z in z_names])
    xb = "c(0," + ",".join([str(x_scale)] * K) + ")"
    zforms = ", ".join([f"{z}~1" for z in z_names])
    xform = "X~" + "+".join(z_names)
    cop_str = "c(" + ",".join([f"{v:g}" for v in cop_vec]) + ")"
    fam_z = ",".join(["1"] * K)
    return f"""
    library(causl)
    pars <- list(
{z_pars}
                 X = list(beta={xb}),
                 Y = list(beta=c({CP[0]},{CP[1]}), phi=0.5),
                 cop = list(beta=matrix({cop_str}, nrow=1)))
    set.seed({seed})
    fams <- list(c({fam_z}), 5, 3, 1)
    data_samples <- causalSamp({n}, formulas=list(list({zforms}), {xform}, Y~X, ~1), family=fams, pars=pars)
    """


def fit_ate(regime, config, seed):
    cop_vec = REGIMES[regime]
    data = causl.generate_data_samples(build_rscript(cop_vec, seed, N))
    X, Y, Zc = data["X"], data["Y"], data["Z_cont"]
    overrides = dict(CONFIGS[config])
    use_mflow = overrides.pop("__marginal_flow__", False)
    hp = base_hyperparams(EPOCHS)
    hp.update(overrides)
    uz = causl.generate_uz_samples(None, Zc, use_mflow, seed, hp)["uz_samples"]
    # log-then-standardize outcome (winning recipe); invert on samples before differencing
    logY = jnp.log(Y)
    m = float(jnp.mean(logY))
    s = float(jnp.std(logY)) or 1.0
    zf = (logY - m) / s
    cargs = model_args("flexible_continuous", X.shape[1])
    k = jr.key(seed)
    k, kf = jr.split(k)
    ff, _ = train_frugal_flow(kf, zf, uz, condition=X,
                              causal_model="flexible_continuous", causal_model_args=cargs, **hp)
    k, ki = jr.split(k)
    z0 = np.asarray(ff.sample(ki, condition=jnp.zeros((N_MC, X.shape[1])))[:, 0])
    z1 = np.asarray(ff.sample(ki, condition=jnp.ones((N_MC, X.shape[1])))[:, 0])
    ate = float(np.mean(np.exp(z1 * s + m)) - np.mean(np.exp(z0 * s + m)))
    if not math.isfinite(ate):
        raise ValueError("non-finite ATE")
    return ate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    jobs = [(r, c, s) for r in REGIMES for c in CONFIGS for s in SEEDS]
    mine = jobs[args.shard::args.nshards]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"shard {args.shard}/{args.nshards}: {len(mine)}/{len(jobs)} jobs  true ATE={TRUE_ATE:.4f}", flush=True)
    with open(args.out, "w") as f:
        f.write("regime,config,seed,ate,bias,secs,error\n")
        for (regime, config, seed) in mine:
            t0 = time.time()
            try:
                ate = fit_ate(regime, config, seed)
                err = ""
            except Exception as e:
                ate = float("nan")
                err = repr(e)[:120]
            bias = ate - TRUE_ATE
            f.write(f"{regime},{config},{seed},{ate},{bias},{time.time()-t0:.1f},\"{err}\"\n")
            f.flush()
            print(f"  {regime:>6}/{config:<10} s{seed}: ate={ate:+.3f} bias={bias:+.3f} "
                  f"({time.time()-t0:.0f}s){' ERR '+err if err else ''}", flush=True)
    print(f"shard {args.shard} DONE", flush=True)


if __name__ == "__main__":
    main()
