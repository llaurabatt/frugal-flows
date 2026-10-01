"""A K-pixel toy for the leftover confounding (2026-09-28).

The images keep ~4-7 % of the confounding in the flexible-continuous estimate, independent of
training length; the 1-pixel toy (toy_gaussian.py) did not (its leftover vanished with training
and at n = 50000). This toy sits in between: a small "image" with a known truth.

Data, for pixel k = 1..K:
    Z ~ N(0, 1);  T ~ Bernoulli(sigmoid(a Z))              (a = 0: randomised)
    Y_k = b_k Z + tau_k T + e_k,   e ~ N(0, Sigma),  Sigma_jk = sigma^2 rho^|j-k|
b_k runs from 0.2 to 2 across pixels (some pixels depend on Z weakly, some strongly);
tau_k = 1 on the first half of the pixels, 0 on the rest. The true ATE is tau exactly.
The covariate ranks passed to the flow are the true ranks Phi(Z).

Measures, per fit:
  confounding map c_k = b_k (mean Z | T=1 - mean Z | T=0)   (what the naive difference adds)
  slope = sum_k err_k c_k / sum_k c_k^2   (share of the confounding left; 1 = naive, 0 = none)
  mae   = mean_k |err_k|
Fit and read-out as exp_ate_recovery.py's plain setting (joint ff, lr 1e-3, margin 48/8,
copula width 16, batch 100, patience 30), paired draws from the margin under do(0), do(1).
"""
import argparse
import json
import os

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from scipy import stats

from frugal_flows.causal_flows import train_frugal_flow
from frugal_flows.interventions import interventional_samples

p = argparse.ArgumentParser()
p.add_argument("--K", type=int, required=True)
p.add_argument("--n", type=int, required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--a", type=float, default=2.0)
p.add_argument("--sigma", type=float, default=1.0)
p.add_argument("--rho", type=float, default=0.7)
p.add_argument("--umarg-weight", type=float, default=0.0)
p.add_argument("--patience", type=int, default=30)
p.add_argument("--max-epochs", type=int, default=1000)
p.add_argument("--out", required=True)
args = p.parse_args()

rng = np.random.default_rng(args.seed)
K, n = args.K, args.n
b = np.linspace(0.2, 2.0, K)
tau = np.where(np.arange(K) < K // 2, 1.0, 0.0)
Sigma = args.sigma ** 2 * args.rho ** np.abs(np.subtract.outer(np.arange(K), np.arange(K)))
Z = rng.standard_normal(n)
T = (rng.random(n) < 1 / (1 + np.exp(-args.a * Z))).astype(float)
Y = np.outer(Z, b) + np.outer(T, tau) + rng.multivariate_normal(np.zeros(K), Sigma, size=n)
u_z = np.clip(stats.norm.cdf(Z), 1e-6, 1 - 1e-6)

t1 = T == 1
conf = b * (Z[t1].mean() - Z[~t1].mean())
naive = Y[t1].mean(0) - Y[~t1].mean(0)
X = np.column_stack([np.ones(n), T, Z])
ols = np.linalg.lstsq(X, Y, rcond=None)[0][1]

flow, losses = train_frugal_flow(
    causal_model="flexible_continuous", key=jr.PRNGKey(args.seed),
    y=jnp.asarray(Y), u_z=jnp.asarray(u_z[:, None]), condition=jnp.asarray(T[:, None]),
    learning_rate=1e-3, max_epochs=args.max_epochs, max_patience=args.patience, batch_size=100,
    causal_model_args={"RQS_knots": 8, "nn_depth": 1, "nn_width": 48, "flow_layers": 4, "conditioner": "mlp"},
    nn_width=16, flow_layers=4, RQS_knots=8, nn_depth=1,
    copula_umarg_weight=args.umarg_weight, show_progress=False,
)
r = interventional_samples(jr.key(0), flow, cond_dim=1, n_mc=50_000, dim_y=K)
y0, y1 = np.asarray(r["y0"]), np.asarray(r["y1"])
ok = np.isfinite(y0).all(1) & np.isfinite(y1).all(1)
ff = (y1[ok] - y0[ok]).mean(0)


def summary(est):
    err = est - tau
    return {"slope": float(err @ conf / (conf @ conf)), "mae": float(np.abs(err).mean()),
            "err_effect_pixels": float(err[tau == 1].mean()), "err_null_pixels": float(err[tau == 0].mean())}


res = {"K": K, "n": n, "seed": args.seed, "a": args.a, "sigma": args.sigma, "rho": args.rho,
       "umarg_weight": args.umarg_weight, "patience": args.patience, "n_epochs": len(losses["val"]),
       "best_epoch": int(np.argmin(losses["val"])) + 1, "best_val": float(min(losses["val"])),
       "ff": summary(ff), "ols": summary(ols), "naive": summary(naive), "frac_dropped": float(1 - ok.mean())}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(res, open(args.out, "w"), indent=1)
print(json.dumps({"K": K, "n": n, "seed": args.seed, "ff_slope": res["ff"]["slope"], "ols_slope": res["ols"]["slope"]}))
