"""Does the flexible-continuous frugal flow recover the ATE under confounding when the
truth is simple and n is large?  (2026-09-27, investigating the ~3.6 % of confounding
left in the E2 estimates.)

Data: Z ~ N(0, 1); T ~ Bernoulli(sigmoid(a Z)) (a = 0 is a randomised trial);
Y = Z + tau T + sigma e, e ~ N(0, 1), tau = 1.  So the true ATE is exactly 1.
The covariate ranks passed to the flow are the TRUE ranks Phi(Z), so the stage that fits
the covariate margins plays no part.

The fit is the library's train_frugal_flow(causal_model="flexible_continuous"), the same
call exp_ate_recovery.py makes, and the ATE is read out the same way
(interventional_samples: paired draws from the causal margin under do(0) and do(1)).

Hypothesis being tested: the flow maximises the joint density of (Y, U_Z) given T, and its
copula models U_Z given the outcome rank R without seeing T.  The true U_Z | T depends on T
whenever assignment depends on Z, so the model cannot match the data with the true causal
margin, and the maximum-likelihood fit moves the margin towards the naive difference.  If
this is right, the bias
  * is zero (up to noise) when a = 0,
  * does not shrink when n grows from 5000 to 50000,
  * is smaller when sigma is small, because then R almost determines U_Z.
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
p.add_argument("--a", type=float, required=True)        # confounding strength, 0 = randomised
p.add_argument("--sigma", type=float, required=True)    # outcome noise sd
p.add_argument("--n", type=int, required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--out", required=True)
args = p.parse_args()

rng = np.random.default_rng(args.seed)
n, tau = args.n, 1.0
Z = rng.standard_normal(n)
T = (rng.random(n) < 1 / (1 + np.exp(-args.a * Z))).astype(float)
Y = Z + tau * T + args.sigma * rng.standard_normal(n)
u_z = np.clip(stats.norm.cdf(Z), 1e-6, 1 - 1e-6)

naive = Y[T == 1].mean() - Y[T == 0].mean()
X = np.column_stack([np.ones(n), T, Z])
ols = np.linalg.lstsq(X, Y, rcond=None)[0][1]

flow, losses = train_frugal_flow(
    causal_model="flexible_continuous",
    key=jr.PRNGKey(args.seed),
    y=jnp.asarray(Y[:, None]),
    u_z=jnp.asarray(u_z[:, None]),
    condition=jnp.asarray(T[:, None]),
    learning_rate=1e-3, max_epochs=400, max_patience=30, batch_size=100,
    causal_model_args={"RQS_knots": 8, "nn_depth": 1, "nn_width": 48, "flow_layers": 4,
                       "conditioner": "mlp"},
    nn_width=16, flow_layers=4, RQS_knots=8, nn_depth=1,
    show_progress=False,
)
r = interventional_samples(jr.key(0), flow, cond_dim=1, n_mc=200_000, dim_y=1)
y0, y1 = r["y0"], r["y1"]
ok = np.isfinite(y0) & np.isfinite(y1)
ff = float(np.mean(y1[ok] - y0[ok]))
res = {"a": args.a, "sigma": args.sigma, "n": n, "seed": args.seed, "tau": tau,
       "naive": float(naive), "ols": float(ols), "ff": ff,
       "ff_share_of_confounding": float((ff - tau) / (naive - tau)) if args.a else None,
       "n_epochs": len(losses["val"]), "best_val": float(min(losses["val"]))}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(res, open(args.out, "w"), indent=1)
print(json.dumps(res))
