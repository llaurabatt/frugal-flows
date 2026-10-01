"""Spread check for the regression-dilution hypothesis (2026-09-30; README, top section).

Hypothesis: the current arm's copula q(u | r) is under-confident (too wide) at n ~ 6000; inverting it
with Bayes attenuates the implied dependence of the image on the covariate, and the missing part stays
in the effect estimate as confounding.

Toy as toy_multi.py: Y(t) = b Z + tau t + e, e ~ N(0, Sigma), Z ~ N(0, 1). In the interventional world
Z | Y(t) = y is Gaussian with
    mean  m(y) = b' (Sigma + b b')^-1 (y - tau t),   variance  v* = 1 / (1 + b' Sigma^-1 b).
The copula models u = Phi(Z) given the image ranks r = F*(y | t), so in z = Phi^-1(u) units its
conditional distribution should have variance v* and mean m(y).

For each held-in row i: draw u from the fitted copula given r_i (n_draw times; forward through the
copula blocks with fresh base noise), convert to z, and record the per-row variance and mean.
Reported: mean model variance / v*  (> 1 = under-confident, the hypothesis), and the slope of the model's
conditional mean on the true m(y) (< 1 = shrunk towards 0, also under-confident).
Also the fit's effect error and leftover slope, for pairing with the spread.
"""
import argparse
import json
import os

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import paramax
from scipy import stats

from frugal_flows.causal_flows import COPULA_BLOCKS, train_frugal_flow
from frugal_flows.interventions import interventional_samples

p = argparse.ArgumentParser()
p.add_argument("--K", type=int, default=8)
p.add_argument("--n", type=int, default=5923)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--n-rows", type=int, default=1000)
p.add_argument("--n-draw", type=int, default=200)
p.add_argument("--out", required=True)
args = p.parse_args()

rng = np.random.default_rng(args.seed)
K, n = args.K, args.n
b = np.linspace(0.2, 2.0, K)
tau = np.where(np.arange(K) < K // 2, 1.0, 0.0)
Sigma = 0.7 ** np.abs(np.subtract.outer(np.arange(K), np.arange(K)))
Z = rng.standard_normal(n)
T = (rng.random(n) < 1 / (1 + np.exp(-2.0 * Z))).astype(float)
Y = np.outer(Z, b) + np.outer(T, tau) + rng.multivariate_normal(np.zeros(K), Sigma, size=n)
u_z = np.clip(stats.norm.cdf(Z), 1e-6, 1 - 1e-6)
t1 = T == 1
conf = b * (Z[t1].mean() - Z[~t1].mean())

flow, losses = train_frugal_flow(
    causal_model="flexible_continuous", key=jr.PRNGKey(args.seed),
    y=jnp.asarray(Y), u_z=jnp.asarray(u_z[:, None]), condition=jnp.asarray(T[:, None]),
    learning_rate=1e-3, max_epochs=1000, max_patience=30, batch_size=100,
    causal_model_args={"RQS_knots": 8, "nn_depth": 1, "nn_width": 48, "flow_layers": 4, "conditioner": "mlp"},
    nn_width=16, flow_layers=4, RQS_knots=8, nn_depth=1, show_progress=False,
)
dist = paramax.unwrap(flow)
B = dist.bijection.bijections

r = interventional_samples(jr.key(0), flow, cond_dim=1, n_mc=50_000, dim_y=K)
y0, y1 = np.asarray(r["y0"]), np.asarray(r["y1"])
ok = np.isfinite(y0).all(1) & np.isfinite(y1).all(1)
err = (y1[ok] - y0[ok]).mean(0) - tau


def apply(b_, xs, cond, inverse):
    f = b_.inverse if inverse else b_.transform
    return jax.vmap(f)(xs, cond) if b_.cond_shape is not None else jax.vmap(f)(xs)


# rows to probe; data -> base (image ranks in base space)
idx = np.random.default_rng(0).choice(n, args.n_rows, replace=False)
x = jnp.hstack([jnp.asarray(Y[idx]), jnp.asarray(u_z[idx, None])])
cond = jnp.asarray(T[idx, None])
xs = x
for b_ in reversed(B):
    xs = apply(b_, xs, cond, inverse=True)
r_base = xs[:, :K]

# draws of u given each row's ranks: forward through the copula blocks with fresh base noise
zdraw = np.empty((args.n_draw, args.n_rows))
key = jr.key(1)
for s in range(args.n_draw):
    key, sub = jr.split(key)
    ys = jnp.hstack([r_base, jr.uniform(sub, (args.n_rows, 1))])
    for i in COPULA_BLOCKS:
        ys = apply(B[i], ys, cond, inverse=False)
    zdraw[s] = stats.norm.ppf(np.clip(np.asarray(ys[:, K]), 1e-6, 1 - 1e-6))

A = np.linalg.inv(Sigma + np.outer(b, b))
m_true = (Y[idx] - np.outer(T[idx], tau)) @ A @ b
v_true = 1.0 / (1.0 + b @ np.linalg.solve(Sigma, b))
v_model = zdraw.var(0, ddof=1)
m_model = zdraw.mean(0)
res = {"seed": args.seed, "K": K, "n": n, "v_true": float(v_true),
       "v_model_mean": float(v_model.mean()), "v_ratio": float(v_model.mean() / v_true),
       "mean_slope_on_truth": float(np.polyfit(m_true, m_model, 1)[0]),
       "mean_corr": float(np.corrcoef(m_true, m_model)[0, 1]),
       "leftover_slope": float(err @ conf / (conf @ conf)), "mae": float(np.abs(err).mean()),
       "best_epoch": int(np.argmin(losses["val"])) + 1}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(res, open(args.out, "w"), indent=1)
print(json.dumps(res))
