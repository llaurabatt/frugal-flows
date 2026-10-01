"""Test of the 'pulled towards independence' explanation of the leftover confounding (2026-09-28).

Hypothesis: at n ~ 6000 the validation-selected copula has learned only part of how the image
depends on the covariate; the copula starts near independence, whose estimate is the naive
difference, so the missing part of the dependence stays in the estimate as confounding.

Prediction, per pixel k (same toy as toy_multi.py: Y_k = b_k Z + tau_k T + noise):
    err_k  ~=  (b_k - bhat_k) * (mean Z | T=1  -  mean Z | T=0)
where bhat_k is the slope on Z of the FITTED model's conditional mean E[Y_k | Z, T=0].

How bhat_k is computed from the fitted flow (no sampling of u needed):
  the model is p(y, u | t) = p*(y | t) q(u | r),  r = F*(y | t).  For a grid of covariate ranks
  u_j, E[Y | u_j, t] = sum_m w_mj y_m / sum_m w_mj  with image ranks r_m drawn uniform,
  y_m = the margin applied to r_m under t, and w_mj = q(u_j | r_m) (copula density, evaluated
  in the density direction: one pass). bhat_k = slope of E[Y_k | u_j, 0] on Z_j = Phi^-1(u_j).
Also reported: the effective sample size of the weights (how peaked they are).
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
p.add_argument("--M", type=int, default=20000)
p.add_argument("--J", type=int, default=41)
p.add_argument("--out", required=True)
args = p.parse_args()

# ---- data, identical to toy_multi.py (a=2, sigma=1, rho=0.7)
rng = np.random.default_rng(args.seed)
K, n = args.K, args.n
b = np.linspace(0.2, 2.0, K)
tau = np.where(np.arange(K) < K // 2, 1.0, 0.0)
Sigma = 0.7 ** np.abs(np.subtract.outer(np.arange(K), np.arange(K)))
Z = rng.standard_normal(n)
T = (rng.random(n) < 1 / (1 + np.exp(-2.0 * Z))).astype(float)
Y = np.outer(Z, b) + np.outer(T, tau) + rng.multivariate_normal(np.zeros(K), Sigma, size=n)
u_z = np.clip(stats.norm.cdf(Z), 1e-6, 1 - 1e-6)
dZ = Z[T == 1].mean() - Z[T == 0].mean()

flow, losses = train_frugal_flow(
    causal_model="flexible_continuous", key=jr.PRNGKey(args.seed),
    y=jnp.asarray(Y), u_z=jnp.asarray(u_z[:, None]), condition=jnp.asarray(T[:, None]),
    learning_rate=1e-3, max_epochs=1000, max_patience=30, batch_size=100,
    causal_model_args={"RQS_knots": 8, "nn_depth": 1, "nn_width": 48, "flow_layers": 4, "conditioner": "mlp"},
    nn_width=16, flow_layers=4, RQS_knots=8, nn_depth=1, show_progress=False,
)
dist = paramax.unwrap(flow)
B = dist.bijection.bijections

# ---- the estimate, read out as usual
r = interventional_samples(jr.key(0), flow, cond_dim=1, n_mc=50_000, dim_y=K)
y0, y1 = np.asarray(r["y0"]), np.asarray(r["y1"])
ok = np.isfinite(y0).all(1) & np.isfinite(y1).all(1)
err = (y1[ok] - y0[ok]).mean(0) - tau


def fwd(b_, xs, cond):
    return jax.vmap(b_.transform)(xs, cond) if b_.cond_shape is not None else jax.vmap(b_.transform)(xs)


def inv_logdet(b_, xs, cond):
    f = b_.inverse_and_log_det
    return jax.vmap(f)(xs, cond) if b_.cond_shape is not None else jax.vmap(f)(xs)


# ---- image ranks r_m in the space between the copula and the margin (after block 2)
M, J = args.M, args.J
base = jr.uniform(jr.key(1), (M, K + 1))
zeros = jnp.zeros((M, 1))
xs = base
for i in COPULA_BLOCKS:
    xs = fwd(B[i], xs, zeros)
r_flow = xs[:, :K]                        # unchanged by blocks 1-2; block 0 only rescales

# y_m under t = 0 and t = 1 (margin + tanh blocks; u column untouched)
def to_y(t):
    ys = xs
    c = jnp.full((M, 1), float(t))
    for i in range(max(COPULA_BLOCKS) + 1, len(B)):
        ys = fwd(B[i], ys, c)
    return np.asarray(ys[:, :K])


Y0m, Y1m = to_y(0), to_y(1)
keep = np.isfinite(Y0m).all(1) & np.isfinite(Y1m).all(1)

# ---- copula density q(u_j | r_m): density direction through blocks 2, 1, 0
u_grid = (np.arange(J) + 0.5) / J
logw = np.empty((M, J))
for j, uj in enumerate(u_grid):
    v = jnp.hstack([r_flow, jnp.full((M, 1), uj)])
    total = jnp.zeros(M)
    for i in reversed(COPULA_BLOCKS):
        v, ld = inv_logdet(B[i], v, zeros)
        total = total + ld
    logw[:, j] = np.asarray(total)
logw = logw[keep]
w = np.exp(logw - logw.max(0, keepdims=True))
ess = (w.sum(0) ** 2) / (w ** 2).sum(0)
m0 = (w.T @ Y0m[keep]) / w.sum(0)[:, None]          # (J, K): E[Y | u_j, t=0]
m1 = (w.T @ Y1m[keep]) / w.sum(0)[:, None]
zg = stats.norm.ppf(u_grid)
bhat0 = np.array([np.polyfit(zg, m0[:, k], 1)[0] for k in range(K)])
bhat1 = np.array([np.polyfit(zg, m1[:, k], 1)[0] for k in range(K)])
pred = (b - bhat0) * dZ

res = {"seed": args.seed, "K": K, "n": n, "dZ": float(dZ), "b": b.tolist(), "bhat_t0": bhat0.tolist(),
       "bhat_t1": bhat1.tolist(), "err": err.tolist(), "pred_err": pred.tolist(),
       "corr_err_pred": float(np.corrcoef(err, pred)[0, 1]),
       "slope_err_on_pred": float(err @ pred / (pred @ pred)),
       "share_of_dependence_missed": float(1 - (bhat0 @ b) / (b @ b)),
       "ess_min": float(ess.min()), "ess_median": float(np.median(ess)), "best_epoch": int(np.argmin(losses["val"])) + 1}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(res, open(args.out, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("seed", "corr_err_pred", "slope_err_on_pred", "share_of_dependence_missed", "ess_min")}))
