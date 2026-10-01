"""Reversed-copula arm on the K-pixel toy (2026-09-30; plan step 3).

Data identical to toy_multi.py / toy_frengression.py for the same (K, n, seed), so each fit pairs
with the current arm's and frengression's fits of that seed. Fit: train_frugal_flow(
causal_model="flexible_reversed") with the same margin and copula sizes as toy_multi.py.

Recorded:
  gformula  effect read out by simulating the intervention (u from the data, w ~ U): the fitted
            model's exact interventional distribution; does not depend on the margin/copula split
  margin    effect read out by r ~ U straight into the margin (the current arm's read-out); equals
            gformula iff the pooled image ranks are uniform
  for each: slope (share of the confounding left; 0 none, 1 naive), mae
  rank uniformity: per-coordinate KS of the observed images' ranks (pooled over arms) from U(-1, 1),
            max over coordinates (noise ~ 1.36 / sqrt(n))
"""
import argparse
import json
import os

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import paramax
from scipy import stats

from frugal_flows.causal_flows import train_frugal_flow
from frugal_flows.reversed_copula import image_ranks, interventional_samples_reversed

p = argparse.ArgumentParser()
p.add_argument("--K", type=int, required=True)
p.add_argument("--n", type=int, required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--a", type=float, default=2.0)
p.add_argument("--sigma", type=float, default=1.0)
p.add_argument("--rho", type=float, default=0.7)
p.add_argument("--rank-weight", type=float, default=0.0)
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
u_z = np.clip(stats.norm.cdf(Z), 1e-6, 1 - 1e-6)[:, None]
t1 = T == 1
conf = b * (Z[t1].mean() - Z[~t1].mean())

flow, losses = train_frugal_flow(
    causal_model="flexible_reversed", key=jr.PRNGKey(args.seed),
    y=jnp.asarray(Y), u_z=jnp.asarray(u_z), condition=jnp.asarray(T[:, None]),
    learning_rate=1e-3, max_epochs=1000, max_patience=30, batch_size=100,
    causal_model_args={"RQS_knots": 8, "nn_depth": 1, "nn_width": 48, "flow_layers": 4, "conditioner": "mlp"},
    nn_width=16, flow_layers=4, RQS_knots=8, nn_depth=1,
    rank_penalty_weight=args.rank_weight, show_progress=False,
)
out = interventional_samples_reversed(jr.key(0), flow, u_z, n_mc=50_000)


def summary(y0, y1):
    ok = np.isfinite(y0).all(1) & np.isfinite(y1).all(1)
    err = (y1[ok] - y0[ok]).mean(0) - tau
    return {"slope": float(err @ conf / (conf @ conf)), "mae": float(np.abs(err).mean()),
            "frac_dropped": float(1 - ok.mean())}


r = np.asarray(image_ranks(paramax.unwrap(flow), jnp.asarray(Y), jnp.hstack([jnp.asarray(T[:, None]), jnp.asarray(u_z)])))
ks = [stats.kstest((r[:, k] + 1) / 2, "uniform").statistic for k in range(K)]
res = {"K": K, "n": n, "seed": args.seed, "a": args.a, "rank_weight": args.rank_weight,
       "best_epoch": int(np.argmin(losses["val"])) + 1, "n_epochs": len(losses["val"]),
       "best_val": float(min(losses["val"])),
       "gformula": summary(out["gformula_y0"], out["gformula_y1"]),
       "margin": summary(out["margin_y0"], out["margin_y1"]),
       "rank_ks_max": float(max(ks)), "rank_ks_noise": float(1.36 / np.sqrt(n))}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(res, open(args.out, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("K", "n", "seed", "rank_weight", "gformula", "margin", "rank_ks_max")}))
