"""Follow-up to toy_gaussian.py (2026-09-27). The toy grid showed a positive bias at n=5000
(about 7 % of the confounding, sigma=1) that disappears at n=50000, so the leftover is not
built into the objective. This script asks whether it is a matter of how far training got:
train WITHOUT early stopping for a fixed number of epochs, and every few epochs record the
ATE estimate and the validation loss. Same data, fit and read-out as toy_gaussian.py.

If the estimate starts near the naive difference and moves towards the truth as training
goes on, and early stopping (patience 30 on the validation loss) stops it before it gets
there, the leftover is caused by stopping too early.
"""
import argparse
import json
import os

import equinox
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from scipy import stats

from frugal_flows.causal_flows import train_frugal_flow
from frugal_flows.interventions import interventional_samples

p = argparse.ArgumentParser()
p.add_argument("--a", type=float, default=2.0)
p.add_argument("--sigma", type=float, default=1.0)
p.add_argument("--n", type=int, required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--epochs", type=int, required=True)
p.add_argument("--every", type=int, default=5)
p.add_argument("--out", required=True)
args = p.parse_args()

rng = np.random.default_rng(args.seed)
n, tau = args.n, 1.0
Z = rng.standard_normal(n)
T = (rng.random(n) < 1 / (1 + np.exp(-args.a * Z))).astype(float)
Y = Z + tau * T + args.sigma * rng.standard_normal(n)
u_z = np.clip(stats.norm.cdf(Z), 1e-6, 1 - 1e-6)
naive = Y[T == 1].mean() - Y[T == 0].mean()


def on_epoch(epoch, params, static):
    if epoch % args.every and epoch != 1:
        return None
    r = interventional_samples(jr.key(0), equinox.combine(params, static), cond_dim=1, n_mc=50_000, dim_y=1)
    ok = np.isfinite(r["y0"]) & np.isfinite(r["y1"])
    return {"ate": float(np.mean(r["y1"][ok] - r["y0"][ok]))}


flow, losses = train_frugal_flow(
    causal_model="flexible_continuous",
    key=jr.PRNGKey(args.seed),
    y=jnp.asarray(Y[:, None]), u_z=jnp.asarray(u_z[:, None]), condition=jnp.asarray(T[:, None]),
    learning_rate=1e-3, max_epochs=args.epochs, max_patience=10**6, batch_size=100,
    causal_model_args={"RQS_knots": 8, "nn_depth": 1, "nn_width": 48, "flow_layers": 4, "conditioner": "mlp"},
    nn_width=16, flow_layers=4, RQS_knots=8, nn_depth=1,
    fit_kwargs={"on_epoch": on_epoch}, show_progress=False,
)
val = losses["val"]
# the epoch patience-30 early stopping would have picked: first time 30 epochs pass without a new minimum
best, stop = 0, len(val)
for e in range(len(val)):
    if val[e] <= val[best]:
        best = e
    if e - best > 30:
        stop = e
        break
res = {"a": args.a, "sigma": args.sigma, "n": n, "seed": args.seed, "tau": tau, "naive": float(naive),
       "early_stop_best_epoch": best + 1, "early_stop_at_epoch": stop + 1,
       "global_best_epoch": int(np.argmin(val)) + 1, "val": val, "track": losses["track"]}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(res, open(args.out, "w"))
print(json.dumps({k: v for k, v in res.items() if k not in ("val", "track")}))
