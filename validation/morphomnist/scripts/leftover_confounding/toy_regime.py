"""Training-regime test on the K-pixel toy (2026-09-30; plan step 3, the fallback suspect).

The reversed copula did not remove the leftover (results_reversed). The other difference with
frengression is how it is trained: all rows, full batch, a fixed 5000 steps, no early stopping.
Here the CURRENT arm is trained without early stopping and the effect is read out along the way:
  --batch 0   full batch (all training rows per step), --epochs = steps
  --batch 100 our usual batches
Same data as toy_multi.py per seed. Recorded every --every epochs: leftover slope and mae; plus the
validation loss per epoch and the epoch our usual early stopping (patience 30) would have kept.
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
p.add_argument("--K", type=int, required=True)
p.add_argument("--n", type=int, default=5923)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--batch", type=int, required=True)
p.add_argument("--epochs", type=int, required=True)
p.add_argument("--every", type=int, required=True)
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
batch = args.batch or (n - round(0.1 * n))   # 0 = all training rows


def on_epoch(epoch, params, static):
    if epoch % args.every:
        return None
    r = interventional_samples(jr.key(0), equinox.combine(params, static), cond_dim=1, n_mc=20_000, dim_y=K)
    y0, y1 = np.asarray(r["y0"]), np.asarray(r["y1"])
    ok = np.isfinite(y0).all(1) & np.isfinite(y1).all(1)
    err = (y1[ok] - y0[ok]).mean(0) - tau
    return {"slope": float(err @ conf / (conf @ conf)), "mae": float(np.abs(err).mean())}


flow, losses = train_frugal_flow(
    causal_model="flexible_continuous", key=jr.PRNGKey(args.seed),
    y=jnp.asarray(Y), u_z=jnp.asarray(u_z[:, None]), condition=jnp.asarray(T[:, None]),
    learning_rate=1e-3, max_epochs=args.epochs, max_patience=10**7, batch_size=batch,
    causal_model_args={"RQS_knots": 8, "nn_depth": 1, "nn_width": 48, "flow_layers": 4, "conditioner": "mlp"},
    nn_width=16, flow_layers=4, RQS_knots=8, nn_depth=1,
    fit_kwargs={"on_epoch": on_epoch}, show_progress=False,
)
val = losses["val"]
best, stop = 0, len(val) - 1
for e in range(len(val)):
    if val[e] <= val[best]:
        best = e
    if e - best > 30:
        stop = e
        break
res = {"K": K, "n": n, "seed": args.seed, "batch": batch, "epochs": args.epochs,
       "early_stop_keeps": best + 1, "lowest_val_epoch": int(np.argmin(val)) + 1,
       "val": [float(v) for v in val], "track": losses["track"]}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(res, open(args.out, "w"))
print(json.dumps({"K": K, "seed": args.seed, "batch": batch, "last": losses["track"][-1] if losses["track"] else None}))
