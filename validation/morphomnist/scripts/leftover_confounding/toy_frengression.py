"""Frengression on the K-pixel toy (2026-09-28). Plan step 1: does the toy separate the two
frugal models the way the images do (frengression E2 slope -0.001, frugal flow ~0.06)?

Data: identical to toy_multi.py for the same (K, n, seed, a, sigma, rho) -- same generator, same
draws -- so each fit pairs with the frugal-flow fit of the same seed.
Fit: exp_frengression_recovery's own prepare_inputs / build_model / fit / sample_margins with
its default Config (the settings of the image runs: 5000 full-batch iterations, lr 1e-3,
hidden 100 x 3 layers, noise_dim 64, per-pixel y scaling, standardised z, 50000 paired draws).
Run in the frugal-flows-frengression environment.
"""
import argparse
import dataclasses
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import exp_frengression_recovery as fr  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--K", type=int, required=True)
p.add_argument("--n", type=int, required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--a", type=float, default=2.0)
p.add_argument("--sigma", type=float, default=1.0)
p.add_argument("--rho", type=float, default=0.7)
p.add_argument("--threads", type=int, default=4)
p.add_argument("--out", required=True)
args = p.parse_args()

# ---- data: same code path as toy_multi.py
rng = np.random.default_rng(args.seed)
K, n = args.K, args.n
b = np.linspace(0.2, 2.0, K)
tau = np.where(np.arange(K) < K // 2, 1.0, 0.0)
Sigma = args.sigma ** 2 * args.rho ** np.abs(np.subtract.outer(np.arange(K), np.arange(K)))
Z = rng.standard_normal(n)
T = (rng.random(n) < 1 / (1 + np.exp(-args.a * Z))).astype(float)
Y = np.outer(Z, b) + np.outer(T, tau) + rng.multivariate_normal(np.zeros(K), Sigma, size=n)

t1 = T == 1
conf = b * (Z[t1].mean() - Z[~t1].mean())
naive = Y[t1].mean(0) - Y[~t1].mean(0)
ols = np.linalg.lstsq(np.column_stack([np.ones(n), T, Z]), Y, rcond=None)[0][1]

cfg = dataclasses.replace(fr.Config(), seed_fit=args.seed, threads=args.threads)
data = {"Y": Y, "X": T[:, None], "Z": Z[:, None], "z_cat_idx": np.array([False])}
inputs = fr.prepare_inputs(data, cfg)
model = fr.build_model(cfg, inputs)
fr.fit(cfg, model, inputs, out_stream=io.StringIO())
y0, y1, diag = fr.sample_margins(cfg, model, inputs)
est = (y1 - y0).mean(0)


def summary(e):
    err = e - tau
    return {"slope": float(err @ conf / (conf @ conf)), "mae": float(np.abs(err).mean()),
            "err_effect_pixels": float(err[tau == 1].mean()), "err_null_pixels": float(err[tau == 0].mean())}


res = {"K": K, "n": n, "seed": args.seed, "a": args.a, "sigma": args.sigma, "rho": args.rho,
       "method": "frengression", "config": dataclasses.asdict(cfg),
       "frengression": summary(est), "ols": summary(ols), "naive": summary(naive),
       "mc_n_used": diag.get("mc_n_used")}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(res, open(args.out, "w"), indent=1, default=str)
print(json.dumps({"K": K, "n": n, "seed": args.seed, "frengression_slope": res["frengression"]["slope"],
                  "ols_slope": res["ols"]["slope"]}))
