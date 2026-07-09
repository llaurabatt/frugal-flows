"""E2: does the spline's spurious effect heterogeneity scale with CONFOUNDING?

Hypothesis H2 (residual confounding / bias): at small n the copula deconfounds
imperfectly, leaving Z-Y structure that the treatment-conditioned margin renders
as apparent effect heterogeneity. If so, the spurious `tau_sd` (on a Gaussian DGP
whose true per-quantile effect is FLAT => true tau_sd = 0) should scale with the
Z->X confounding strength beta.

We sweep beta over the propensity backdoor coefficient (via
`outcome_families.make_gaussian_family(beta)`):
  beta = 0    -> X ⟂ Z (unconfounded)
  beta = 1    -> the original confounded DGP
  beta > 1    -> stronger backdoor (note: the logit saturates / overlap degrades
                 for large beta, which the pre-fit observational contrast reports).

Per fit we log ate, tau_sd, val_loss, and the pre-fit observational contrast
E[Y|X=1]-E[Y|X=0] (a generator sanity check: it should grow with beta while the
true ATE stays 1).

Sharded by --seeds. Usage (from validation/):
  micromamba run -n frugal-flows-flowjax python -m diagnostics.spline_confounding \
      --seeds 0 --out outputs/confound_shard0.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402  (parity)
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402

_VALIDATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VALIDATION_DIR not in sys.path:
    sys.path.insert(0, _VALIDATION_DIR)

import data_processing_and_simulations.causl_sim_data_generation as causl_py  # noqa: E402
from frugal_flows.causal_flows import train_frugal_flow  # noqa: E402
from diagnostics.ate_extraction_suite import intervene  # noqa: E402
from diagnostics.outcome_families import make_gaussian_family  # noqa: E402
from diagnostics.quick_sense_check import base_hyperparams, final_val_loss, model_args  # noqa: E402

FIELDNAMES = ["beta", "n", "seed", "true_ate", "obs_contrast", "treated_frac",
              "ate", "tau_sd", "val_loss", "secs", "error"]


def run(args):
    ns = [int(x) for x in args.ns.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    betas = [float(x) for x in args.betas.split(",")]
    cp = [args.const, args.ate]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    total = len(betas) * len(ns) * len(seeds)
    done = 0
    print(f"[shard {args.out}] arm=flexible_continuous confounding sweep betas={betas} "
          f"ns={ns} seeds={seeds} => {total} fits  epochs={args.epochs}", flush=True)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        w.writeheader(); fh.flush()
        for beta in betas:
            fam = make_gaussian_family(beta)
            true_ate = fam.true_ate(cp)
            for n in ns:
                for seed in seeds:
                    data = fam.generate(n, causal_params=cp, seed=seed)
                    X, Y, Z_disc, Z_cont = data["X"], data["Y"], data["Z_disc"], data["Z_cont"]
                    Xn, Yn = np.asarray(X).ravel(), np.asarray(Y).ravel()
                    # pre-fit generator sanity: observational contrast + treated frac
                    if (Xn == 1).any() and (Xn == 0).any():
                        obs_contrast = float(Yn[Xn == 1].mean() - Yn[Xn == 0].mean())
                    else:
                        obs_contrast = float("nan")
                    treated_frac = float(Xn.mean())
                    uz = causl_py.generate_uz_samples(Z_disc, Z_cont, False, seed,
                                                      base_hyperparams(args.epochs))
                    u_z = uz["uz_samples"]
                    key = jr.key(1000 * seed + 1)
                    t0 = time.time()
                    row = {k: "" for k in FIELDNAMES}
                    row.update(beta=beta, n=n, seed=seed, true_ate=true_ate,
                               obs_contrast=round(obs_contrast, 4), treated_frac=round(treated_frac, 4))
                    try:
                        ff, losses = train_frugal_flow(
                            jr.fold_in(key, 1), Y, u_z, condition=X,
                            causal_model="flexible_continuous",
                            causal_model_args=model_args("flexible_continuous", X.shape[1]),
                            **base_hyperparams(args.epochs),
                        )
                        m = intervene(jr.fold_in(key, 2), ff, X.shape[1], args.n_mc)
                        row.update(ate=m["ate"], tau_sd=m["tau_sd"],
                                   val_loss=final_val_loss(losses))
                    except Exception as e:  # noqa: BLE001
                        row.update(ate=float("nan"), error=repr(e)[:200])
                    row["secs"] = round(time.time() - t0, 1)
                    w.writerow(row); fh.flush()
                    done += 1
                    tag = "ERR" if row["error"] else f"ate={float(row['ate']):+.3f} tau_sd={float(row['tau_sd']):.3f}"
                    print(f"[{args.out}] {done}/{total} beta={beta} n={n} seed={seed} "
                          f"obs={obs_contrast:+.2f} {tag} ({row['secs']}s)", flush=True)
    print(f"[shard {args.out}] DONE {done}/{total}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--betas", default="0,0.5,1.0,1.5")
    p.add_argument("--ns", default="200,1000")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--n-mc", type=int, default=20000)
    p.add_argument("--const", type=float, default=0.0)
    p.add_argument("--ate", type=float, default=1.0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
