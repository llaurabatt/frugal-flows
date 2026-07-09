"""Diagnostics for the SPLINE arm (`flexible_continuous`), parallel to bias_ablation.

The spline has no explicit `ate`/`scale` parameter, so the additive arm's
init-at-truth / scale-inflation diagnostics don't apply. Its small-n failure mode
is different: SPURIOUS effect heterogeneity (`tau_sd` > 0) on a homogeneous DGP.
And because the raw outcome is fed into a FIXED `RationalQuadraticSpline(interval=1)`
+ tanh frame (the pipeline does NOT standardise Y), the outcome's scale/offset can
eat into the spline's resolution. So the three conditions here are:

  baseline      confounded gaussian outcome, raw Y
  unconfounded  X ⟂ Z, raw Y
  standardized  confounded gaussian outcome, Y standardised to mean 0 / unit var
                before fitting (ATE + tau rescaled back to original units)

Metrics logged per fit: the paired-CRN ATE and `tau_sd` (effect heterogeneity;
the TRUE value is 0 here — the gaussian outcome has a constant location-shift
effect). Sharded by --seeds like the other diagnostics.

Usage (from validation/):
  micromamba run -n frugal-flows-flowjax python -m diagnostics.spline_ablation \
      --seeds 0 --out outputs/spline_shard0.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402

_VALIDATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VALIDATION_DIR not in sys.path:
    sys.path.insert(0, _VALIDATION_DIR)

import data_processing_and_simulations.causl_sim_data_generation as causl_py  # noqa: E402
from frugal_flows.causal_flows import train_frugal_flow  # noqa: E402
from diagnostics.ate_extraction_suite import intervene  # noqa: E402
from diagnostics.outcome_families import FAMILIES  # noqa: E402
from diagnostics.quick_sense_check import base_hyperparams, final_val_loss, model_args  # noqa: E402

FIELDNAMES = ["condition", "family", "standardized", "n", "seed", "true_ate",
              "ate_crn", "tau_sd", "frac_neg", "val_loss", "secs", "error"]

# condition -> (family name, standardize-Y flag)
CONDITIONS = {
    "baseline": ("gaussian", False),
    "unconfounded": ("gaussian_unconfounded", False),
    "standardized": ("gaussian", True),
}


def run(args):
    ns = [int(x) for x in args.ns.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    cp = [args.const, args.ate]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    total = len(conditions) * len(ns) * len(seeds)
    done = 0
    print(f"[shard {args.out}] arm=flexible_continuous conditions={conditions} ns={ns} "
          f"seeds={seeds} => {total} fits  epochs={args.epochs}", flush=True)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        w.writeheader(); fh.flush()
        for cond in conditions:
            fam_name, standardize = CONDITIONS[cond]
            fam = FAMILIES[fam_name]
            true_ate = fam.true_ate(cp)
            for n in ns:
                for seed in seeds:
                    data = fam.generate(n, causal_params=cp, seed=seed)
                    X, Y, Z_disc, Z_cont = data["X"], data["Y"], data["Z_disc"], data["Z_cont"]
                    # standardise Y for fitting; rescale ATE/tau back to original units
                    if standardize:
                        y_sd = float(jnp.std(Y))
                        Y_fit = (Y - jnp.mean(Y)) / y_sd
                    else:
                        y_sd = 1.0
                        Y_fit = Y
                    uz = causl_py.generate_uz_samples(Z_disc, Z_cont, False, seed,
                                                      base_hyperparams(args.epochs))
                    u_z = uz["uz_samples"]
                    key = jr.key(1000 * seed + 1)
                    t0 = time.time()
                    row = {k: "" for k in FIELDNAMES}
                    row.update(condition=cond, family=fam_name, standardized=standardize,
                               n=n, seed=seed, true_ate=true_ate)
                    try:
                        ff, losses = train_frugal_flow(
                            jr.fold_in(key, 1), Y_fit, u_z, condition=X,
                            causal_model="flexible_continuous",
                            causal_model_args=model_args("flexible_continuous", X.shape[1]),
                            **base_hyperparams(args.epochs),
                        )
                        m = intervene(jr.fold_in(key, 2), ff, X.shape[1], args.n_mc)
                        row.update(
                            ate_crn=m["ate"] * y_sd,       # rescale to original units
                            tau_sd=m["tau_sd"] * y_sd,
                            frac_neg=m["frac_neg"],
                            val_loss=final_val_loss(losses),
                        )
                    except Exception as e:  # noqa: BLE001
                        row.update(ate_crn=float("nan"), error=repr(e)[:200])
                    row["secs"] = round(time.time() - t0, 1)
                    w.writerow(row); fh.flush()
                    done += 1
                    tag = "ERR" if row["error"] else f"ate={float(row['ate_crn']):+.3f} tau_sd={float(row['tau_sd']):.3f}"
                    print(f"[{args.out}] {done}/{total} {cond} n={n} seed={seed} {tag} ({row['secs']}s)", flush=True)
    print(f"[shard {args.out}] DONE {done}/{total}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--conditions", default="baseline,unconfounded,standardized")
    p.add_argument("--ns", default="100,200,1000,5000")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--n-mc", type=int, default=10000)
    p.add_argument("--const", type=float, default=0.0)
    p.add_argument("--ate", type=float, default=1.0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
