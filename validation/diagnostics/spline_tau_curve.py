"""E3 (+E1b): quantile-resolved effect curve tau(u) for the SPLINE arm, decomposed
into BIAS (seed-mean) and VARIANCE (seed-SD) — the centrepiece diagnostic for the
spline's spurious effect heterogeneity.

The paired-CRN readout gives, per fit, a curve tau(u)=Q_1(u)-Q_0(u) over the
outcome quantile u (see `ate_extraction_suite.tau_curve`). We fit the spline
(`flexible_continuous`) across many seeds and stack the per-seed curves on a fixed
u-grid, so we can separate:

  * BIAS      = seed-MEAN tau(u) minus the analytic truth. If this is flat at the
                ATE on a homogeneous DGP, the spline is UNBIASED in shape.
  * VARIANCE  = seed-SD envelope of tau(u). A wide envelope with a flat mean means
                the spurious heterogeneity is per-fit NOISE (over-flexibility, H1),
                not a systematic artefact.

Conditions (each with its analytic true tau(u)):
  baseline      confounded Gaussian outcome   -> truth FLAT at ATE (tau_sd should be 0)
  unconfounded  X ⟂ Z Gaussian outcome        -> truth FLAT at ATE; isolates the
                                                  confounding contribution (H2) as the
                                                  baseline-minus-unconfounded curve gap
  gamma         confounded Gamma outcome       -> truth INCREASING (genuine multiplicative
                (E1b positive control)            heterogeneity): the spline SHOULD track it.
                                                  Distinguishes "spline always inflates"
                                                  from "spline fits real heterogeneity".

Per fit we log ate, tau_sd, val_loss and the 40-bin tau(u) as a ';'-joined string.
Sharded by --seeds like the other diagnostics.

Usage (from validation/, frugal-flows-flowjax env):
  micromamba run -n frugal-flows-flowjax python -m diagnostics.spline_tau_curve \
      --seeds 0 --out outputs/taucurve_shard0.csv
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
from diagnostics.ate_extraction_suite import TAU_CURVE_BINS, intervene, tau_curve  # noqa: E402
from diagnostics.outcome_families import FAMILIES  # noqa: E402
from diagnostics.quick_sense_check import base_hyperparams, final_val_loss, model_args  # noqa: E402

FIELDNAMES = ["condition", "family", "n", "seed", "true_ate", "ate", "tau_sd",
              "val_loss", "n_bins", "tau_curve", "secs", "error"]

# condition -> family key in FAMILIES
CONDITIONS = {
    "baseline": "gaussian",
    "unconfounded": "gaussian_unconfounded",
    "gamma": "gamma",
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
          f"seeds={seeds} => {total} fits  epochs={args.epochs}  bins={TAU_CURVE_BINS}", flush=True)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        w.writeheader(); fh.flush()
        for cond in conditions:
            fam = FAMILIES[CONDITIONS[cond]]
            true_ate = fam.true_ate(cp)
            for n in ns:
                for seed in seeds:
                    data = fam.generate(n, causal_params=cp, seed=seed)
                    X, Y, Z_disc, Z_cont = data["X"], data["Y"], data["Z_disc"], data["Z_cont"]
                    uz = causl_py.generate_uz_samples(Z_disc, Z_cont, False, seed,
                                                      base_hyperparams(args.epochs))
                    u_z = uz["uz_samples"]
                    key = jr.key(1000 * seed + 1)
                    t0 = time.time()
                    row = {k: "" for k in FIELDNAMES}
                    row.update(condition=cond, family=fam.name, n=n, seed=seed,
                               true_ate=true_ate, n_bins=TAU_CURVE_BINS)
                    try:
                        ff, losses = train_frugal_flow(
                            jr.fold_in(key, 1), Y, u_z, condition=X,
                            causal_model="flexible_continuous",
                            causal_model_args=model_args("flexible_continuous", X.shape[1]),
                            **base_hyperparams(args.epochs),
                        )
                        m = intervene(jr.fold_in(key, 2), ff, X.shape[1], args.n_mc)
                        _, tau_of_u = tau_curve(m["y0"], m["y1"])
                        row.update(
                            ate=m["ate"], tau_sd=m["tau_sd"],
                            val_loss=final_val_loss(losses),
                            tau_curve=";".join(f"{v:.5f}" for v in tau_of_u),
                        )
                    except Exception as e:  # noqa: BLE001
                        row.update(ate=float("nan"), error=repr(e)[:200])
                    row["secs"] = round(time.time() - t0, 1)
                    w.writerow(row); fh.flush()
                    done += 1
                    tag = "ERR" if row["error"] else f"ate={float(row['ate']):+.3f} tau_sd={float(row['tau_sd']):.3f}"
                    print(f"[{args.out}] {done}/{total} {cond} n={n} seed={seed} {tag} ({row['secs']}s)", flush=True)
    print(f"[shard {args.out}] DONE {done}/{total}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--conditions", default="baseline,unconfounded,gamma")
    p.add_argument("--ns", default="200,1000")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--n-mc", type=int, default=20000)
    p.add_argument("--const", type=float, default=0.0)
    p.add_argument("--ate", type=float, default=1.0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
