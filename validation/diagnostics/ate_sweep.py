"""Sample-size sweep for ATE extraction: mean ATE + uncertainty over seeds.

For a grid of (outcome family x causal-margin arm x n x seed) this fits the frugal
flow, extracts the ATE model-agnostically (paired-CRN, dim 0 of the fitted flow at
fixed T), and writes ONE CSV ROW PER FIT (summary stats only — no sample arrays).
Rows are flushed incrementally so partial progress survives an interruption.

It is designed to be SHARDED: run several instances over disjoint `--seeds`, each
writing its own `--out` CSV; `ate_sweep_plot.py` concatenates `sweep_*.csv` and
draws the mean +/- spread bands. (Separate files per shard => no write contention.)

Usage (from validation/, in the frugal-flows-flowjax env):
  micromamba run -n frugal-flows-flowjax python -m diagnostics.ate_sweep \
      --ns 100,200,1000,2000,5000 --seeds 0,1 --out outputs/sweep_shard0.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402,F401
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402

_VALIDATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VALIDATION_DIR not in sys.path:
    sys.path.insert(0, _VALIDATION_DIR)

import data_processing_and_simulations.causl_sim_data_generation as causl_py  # noqa: E402
from diagnostics.ate_extraction_suite import intervene  # noqa: E402
from diagnostics.outcome_families import FAMILIES  # noqa: E402
from diagnostics.quick_sense_check import base_hyperparams, fit_model  # noqa: E402

FIELDNAMES = [
    "family", "arm", "n", "seed", "true_ate", "ate", "ate_relerr", "tau_sd",
    "mean0", "mean1", "true_m0", "true_m1", "var0", "var1", "frac_neg",
    "val_loss", "secs", "error",
]


def run(args):
    ns = [int(x) for x in args.ns.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    cp = [args.const, args.ate]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    total = len(families) * len(ns) * len(seeds) * len(arms)
    done = 0
    print(f"[shard {args.out}] grid: families={families} arms={arms} ns={ns} seeds={seeds} "
          f"=> {total} fits  epochs={args.epochs}", flush=True)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        w.writeheader()
        fh.flush()
        for fname in families:
            fam = FAMILIES[fname]
            true_ate = fam.true_ate(cp)
            true_m0, true_m1 = fam.mean_do(cp, 0), fam.mean_do(cp, 1)
            for n in ns:
                for seed in seeds:
                    # one dataset + one u_z per (family, n, seed); reused across arms
                    data = fam.generate(n, causal_params=cp, seed=seed)
                    X, Y, Z_disc, Z_cont = data["X"], data["Y"], data["Z_disc"], data["Z_cont"]
                    uz = causl_py.generate_uz_samples(Z_disc, Z_cont, False, seed,
                                                      base_hyperparams(args.epochs))
                    u_z = uz["uz_samples"]
                    for ai, arm in enumerate(arms):
                        key = jr.key(1000 * seed + 13 * ai + 1)
                        t0 = time.time()
                        row = {k: "" for k in FIELDNAMES}
                        row.update(family=fname, arm=arm, n=n, seed=seed,
                                   true_ate=true_ate, true_m0=true_m0, true_m1=true_m1)
                        try:
                            ff, val_loss = fit_model(jr.fold_in(key, 1), Y, u_z, X, arm, args.epochs)
                            m = intervene(jr.fold_in(key, 2), ff, X.shape[1], args.n_mc)
                            ate = m["ate"]
                            row.update(
                                ate=ate,
                                ate_relerr=abs(ate - true_ate) / max(abs(true_ate), 1e-8),
                                tau_sd=m["tau_sd"], mean0=m["mean0"], mean1=m["mean1"],
                                var0=m["var0"], var1=m["var1"], frac_neg=m["frac_neg"],
                                val_loss=val_loss,
                            )
                        except Exception as e:  # noqa: BLE001 — keep sweeping
                            row.update(ate=float("nan"), error=repr(e)[:200])
                        row["secs"] = round(time.time() - t0, 1)
                        w.writerow(row)
                        fh.flush()
                        done += 1
                        tag = "ERR" if row["error"] else f"ate={float(row['ate']):+.3f}"
                        print(f"[{args.out}] {done}/{total} {fname}/{arm} n={n} seed={seed} "
                              f"{tag} ({row['secs']}s)", flush=True)
    print(f"[shard {args.out}] DONE {done}/{total}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--families", default="gaussian,gamma")
    p.add_argument("--arms", default="gaussian,flexible_continuous")
    p.add_argument("--ns", default="100,200,1000,2000,5000")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--n-mc", type=int, default=10000)
    p.add_argument("--const", type=float, default=0.0)
    p.add_argument("--ate", type=float, default=1.0)
    p.add_argument("--out", required=True, help="output CSV path for THIS shard")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
