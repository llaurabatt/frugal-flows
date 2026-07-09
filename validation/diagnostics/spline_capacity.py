"""E1: does the spline's spurious effect heterogeneity scale with CAPACITY?

Hypothesis H1 (over-flexibility / variance): the treatment-conditioned RQS margin
overfits per-treatment-arm sampling noise, so tau(u) wobbles differently for T=0
vs T=1. If so, the spurious `tau_sd` (std of the paired-CRN effect on a HOMOGENEOUS
Gaussian DGP, where the true per-quantile effect is flat => true tau_sd = 0) should
GROW with the spline's capacity and DECAY with n.

We fix n and vary one capacity axis at a time off the default config
(RQS_knots=8, nn_depth=4, nn_width=50, flow_layers=4), all exposed via
`model_args`, so this needs no core changes:

  axis 'knots'   RQS_knots  in {4, 8, 12}
  axis 'depth'   nn_depth   in {2, 4, 6}
  axis 'layers'  flow_layers in {2, 4, 6}

The shared default (knots=8, depth=4, layers=4) is run once and reused as the
common point on every axis. Per fit we log ate, tau_sd, val_loss.

Sharded by --seeds. Usage (from validation/):
  micromamba run -n frugal-flows-flowjax python -m diagnostics.spline_capacity \
      --seeds 0 --out outputs/capacity_shard0.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402  (kept for parity)
import jax.random as jr  # noqa: E402

_VALIDATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VALIDATION_DIR not in sys.path:
    sys.path.insert(0, _VALIDATION_DIR)

import data_processing_and_simulations.causl_sim_data_generation as causl_py  # noqa: E402
from frugal_flows.causal_flows import train_frugal_flow  # noqa: E402
from diagnostics.ate_extraction_suite import intervene  # noqa: E402
from diagnostics.outcome_families import FAMILIES  # noqa: E402
from diagnostics.quick_sense_check import base_hyperparams, final_val_loss  # noqa: E402

FIELDNAMES = ["axis", "level", "config", "RQS_knots", "nn_depth", "nn_width", "flow_layers",
              "n", "seed", "true_ate", "ate", "tau_sd", "val_loss", "secs", "error"]

# Default capacity (the common point shared by every axis).
DEFAULT = {"RQS_knots": 8, "nn_depth": 4, "nn_width": 50, "flow_layers": 4}

# axis -> (param name, list of levels). One line per axis in the plot; the level
# equal to the default is the shared point.
AXES = {
    "knots": ("RQS_knots", [4, 8, 12]),
    "depth": ("nn_depth", [2, 4, 6]),
    "layers": ("flow_layers", [2, 4, 6]),
}


def _configs():
    """Yield (axis, level, param_dict, label) — one per axis level.

    The default-level point is run once per axis (small redundancy) so each axis
    line is self-contained for plotting; no cross-axis bookkeeping needed.
    """
    for axis, (param, levels) in AXES.items():
        for lvl in levels:
            cfg = dict(DEFAULT)
            cfg[param] = lvl
            yield axis, lvl, cfg, f"{param}={lvl}"


def run(args):
    ns = [int(x) for x in args.ns.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    cp = [args.const, args.ate]
    fam = FAMILIES["gaussian"]  # homogeneous DGP: true tau(u) flat => true tau_sd = 0
    true_ate = fam.true_ate(cp)
    configs = list(_configs())

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    total = len(configs) * len(ns) * len(seeds)
    done = 0
    print(f"[shard {args.out}] arm=flexible_continuous capacity sweep: {len(configs)} configs "
          f"ns={ns} seeds={seeds} => {total} fits  epochs={args.epochs}", flush=True)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        w.writeheader(); fh.flush()
        for n in ns:
            for seed in seeds:
                # one dataset + u_z per (n, seed), reused across capacity configs
                data = fam.generate(n, causal_params=cp, seed=seed)
                X, Y, Z_disc, Z_cont = data["X"], data["Y"], data["Z_disc"], data["Z_cont"]
                uz = causl_py.generate_uz_samples(Z_disc, Z_cont, False, seed,
                                                  base_hyperparams(args.epochs))
                u_z = uz["uz_samples"]
                for axis, lvl, cfg, label in configs:
                    key = jr.key(1000 * seed + 1)
                    t0 = time.time()
                    row = {k: "" for k in FIELDNAMES}
                    row.update(axis=axis, level=lvl, config=label, n=n, seed=seed,
                               true_ate=true_ate, **cfg)
                    try:
                        hp = base_hyperparams(args.epochs)
                        # override the capacity knobs on both the training-loop args
                        # (copula/base flow) and the causal-margin model_args
                        hp.update(cfg)
                        ff, losses = train_frugal_flow(
                            jr.fold_in(key, 1), Y, u_z, condition=X,
                            causal_model="flexible_continuous",
                            causal_model_args=dict(cfg),
                            **hp,
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
                    print(f"[{args.out}] {done}/{total} n={n} seed={seed} {axis}:{label} {tag} ({row['secs']}s)", flush=True)
    print(f"[shard {args.out}] DONE {done}/{total}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
