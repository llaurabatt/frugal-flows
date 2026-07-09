"""Does more COPULA capacity / better u_z reduce the spline's confounding-induced bias?

The confounding-bias hypothesis says: the copula deconfounds imperfectly at small n, and
because X overlaps Z the residual leaks into the treatment-conditioned margin as apparent
effect heterogeneity. If that leak is COPULA-UNDERFITTING, then giving the copula more
capacity — or feeding it better confounder ranks u_z — should shrink the SYSTEMATIC
(seed-averaged) shape of τ(u) on the confounded baseline. If it doesn't, the leak is a
deeper joint margin/copula finite-sample identifiability effect that only n fixes.

The margin and copula capacities are separately controlled in
`train_frugal_flow_flexible_continuous`:
  * margin  <- causal_model_args (held FIXED at the default here)
  * copula  <- the top-level nn_depth / nn_width / flow_layers / RQS_knots args
and the confounder ranks come from `generate_uz_samples(..., use_marginal_flow, ...)`
(empirical-CDF ranks when False; per-column univariate flows when True).

Each config is fit on the CONFOUNDED gaussian baseline (true τ(u) flat at ATE=1, so any
seed-mean shape is spurious BIAS). We log ate, tau_sd and the 40-bin τ(u); the plotter
computes, per config, the seed-mean bias shape vs the per-seed variance.

Sharded by --seeds. Usage (from validation/):
  micromamba run -n frugal-flows-flowjax python -m diagnostics.spline_copula_hp \
      --seeds 0 --out outputs/copulahp_shard0.csv
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

_VALIDATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VALIDATION_DIR not in sys.path:
    sys.path.insert(0, _VALIDATION_DIR)

import data_processing_and_simulations.causl_sim_data_generation as causl_py  # noqa: E402
from frugal_flows.causal_flows import train_frugal_flow  # noqa: E402
from diagnostics.ate_extraction_suite import TAU_CURVE_BINS, intervene, tau_curve  # noqa: E402
from diagnostics.outcome_families import FAMILIES  # noqa: E402
from diagnostics.quick_sense_check import base_hyperparams, final_val_loss, model_args  # noqa: E402

FIELDNAMES = ["config", "cop_nn_depth", "cop_nn_width", "cop_flow_layers", "cop_RQS_knots",
              "use_marginal_flow", "lr", "n", "seed", "true_ate", "ate", "tau_sd",
              "val_loss", "n_bins", "tau_curve", "secs", "error"]

# Copula defaults (what base_hyperparams currently sends the copula).
COP_DEFAULT = {"nn_depth": 4, "nn_width": 50, "flow_layers": 4, "RQS_knots": 8}

# Each config: label -> (copula-arg overrides, use_marginal_flow, lr override or None).
# The MARGIN is always held at model_args default, so only the copula / u_z / optimiser move.
CONFIGS = {
    "baseline":       ({}, False, None),
    "cop_deep":       ({"nn_depth": 8}, False, None),
    "cop_wide":       ({"nn_width": 100}, False, None),
    "cop_layers":     ({"flow_layers": 8}, False, None),
    "cop_knots":      ({"RQS_knots": 12}, False, None),
    "cop_big":        ({"nn_depth": 8, "nn_width": 100, "flow_layers": 8, "RQS_knots": 12}, False, None),
    "marginal_flow":  ({}, True, None),
    "mflow_cop_big":  ({"nn_depth": 8, "nn_width": 100, "flow_layers": 8, "RQS_knots": 12}, True, None),
    "lr_low":         ({}, False, 1e-3),
}


def run(args):
    ns = [int(x) for x in args.ns.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    cp = [args.const, args.ate]
    fam = FAMILIES["gaussian"]  # CONFOUNDED baseline; true τ(u) flat => any mean shape is bias
    true_ate = fam.true_ate(cp)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    total = len(configs) * len(ns) * len(seeds)
    done = 0
    print(f"[shard {args.out}] copula-hp sweep configs={configs} ns={ns} seeds={seeds} "
          f"=> {total} fits  epochs={args.epochs}", flush=True)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        w.writeheader(); fh.flush()
        for n in ns:
            for seed in seeds:
                data = fam.generate(n, causal_params=cp, seed=seed)
                X, Y, Z_disc, Z_cont = data["X"], data["Y"], data["Z_disc"], data["Z_cont"]
                for cname in configs:
                    cop_over, use_mflow, lr = CONFIGS[cname]
                    cop = {**COP_DEFAULT, **cop_over}
                    t0 = time.time()
                    row = {k: "" for k in FIELDNAMES}
                    row.update(config=cname, cop_nn_depth=cop["nn_depth"], cop_nn_width=cop["nn_width"],
                               cop_flow_layers=cop["flow_layers"], cop_RQS_knots=cop["RQS_knots"],
                               use_marginal_flow=use_mflow, lr=lr if lr else 5e-3,
                               n=n, seed=seed, true_ate=true_ate, n_bins=TAU_CURVE_BINS)
                    try:
                        # u_z: empirical CDF (False) or per-column marginal flow (True)
                        uz = causl_py.generate_uz_samples(Z_disc, Z_cont, use_mflow, seed,
                                                          base_hyperparams(args.epochs))
                        u_z = uz["uz_samples"]
                        hp = base_hyperparams(args.epochs)
                        hp.update(cop)                    # copula capacity (top-level args)
                        if lr is not None:
                            hp["learning_rate"] = lr
                        key = jr.key(1000 * seed + 1)
                        ff, losses = train_frugal_flow(
                            jr.fold_in(key, 1), Y, u_z, condition=X,
                            causal_model="flexible_continuous",
                            causal_model_args=model_args("flexible_continuous", X.shape[1]),  # margin FIXED
                            **hp,
                        )
                        m = intervene(jr.fold_in(key, 2), ff, X.shape[1], args.n_mc)
                        _, tau_of_u = tau_curve(m["y0"], m["y1"])
                        row.update(ate=m["ate"], tau_sd=m["tau_sd"], val_loss=final_val_loss(losses),
                                   tau_curve=";".join(f"{v:.5f}" for v in tau_of_u))
                    except Exception as e:  # noqa: BLE001
                        row.update(ate=float("nan"), error=repr(e)[:200])
                    row["secs"] = round(time.time() - t0, 1)
                    w.writerow(row); fh.flush(); done += 1
                    tag = "ERR" if row["error"] else f"ate={float(row['ate']):+.3f} tau_sd={float(row['tau_sd']):.3f}"
                    print(f"[{args.out}] {done}/{total} n={n} seed={seed} {cname} {tag} ({row['secs']}s)", flush=True)
    print(f"[shard {args.out}] DONE {done}/{total}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", default=",".join(CONFIGS))
    p.add_argument("--ns", default="200")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--n-mc", type=int, default=20000)
    p.add_argument("--const", type=float, default=0.0)
    p.add_argument("--ate", type=float, default=1.0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
