"""Does the gaussian-arm ATE move with MORE TRAINING (epochs), or is it stuck?

Distinguishes two stories for the small-n bias:
  * under-training: ATE is still climbing toward truth; more epochs would fix it.
  * flat likelihood: the NLL is ~flat along `ate` at small n, so `ate` barely moves
    from its 0 init regardless of epochs — only more DATA sharpens the curvature.

For the baseline condition (confounded gaussian outcome, init ate=0) at a fixed n,
sweep the epoch budget with EARLY STOPPING DISABLED (`max_patience = max_epochs`,
so the full budget is always spent) and log the final learned `ate`. A flat curve
(ate independent of epochs, well below truth) => flat-likelihood, not under-training.

Sharded by --seeds like the other diagnostics.

Usage (from validation/):
  micromamba run -n frugal-flows-flowjax python -m diagnostics.bias_epochs \
      --seeds 0 --out outputs/biasep_shard0.csv
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
from diagnostics.bias_ablation import find_margin, _scalar  # noqa: E402
from diagnostics.outcome_families import FAMILIES  # noqa: E402
from diagnostics.quick_sense_check import base_hyperparams, final_val_loss, model_args  # noqa: E402

FIELDNAMES = ["arm", "n", "seed", "epochs", "true_ate", "ate_crn", "tau_sd", "ate_param",
              "scale_param", "val_loss", "secs", "error"]


def hyperparams_no_earlystop(epochs: int) -> dict:
    hp = base_hyperparams(epochs)
    hp["max_patience"] = epochs  # never early-stop: always spend the full budget
    return hp


def run(args):
    ns = [int(x) for x in args.ns.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    epochs_grid = [int(x) for x in args.epochs_grid.split(",")]
    cp = [args.const, args.ate]
    fam = FAMILIES["gaussian"]
    true_ate = fam.true_ate(cp)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    total = len(ns) * len(seeds) * len(epochs_grid)
    done = 0
    print(f"[shard {args.out}] ns={ns} seeds={seeds} epochs_grid={epochs_grid} "
          f"=> {total} fits (early-stop OFF)", flush=True)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        w.writeheader(); fh.flush()
        for n in ns:
            for seed in seeds:
                # one dataset + u_z per (n, seed), reused across epoch budgets
                data = fam.generate(n, causal_params=cp, seed=seed)
                X, Y, Z_disc, Z_cont = data["X"], data["Y"], data["Z_disc"], data["Z_cont"]
                uz = causl_py.generate_uz_samples(Z_disc, Z_cont, False, seed,
                                                  base_hyperparams(max(epochs_grid)))
                u_z = uz["uz_samples"]
                for epochs in epochs_grid:
                    key = jr.key(1000 * seed + 1)
                    t0 = time.time()
                    row = {k: "" for k in FIELDNAMES}
                    row.update(arm=args.arm, n=n, seed=seed, epochs=epochs, true_ate=true_ate)
                    if args.arm == "gaussian":
                        cm_args = {"ate": jnp.zeros(1), "scale": 1.0, "const": 0.0}
                    else:
                        cm_args = model_args(args.arm, X.shape[1])
                    try:
                        ff, losses = train_frugal_flow(
                            jr.fold_in(key, 1), Y, u_z, condition=X,
                            causal_model=args.arm,
                            causal_model_args=cm_args,
                            **hyperparams_no_earlystop(epochs),
                        )
                        m = intervene(jr.fold_in(key, 2), ff, X.shape[1], args.n_mc)
                        margin = find_margin(ff.bijection)
                        row.update(
                            ate_crn=m["ate"], tau_sd=m["tau_sd"],
                            ate_param=_scalar(margin.ate) if margin is not None else "",
                            scale_param=_scalar(margin.scale) if margin is not None else "",
                            val_loss=final_val_loss(losses),
                        )
                    except Exception as e:  # noqa: BLE001
                        row.update(ate_crn=float("nan"), error=repr(e)[:200])
                    row["secs"] = round(time.time() - t0, 1)
                    w.writerow(row); fh.flush()
                    done += 1
                    tag = "ERR" if row["error"] else f"ate={float(row['ate_crn']):+.3f}"
                    print(f"[{args.out}] {done}/{total} n={n} seed={seed} epochs={epochs} {tag} ({row['secs']}s)", flush=True)
    print(f"[shard {args.out}] DONE {done}/{total}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", default="gaussian",
                   help="causal_model: gaussian | flexible_continuous | location_translation")
    p.add_argument("--ns", default="200,1000")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--epochs-grid", default="100,200,400,800,1600")
    p.add_argument("--n-mc", type=int, default=10000)
    p.add_argument("--const", type=float, default=0.0)
    p.add_argument("--ate", type=float, default=1.0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
