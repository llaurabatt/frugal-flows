"""E5 (additive contrast): profile likelihood of the gaussian-arm causal-margin ATE.

The additive (`gaussian`) arm is biased low at small n. Is the true ATE a LOCAL vs
GLOBAL optimum of the (profiled) likelihood? We freeze the causal-margin `ate` at a
grid of values, re-optimise ALL OTHER parameters (copula + margin scale/const), and
record the resulting validation NLL. The curve NLL(ate) is the profile likelihood.

  * If its minimum sits BELOW the true ATE at small n, the biased estimate is the
    genuine finite-sample optimum (non-identifiability) — not just an optimiser that
    failed to reach truth.
  * If truth becomes the minimum as n grows, the small-n bias is a finite-sample /
    trap effect that data (not epochs) fixes.

This frames why the SPLINE arm (flexible margin) escapes the scalar-ATE trap — at the
cost of the spurious effect heterogeneity the other experiments characterise.

Implementation touches NO core code: it reuses the existing `NonTrainable` + `eqx.tree_at`
freezing pattern from `frugal_flows.causal_flows`. For each (n, seed) the copula is
initialised once (identical across grid points via the same key), then for each grid
`ate` we pin `ate`, refit the rest with `flowjax.train.fit_to_data`, and log the NLL.
The unconstrained free fit is logged too as the reference optimum.

Sharded by --seeds. Usage (from validation/):
  micromamba run -n frugal-flows-flowjax python -m diagnostics.bias_profile \
      --seeds 0 --out outputs/profile_shard0.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402
from paramax import NonTrainable  # noqa: E402

_VALIDATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VALIDATION_DIR not in sys.path:
    sys.path.insert(0, _VALIDATION_DIR)

import data_processing_and_simulations.causl_sim_data_generation as causl_py  # noqa: E402
from flowjax.train import fit_to_data  # noqa: E402
from frugal_flows.causal_flows import train_frugal_flow  # noqa: E402
from diagnostics.ate_extraction_suite import intervene  # noqa: E402
from diagnostics.bias_ablation import find_margin, _scalar  # noqa: E402
from diagnostics.outcome_families import FAMILIES  # noqa: E402
from diagnostics.quick_sense_check import base_hyperparams, final_val_loss  # noqa: E402

FIELDNAMES = ["n", "seed", "kind", "grid_ate", "true_ate", "val_loss",
              "ate_after", "ate_crn", "scale_after", "secs", "error"]

# path to the UnivariateNormalCDF inside a trained gaussian frugal flow
_CDF = lambda f: f.bijection.bijections[-1].bijection.bijections[0]  # noqa: E731


def _build_gaussian_flow(key, Y, u_z, X, ate_init, epochs):
    """Structured gaussian frugal flow, ~untrained (1 epoch), ate initialised at ate_init.
    Same key => identical copula init across grid points, so profiles are comparable."""
    hp = base_hyperparams(epochs)
    hp["max_epochs"] = 1  # just to obtain a correctly-structured object
    ff, _ = train_frugal_flow(
        jr.fold_in(key, 1), Y, u_z, condition=X, causal_model="gaussian",
        causal_model_args={"ate": jnp.array([ate_init]), "scale": 1.0, "const": 0.0}, **hp,
    )
    return ff


def _profile_point(key, Y, u_z, X, g, epochs, n_mc):
    """Pin ate=g, refit the rest, return (val_loss, ate_after, scale_after, ate_crn)."""
    ff = _build_gaussian_flow(key, Y, u_z, X, g, epochs)

    def set_and_freeze(cdf):
        cdf = eqx.tree_at(lambda c: c.ate, cdf, jnp.array([g]))
        return eqx.tree_at(lambda c: c.ate, cdf, NonTrainable(cdf.ate))

    ff = eqx.tree_at(_CDF, ff, replace_fn=set_and_freeze)
    iv = jnp.hstack([Y, u_z])
    hp = base_hyperparams(epochs)
    ff2, losses = fit_to_data(
        key=jr.fold_in(key, 2), dist=ff, data=(iv, X),
        learning_rate=hp["learning_rate"], max_epochs=epochs,
        max_patience=epochs, batch_size=hp["batch_size"], show_progress=False,
    )
    # ate is frozen at g; only `scale` (and copula) moved. `margin.ate` is now a
    # NonTrainable wrapper, so don't read it — ate_after is g by construction.
    margin = find_margin(ff2.bijection)
    m = intervene(jr.fold_in(key, 3), ff2, X.shape[1], n_mc)
    return final_val_loss(losses), g, \
        _scalar(margin.scale) if margin is not None else float("nan"), m["ate"]


def _free_fit(key, Y, u_z, X, epochs, n_mc):
    """Unconstrained gaussian fit (ate free): the reference optimum."""
    ff, losses = train_frugal_flow(
        jr.fold_in(key, 1), Y, u_z, condition=X, causal_model="gaussian",
        causal_model_args={"ate": jnp.zeros(1), "scale": 1.0, "const": 0.0},
        **base_hyperparams(epochs),
    )
    margin = find_margin(ff.bijection)
    m = intervene(jr.fold_in(key, 2), ff, X.shape[1], n_mc)
    return final_val_loss(losses), _scalar(margin.ate) if margin is not None else float("nan"), \
        _scalar(margin.scale) if margin is not None else float("nan"), m["ate"]


def run(args):
    ns = [int(x) for x in args.ns.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    grid = [float(x) for x in np.linspace(args.grid_lo, args.grid_hi, args.grid_n)]
    cp = [args.const, args.ate]
    fam = FAMILIES["gaussian"]
    true_ate = fam.true_ate(cp)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    total = len(ns) * len(seeds) * (len(grid) + 1)  # +1 free fit per (n,seed)
    done = 0
    print(f"[shard {args.out}] gaussian-arm profile likelihood ns={ns} seeds={seeds} "
          f"grid={grid} => {total} fits  epochs={args.epochs}", flush=True)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        w.writeheader(); fh.flush()
        for n in ns:
            for seed in seeds:
                data = fam.generate(n, causal_params=cp, seed=seed)
                X, Y, Z_disc, Z_cont = data["X"], data["Y"], data["Z_disc"], data["Z_cont"]
                uz = causl_py.generate_uz_samples(Z_disc, Z_cont, False, seed,
                                                  base_hyperparams(args.epochs))
                u_z = uz["uz_samples"]
                key0 = jr.key(1000 * seed + 1)

                # free reference fit
                t0 = time.time()
                row = {k: "" for k in FIELDNAMES}
                row.update(n=n, seed=seed, kind="free", grid_ate="", true_ate=true_ate)
                try:
                    vl, a_after, s_after, a_crn = _free_fit(key0, Y, u_z, X, args.epochs, args.n_mc)
                    row.update(val_loss=vl, ate_after=a_after, scale_after=s_after, ate_crn=a_crn)
                except Exception as e:  # noqa: BLE001
                    row.update(val_loss=float("nan"), error=repr(e)[:200])
                row["secs"] = round(time.time() - t0, 1)
                w.writerow(row); fh.flush(); done += 1
                print(f"[{args.out}] {done}/{total} n={n} seed={seed} FREE "
                      f"ate={row['ate_after']} nll={row['val_loss']} ({row['secs']}s)", flush=True)

                # profile grid
                for g in grid:
                    t0 = time.time()
                    row = {k: "" for k in FIELDNAMES}
                    row.update(n=n, seed=seed, kind="profile", grid_ate=g, true_ate=true_ate)
                    try:
                        vl, a_after, s_after, a_crn = _profile_point(key0, Y, u_z, X, g, args.epochs, args.n_mc)
                        row.update(val_loss=vl, ate_after=a_after, scale_after=s_after, ate_crn=a_crn)
                    except Exception as e:  # noqa: BLE001
                        row.update(val_loss=float("nan"), error=repr(e)[:200])
                    row["secs"] = round(time.time() - t0, 1)
                    w.writerow(row); fh.flush(); done += 1
                    tag = "ERR" if row["error"] else f"nll={float(row['val_loss']):.4f}"
                    print(f"[{args.out}] {done}/{total} n={n} seed={seed} ate*={g:.2f} {tag} ({row['secs']}s)", flush=True)
    print(f"[shard {args.out}] DONE {done}/{total}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ns", default="100,200,1000")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--grid-lo", type=float, default=0.0)
    p.add_argument("--grid-hi", type=float, default=2.0)
    p.add_argument("--grid-n", type=int, default=9)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--n-mc", type=int, default=20000)
    p.add_argument("--const", type=float, default=0.0)
    p.add_argument("--ate", type=float, default=1.0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
