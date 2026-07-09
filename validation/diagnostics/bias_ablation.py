"""Ablation sweep to diagnose the small-n ATE bias of the gaussian (additive) arm.

For the gaussian outcome (correctly specified — so any bias is NOT shape
misspecification), fit the `causal_model='gaussian'` arm under three conditions
and log, per fit, BOTH the model-agnostic CRN-readout ATE and the learned causal
margin parameters (`ate`, `const`, `scale`):

  baseline      confounded data, init (ate=0, scale=1, const=0)  -> reproduces the sweep
  unconfounded  X ⟂ Z data, init at 0                            -> tests "copula steals
                                                                     the confounded effect"
  init_truth    confounded data, init AT the truth (ate=1, ...)  -> tests "optimisation can't
                                                                     reach truth from 0" vs
                                                                     "biased value is the MLE"

Decision rules:
  * unconfounded removes the bias            => copula-absorbs-confounded-effect (hyp 3)
  * init_truth stays at truth, baseline low  => optimisation can't reach truth from 0 (hyp 1)
  * init_truth drifts down to baseline value => biased value is the genuine NLL optimum (hyp 2)

Sharded like ate_sweep.py: run several instances over disjoint --seeds, each
writing its own --out CSV; bias_plot.py concatenates them.

Usage (from validation/, in the frugal-flows-flowjax env):
  micromamba run -n frugal-flows-flowjax python -m diagnostics.bias_ablation \
      --seeds 0 --out outputs/bias_shard0.csv
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
from frugal_flows.bijections.univariate_normal_cdf import UnivariateNormalCDF  # noqa: E402
from frugal_flows.causal_flows import train_frugal_flow  # noqa: E402
from diagnostics.ate_extraction_suite import intervene  # noqa: E402
from diagnostics.outcome_families import FAMILIES  # noqa: E402
from diagnostics.quick_sense_check import base_hyperparams, final_val_loss  # noqa: E402

FIELDNAMES = [
    "condition", "family", "n", "seed", "true_ate",
    "ate_crn", "ate_param", "const_param", "scale_param",
    "frac_neg", "val_loss", "secs", "error",
]

# condition -> (family name, init-at-truth flag)
CONDITIONS = {
    "baseline": ("gaussian", False),
    "unconfounded": ("gaussian_unconfounded", False),
    "init_truth": ("gaussian", True),
}


def _scalar(x) -> float:
    return float(np.ravel(np.asarray(x))[0])


def find_margin(obj, _depth=0):
    """Recursively locate the UnivariateNormalCDF causal margin inside a fitted flow."""
    if isinstance(obj, UnivariateNormalCDF):
        return obj
    if _depth > 12:
        return None
    for attr in ("bijection", "bijections"):
        child = getattr(obj, attr, None)
        if child is None:
            continue
        children = child if isinstance(child, (list, tuple)) else [child]
        for c in children:
            found = find_margin(c, _depth + 1)
            if found is not None:
                return found
    return None


def init_args(true_ate, true_const, true_scale, at_truth):
    if at_truth:
        return {"ate": jnp.array([float(true_ate)]),
                "scale": float(true_scale), "const": float(true_const)}
    return {"ate": jnp.zeros(1), "scale": 1.0, "const": 0.0}


def run(args):
    ns = [int(x) for x in args.ns.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    cp = [args.const, args.ate]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    total = len(conditions) * len(ns) * len(seeds)
    done = 0
    print(f"[shard {args.out}] conditions={conditions} ns={ns} seeds={seeds} "
          f"=> {total} fits  epochs={args.epochs}", flush=True)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        w.writeheader(); fh.flush()
        for cond in conditions:
            fam_name, at_truth = CONDITIONS[cond]
            fam = FAMILIES[fam_name]
            true_ate = fam.true_ate(cp)
            true_const = fam.mean_do(cp, 0)            # identity link => const = mean_do(0)
            true_scale = float(np.sqrt(fam.phi))       # gaussian: scale = sqrt(phi)
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
                    row.update(condition=cond, family=fam_name, n=n, seed=seed, true_ate=true_ate)
                    try:
                        ff, losses = train_frugal_flow(
                            jr.fold_in(key, 1), Y, u_z, condition=X,
                            causal_model="gaussian",
                            causal_model_args=init_args(true_ate, true_const, true_scale, at_truth),
                            **base_hyperparams(args.epochs),
                        )
                        m = intervene(jr.fold_in(key, 2), ff, X.shape[1], args.n_mc)
                        margin = find_margin(ff.bijection)
                        row.update(
                            ate_crn=m["ate"], frac_neg=m["frac_neg"],
                            ate_param=_scalar(margin.ate) if margin is not None else "",
                            const_param=_scalar(margin.const) if margin is not None else "",
                            scale_param=_scalar(margin.scale) if margin is not None else "",
                            val_loss=final_val_loss(losses),
                        )
                    except Exception as e:  # noqa: BLE001
                        row.update(ate_crn=float("nan"), error=repr(e)[:200])
                    row["secs"] = round(time.time() - t0, 1)
                    w.writerow(row); fh.flush()
                    done += 1
                    tag = "ERR" if row["error"] else f"crn={float(row['ate_crn']):+.3f} param={row['ate_param']}"
                    print(f"[{args.out}] {done}/{total} {cond} n={n} seed={seed} {tag} ({row['secs']}s)", flush=True)
    print(f"[shard {args.out}] DONE {done}/{total}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--conditions", default="baseline,unconfounded,init_truth")
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
