"""Confirm the propensity score (stage 3) does NOT affect the ATE / its bias.

The frugal-flow ATE is a STAGE-2 quantity (a parameter of the frugal flow =
copula ∘ causal margin). The quantile propensity score is STAGE 3
(`train_propensity_flow`), fit AFTER stage 2 and used only to resample X from Z
when generating synthetic OBSERVATIONAL data. A do(X=t) query fixes X, so the
propensity is bypassed and cannot touch the ATE.

This script demonstrates that concretely on the full `FrugalFlowModel` pipeline:
for a few gaussian-outcome cells it fits stage 1 (marginals) + stage 2 (frugal),
reads the ATE, THEN fits stage 3 (propensity) and reads the ATE again from the
SAME `model.frugal_flow`. The two readings are identical => the fitted propensity
leaves the ATE (and its small-n bias) untouched.

Usage (from validation/, in the frugal-flows-flowjax env):
  micromamba run -n frugal-flows-flowjax python -m diagnostics.bias_propensity_check
"""

from __future__ import annotations

import argparse
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402

_VALIDATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VALIDATION_DIR not in sys.path:
    sys.path.insert(0, _VALIDATION_DIR)

from frugal_flows.benchmarking import FrugalFlowModel  # noqa: E402
from diagnostics.ate_extraction_suite import intervene  # noqa: E402
from diagnostics.bias_ablation import _scalar, find_margin  # noqa: E402
from diagnostics.outcome_families import FAMILIES  # noqa: E402
from diagnostics.quick_sense_check import base_hyperparams  # noqa: E402


def read_ate(key, model, n_mc):
    """ATE via the stage-2 readout (intervene on model.frugal_flow) + learned param."""
    m = intervene(key, model.frugal_flow, 1, n_mc)
    margin = find_margin(model.frugal_flow.bijection)
    return m["ate"], (_scalar(margin.ate) if margin is not None else float("nan"))


def run(args):
    ns = [int(x) for x in args.ns.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    cp = [args.const, args.ate]
    fam = FAMILIES["gaussian"]
    true_ate = fam.true_ate(cp)

    flow_hp = base_hyperparams(args.epochs)
    marg_hp = {"max_epochs": args.epochs, "max_patience": max(20, args.epochs // 10)}
    prop_hp = dict(flow_hp)  # same MAF hyperparams for the propensity flow

    init0 = {"ate": jnp.zeros(1), "scale": 1.0, "const": 0.0}

    print(f"families=gaussian  ns={ns}  seeds={seeds}  epochs={args.epochs}  true_ate={true_ate:+.3f}\n")
    rows = []
    for n in ns:
        for seed in seeds:
            data = fam.generate(n, causal_params=cp, seed=seed)
            X, Y, Z_disc, Z_cont = data["X"], data["Y"], data["Z_disc"], data["Z_cont"]
            model = FrugalFlowModel(Y=Y, X=X, Z_disc=Z_disc, Z_cont=Z_cont)
            tseeds = jr.split(jr.key(1000 * seed + 1), 20)

            # stage 1 + 2 (NO propensity yet)
            model.train_marginal_cdfs(tseeds[0], marg_hp)
            model.train_frugal_flow(tseeds[1], flow_hp, "gaussian", init0)
            has_prop_before = hasattr(model, "prop_flow")
            ate_no_crn, ate_no_param = read_ate(tseeds[5], model, args.n_mc)

            # stage 3 (propensity) — does NOT touch model.frugal_flow
            model.train_propensity_flow(tseeds[2], prop_hp)
            has_prop_after = hasattr(model, "prop_flow")
            ate_yes_crn, ate_yes_param = read_ate(tseeds[5], model, args.n_mc)

            rows.append(dict(n=n, seed=seed,
                             ate_no_crn=ate_no_crn, ate_yes_crn=ate_yes_crn,
                             ate_no_param=ate_no_param, ate_yes_param=ate_yes_param,
                             prop_before=has_prop_before, prop_after=has_prop_after,
                             delta=abs(ate_yes_crn - ate_no_crn)))
            print(f"n={n} seed={seed}: ATE(no prop)={ate_no_crn:+.4f}  ATE(with prop)={ate_yes_crn:+.4f}  "
                  f"|delta|={abs(ate_yes_crn - ate_no_crn):.2e}  prop_flow trained: {has_prop_before}->{has_prop_after}")

    print("\n" + "=" * 78)
    print(f"{'n':>6}{'seed':>5}{'ATE no-prop':>13}{'ATE with-prop':>14}{'|delta|':>11}{'prop fitted':>13}")
    print("-" * 78)
    for r in rows:
        print(f"{r['n']:>6}{r['seed']:>5}{r['ate_no_crn']:>13.4f}{r['ate_yes_crn']:>14.4f}"
              f"{r['delta']:>11.2e}{'no->yes':>13}")
    max_delta = max(r["delta"] for r in rows)
    print("-" * 78)
    print(f"max |ATE(with prop) - ATE(no prop)| across cells = {max_delta:.2e}")
    verdict = ("IDENTICAL — the fitted propensity does NOT change the ATE or its small-n bias."
               if max_delta < 1e-6 else
               f"NON-ZERO delta {max_delta:.2e} — investigate (unexpected).")
    print(verdict)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ns", default="200,5000")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--n-mc", type=int, default=10000)
    p.add_argument("--const", type=float, default=0.0)
    p.add_argument("--ate", type=float, default=1.0)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
