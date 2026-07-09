"""Run all three identification diagnostics for one generator + causal config.

Convenience orchestrator: fits/draws once per diagnostic and writes every CSV and
PDF into ``validation/diagnostics/outputs/``. Each diagnostic is also runnable on
its own (``python -m diagnostics.d1_ate_recovery`` etc.) — this just chains them
with a shared config so you get the full picture in one command.

Run (from validation/):
    micromamba run -n frugal-flows-flowjax python -m diagnostics.run_diagnostics --smoke
    micromamba run -n frugal-flows-flowjax python -m diagnostics.run_diagnostics \
        --generator mixed --const 1 --ate 1 --n-samples 20000 --n-iter 25
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from . import d1_ate_recovery as d1
from . import d2_margin_shape as d2
from . import d3_moment_match as d3
from ._harness import DEFAULT_HYPERPARAMS, SMOKE_HYPERPARAMS, fit_once, outputs_dir, true_params_from_causal_params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--generator", default="mixed", choices=["gaussian", "mixed", "discrete", "many_discrete"]
    )
    parser.add_argument("--const", type=float, default=1.0, help="true const (causal_params[0])")
    parser.add_argument("--ate", type=float, default=1.0, help="true ate (causal_params[1])")
    parser.add_argument("--n-samples", type=int, default=20000)
    parser.add_argument("--n-iter", type=int, default=25, help="seeds for d1 recovery")
    parser.add_argument("--ref-n", type=int, default=100000, help="d3 causl reference size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="fast feedback config across all three")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    causal_params = [args.const, args.ate]
    true_params = true_params_from_causal_params(causal_params)
    outdir = args.outdir or outputs_dir()
    os.makedirs(outdir, exist_ok=True)

    if args.smoke:
        hyperparams = dict(SMOKE_HYPERPARAMS)
        n_samples = min(args.n_samples, 2000) if args.n_samples == 20000 else args.n_samples
        n_iter = min(args.n_iter, 3) if args.n_iter == 25 else args.n_iter
        ref_n = min(args.ref_n, 20000) if args.ref_n == 100000 else args.ref_n
    else:
        hyperparams = dict(DEFAULT_HYPERPARAMS)
        n_samples, n_iter, ref_n = args.n_samples, args.n_iter, args.ref_n

    print(f"\n{'=' * 70}\n[run] generator={args.generator}  truth={true_params}\n"
          f"      n_samples={n_samples}  n_iter={n_iter}  ref_n={ref_n}\n{'=' * 70}")

    # --- d1: ATE recovery (n_iter fits) ---
    print("\n>>> d1 ATE recovery")
    results = d1.run_recovery(args.generator, causal_params, n_samples, n_iter, args.seed, hyperparams)
    results.to_csv(os.path.join(outdir, f"d1_{args.generator}_recovery.csv"), index=False)
    print(d1.summarise(results, true_params).to_string(float_format=lambda v: f"{v:+.4f}"))
    d1.plot_recovery(results, true_params, os.path.join(outdir, f"d1_{args.generator}_recovery.pdf"))

    # --- d2: margin shape (one fit; reuse seed 0) ---
    print("\n>>> d2 margin shape")
    res = fit_once(args.generator, causal_params, n_samples, seed=args.seed, hyperparams=hyperparams)
    print(f"    recovered={res.recovered}")
    for k, v in d2.sense_checks(res.recovered, np.linspace(1e-3, 1 - 1e-3, 400)).items():
        print(f"    {k}: {v}")
    d2.plot_margin(res.recovered, res.true_params, os.path.join(outdir, f"d2_{args.generator}_margin.pdf"))

    # --- d3: moment match (no fit) ---
    print("\n>>> d3 moment match")
    table, reference, sample = d3.run_moment_match(args.generator, causal_params, n_samples, ref_n, args.seed)
    table.to_csv(os.path.join(outdir, f"d3_{args.generator}_moments.csv"))
    print(table.to_string(float_format=lambda v: f"{v:+.4f}"))
    d3.plot_moments(reference, sample, os.path.join(outdir, f"d3_{args.generator}_moments.pdf"), ref_n, n_samples)

    print(f"\n[run] all diagnostics written to {outdir}")


if __name__ == "__main__":
    main()
