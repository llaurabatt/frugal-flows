"""Diagnostic 1 — does the frugal flow identify the causal-margin parameters?

Reproduces the core identification finding of ``Continous_Frugal_Flows.ipynb``:
generate data with a KNOWN causal margin ``(ate, const, scale)``, fit the flow
over several seeds, pull the learned margin params, and check they recover the
truth.

Output (written to ``validation/diagnostics/outputs/``):
  * ``d1_<generator>_recovery.csv`` — per-seed recovered params + the truth.
  * a printed summary table: recovered mean/std and bias vs truth.
  * ``d1_<generator>_recovery.pdf`` — the notebook's boxplot of recovered
    ate/const/scale with dashed lines at the true values.

This is a *sense-check*, not a hard pass/fail test: a good fit shows the boxes
centred on the dashed truth lines with the truth inside the spread.

Run:
    micromamba run -n frugal-flows-flowjax python -m diagnostics.d1_ate_recovery --smoke
(from the ``validation/`` directory), or with explicit knobs:
    ... d1_ate_recovery --generator mixed --const 1 --ate 1 --n-samples 20000 --n-iter 25
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # headless: write figures, never try to open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._harness import (
    DEFAULT_HYPERPARAMS,
    PARAM_NAMES,
    SMOKE_HYPERPARAMS,
    fit_once,
    outputs_dir,
    true_params_from_causal_params,
)


def run_recovery(
    generator_name: str,
    causal_params,
    n_samples: int,
    n_iter: int,
    seed: int,
    hyperparams: dict,
) -> pd.DataFrame:
    """Fit ``n_iter`` times (seeds ``seed..seed+n_iter-1``); return recovered params.

    Columns: ``ate, const, scale, seed, val_loss``. One row per fit.
    """
    rows = []
    for i in range(n_iter):
        s = seed + i
        print(f"[d1] fit {i + 1}/{n_iter}  (generator={generator_name}, seed={s})")
        res = fit_once(
            generator_name=generator_name,
            causal_params=causal_params,
            n_samples=n_samples,
            seed=s,
            hyperparams=hyperparams,
        )
        rows.append({**res.recovered, "seed": s, "val_loss": res.val_loss})
        r = res.recovered
        print(
            f"      recovered: ate={r['ate']:+.4f}  const={r['const']:+.4f}  "
            f"scale={r['scale']:.4f}  (val_loss={res.val_loss:.4f})"
        )
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame, true_params: dict) -> pd.DataFrame:
    """Build a recovered-vs-truth summary table (mean, std, bias)."""
    summary = pd.DataFrame(
        {
            "true": [true_params[p] for p in PARAM_NAMES],
            "recovered_mean": [results[p].mean() for p in PARAM_NAMES],
            "recovered_std": [results[p].std(ddof=1) for p in PARAM_NAMES],
        },
        index=list(PARAM_NAMES),
    )
    summary["bias"] = summary["recovered_mean"] - summary["true"]
    return summary


def plot_recovery(results: pd.DataFrame, true_params: dict, save_path: str) -> None:
    """Boxplot of recovered ate/const/scale with dashed lines at the truth.

    Mirrors the notebook's ``plot_simulation_results`` (kept script-local rather
    than importing it, so this folder stays self-contained and we are not coupled
    to that module's globals).
    """
    plt.figure(figsize=(10, 6))
    results.boxplot(column=list(PARAM_NAMES), grid=False)
    colors = {"ate": "r", "const": "g", "scale": "b"}
    for p in PARAM_NAMES:
        plt.axhline(y=true_params[p], color=colors[p], linestyle="--", label=f"true {p}")
    plt.title("Causal-margin parameter recovery")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--generator",
        default="mixed",
        choices=["gaussian", "mixed", "discrete", "many_discrete"],
        help="which causl ground-truth generator to use (default: mixed)",
    )
    parser.add_argument("--const", type=float, default=1.0, help="true const (causal_params[0])")
    parser.add_argument("--ate", type=float, default=1.0, help="true ate (causal_params[1])")
    parser.add_argument("--n-samples", type=int, default=20000, help="samples per fit")
    parser.add_argument("--n-iter", type=int, default=25, help="number of seeds/fits")
    parser.add_argument("--seed", type=int, default=0, help="base seed")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="fast feedback config: small n-samples/n-iter and short training",
    )
    parser.add_argument("--max-epochs", type=int, default=None, help="override max_epochs")
    parser.add_argument("--max-patience", type=int, default=None, help="override max_patience")
    parser.add_argument("--outdir", default=None, help="output directory (default: diagnostics/outputs)")
    args = parser.parse_args()

    causal_params = [args.const, args.ate]
    true_params = true_params_from_causal_params(causal_params)

    if args.smoke:
        hyperparams = dict(SMOKE_HYPERPARAMS)
        n_samples = min(args.n_samples, 2000) if args.n_samples == 20000 else args.n_samples
        n_iter = min(args.n_iter, 3) if args.n_iter == 25 else args.n_iter
    else:
        hyperparams = dict(DEFAULT_HYPERPARAMS)
        n_samples, n_iter = args.n_samples, args.n_iter
    if args.max_epochs is not None:
        hyperparams["max_epochs"] = args.max_epochs
    if args.max_patience is not None:
        hyperparams["max_patience"] = args.max_patience

    outdir = args.outdir or outputs_dir()
    os.makedirs(outdir, exist_ok=True)

    print(
        f"[d1] generator={args.generator}  causal_params=[const={args.const}, ate={args.ate}]  "
        f"-> truth {true_params}\n"
        f"     n_samples={n_samples}  n_iter={n_iter}  "
        f"max_epochs={hyperparams['max_epochs']}  max_patience={hyperparams['max_patience']}"
    )

    results = run_recovery(args.generator, causal_params, n_samples, n_iter, args.seed, hyperparams)

    csv_path = os.path.join(outdir, f"d1_{args.generator}_recovery.csv")
    results.to_csv(csv_path, index=False)

    summary = summarise(results, true_params)
    print("\n[d1] recovered vs truth:\n")
    print(summary.to_string(float_format=lambda v: f"{v:+.4f}"))

    fig_path = os.path.join(outdir, f"d1_{args.generator}_recovery.pdf")
    plot_recovery(results, true_params, fig_path)

    print(f"\n[d1] wrote:\n  {csv_path}\n  {fig_path}")


if __name__ == "__main__":
    main()
