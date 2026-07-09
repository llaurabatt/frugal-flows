"""Diagnostic 3 — are the treatment/covariate summary stats what causl intended?

Goal 3 from the meeting: check the mean/variance of the treatment X and the
confounders Z against their causl ground truth.

Ground truth comes from causl itself, two ways:
  * **Monte-Carlo reference (primary):** a large ``causalSamp`` draw (default 100k)
    from the SAME generator. Its empirical moments are the population values for
    *every* variable — including the treatment X, which is binomial *conditional*
    on Z (``X~Z``) and so has no closed-form marginal mean. This is the universal,
    robust route and is what we compare against.
  * **Closed-form cross-check (where it applies):** for intercept-only covariates
    (``Z~1``) the causl conventions give exact moments — Gamma: mean ``exp(beta)``,
    var ``phi*exp(beta)^2``; binomial: ``p=expit(beta)``, var ``p(1-p)``; gaussian:
    mean ``beta``, var ``phi`` (see ``reference_causl-conventions`` in memory /
    the README). Generators with conditional covariates (e.g. ``gaussian``:
    ``Zc2~Zc1``) have no per-covariate closed form — the MC reference covers them.

What this measures: how faithfully a *fit-sized* dataset (``n_samples``) reproduces
the population treatment/covariate distribution. Large gaps at small ``n`` mean the
inputs the flow is trained on are noisy — directly relevant to the ATE-recovery /
data-efficiency question (noisy inputs -> noisier ATE).

NB: this v1 compares causl-truth vs the *generated data*. A later mode 2 will swap
the data side for the flow's *reconstruction* of X/Z (does the model distort the
covariates?) — that needs the FrugalFlowModel sampler and is added separately,
still without touching core code.

Output:
  * ``d3_<generator>_moments.csv`` — per-variable causl-ref vs sample mean/var + deltas.
  * printed table.
  * ``d3_<generator>_moments.pdf`` — per-variable overlaid histograms (ref vs sample).

Run (from validation/):
    micromamba run -n frugal-flows-flowjax python -m diagnostics.d3_moment_match --smoke
"""

from __future__ import annotations

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._harness import generate_dataset, outputs_dir


def labelled_columns(data: dict) -> dict[str, np.ndarray]:
    """Flatten a generator's data dict into ``{label: 1d array}`` for X and each Z.

    Y is excluded — this diagnostic is about the treatment and confounders only.
    """
    cols: dict[str, np.ndarray] = {}
    cols["X (treatment)"] = np.asarray(data["X"]).reshape(-1)
    if data.get("Z_cont") is not None:
        zc = np.asarray(data["Z_cont"])
        for j in range(zc.shape[1]):
            cols[f"Zc{j + 1}"] = zc[:, j]
    if data.get("Z_disc") is not None:
        zd = np.asarray(data["Z_disc"])
        for j in range(zd.shape[1]):
            cols[f"Zd{j + 1}"] = zd[:, j]
    return cols


def moment_table(reference: dict, sample: dict) -> pd.DataFrame:
    """Per-variable causl-reference vs sample mean/var, with absolute deltas."""
    rows = []
    for label in reference:
        r, s = reference[label], sample[label]
        r_mean, r_var = float(np.mean(r)), float(np.var(r, ddof=1))
        s_mean, s_var = float(np.mean(s)), float(np.var(s, ddof=1))
        rows.append(
            {
                "variable": label,
                "causl_mean": r_mean,
                "sample_mean": s_mean,
                "d_mean": s_mean - r_mean,
                "causl_var": r_var,
                "sample_var": s_var,
                "d_var": s_var - r_var,
            }
        )
    return pd.DataFrame(rows).set_index("variable")


def plot_moments(reference: dict, sample: dict, save_path: str, ref_n: int, samp_n: int) -> None:
    labels = list(reference)
    ncols = min(3, len(labels))
    nrows = math.ceil(len(labels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.4 * nrows), squeeze=False)
    for ax, label in zip(axes.ravel(), labels):
        r, s = reference[label], sample[label]
        lo, hi = float(min(r.min(), s.min())), float(max(r.max(), s.max()))
        # discrete-ish (few unique values) -> one bin centred on each value (edges at
        # value +/- 0.5) so e.g. binary {0,1} shows two separated bars; else 40 bins.
        uniq = np.unique(np.concatenate([np.unique(r), np.unique(s)]))
        if uniq.size <= 10:
            bins = np.concatenate([uniq - 0.5, [uniq[-1] + 0.5]])
        else:
            bins = np.linspace(lo, hi, 41)
        ax.hist(r, bins=bins, density=True, alpha=0.45, label=f"causl ref (N={ref_n})", color="tab:gray")
        ax.hist(s, bins=bins, density=True, alpha=0.55, label=f"sample (n={samp_n})", color="tab:blue")
        ax.axvline(np.mean(r), color="k", linestyle="--", linewidth=1)
        ax.axvline(np.mean(s), color="tab:blue", linestyle="--", linewidth=1)
        ax.set_title(label)
    # blank any unused panels
    for ax in axes.ravel()[len(labels):]:
        ax.set_visible(False)
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle("Treatment / covariate moments: causl reference vs fit-sized sample")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def run_moment_match(
    generator_name: str, causal_params, n_samples: int, ref_n: int, seed: int
) -> tuple[pd.DataFrame, dict, dict]:
    """Draw a large causl reference and a fit-sized sample; return table + raw columns."""
    print(f"[d3] drawing causl reference (N={ref_n}) ...")
    ref_data = generate_dataset(generator_name, causal_params, ref_n, seed=seed + 9999)
    print(f"[d3] drawing fit-sized sample (n={n_samples}) ...")
    samp_data = generate_dataset(generator_name, causal_params, n_samples, seed=seed)

    reference = labelled_columns(ref_data)
    sample = labelled_columns(samp_data)
    table = moment_table(reference, sample)
    return table, reference, sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--generator", default="mixed", choices=["gaussian", "mixed", "discrete", "many_discrete"]
    )
    parser.add_argument("--const", type=float, default=1.0, help="true const (causal_params[0])")
    parser.add_argument("--ate", type=float, default=1.0, help="true ate (causal_params[1])")
    parser.add_argument("--n-samples", type=int, default=20000, help="fit-sized sample")
    parser.add_argument("--ref-n", type=int, default=100000, help="causl Monte-Carlo reference size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="fast: small sample + reference")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    causal_params = [args.const, args.ate]
    if args.smoke:
        n_samples = min(args.n_samples, 2000) if args.n_samples == 20000 else args.n_samples
        ref_n = min(args.ref_n, 20000) if args.ref_n == 100000 else args.ref_n
    else:
        n_samples, ref_n = args.n_samples, args.ref_n

    outdir = args.outdir or outputs_dir()
    os.makedirs(outdir, exist_ok=True)

    print(f"[d3] generator={args.generator}  causal_params={causal_params}  n_samples={n_samples}  ref_n={ref_n}")
    table, reference, sample = run_moment_match(args.generator, causal_params, n_samples, ref_n, args.seed)

    csv_path = os.path.join(outdir, f"d3_{args.generator}_moments.csv")
    table.to_csv(csv_path)
    print("\n[d3] treatment/covariate moments (causl reference vs sample):\n")
    print(table.to_string(float_format=lambda v: f"{v:+.4f}"))

    fig_path = os.path.join(outdir, f"d3_{args.generator}_moments.pdf")
    plot_moments(reference, sample, fig_path, ref_n, n_samples)
    print(f"\n[d3] wrote:\n  {csv_path}\n  {fig_path}")


if __name__ == "__main__":
    main()
