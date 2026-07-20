"""Diagnostic 2 — what does the learned causal margin look like?

Goal 2 from the meeting: don't just read the scalar ATE, *look* at the margin.
For the current Gaussian/location-translation model the margin is a
``UnivariateNormalCDF`` whose inverse is the outcome quantile function under an
intervention ``do(X=x)``:

    Q_x(u) = Phi^{-1}(u) * scale + (ate * x + const)

So the implied interventional outcome is Gaussian with mean ``ate*x + const`` and
sd ``scale``. Plotting ``Q_0`` and ``Q_1`` (i.e. ``do(X=0)`` vs ``do(X=1)``) makes
the treatment effect visible as the *gap* between the two curves. For this
additive-shift model the gap is constant in ``u`` and equals ``ate``.

Why this diagnostic matters going forward: when the margin is later replaced by a
treatment-conditioned spline, the gap will no longer be constant — you will be
able to *see* heterogeneity that a single scalar ATE cannot express. This script
is the tool that will show that.

Sense-checks printed:
  * scale > 0 (the bijection's scale is unconstrained — see UnivariateNormalCDF
    warning; a fit can in principle drive it negative).
  * Q_x monotone increasing in u (well-posed quantile function).
  * measured gap Q_1 - Q_0 is (near-)constant and equals the recovered ate.

Output: ``d2_<generator>_margin.pdf`` — left: the two quantile curves; right: the
two implied outcome densities.

Run (from validation/):
    micromamba run -n frugal-flows-flowjax python -m diagnostics.d2_margin_shape --smoke
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from ._harness import DEFAULT_HYPERPARAMS, SMOKE_HYPERPARAMS, fit_once, outputs_dir


def margin_quantiles(recovered: dict, u: np.ndarray, x: float) -> np.ndarray:
    """Interventional outcome quantile function ``Q_x(u)`` for ``do(X=x)``.

    Closed form of ``UnivariateNormalCDF.inverse`` (see its docstring):
    ``ndtri(u) * scale + (ate * x + const)``.
    """
    loc = recovered["ate"] * x + recovered["const"]
    return norm.ppf(u) * recovered["scale"] + loc


def sense_checks(recovered: dict, u: np.ndarray) -> dict:
    """Return the diagnostic sense-checks as a dict (also printed by main)."""
    q0 = margin_quantiles(recovered, u, 0.0)
    q1 = margin_quantiles(recovered, u, 1.0)
    gap = q1 - q0  # constant == ate for the Gaussian margin
    return {
        "scale_positive": bool(recovered["scale"] > 0),
        "q0_monotone": bool(np.all(np.diff(q0) > 0)),
        "q1_monotone": bool(np.all(np.diff(q1) > 0)),
        "gap_mean": float(np.mean(gap)),
        "gap_max_dev": float(np.max(np.abs(gap - np.mean(gap)))),
        "gap_matches_ate": bool(np.allclose(gap, recovered["ate"], atol=1e-6)),
    }


def plot_margin(recovered: dict, true_params: dict, save_path: str) -> None:
    u = np.linspace(1e-3, 1 - 1e-3, 400)
    q0 = margin_quantiles(recovered, u, 0.0)
    q1 = margin_quantiles(recovered, u, 1.0)

    fig, (ax_q, ax_d) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: quantile functions, gap = ATE.
    ax_q.plot(u, q0, label="do(X=0):  $Q_0(u)$", color="tab:blue")
    ax_q.plot(u, q1, label="do(X=1):  $Q_1(u)$", color="tab:orange")
    mid = len(u) // 2
    ax_q.annotate(
        "",
        xy=(u[mid], q1[mid]),
        xytext=(u[mid], q0[mid]),
        arrowprops=dict(arrowstyle="<->", color="k"),
    )
    ax_q.text(
        u[mid] + 0.02,
        0.5 * (q0[mid] + q1[mid]),
        f"gap = ATE = {recovered['ate']:.3f}\n(true {true_params['ate']:.3f})",
        va="center",
    )
    ax_q.set_xlabel("quantile $u$")
    ax_q.set_ylabel("outcome $Y$")
    ax_q.set_title("Interventional outcome quantile functions")
    ax_q.legend()

    # Right: implied outcome densities (Gaussian: mean = ate*x + const, sd = scale).
    lo = min(q0.min(), q1.min())
    hi = max(q0.max(), q1.max())
    grid = np.linspace(lo, hi, 400)
    for x, color, lab in [(0.0, "tab:blue", "do(X=0)"), (1.0, "tab:orange", "do(X=1)")]:
        loc = recovered["ate"] * x + recovered["const"]
        ax_d.plot(grid, norm.pdf(grid, loc=loc, scale=recovered["scale"]), color=color, label=lab)
        ax_d.axvline(loc, color=color, linestyle=":", alpha=0.7)
    ax_d.set_xlabel("outcome $Y$")
    ax_d.set_ylabel("density")
    ax_d.set_title("Implied $p(Y \\mid do(X))$")
    ax_d.legend()

    fig.suptitle(
        f"Learned causal margin  "
        f"(ate={recovered['ate']:.3f}, const={recovered['const']:.3f}, scale={recovered['scale']:.3f})"
    )
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--generator", default="mixed", choices=["gaussian", "mixed", "discrete", "many_discrete"]
    )
    parser.add_argument("--const", type=float, default=1.0, help="true const (causal_params[0])")
    parser.add_argument("--ate", type=float, default=1.0, help="true ate (causal_params[1])")
    parser.add_argument("--n-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="fast feedback config")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-patience", type=int, default=None)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    causal_params = [args.const, args.ate]
    hyperparams = dict(SMOKE_HYPERPARAMS if args.smoke else DEFAULT_HYPERPARAMS)
    n_samples = (min(args.n_samples, 2000) if (args.smoke and args.n_samples == 20000) else args.n_samples)
    if args.max_epochs is not None:
        hyperparams["max_epochs"] = args.max_epochs
    if args.max_patience is not None:
        hyperparams["max_patience"] = args.max_patience

    outdir = args.outdir or outputs_dir()
    os.makedirs(outdir, exist_ok=True)

    print(f"[d2] fitting once: generator={args.generator}, causal_params={causal_params}, n={n_samples}")
    res = fit_once(
        generator_name=args.generator,
        causal_params=causal_params,
        n_samples=n_samples,
        seed=args.seed,
        hyperparams=hyperparams,
    )
    print(f"[d2] recovered: {res.recovered}   true: {res.true_params}")

    u = np.linspace(1e-3, 1 - 1e-3, 400)
    checks = sense_checks(res.recovered, u)
    print("\n[d2] sense-checks:")
    for k, v in checks.items():
        print(f"      {k}: {v}")

    fig_path = os.path.join(outdir, f"d2_{args.generator}_margin.pdf")
    plot_margin(res.recovered, res.true_params, fig_path)
    print(f"\n[d2] wrote:\n  {fig_path}")


if __name__ == "__main__":
    main()
