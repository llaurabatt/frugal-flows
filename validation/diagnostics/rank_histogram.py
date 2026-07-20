"""Histogram of the ranks after fitting the frugal flow (calibration / Rosenblatt check).

Two sets of "ranks":
  INPUT u_z      -- the retained stage-1 marginal ranks (empirical CDF of Z, the
                    `use_marginal_flow=False` default). Uniform ~by construction.
  ROSENBLATT     -- push the fitted data (Y, u_z) BACKWARD through the trained flow
                    to its base, `ff.bijection.inverse(data, condition=X)`. The base
                    is independent Uniform(-1,1); rescaled to (0,1) these are the
                    "iid Rosenblatt ranks". If the margin+copula fit, every component
                    is ~Uniform. Deviations = miscalibration (margin or copula misfit).

Fits the SPLINE arm (`flexible_continuous`) on a gaussian-outcome causl dataset.

Usage (from validation/):
  micromamba run -n frugal-flows-flowjax python -m diagnostics.rank_histogram \
      --n 2000 --arm flexible_continuous
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_VALIDATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VALIDATION_DIR not in sys.path:
    sys.path.insert(0, _VALIDATION_DIR)

import data_processing_and_simulations.causl_sim_data_generation as causl_py  # noqa: E402
from frugal_flows.causal_flows import train_frugal_flow  # noqa: E402
from diagnostics.outcome_families import FAMILIES  # noqa: E402
from diagnostics.quick_sense_check import base_hyperparams, model_args  # noqa: E402


def ks_uniform(u):
    """KS distance of samples u in [0,1] from Uniform(0,1)."""
    u = np.sort(np.asarray(u))
    n = len(u)
    cdf = np.arange(1, n + 1) / n
    return float(np.max(np.abs(cdf - u)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", default="flexible_continuous")
    p.add_argument("--family", default="gaussian")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--const", type=float, default=0.0)
    p.add_argument("--ate", type=float, default=1.0)
    args = p.parse_args()

    fam = FAMILIES[args.family]
    data = fam.generate(args.n, causal_params=[args.const, args.ate], seed=args.seed)
    X, Y, Z_disc, Z_cont = data["X"], data["Y"], data["Z_disc"], data["Z_cont"]
    uz = causl_py.generate_uz_samples(Z_disc, Z_cont, False, args.seed, base_hyperparams(args.epochs))
    u_z = uz["uz_samples"]
    k_z = u_z.shape[1]
    print(f"fitting {args.arm} on {args.family}: X{tuple(X.shape)} Y{tuple(Y.shape)} u_z{tuple(u_z.shape)}")

    key = jr.key(1000 * args.seed + 1)
    ff, _ = train_frugal_flow(
        jr.fold_in(key, 1), Y, u_z, condition=X,
        causal_model=args.arm, causal_model_args=model_args(args.arm, X.shape[1]),
        **base_hyperparams(args.epochs),
    )

    # ---- Rosenblatt ranks: push fitted data (Y,u_z) backward to the base ----
    data_full = jnp.hstack([Y, u_z])  # dim 0 = outcome, dims 1..k = confounder ranks
    # base is _StandardUniform on (0,1); inverse maps data -> base directly (no rescale)
    ros = np.asarray(jax.vmap(ff.bijection.inverse, in_axes=(0, 0))(data_full, X))
    u_z_np = np.asarray(u_z)

    ros_y = ros[:, 0]                  # outcome Rosenblatt rank
    ros_z = ros[:, 1:].ravel()         # confounder Rosenblatt ranks (pooled)
    finite = np.isfinite(ros).all(axis=1)
    print(f"finite Rosenblatt rows: {finite.sum()}/{len(finite)}")
    print(f"KS-from-uniform: input u_z={ks_uniform(u_z_np.ravel()):.4f}  "
          f"outcome Rosenblatt={ks_uniform(ros_y[np.isfinite(ros_y)]):.4f}  "
          f"confounder Rosenblatt={ks_uniform(ros_z[np.isfinite(ros_z)]):.4f}")

    # ---- figure ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    bins = np.linspace(0, 1, 31)

    def _hist(ax, vals, title, color):
        vals = np.asarray(vals); vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=bins, density=True, color=color, alpha=0.8, edgecolor="white")
        ax.axhline(1.0, color="black", ls="--", lw=1.6, label="Uniform(0,1)")
        ax.set_ylim(0, 2.2); ax.set_xlabel("rank"); ax.set_ylabel("density")
        ax.set_title(f"{title}\nKS={ks_uniform(vals):.3f}", fontsize=12)
        ax.legend(fontsize=9)

    _hist(axes[0, 0], u_z_np.ravel(), "INPUT u_z ranks (retained stage-1 ECDF, all Z dims)", "#7f7f7f")
    _hist(axes[0, 1], ros_y, "OUTCOME Rosenblatt rank (dim 0, after fit)", "#1f77b4")
    _hist(axes[1, 0], ros_z, "CONFOUNDER Rosenblatt ranks (dims 1..k, after fit)", "#2ca02c")
    _hist(axes[1, 1], ros.ravel(), "ALL Rosenblatt ranks pooled (after fit)", "#9467bd")

    fig.suptitle(f"Ranks after fitting the {args.arm} frugal flow ({args.family} outcome, n={args.n})\n"
                 "flat at density 1 => calibrated (iid Rosenblatt ranks ~ Uniform)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs",
                       f"rank_hist_{args.arm}_{args.family}_n{args.n}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"figure: {out}")


if __name__ == "__main__":
    main()
