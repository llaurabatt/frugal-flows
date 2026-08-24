"""Interventional read-out of a fitted frugal flow's causal margin.

The frugal flow stores the causal margin in the **leading output dimensions**
(dim 0 for a scalar outcome, dims ``0..K-1`` for a K-dimensional outcome), so
sampling the fitted flow at a fixed treatment ``T = t`` and reading those dims
gives draws from the interventional outcome ``Y | do(T = t)``. Differencing the
do(1) and do(0) draws under COMMON RANDOM NUMBERS (the same base ``key``) yields
the paired quantile effect ``tau(u) = Q_1(u) - Q_0(u)`` per outcome dimension;
its mean is the ATE and its spread is genuine effect heterogeneity across
quantiles (~0 for a pure location shift, > 0 for a real treatment-conditioned
spline effect).

This read-out is **model-agnostic**: it works for every ``causal_model`` arm
(``gaussian``, ``flexible_continuous``, ...), unlike reading a parametric ``.ate``
field that only the additive/``gaussian`` arm exposes.

If the flow was fitted on a TRANSFORMED outcome (see
``frugal_flows.outcome_transforms``), pass that transform so the samples are
inverted back to the ORIGINAL ``Y`` scale BEFORE any contrast is taken -- a
nonlinear transform is not estimand-preserving, so the inverse must act on the
samples, not on the difference of means.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from frugal_flows.outcome_transforms import as_outcome_transform

Y_INDEX = 0  # the frugal flow stores the causal margin (Y) in output dim 0


def interventional_samples(
    key, flow, cond_dim, n_mc, outcome_transform=None, y_index=Y_INDEX, dim_y=1
):
    """Paired common-random-number draws of ``Y | do(T=0)`` and ``Y | do(T=1)``.

    Args:
        key: a **typed** JAX PRNG key (``jax.random.key(...)``, not the legacy
            ``PRNGKey``) -- flowjax's ``.sample`` requires the new-style key.
        flow: a fitted frugal flow (a flowjax distribution) with the causal margin
            in output dims ``y_index .. y_index + dim_y - 1``.
        cond_dim: treatment / condition dimensionality; ``T`` is set to all-zeros
            for do(0) and all-ones for do(1).
        n_mc: number of Monte-Carlo base draws, shared across do(0)/do(1) so the
            effect is paired.
        outcome_transform: ``None`` / kind-string / ``OutcomeTransform`` used at fit
            time; its inverse maps the sampled margin back to the original ``Y``
            scale before differencing. ``None`` -> identity (no-op).
        y_index: first output dim holding the causal margin (default 0).
        dim_y: outcome dimensionality K (default 1). With ``dim_y == 1`` the
            samples are 1-D and every statistic a float; with ``dim_y > 1`` the
            samples are ``(n_mc, K)`` and every statistic a length-K vector, one
            entry per outcome dimension (still paired draws from the joint margin).

    Returns:
        dict with ``y0``/``y1`` sample arrays, their ``mean``/``var``,
        ``ate = mean(y1 - y0)``, ``tau_sd = std(y1 - y0)``, ``frac_neg`` (fraction
        of pooled draws <= 0), and ``anynan`` (True when ANY draw is non-finite —
        NaN or +/-inf; a saturated margin produces inf draws whose statistics then
        surface as NaN).
    """
    t = as_outcome_transform(outcome_transform)
    cols = slice(y_index, y_index + dim_y)
    y0 = np.asarray(t.inverse(flow.sample(key, condition=jnp.zeros((n_mc, cond_dim)))[:, cols]))
    y1 = np.asarray(t.inverse(flow.sample(key, condition=jnp.ones((n_mc, cond_dim)))[:, cols]))
    if dim_y == 1:
        y0, y1 = y0[:, 0], y1[:, 0]
    tau = y1 - y0
    stat = float if dim_y == 1 else (lambda a: np.asarray(a))
    return {
        "y0": y0, "y1": y1,
        "mean0": stat(np.mean(y0, axis=0)), "mean1": stat(np.mean(y1, axis=0)),
        "var0": stat(np.var(y0, axis=0)), "var1": stat(np.var(y1, axis=0)),
        "ate": stat(np.mean(tau, axis=0)), "tau_sd": stat(np.std(tau, axis=0)),
        "frac_neg": stat(np.mean(np.concatenate([y0, y1]) <= 0, axis=0)),
        "anynan": bool(not (np.all(np.isfinite(y0)) and np.all(np.isfinite(y1)))),
    }


TAU_CURVE_BINS = 40  # fixed => identical u-grid across seeds (stackable curves)


def tau_curve(y0, y1, n_bins=TAU_CURVE_BINS):
    """Quantile-resolved paired effect ``tau(u) = Q_1(u) - Q_0(u)`` on a fixed u-grid.

    Pairs are aligned by base draw (same ``key`` in ``interventional_samples``), so
    ``tau[i] = y1[i] - y0[i]`` is the effect at draw ``i``'s latent quantile. The
    causal margin is monotone, so ranking by the control outcome ``y0`` recovers that
    quantile; binning ``tau`` by ``y0``-rank then gives ``tau`` as a function of
    ``u in (0, 1)``. A FIXED ``n_bins`` yields an identical u-grid across seeds, so
    per-seed curves stack directly for a bias (seed-mean) vs variance (seed-SD)
    decomposition. For a pure location shift the truth is flat at the ATE; any
    slope/curvature the spline shows on such a DGP is spurious.

    Accepts ``(n,)`` samples (scalar outcome) or ``(n, K)`` samples (multivariate
    outcome); in the K-dim case each dimension is ranked by ITS OWN ``y0`` column
    (each margin is monotone in its own latent quantile) and the curves share one
    u-grid.

    Returns ``(u_centers[n_bins], tau_of_u[n_bins])``, with ``tau_of_u`` of shape
    ``(n_bins, K)`` for K-dim input.
    """
    y0 = np.asarray(y0)
    y1 = np.asarray(y1)
    if y0.ndim == 2:
        curves = [tau_curve(y0[:, k], y1[:, k], n_bins) for k in range(y0.shape[1])]
        return curves[0][0], np.column_stack([c[1] for c in curves])
    tau = (y1 - y0)[np.argsort(np.asarray(y0), kind="stable")]
    n = len(tau)
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    tau_of_u = np.array([tau[a:b].mean() if b > a else np.nan
                         for a, b in zip(edges[:-1], edges[1:])])
    u_centers = (np.arange(n_bins) + 0.5) / n_bins
    return u_centers, tau_of_u
