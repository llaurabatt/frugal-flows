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
(``gaussian``, ``flexible_continuous``, ``flexible_continuous_gaussian``, ...), unlike
reading a parametric ``.ate`` field that only the additive ``gaussian`` arm exposes. (The
``gaussian`` arm is the paper's parametric scalar margin; it is unrelated to the Gaussian-scale
multivariate flow in ``frugal_flows.gaussian_scale``.) For the Gaussian-scale flow, unit-level
counterfactuals are ``gaussian_scale.counterfactual_gaussian``.

If the flow was fitted on a TRANSFORMED outcome (see
``frugal_flows.outcome_transforms``), pass that transform so the samples are
inverted back to the ORIGINAL ``Y`` scale BEFORE any contrast is taken -- a
nonlinear transform is not estimand-preserving, so the inverse must act on the
samples, not on the difference of means.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from paramax import unwrap

from frugal_flows.outcome_transforms import as_outcome_transform

Y_INDEX = 0  # the frugal flow stores the causal margin (Y) in output dim 0

# How far inside the base support a base draw is pushed. ``jax.random.uniform`` draws
# on [0, 1) and can return exactly 0, which the Uniform(-1, 1) base maps to exactly -1;
# the chain then sends that boundary point through arctanh and returns -inf. 2**-25 of
# the base range (2**-24 on [-1, 1]) is the smallest floor whose image stays strictly
# inside the support in float32; the sampler's grid is 2**-23, so only exact endpoints
# move.
BASE_CLAMP = 2.0**-25


def sample_clamped(key, flow, n, condition=None, clamp=BASE_CLAMP):
    """``flow.sample`` with the base draws clamped inward, plus how many were clamped.

    Reproduces ``flow.sample(key, (n,), condition)`` exactly (same key splitting, same
    base draws) except that base coordinates on the support boundary are moved inward
    by ``clamp`` times the support width. Returns ``(samples, n_clamped)`` where
    ``n_clamped`` counts the base coordinates that were moved (0 for almost every call).

    Only a flow whose base distribution is a Uniform (flowjax's ``Uniform`` or, after
    ``merge_transforms``, its ``_StandardUniform`` on [0, 1)) has a boundary to clamp at.
    Any other flow, including one without a ``base_dist`` (a test double, a Normal-based
    flow), is sampled with its own ``sample`` and ``n_clamped`` is 0.
    """
    flow = unwrap(flow)
    base = getattr(flow, "base_dist", None)
    base = unwrap(base) if base is not None else None
    if base is None or type(base).__name__ not in ("Uniform", "_StandardUniform"):
        y = flow.sample(key, condition=condition) if condition is not None else flow.sample(key, (n,))
        return y, 0
    # after merge_transforms the base is flowjax's _StandardUniform on [0, 1) (its affine
    # to [-1, 1] has become the first block of the chain); a plain Uniform keeps its bounds
    lo, hi = jnp.asarray(getattr(base, "minval", 0.0)), jnp.asarray(getattr(base, "maxval", 1.0))
    eps = clamp * (hi - lo)
    if condition is None:
        # unconditional: n draws come from sample_shape=(n,)
        keys = flow._get_sample_keys(key, (n,), None)
    else:
        # conditional: flowjax makes one draw per condition row, so the n rows of the
        # condition ARE the sample dimension and sample_shape must stay ()
        condition = jnp.asarray(condition)
        assert condition.shape[0] == n, (condition.shape, n)
        keys = flow._get_sample_keys(key, (), condition)
    u = jax.vmap(base._sample)(keys)
    # upper bound: 2 eps, because in float32 (1 - 2**-25) rounds back up to exactly 1
    u_c = jnp.clip(u, lo + eps, hi - 2 * eps)
    n_clamped = int(jnp.sum(u_c != u))
    if condition is None:
        y = jax.vmap(flow.bijection.transform)(u_c)
    else:
        y = jax.vmap(flow.bijection.transform)(u_c, condition)
    return y, n_clamped


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
    s0, c0 = sample_clamped(key, flow, n_mc, jnp.zeros((n_mc, cond_dim)))
    s1, c1 = sample_clamped(key, flow, n_mc, jnp.ones((n_mc, cond_dim)))
    y0 = np.asarray(t.inverse(s0[:, cols]))
    y1 = np.asarray(t.inverse(s1[:, cols]))
    if dim_y == 1:
        y0, y1 = y0[:, 0], y1[:, 0]
    tau = y1 - y0
    stat = float if dim_y == 1 else (lambda a: np.asarray(a))
    return {
        "y0": y0, "y1": y1,
        "n_clamped": c0 + c1,   # base coordinates moved off the support boundary
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


def counterfactual_flexible(flow, y, u_z, t, t_new):
    """Counterfactual images for observed units under the flexible-continuous (uniform-base) arm.

    Rank-preserving margin transport: each image ``y`` (rows) is mapped to its ranks under the causal
    margin at its observed treatment ``t``, and back to an image at ``t_new``:
    ``y' = F*^{-1}(F*(y | t) | t_new)``. The copula is blind to the treatment, so the ranks carry
    over unchanged; this is the abduction-action-prediction counterfactual under rank preservation.

    The flexible arm's merged chain is, base -> data: [0] affine, [1] copula, [2] u_z affine,
    [3] causal margin (Concatenate: margin on the image, identity on u_z), [4] Tanh stack. Only
    blocks [4] and [3] are used. ``u_z`` is passed only because those blocks act on the full
    (image, covariate-rank) vector; it does not change the result.

    Args: ``y`` (n, K) observed images on the model's scale; ``u_z`` (n, d) the run's covariate ranks;
    ``t``, ``t_new`` (n,) or (n, 1) treatments. Returns (n, K) counterfactual images.
    """
    blocks = unwrap(flow).bijection.bijections
    margin, tanh = blocks[3], blocks[4]
    y = jnp.asarray(y)
    k = y.shape[1]
    x = jnp.hstack([y, jnp.asarray(u_z, dtype=y.dtype)])
    c_old = jnp.asarray(t, dtype=y.dtype).reshape(len(y), -1)
    c_new = jnp.asarray(t_new, dtype=y.dtype).reshape(len(y), -1)
    r = jax.vmap(tanh.inverse)(x)
    r = jax.vmap(margin.inverse)(r, c_old)
    out = jax.vmap(margin.transform)(r, c_new)
    out = jax.vmap(tanh.transform)(out)
    return np.asarray(out[:, :k])
