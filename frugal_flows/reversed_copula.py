"""Reversed-copula frugal flow (2026-09-30; PROPOSED, being tested).

Model:  p(y | u, t) = p*(y | t) * c(r | u),   r = F*(y | t),
the Evans-Didelez conditional density with the copula written as a density over the image ranks
r given the covariate ranks u (the current flexible arm has it the other way round: u given r).
Reasoning, invariances and open questions: validation/morphomnist/docs/leftover_confounding/README.md.

Chain, base -> data (all on [-1, 1] until the last block):
  [0] copula:  SelectCondition(MAF over the K image ranks, reads u only)      w -> r
  [1] margin:  SelectCondition(the flexible arm's causal margin, reads t only) r -> y on [-1, 1]
  [2] Invert(Tanh) per pixel                                                   -> y on R
The flow is fitted to y with condition = [t, u]; u is conditioned on, never modelled.

What does not depend on how the work is split between margin and copula (see README):
  * the ATE read out by simulating the intervention: u from the data, w ~ U, y(t) = flow(w; t, u);
  * counterfactuals y' = F*^-1(F*(y | t) | t') (margin transport; the copula is blind to t).
What does: p* alone being the causal margin (sampling r ~ U straight into the margin). That holds
when the ranks of the observed images, pooled over both arms, are uniform; ``RankUniformityLoss``
adds a soft penalty for it (energy distance between each batch's ranks and uniform draws). The
penalty's gradient reaches only the margin (the ranks are computed through blocks [2] and [1]).
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import paramax
from flowjax.bijections import Chain, Invert, Stack, Tanh
from flowjax.distributions import Transformed, Uniform
from flowjax.train.losses import MaximumLikelihoodLoss

from frugal_flows.basic_flows import masked_autoregressive_bijection
from frugal_flows.bijections import SelectCondition
from frugal_flows.training import fit_to_data

# block positions in the chain (base -> data)
COPULA_BLOCK, MARGIN_BLOCK, TANH_BLOCK = 0, 1, 2
BASE_CLAMP = 2.0**-25   # keep base draws off +-1 (as frugal_flows.interventions)


def _energy_distance(a, b):
    """2 E|a-b| - E|a-a'| - E|b-b'| (within-sample terms over distinct pairs)."""
    def mean_dist(x, y, exclude_diag):
        d = jnp.sqrt(jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1) + 1e-12)
        if exclude_diag:
            n = x.shape[0]
            return (d.sum() - jnp.trace(d)) / (n * (n - 1))
        return d.mean()
    return 2 * mean_dist(a, b, False) - mean_dist(a, a, True) - mean_dist(b, b, True)


def build_reversed_flow(key, dim_y: int, t_dim: int, u_dim: int, causal_model_args: dict,
                        copula_nn_width: int = 16, copula_nn_depth: int = 1,
                        copula_flow_layers: int = 4, copula_rqs_knots: int = 8):
    """The unfitted reversed-copula flow over y (dim_y), conditioned on [t, u]."""
    from frugal_flows.causal_flows import (
        _build_flexible_margin,  # the current arm's margin builder
    )

    k_m, k_c = jr.split(key)
    cond_dim = t_dim + u_dim
    margin = _build_flexible_margin(k_m, dim_y, jnp.zeros((1, t_dim)), causal_model_args)
    copula = masked_autoregressive_bijection(
        key=k_c, dim=dim_y, condition=jnp.zeros((1, u_dim)), nn_depth=copula_nn_depth,
        nn_width=copula_nn_width, RQS_knots=copula_rqs_knots, flow_layers=copula_flow_layers)
    chain = Chain([
        SelectCondition(copula, range(t_dim, cond_dim), cond_dim),
        SelectCondition(margin, range(t_dim), cond_dim),
        Stack([Invert(Tanh(()))] * dim_y),
    ])
    return Transformed(Uniform(-jnp.ones(dim_y), jnp.ones(dim_y)), chain)


def image_ranks(dist, y, condition):
    """Ranks r = F*(y | t) of observed images (fast direction: blocks [2], [1] inverted)."""
    blocks = dist.bijection.bijections
    r = jax.vmap(blocks[TANH_BLOCK].inverse)(y)
    return jax.vmap(blocks[MARGIN_BLOCK].inverse)(r, condition)


class RankUniformityLoss(eqx.Module):
    """Negative log likelihood of y given [t, u], plus ``weight`` x energy distance between the
    batch's image ranks (pooled over both arms) and ``n_ref`` uniform draws on [-1, 1]^K."""
    weight: float = eqx.field(static=True)
    n_ref: int = eqx.field(static=True, default=500)

    @eqx.filter_jit
    def __call__(self, params, static, x, condition=None, key=None):
        dist = paramax.unwrap(eqx.combine(params, static))
        nll = -dist.log_prob(x, condition).mean()
        if not self.weight:
            return nll
        r = image_ranks(dist, x, condition)
        ref = jr.uniform(key, (self.n_ref, x.shape[1]), minval=-1.0, maxval=1.0)
        return nll + self.weight * _energy_distance(r, ref)


def train_frugal_flow_reversed(key, y, u_z, condition, causal_model_args: dict,
                               nn_width: int = 16, nn_depth: int = 1, flow_layers: int = 4,
                               RQS_knots: int = 8, learning_rate: float = 1e-3,
                               max_epochs: int = 1000, max_patience: int = 30, batch_size: int = 100,
                               rank_penalty_weight: float = 0.0, fit_kwargs: dict | None = None,
                               show_progress: bool = False, optimizer=None):
    """Fit the reversed-copula flow. ``nn_width`` etc. size the copula (as in the dispatcher);
    ``causal_model_args`` sizes the margin. Validation / early stopping use the plain likelihood."""
    y = jnp.asarray(y)
    u_z = jnp.asarray(u_z, dtype=y.dtype)
    condition = jnp.asarray(condition, dtype=y.dtype)
    key, sub = jr.split(key)
    flow = build_reversed_flow(sub, y.shape[1], condition.shape[1], u_z.shape[1], causal_model_args,
                               copula_nn_width=nn_width, copula_nn_depth=nn_depth,
                               copula_flow_layers=flow_layers, copula_rqs_knots=RQS_knots)
    kw = dict(fit_kwargs or {})
    kw["loss_fn"] = RankUniformityLoss(weight=float(rank_penalty_weight))
    kw["val_loss_fn"] = MaximumLikelihoodLoss()
    key, sub = jr.split(key)
    flow, losses = fit_to_data(
        key=sub, dist=flow, data=(y, jnp.hstack([condition, u_z])), optimizer=optimizer,
        learning_rate=learning_rate, max_epochs=max_epochs, max_patience=max_patience,
        batch_size=batch_size, show_progress=show_progress, **kw)
    return flow, losses


def interventional_samples_reversed(key, flow, u_pool, n_mc: int, t_dim: int = 1):
    """Paired draws under do(t=0), do(t=1), two ways:
      * "gformula": u resampled from ``u_pool`` (the observed covariate ranks), w ~ U, y = flow(w; t, u)
        -- the exact interventional distribution of the fitted model;
      * "margin": r ~ U straight into the margin -- equals the above iff the copula's r-marginal is
        uniform (what the rank penalty targets).
    Same base draws for t = 0 and t = 1 within each way. Returns dict of (n_mc, K) arrays."""
    dist = paramax.unwrap(flow)
    blocks = dist.bijection.bijections
    K = dist.shape[0]
    u_pool = jnp.asarray(u_pool)
    k_u, k_w, k_r = jr.split(key, 3)
    u = u_pool[jr.choice(k_u, u_pool.shape[0], (n_mc,), replace=True)]
    w = jr.uniform(k_w, (n_mc, K), minval=-1 + BASE_CLAMP, maxval=1 - BASE_CLAMP)
    r = jr.uniform(k_r, (n_mc, K), minval=-1 + BASE_CLAMP, maxval=1 - BASE_CLAMP)
    out = {}
    for t in (0, 1):
        cond = jnp.hstack([jnp.full((n_mc, t_dim), float(t)), u])
        out[f"gformula_y{t}"] = np.asarray(jax.vmap(dist.bijection.transform)(w, cond))
        ym = jax.vmap(blocks[MARGIN_BLOCK].transform)(r, cond)
        out[f"margin_y{t}"] = np.asarray(jax.vmap(blocks[TANH_BLOCK].transform)(ym))
    return out


def counterfactual_reversed(flow, y, t, t_new, u_dim: int):
    """y' = F*^-1(F*(y | t) | t_new): margin transport of observed images (rows of y)."""
    dist = paramax.unwrap(flow)
    blocks = dist.bijection.bijections
    n = y.shape[0]
    pad = jnp.zeros((n, u_dim))
    c_old = jnp.hstack([jnp.asarray(t, dtype=float).reshape(n, -1), pad])
    c_new = jnp.hstack([jnp.asarray(t_new, dtype=float).reshape(n, -1), pad])
    r = image_ranks(dist, jnp.asarray(y), c_old)
    y_new = jax.vmap(blocks[MARGIN_BLOCK].transform)(r, c_new)
    return np.asarray(jax.vmap(blocks[TANH_BLOCK].transform)(y_new))
