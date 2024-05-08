from collections.abc import Callable

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from flowjax.bijections import (
    AbstractBijection,
    Affine,
    Chain,
    Invert,
    Loc,
    Permute,
    Scan,
    SoftPlus,
)
from flowjax.distributions import AbstractDistribution, Transformed
from flowjax.flows import _add_default_permute
from flowjax.wrappers import BijectionReparam, NonTrainable
from jax import Array

from frugal_flows.bijections import MaskedAutoregressiveFirstUniform, MaskedIndependent


def masked_independent_flow(
    key: Array,
    *,
    base_dist: AbstractDistribution,
    transformer: AbstractBijection | None = None,
    cond_dim: int | None = None,
    flow_layers: int = 8,
    nn_width: int = 50,
    nn_depth: int = 1,
    nn_activation: Callable = jnn.relu,
    invert: bool = True,
) -> Transformed:
    """Masked autoregressive flow.

    Parameterises a transformer bijection with an autoregressive neural network.
    Refs: https://arxiv.org/abs/1606.04934; https://arxiv.org/abs/1705.07057v4.

    Args:
        key: Random seed.
        base_dist: Base distribution, with ``base_dist.ndim==1``.
        transformer: Bijection parameterised by autoregressive network. Defaults to
            affine.
        cond_dim: Dimension of the conditioning variable. Defaults to None.
        flow_layers: Number of flow layers. Defaults to 8.
        nn_width: Number of hidden layers in neural network. Defaults to 50.
        nn_depth: Depth of neural network. Defaults to 1.
        nn_activation: _description_. Defaults to jnn.relu.
        invert: Whether to invert the bijection. Broadly, True will prioritise a faster
            inverse, leading to faster `log_prob`, False will prioritise faster forward,
            leading to faster `sample`. Defaults to True.
    """
    if transformer is None:
        transformer = eqx.tree_at(
            lambda aff: aff.scale,
            Affine(),
            BijectionReparam(1, Chain([SoftPlus(), NonTrainable(Loc(1e-2))])),
        )
    dim = base_dist.shape[-1]

    def make_layer(key):  # masked autoregressive layer + permutation
        bij_key, perm_key = jr.split(key)
        bijection = MaskedIndependent(
            key=bij_key,
            transformer=transformer,
            dim=dim,
            cond_dim=cond_dim,
            nn_width=nn_width,
            nn_depth=nn_depth,
            nn_activation=nn_activation,
        )
        return _add_default_permute(bijection, dim, perm_key)

    keys = jr.split(key, flow_layers)
    layers = eqx.filter_vmap(make_layer)(keys)
    bijection = Invert(Scan(layers)) if invert else Scan(layers)
    return Transformed(base_dist, bijection)


def masked_autoregressive_flow_first_uniform(
    key: Array,
    *,
    base_dist: AbstractDistribution,
    transformer: AbstractBijection | None = None,
    # cond_dim: int | None = None,
    cond_dim_mask: int | None = None,
    cond_dim_nomask: int | None = None,
    cond_u_y_dim: int = 1,
    flow_layers: int = 8,
    nn_width: int = 50,
    nn_depth: int = 1,
    nn_activation: Callable = jnn.relu,
    invert: bool = True,
) -> Transformed:
    """Masked autoregressive flow.

    Parameterises a transformer bijection with an autoregressive neural network.
    Refs: https://arxiv.org/abs/1606.04934; https://arxiv.org/abs/1705.07057v4.

    Args:
        key: Random seed.
        base_dist: Base distribution, with ``base_dist.ndim==1``.
        transformer: Bijection parameterised by autoregressive network. Defaults to
            affine.
        cond_dim: Dimension of the conditioning variable. Defaults to None.
        flow_layers: Number of flow layers. Defaults to 8.
        nn_width: Number of hidden layers in neural network. Defaults to 50.
        nn_depth: Depth of neural network. Defaults to 1.
        nn_activation: _description_. Defaults to jnn.relu.
        invert: Whether to invert the bijection. Broadly, True will prioritise a faster
            inverse, leading to faster `log_prob`, False will prioritise faster forward,
            leading to faster `sample`. Defaults to True.
    """
    if transformer is None:
        transformer = eqx.tree_at(
            lambda aff: aff.scale,
            Affine(),
            BijectionReparam(1, Chain([SoftPlus(), NonTrainable(Loc(1e-2))])),
        )
    dim = base_dist.shape[-1]

    # assert cond_dim >= cond_u_y_dim

    def make_layer(key):  # masked autoregressive layer + permutation
        bij_key, perm_key = jr.split(key)
        # list_bijections = [Identity((cond_u_y_dim,))]
        MAF_bijection = MaskedAutoregressiveFirstUniform(
            key=bij_key,
            transformer=transformer,
            dim=dim,  # dim - cond_u_y_dim,
            # cond_dim=cond_dim,
            cond_dim_mask=cond_dim_mask,
            cond_dim_nomask=cond_dim_nomask,
            # cond_dim=cond_dim,
            nn_width=nn_width,
            nn_depth=nn_depth,
            nn_activation=nn_activation,
        )
        # list_bijections.append(MAF_bijection)
        # bijection = Concatenate(list_bijections)
        bijection = MAF_bijection
        return _add_default_permute_but_first(bijection, dim, perm_key)

    keys = jr.split(key, flow_layers)
    layers = eqx.filter_vmap(make_layer)(keys)
    bijection = Invert(Scan(layers)) if invert else Scan(layers)
    return Transformed(base_dist, bijection)


def _add_default_permute_but_first(bijection: AbstractBijection, dim: int, key: Array):
    if (dim == 1) or (dim == 2):
        return bijection

    perm = Permute(
        jnp.hstack(
            [jnp.expand_dims(0, axis=-1), jr.permutation(key, jnp.arange(1, dim))]
        )
    )
    return Chain([bijection, perm]).merge_chains()
