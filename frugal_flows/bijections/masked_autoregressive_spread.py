"""flowjax's ``MaskedAutoregressive`` with hidden-unit ranks that reach every input.

The constructor below is flowjax 19.1.0's ``MaskedAutoregressive.__init__`` line for line,
except for the hidden ranks. flowjax uses ``arange(nn_width) % (dim - 1)`` (unconditional)
and ``(arange(nn_width) % dim) - 1`` (conditional); with fewer hidden units than
coordinates the highest ranks are never assigned, so in each layer the last outputs
cannot depend on the inputs just before them (measured on the outcome margin at 64 pixels
and width 48: 137 of 2016 allowed links missing per layer). Here the ranks come from
``frugal_flows.bijections.ranks.autoregressive_hidden_ranks``, which reproduces flowjax's
ranks exactly whenever the width covers them, and otherwise spreads them over the full
range. Everything else (transform, inverse, parameter layout) is inherited unchanged.
"""
from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import paramax
from flowjax.bijections import MaskedAutoregressive
from flowjax.bijections.bijection import AbstractBijection
from flowjax.bijections.masked_autoregressive import masked_autoregressive_mlp
from flowjax.utils import get_ravelled_pytree_constructor
from jaxtyping import PRNGKeyArray

from frugal_flows.bijections.ranks import autoregressive_hidden_ranks


class MaskedAutoregressiveSpread(MaskedAutoregressive):
    """``flowjax.bijections.MaskedAutoregressive`` whose hidden units reach every input.

    Same arguments and behaviour as flowjax's class; see the module docstring for the
    single difference.
    """

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        transformer: AbstractBijection,
        dim: int,
        cond_dim: int | None = None,
        nn_width: int,
        nn_depth: int,
        nn_activation: Callable = jnn.relu,
    ) -> None:
        if transformer.shape != () or transformer.cond_shape is not None:
            raise ValueError(
                "Only unconditional transformers with shape () are supported.",
            )

        constructor, num_params = get_ravelled_pytree_constructor(
            transformer,
            filter_spec=eqx.is_inexact_array,
            is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
        )

        if cond_dim is None:
            in_ranks = jnp.arange(dim)
            hidden_ranks = autoregressive_hidden_ranks(nn_width, dim, lo=0)
        else:
            # conditioning variables have rank -1: visible to every hidden unit
            in_ranks = jnp.hstack((jnp.arange(dim), -jnp.ones(cond_dim, int)))
            hidden_ranks = autoregressive_hidden_ranks(nn_width, dim, lo=-1)
        out_ranks = jnp.repeat(jnp.arange(dim), num_params)

        self.masked_autoregressive_mlp = masked_autoregressive_mlp(
            in_ranks,
            hidden_ranks,
            out_ranks,
            depth=nn_depth,
            activation=nn_activation,
            key=key,
        )

        self.transformer_constructor = constructor
        self.shape = (dim,)
        self.cond_shape = None if cond_dim is None else (cond_dim,)
