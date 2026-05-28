"""Masked autoregressive network and bijection."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from flowjax.bijections import AbstractBijection, Concatenate
from flowjax.bijections.bijection import AbstractBijection
from flowjax.bijections.jax_transforms import Vmap
from flowjax.bijections.utils import Identity
from flowjax.masks import rank_based_mask
from flowjax.utils import get_ravelled_pytree_constructor
from jax import Array
from paramax import NonTrainable, Parameterize
from jaxtyping import Array, Int


class MaskedAutoregressiveFirstUniform(AbstractBijection):
    """Masked autoregressive bijection that holds the first coordinate fixed.

    A MADE-masked MLP parameterises a per-dimension ``transformer``, giving the
    usual autoregressive triangular Jacobian. This variant replaces the
    transformer of coordinate 0 with the identity, so the first coordinate is
    passed through unchanged: if the base distribution's first marginal is
    uniform it stays uniform after the flow. That slot carries the causal
    effect in the frugal parametrisation.

    Conditioning variables can enter in two ways:
        - ``cond_dim_nomask``: given input rank -1, so every output may depend
          on them (standard conditional MAF).
        - ``cond_dim_mask``: given input rank ``dim``, so they are masked from
          all outputs (present but not used by the autoregressive net).
    Both may be supplied together; the condition is then their concatenation.

    Refs:
        - https://arxiv.org/abs/1705.07057v4
        - https://arxiv.org/abs/1502.03509

    Args:
        key: Jax PRNGKey
        transformer: Bijection with shape () to be parameterised by the autoregressive
            network. Parameters wrapped with ``NonTrainable`` are exluded.
        dim: Dimension.
        cond_dim_nomask: Size of the unmasked conditioning block. Defaults to None.
        cond_dim_mask: Size of the masked conditioning block. Defaults to None.
        nn_width: Neural network width.
        nn_depth: Neural network depth.
        nn_activation: Neural network activation. Defaults to jnn.relu.
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None
    transformer_constructor: Callable
    masked_autoregressive_mlp: eqx.nn.MLP

    def __init__(
        self,
        key: Array,
        *,
        transformer: AbstractBijection,
        dim: int,
        cond_dim_nomask: int | None = None,
        cond_dim_mask: int | None = None,
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
            is_leaf=lambda leaf: isinstance(leaf, NonTrainable),
        )

        if cond_dim_mask is None:
            if cond_dim_nomask is None:
                self.cond_shape = None
                in_ranks = jnp.arange(dim)
            else:
                self.cond_shape = (cond_dim_nomask,)
                # we give conditioning variables rank -1 (no masking of edges to output)
                in_ranks = jnp.hstack((jnp.arange(dim), -jnp.ones(cond_dim_nomask)))
        else:
            if cond_dim_nomask is None:
                self.cond_shape = (cond_dim_mask,)
                # we give conditioning variables rank dim (masking of all edges to output)
                in_ranks = jnp.hstack(
                    (jnp.arange(dim), jnp.ones(cond_dim_mask) * (dim))
                )
            else:
                self.cond_shape = (cond_dim_mask + cond_dim_nomask,)
                # we give conditioning variables rank -1 (no masking of edges to output)
                in_ranks = jnp.hstack(
                    (
                        jnp.arange(dim),
                        -jnp.ones(cond_dim_nomask),
                        jnp.ones(cond_dim_mask) * (dim),
                    )
                )

        hidden_ranks = jnp.arange(nn_width) % dim
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

    def transform_and_log_det(self, x, condition=None):
        nn_input = x if condition is None else jnp.hstack((x, condition))
        transformer_params = self.masked_autoregressive_mlp(nn_input)
        transformer = self._flat_params_to_transformer(transformer_params)
        return transformer.transform_and_log_det(x)

    def inv_scan_fn(self, init, _, condition):
        """One 'step' in computing the inverse."""
        y, rank = init
        nn_input = y if condition is None else jnp.hstack((y, condition))
        transformer_params = self.masked_autoregressive_mlp(nn_input)
        transformer = self._flat_params_to_transformer(transformer_params)
        x = transformer.inverse(y)
        x = y.at[rank].set(x[rank])
        return (x, rank + 1), None

    def inverse_and_log_det(self, y, condition=None):
        init = (y, 0)
        fn = partial(self.inv_scan_fn, condition=condition)
        (x, _), _ = jax.lax.scan(fn, init, None, length=len(y))
        log_det = self.transform_and_log_det(x, condition)[1]
        return x, -log_det

    def _flat_params_to_transformer(self, params: Array, cond_u_y_dim=1):
        """Reshape to dim X params_per_dim, then vmap."""
        dim = self.shape[-1]
        transformer_params = jnp.reshape(params, (dim, -1))
        transformer_params = transformer_params[cond_u_y_dim:, :]
        transformer = eqx.filter_vmap(self.transformer_constructor)(transformer_params)
        return Concatenate(
            [Identity((cond_u_y_dim,)), Vmap(transformer, in_axes=eqx.if_array(0))]
        )


def masked_autoregressive_mlp(
    in_ranks: Int[Array, " in_size"],
    hidden_ranks: Int[Array, " hidden_size"],
    out_ranks: Int[Array, " out_size"],
    **kwargs,
) -> eqx.nn.MLP:
    """Returns an equinox multilayer perceptron, with autoregressive masks.

    Masked positions are enforced at network-use time via a parameter wrapper,
    not at construction. Training updates the underlying weight; masked entries
    remain 0 in the forward pass. For mask construction details, see
    https://arxiv.org/pdf/1502.03509.pdf.

    Args:
        in_ranks: The ranks of the inputs.
        hidden_ranks: The ranks of the hidden dimensions.
        out_ranks: The ranks of the output dimensions.
        **kwargs: Keyword arguments passed to equinox.nn.MLP.
    """
    in_ranks, hidden_ranks, out_ranks = (
        jnp.asarray(a, jnp.int32) for a in (in_ranks, hidden_ranks, out_ranks)
    )
    mlp = eqx.nn.MLP(
        in_size=len(in_ranks),
        out_size=len(out_ranks),
        width_size=len(hidden_ranks),
        **kwargs,
    )
    ranks = [in_ranks, *[hidden_ranks] * mlp.depth, out_ranks]

    masked_layers = []
    for i, linear in enumerate(mlp.layers):
        mask = rank_based_mask(ranks[i], ranks[i + 1], eq=i != len(mlp.layers) - 1)
        masked_linear = eqx.tree_at(
            lambda linear: linear.weight,
            linear,
            Parameterize(jnp.where, mask, linear.weight, 0),
        )
        masked_layers.append(masked_linear)

    mlp = eqx.tree_at(lambda mlp: mlp.layers, mlp, replace=tuple(masked_layers))

    return mlp
