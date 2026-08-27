"""Transformer-conditioned autoregressive bijection (TarFlow-style conditioner).

The same mathematical object as ``MaskedAutoregressive`` — an autoregressive
flow whose per-dimension ``transformer`` (e.g. an RQS spline) is parameterised
by a conditioner network — with the MADE-masked MLP conditioner replaced by a
CAUSAL TRANSFORMER over one token per dimension, following TarFlow (Zhai et
al., "Normalizing Flows are Capable Generative Models"; see REF-012/REF-013 in
the project references).

Autoregressive structure. Each input coordinate becomes one token. The token
sequence fed to the transformer is right-shifted with a learned start token:
position ``i`` receives ``y_{i-1}`` (position 0 the start token), so with a
causal attention mask the parameters emitted at position ``i`` depend on
``y_{<i}`` and the condition only — the standard MAF factorisation. The start
token makes dimension 0 fully learnable and condition-dependent (TarFlow
instead zeroes the first position; a start token gives every dimension a
treatment-conditioned transform, which the causal margin needs).

Conditioning. The condition (e.g. treatment) is linearly embedded and added to
every token — the continuous generalisation of TarFlow's additive class
embedding (for a binary condition the two are equivalent).

Initialisation. The parameter head ``proj_out`` is ZERO-INITIALISED, so every
block starts as (close to) the identity map — TarFlow's trainability trick,
carried over.

Directions. ``transform_and_log_det`` is one parallel pass (fast density);
``inverse_and_log_det`` solves one coordinate per step via ``lax.scan``
(sampling), mirroring ``MaskedAutoregressiveFirstUniform``.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from flowjax.bijections import Chain, Flip, Invert, RationalQuadraticSpline, Scan
from flowjax.bijections.bijection import AbstractBijection
from flowjax.bijections.jax_transforms import Vmap
from flowjax.utils import get_ravelled_pytree_constructor
from jax import Array
from paramax import NonTrainable


class _TransformerLayer(eqx.Module):
    """One pre-LayerNorm causal-attention block: attention + MLP, both residual."""

    norm1: eqx.nn.LayerNorm
    attn: eqx.nn.MultiheadAttention
    norm2: eqx.nn.LayerNorm
    mlp_in: eqx.nn.Linear
    mlp_out: eqx.nn.Linear

    def __init__(self, channels: int, num_heads: int, expansion: int, key: Array):
        k1, k2, k3 = jr.split(key, 3)
        self.norm1 = eqx.nn.LayerNorm(channels)
        self.attn = eqx.nn.MultiheadAttention(num_heads, channels, key=k1)
        self.norm2 = eqx.nn.LayerNorm(channels)
        self.mlp_in = eqx.nn.Linear(channels, expansion * channels, key=k2)
        self.mlp_out = eqx.nn.Linear(expansion * channels, channels, key=k3)

    def __call__(self, tokens: Array, mask: Array) -> Array:
        h = jax.vmap(self.norm1)(tokens)
        tokens = tokens + self.attn(h, h, h, mask=mask)
        h = jax.vmap(self.norm2)(tokens)
        return tokens + jax.vmap(self.mlp_out)(jnn.gelu(jax.vmap(self.mlp_in)(h)))


class TransformerAutoregressive(AbstractBijection):
    """Autoregressive bijection with a causal-transformer conditioner.

    Args:
        key: Jax PRNGKey.
        transformer: bijection with shape ``()`` to be parameterised per
            dimension (parameters wrapped in ``NonTrainable`` are excluded),
            e.g. ``RationalQuadraticSpline``.
        dim: dimension of the bijection.
        cond_dim: dimension of the conditioning variable, or ``None`` for an
            unconditional bijection.
        nn_width: transformer channel width.
        nn_depth: number of attention blocks.
        nn_heads: attention heads (must divide ``nn_width``).
        expansion: MLP hidden width as a multiple of ``nn_width``.
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None
    transformer_constructor: Callable
    proj_in: eqx.nn.Linear
    cond_proj: eqx.nn.Linear | None
    start_token: Array
    pos_embed: Array
    layers: tuple[_TransformerLayer, ...]
    proj_out: eqx.nn.Linear

    def __init__(
        self,
        key: Array,
        *,
        transformer: AbstractBijection,
        dim: int,
        cond_dim: int | None = None,
        nn_width: int = 64,
        nn_depth: int = 1,
        nn_heads: int = 4,
        expansion: int = 2,
    ) -> None:
        if transformer.shape != () or transformer.cond_shape is not None:
            raise ValueError(
                "Only unconditional transformers with shape () are supported.",
            )
        if nn_width % nn_heads:
            raise ValueError(f"nn_width {nn_width} not divisible by nn_heads {nn_heads}.")

        constructor, num_params = get_ravelled_pytree_constructor(
            transformer,
            filter_spec=eqx.is_inexact_array,
            is_leaf=lambda leaf: isinstance(leaf, NonTrainable),
        )
        self.transformer_constructor = constructor
        self.shape = (dim,)
        self.cond_shape = None if cond_dim is None else (cond_dim,)

        keys = jr.split(key, nn_depth + 4)
        self.proj_in = eqx.nn.Linear(1, nn_width, key=keys[0])
        self.cond_proj = (
            None if cond_dim is None else eqx.nn.Linear(cond_dim, nn_width, key=keys[1])
        )
        self.start_token = jr.normal(keys[2], (nn_width,)) * 1e-2
        self.pos_embed = jr.normal(keys[3], (dim, nn_width)) * 1e-2
        self.layers = tuple(
            _TransformerLayer(nn_width, nn_heads, expansion, k) for k in keys[4:]
        )
        proj_out = eqx.nn.Linear(nn_width, num_params, key=keys[0])
        # Zero-init the parameter head: with the transformer's zero parameter
        # vector the per-dim transform starts at (close to) the identity, so a
        # freshly built block barely perturbs its input regardless of depth.
        self.proj_out = eqx.tree_at(
            lambda l: (l.weight, l.bias),
            proj_out,
            (jnp.zeros_like(proj_out.weight), jnp.zeros_like(proj_out.bias)),
        )

    def _params(self, x: Array, condition: Array | None) -> Array:
        """Per-dimension transformer parameters from a causal pass over tokens.

        Token ``i`` carries ``x_{i-1}`` (token 0 the start token), so with the
        causal mask the parameters for dimension ``i`` are a function of
        ``x_{<i}`` and the condition only.
        """
        dim = self.shape[0]
        emb = jax.vmap(self.proj_in)(x[: dim - 1, None])          # (dim-1, ch)
        tokens = jnp.vstack([self.start_token[None], emb]) + self.pos_embed
        if self.cond_proj is not None:
            tokens = tokens + self.cond_proj(condition)
        mask = jnp.tril(jnp.ones((dim, dim), dtype=bool))
        for layer in self.layers:
            tokens = layer(tokens, mask)
        return jax.vmap(self.proj_out)(tokens)                    # (dim, n_params)

    def _flat_params_to_transformer(self, params: Array):
        transformer = eqx.filter_vmap(self.transformer_constructor)(params)
        return Vmap(transformer, in_axes=eqx.if_array(0))

    def transform_and_log_det(self, x, condition=None):
        transformer = self._flat_params_to_transformer(self._params(x, condition))
        return transformer.transform_and_log_det(x)

    def inv_scan_fn(self, init, _, condition):
        """Solve one coordinate: parameters at the current rank depend only on
        already-solved coordinates, so sequential substitution is exact."""
        y, rank = init
        transformer = self._flat_params_to_transformer(self._params(y, condition))
        x = transformer.inverse(y)
        x = y.at[rank].set(x[rank])
        return (x, rank + 1), None

    def inverse_and_log_det(self, y, condition=None):
        init = (y, 0)
        fn = partial(self.inv_scan_fn, condition=condition)
        (x, _), _ = jax.lax.scan(fn, init, None, length=len(y))
        log_det = self.transform_and_log_det(x, condition)[1]
        return x, -log_det


def transformer_autoregressive_bijection(
    key: Array,
    dim: int,
    condition: Array,
    RQS_knots: int = 8,
    nn_depth: int = 1,
    nn_width: int = 64,
    flow_layers: int = 4,
    nn_heads: int = 4,
    expansion: int = 2,
):
    """Stack of transformer-conditioned autoregressive RQS blocks.

    The drop-in counterpart of ``basic_flows.masked_autoregressive_bijection``
    with the MADE-MLP conditioner replaced by the causal transformer: same call
    convention (``condition`` supplies ``cond_dim = condition.shape[1]``), same
    ``Invert(Scan(...))`` return (fast ``log_prob``, sequential ``sample``).
    Each block is followed by a ``Flip``, so consecutive blocks run their
    autoregression in opposite directions (TarFlow's alternating permutation).
    """
    transformer = RationalQuadraticSpline(knots=RQS_knots, interval=1)

    def make_layer(key):
        bijection = TransformerAutoregressive(
            key=key,
            transformer=transformer,
            dim=dim,
            cond_dim=condition.shape[1],
            nn_width=nn_width,
            nn_depth=nn_depth,
            nn_heads=nn_heads,
            expansion=expansion,
        )
        if dim == 1:
            return bijection
        return Chain([bijection, Flip((dim,))]).merge_chains()

    keys = jr.split(key, flow_layers)
    layers = eqx.filter_vmap(make_layer)(keys)
    return Invert(Scan(layers))
