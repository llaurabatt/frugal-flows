from collections.abc import Callable

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import optax
from flowjax.bijections import (
    AbstractBijection,
    Affine,
    Chain,
    Invert,
    Loc,
    MaskedAutoregressive,
    Permute,
    RationalQuadraticSpline,
    Scan,
    SoftPlus,
    Tanh,
)
from flowjax.distributions import AbstractDistribution, Transformed, Uniform
from flowjax.flows import _add_default_permute, masked_autoregressive_flow
from flowjax.train import fit_to_data
from flowjax.wrappers import BijectionReparam, NonTrainable
from jax import Array
from jax.typing import ArrayLike

from frugal_flows.bijections import (
    MaskedAutoregressiveFirstUniform,
    MaskedAutoregressiveHeterogeneous,
    MaskedAutoregressiveMaskedCond,
    MaskedAutoregressiveTransformerCond,
    MaskedIndependent,
)


def univariate_marginal_flow(
    key: jr.PRNGKey,
    z_cont: Array,
    optimizer: optax.GradientTransformation | None = None,
    RQS_knots: int = 8,
    flow_layers: int = 8,
    nn_width: int = 50,
    nn_depth: int = 1,
    show_progress: bool = True,
    learning_rate: float = 5e-4,
    max_epochs: int = 100,
    max_patience: int = 5,
    batch_size: int = 100,
    val_prop: float = 0.1,
):
    if z_cont.ndim == 1:
        # Reshape one-dimensional array to two dimensions with second dim as 1
        z_cont = z_cont.reshape(-1, 1)
    elif z_cont.ndim == 2:
        if z_cont.shape[1] > 1:
            raise ValueError(
                "Univariate input with shape (n_samples,) or (n_samples,1) is required"
            )
    else:
        raise ValueError(
            "Univariate input with shape (n_samples,) or (n_samples,1) is required"
        )

    nvars = z_cont.shape[1]
    assert nvars == 1
    key, subkey = jr.split(key)

    base_dist = Uniform(-jnp.ones(nvars), jnp.ones(nvars))

    transformer = RationalQuadraticSpline(knots=RQS_knots, interval=1)

    flow = masked_autoregressive_flow(
        key=subkey,
        base_dist=base_dist,
        transformer=transformer,
        flow_layers=flow_layers,
        nn_width=nn_width,
        nn_depth=nn_depth,
    )  # Support on [-1, 1]

    flow = Transformed(flow, Invert(Tanh(flow.shape)))  # Unbounded support

    flow = flow.merge_transforms()

    flow = eqx.tree_at(
        where=lambda flow: flow.bijection.bijections[0],
        pytree=flow,
        replace_fn=NonTrainable,
    )

    key, subkey = jr.split(key)

    # Train
    flow, losses = fit_to_data(
        key=subkey,
        dist=flow,
        x=z_cont,
        learning_rate=learning_rate,
        max_patience=max_patience,
        max_epochs=max_epochs,
        batch_size=batch_size,
        show_progress=show_progress,
        optimizer=optimizer,
        val_prop=val_prop,
    )

    return flow, losses


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
    stop_grad_until: int | None = None,
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
            stop_grad_until=stop_grad_until,
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


def _affine_with_min_scale(min_scale: float = 1e-2) -> Affine:
    scale_reparam = Chain([SoftPlus(), NonTrainable(Loc(min_scale))])
    return eqx.tree_at(
        where=lambda aff: aff.scale,
        pytree=Affine(),
        replace=BijectionReparam(jnp.array(1), scale_reparam),
    )


def masked_autoregressive_flow_heterogeneous(
    key: Array,
    *,
    base_dist: AbstractDistribution,
    transformer: AbstractBijection | None = None,
    # cond_dim: int | None = None,
    cond_dim_mask: int | None = None,
    cond_dim_nomask: int | None = None,
    causal_effect_idx: int = 0,
    flow_layers: int = 8,
    nn_width: int = 50,
    nn_depth: int = 1,
    nn_activation: Callable = jnn.relu,
    invert: bool = True,
    stop_grad_until: int | None = None,
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
        MAF_bijection = MaskedAutoregressiveHeterogeneous(
            key=bij_key,
            transformer=transformer,
            dim=dim,
            cond_dim_mask=cond_dim_mask,
            cond_dim_nomask=cond_dim_nomask,
            identity_idx=causal_effect_idx,
            nn_width=nn_width,
            nn_depth=nn_depth,
            nn_activation=nn_activation,
            stop_grad_until=stop_grad_until,
        )
        # list_bijections.append(MAF_bijection)
        # bijection = Concatenate(list_bijections)
        bijection = MAF_bijection
        return _add_default_permute_but_specific_idx(
            bijection, dim, perm_key, causal_effect_idx
        )

    keys = jr.split(key, flow_layers)
    layers = eqx.filter_vmap(make_layer)(keys)
    bijection = Invert(Scan(layers)) if invert else Scan(layers)
    return Transformed(base_dist, bijection)


def _add_default_permute_but_specific_idx(
    bijection: AbstractBijection, dim: int, key: Array, idx: int
):
    if (dim == 1) or (dim == 2):
        return bijection

    else:
        perm_key1, perm_key2 = jr.split(key)
        perm = Permute(
            jnp.hstack(
                [
                    jr.permutation(perm_key1, jnp.arange(0, idx)),
                    jnp.expand_dims(idx, axis=-1),
                    jr.permutation(perm_key2, jnp.arange(idx + 1, dim)),
                ]
            )
        )
        return Chain([bijection, perm]).merge_chains()


def masked_autoregressive_flow_transformer_cond(
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
        transformer = _affine_with_min_scale()

    dim = base_dist.shape[-1]

    def make_layer(key):  # masked autoregressive layer + permutation
        bij_key, perm_key = jr.split(key)
        bijection = MaskedAutoregressiveTransformerCond(
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


def masked_autoregressive_flow_masked_cond(
    key: Array,
    *,
    base_dist: AbstractDistribution,
    transformer: AbstractBijection | None = None,
    cond_dim_mask: int | None = None,
    cond_dim_nomask: int | None = None,
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
        bijection = MaskedAutoregressiveMaskedCond(
            key=bij_key,
            transformer=transformer,
            dim=dim,
            cond_dim_mask=cond_dim_mask,
            cond_dim_nomask=cond_dim_nomask,
            nn_width=nn_width,
            nn_depth=nn_depth,
            nn_activation=nn_activation,
        )
        return _add_default_permute(bijection, dim, perm_key)

    keys = jr.split(key, flow_layers)
    layers = eqx.filter_vmap(make_layer)(keys)
    bijection = Invert(Scan(layers)) if invert else Scan(layers)
    return Transformed(base_dist, bijection)


def masked_autoregressive_bijection_masked_condition(
    key: jr.PRNGKey,
    dim: ArrayLike,
    condition: ArrayLike,
    RQS_knots: int = 8,
    nn_depth: int = 1,
    nn_width: int = 50,
    flow_layers: int = 4,
):
    invert = True
    transformer = RationalQuadraticSpline(knots=RQS_knots, interval=1)

    def make_layer(key):
        bij_key, perm_key = jr.split(key)
        bijection = MaskedAutoregressiveMaskedCond(
            key=bij_key,
            transformer=transformer,
            dim=dim,
            cond_dim_mask=condition.shape[1],
            nn_width=nn_width,
            nn_depth=nn_depth,
        )
        return _add_default_permute(bijection, dim, perm_key)

    keys = jr.split(key, flow_layers)
    layers = eqx.filter_vmap(make_layer)(keys)
    maf_bijection = Invert(Scan(layers)) if invert else Scan(layers)

    return maf_bijection


def masked_autoregressive_bijection(
    key: jr.PRNGKey,
    dim: ArrayLike,
    condition: ArrayLike,
    RQS_knots: int = 8,
    nn_depth: int = 1,
    nn_width: int = 50,
    flow_layers: int = 4,
):
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
    invert = True
    transformer = RationalQuadraticSpline(knots=RQS_knots, interval=1)

    def make_layer(key):  # masked autoregressive layer + permutation
        bij_key, perm_key = jr.split(key)
        bijection = MaskedAutoregressive(
            key=bij_key,
            transformer=transformer,
            dim=dim,
            cond_dim=condition.shape[1],
            nn_width=nn_width,
            nn_depth=nn_depth,
        )
        return _add_default_permute(bijection, dim, perm_key)

    keys = jr.split(key, flow_layers)
    layers = eqx.filter_vmap(make_layer)(keys)
    maf_bijection = Invert(Scan(layers)) if invert else Scan(layers)
    return maf_bijection
