from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from flowjax.bijections import (
    Affine,
    Concatenate,
    Invert,
    RationalQuadraticSpline,
    Stack,
    Tanh,
)
from flowjax.bijections.utils import Identity
from flowjax.distributions import Transformed, Uniform, _StandardUniform
from flowjax.flows import masked_autoregressive_flow
from flowjax.train import fit_to_data
from flowjax.utils import get_ravelled_pytree_constructor
from flowjax.wrappers import NonTrainable
from jaxtyping import ArrayLike

from frugal_flows.basic_flows import (
    masked_autoregressive_bijection_masked_condition,
    masked_autoregressive_flow_first_uniform,
    masked_independent_flow,
    univariate_marginal_flow,
)
from frugal_flows.bijections import LocCond, UnivariateNormalCDF


def train_copula_flow(
    key: jr.PRNGKey,
    u_z: ArrayLike,  # impose discrete
    optimizer: optax.GradientTransformation | None = None,
    RQS_knots: int = 8,
    nn_depth: int = 1,
    nn_width: int = 50,
    flow_layers: int = 4,
    show_progress: bool = True,
    learning_rate: float = 5e-4,
    max_epochs: int = 100,
    max_patience: int = 5,
    batch_size: int = 100,
):
    nvars = u_z.shape[1]
    key, subkey = jr.split(key)

    base_dist = Uniform(-jnp.ones(nvars), jnp.ones(nvars))

    transformer = RationalQuadraticSpline(knots=RQS_knots, interval=1)

    copula_flow = masked_autoregressive_flow(  # masked_autoregressive_flow(
        key=subkey,
        base_dist=base_dist,
        transformer=transformer,
        nn_depth=nn_depth,
        nn_width=nn_width,
        flow_layers=flow_layers,
    )  # Support on [-1, 1]

    copula_flow = Transformed(
        copula_flow, Invert(Affine(loc=-jnp.ones(nvars), scale=jnp.ones(nvars) * 2))
    )  # Unbounded support

    copula_flow = copula_flow.merge_transforms()

    assert isinstance(copula_flow.base_dist, _StandardUniform)

    copula_flow = eqx.tree_at(
        where=lambda copula_flow: copula_flow.bijection.bijections[0],
        pytree=copula_flow,
        replace_fn=NonTrainable,
    )

    copula_flow = eqx.tree_at(
        where=lambda copula_flow: copula_flow.bijection.bijections[-1],
        pytree=copula_flow,
        replace_fn=NonTrainable,
    )

    key, subkey = jr.split(key)

    # Train
    copula_flow, losses = fit_to_data(
        key=subkey,
        dist=copula_flow,
        x=u_z,
        optimizer=optimizer,
        show_progress=show_progress,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        max_patience=max_patience,
        batch_size=batch_size,
    )

    return copula_flow, losses


def train_frugal_flow_location_translation(
    key: jr.PRNGKey,
    y: ArrayLike,
    u_z: ArrayLike,  # impose discrete
    optimizer: optax.GradientTransformation | None = None,
    RQS_knots: int = 8,
    nn_depth: int = 1,
    nn_width: int = 50,
    flow_layers: int = 4,
    show_progress: bool = True,
    learning_rate: float = 5e-4,
    max_epochs: int = 100,
    max_patience: int = 5,
    batch_size: int = 100,
    condition: ArrayLike | None = None,
    mask_condition: bool = True,
    stop_grad_until_active: bool = False,
    causal_model_args: dict | None = None,
):
    nvars = u_z.shape[1]

    if condition is None:
        cond_dim = None
    else:
        cond_dim = condition.shape[1]
    if mask_condition:
        cond_dim_mask = cond_dim
        cond_dim_nomask = None
    else:
        cond_dim_mask = None
        cond_dim_nomask = cond_dim

    list_bijections_affine = [Identity((1,))] + [
        Invert(Affine(loc=-jnp.ones(nvars), scale=jnp.ones(nvars) * 2))
    ]
    bijections_affine = Concatenate(list_bijections_affine)

    key, subkey = jr.split(key)
    ate_maf_bijection = masked_autoregressive_bijection_masked_condition(
        key=subkey,
        dim=1,
        condition=condition,
        RQS_knots=causal_model_args["RQS_knots"],
        nn_depth=causal_model_args["nn_depth"],
        nn_width=causal_model_args["nn_width"],
        flow_layers=causal_model_args["flow_layers"],
    )

    list_bijections_ate_maf = [ate_maf_bijection] + [Identity((1,))] * nvars
    bijections_ate_maf = Concatenate(list_bijections_ate_maf)

    list_bijections_tanh = [Invert(Tanh(()))] + [Identity(())] * nvars
    bijections_tanh = Stack(list_bijections_tanh)

    list_bijections_loccond = [LocCond(ate=causal_model_args["ate"])] + [
        Identity(())
    ] * nvars
    bijections_loccond = Stack(list_bijections_loccond)

    base_dist = Uniform(-jnp.ones(nvars + 1), jnp.ones(nvars + 1))

    transformer = RationalQuadraticSpline(knots=RQS_knots, interval=1)
    if stop_grad_until_active:
        _, stop_grad_until = get_ravelled_pytree_constructor(transformer)
    else:
        stop_grad_until = None

    key, subkey = jr.split(key)
    frugal_flow = masked_autoregressive_flow_first_uniform(
        key=subkey,
        base_dist=base_dist,
        transformer=transformer,
        invert=True,
        cond_dim_mask=cond_dim_mask,
        cond_dim_nomask=cond_dim_nomask,
        nn_depth=nn_depth,
        nn_width=nn_width,
        flow_layers=flow_layers,
        stop_grad_until=stop_grad_until,
    )  # Support on [-1, 1]

    frugal_flow = Transformed(
        frugal_flow,
        bijections_affine,
    )

    frugal_flow = Transformed(
        frugal_flow,
        bijections_ate_maf,
    )

    frugal_flow = Transformed(
        frugal_flow,
        bijections_tanh,
    )
    frugal_flow = Transformed(
        frugal_flow,
        bijections_loccond,
    )
    frugal_flow = frugal_flow.merge_transforms()
    frugal_flow = eqx.tree_at(
        where=lambda frugal_flow: frugal_flow.bijection.bijections[-4],
        pytree=frugal_flow,
        replace_fn=NonTrainable,
    )

    frugal_flow = eqx.tree_at(
        where=lambda frugal_flow: frugal_flow.bijection.bijections[0],
        pytree=frugal_flow,
        replace_fn=NonTrainable,
    )

    key, subkey = jr.split(key)

    # Train
    key, subkey = jr.split(key)
    frugal_flow, losses = fit_to_data(
        key=subkey,
        dist=frugal_flow,
        x=jnp.hstack([y, u_z]),
        condition=condition,
        optimizer=optimizer,
        show_progress=show_progress,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        max_patience=max_patience,
        batch_size=batch_size,
    )

    return frugal_flow, losses


def train_frugal_flow_flexible_discrete(
    key: jr.PRNGKey,
    y: ArrayLike,
    u_z: ArrayLike,  # impose discrete
    optimizer: optax.GradientTransformation | None = None,
    RQS_knots: int = 8,
    nn_depth: int = 1,
    nn_width: int = 50,
    flow_layers: int = 4,
    show_progress: bool = True,
    learning_rate: float = 5e-4,
    max_epochs: int = 100,
    max_patience: int = 5,
    batch_size: int = 100,
    condition: ArrayLike | None = None,
    mask_condition: bool = True,
    stop_grad_until_active: bool = False,
    causal_model_args: dict | None = None,
):
    nvars = u_z.shape[1]

    key, subkey = jr.split(key)
    outcome, _ = univariate_discrete_cdf(
        key=subkey, z_discr=y.flatten(), max_unique_z_discr_size=len(jnp.unique(y))
    )
    outcome = jnp.expand_dims(outcome, axis=1)

    if condition is None:
        cond_dim = None
    else:
        cond_dim = condition.shape[1]
    if mask_condition:
        cond_dim_mask = cond_dim
        cond_dim_nomask = None
    else:
        cond_dim_mask = None
        cond_dim_nomask = cond_dim

    list_bijections_affine = [Identity((1,))] + [
        Invert(Affine(loc=-jnp.ones(nvars), scale=jnp.ones(nvars) * 2))
    ]
    bijections_affine = Concatenate(list_bijections_affine)

    key, subkey = jr.split(key)
    ate_maf_bijection = masked_autoregressive_bijection_masked_condition(
        key=subkey,
        dim=1,
        condition=condition,
        RQS_knots=causal_model_args["RQS_knots"],
        nn_depth=causal_model_args["nn_depth"],
        nn_width=causal_model_args["nn_width"],
        flow_layers=causal_model_args["flow_layers"],
    )

    list_bijections_ate_maf = [ate_maf_bijection] + [Identity((1,))] * nvars
    bijections_ate_maf = Concatenate(list_bijections_ate_maf)

    list_bijections_affine_output = [
        Invert(Affine(loc=-jnp.ones(1), scale=jnp.ones(1) * 2))
    ] + [Identity((nvars,))]
    bijections_affine_output = Concatenate(list_bijections_affine_output)

    base_dist = Uniform(-jnp.ones(nvars + 1), jnp.ones(nvars + 1))

    transformer = RationalQuadraticSpline(knots=RQS_knots, interval=1)
    if stop_grad_until_active:
        _, stop_grad_until = get_ravelled_pytree_constructor(transformer)
    else:
        stop_grad_until = None

    key, subkey = jr.split(key)
    frugal_flow = masked_autoregressive_flow_first_uniform(
        key=subkey,
        base_dist=base_dist,
        transformer=transformer,
        invert=True,
        cond_dim_mask=cond_dim_mask,
        cond_dim_nomask=cond_dim_nomask,
        nn_depth=nn_depth,
        nn_width=nn_width,
        flow_layers=flow_layers,
        stop_grad_until=stop_grad_until,
    )  # Support on [-1, 1]

    frugal_flow = Transformed(
        frugal_flow,
        bijections_affine,
    )

    frugal_flow = Transformed(
        frugal_flow,
        bijections_ate_maf,
    )

    frugal_flow = Transformed(
        frugal_flow,
        bijections_affine_output,
    )

    frugal_flow = frugal_flow.merge_transforms()
    frugal_flow = eqx.tree_at(
        where=lambda frugal_flow: frugal_flow.bijection.bijections[-3],
        pytree=frugal_flow,
        replace_fn=NonTrainable,
    )

    frugal_flow = eqx.tree_at(
        where=lambda frugal_flow: frugal_flow.bijection.bijections[-1],
        pytree=frugal_flow,
        replace_fn=NonTrainable,
    )

    frugal_flow = eqx.tree_at(
        where=lambda frugal_flow: frugal_flow.bijection.bijections[0],
        pytree=frugal_flow,
        replace_fn=NonTrainable,
    )

    key, subkey = jr.split(key)

    # Train
    key, subkey = jr.split(key)
    frugal_flow, losses = fit_to_data(
        key=subkey,
        dist=frugal_flow,
        x=jnp.hstack([outcome, u_z]),
        condition=condition,
        optimizer=optimizer,
        show_progress=show_progress,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        max_patience=max_patience,
        batch_size=batch_size,
    )

    return frugal_flow, losses


def train_frugal_flow_gaussian(
    key: jr.PRNGKey,
    y: ArrayLike,
    u_z: ArrayLike,  # impose discrete
    optimizer: optax.GradientTransformation | None = None,
    RQS_knots: int = 8,
    nn_depth: int = 1,
    nn_width: int = 50,
    flow_layers: int = 4,
    show_progress: bool = True,
    learning_rate: float = 5e-4,
    max_epochs: int = 100,
    max_patience: int = 5,
    batch_size: int = 100,
    condition: ArrayLike | None = None,
    mask_condition: bool = True,
    stop_grad_until_active: bool = False,
    causal_model_args: dict | None = None,
):
    nvars = u_z.shape[1]

    if condition is None:
        cond_dim = None
    else:
        cond_dim = condition.shape[1]
    if mask_condition:
        cond_dim_mask = cond_dim
        cond_dim_nomask = None
    else:
        cond_dim_mask = None
        cond_dim_nomask = cond_dim

    list_bijections = [
        UnivariateNormalCDF(
            ate=causal_model_args["ate"],
            scale=causal_model_args["scale"],
            const=causal_model_args["const"],
            cond_dim=cond_dim,
        )
    ] + [Identity(())] * nvars

    marginal_transform = Stack(list_bijections)
    base_dist = Uniform(-jnp.ones(nvars + 1), jnp.ones(nvars + 1))

    transformer = RationalQuadraticSpline(knots=RQS_knots, interval=1)
    if stop_grad_until_active:
        _, stop_grad_until = get_ravelled_pytree_constructor(transformer)
    else:
        stop_grad_until = None

    key, subkey = jr.split(key)
    frugal_flow = masked_autoregressive_flow_first_uniform(
        key=subkey,
        base_dist=base_dist,
        transformer=transformer,
        invert=True,
        cond_dim_mask=cond_dim_mask,
        cond_dim_nomask=cond_dim_nomask,
        nn_depth=nn_depth,
        nn_width=nn_width,
        flow_layers=flow_layers,
        stop_grad_until=stop_grad_until,
    )  # Support on [-1, 1]

    frugal_flow = Transformed(
        frugal_flow,
        Invert(Affine(loc=-jnp.ones(nvars + 1), scale=jnp.ones(nvars + 1) * 2)),
    )
    frugal_flow = Transformed(
        frugal_flow,
        Invert(marginal_transform),
    )
    frugal_flow = frugal_flow.merge_transforms()
    frugal_flow = eqx.tree_at(
        where=lambda frugal_flow: frugal_flow.bijection.bijections[-2],
        pytree=frugal_flow,
        replace_fn=NonTrainable,
    )

    frugal_flow = eqx.tree_at(
        where=lambda frugal_flow: frugal_flow.bijection.bijections[0],
        pytree=frugal_flow,
        replace_fn=NonTrainable,
    )

    key, subkey = jr.split(key)

    # Train
    key, subkey = jr.split(key)
    frugal_flow, losses = fit_to_data(
        key=subkey,
        dist=frugal_flow,
        x=jnp.hstack([y, u_z]),
        condition=condition,
        optimizer=optimizer,
        show_progress=show_progress,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        max_patience=max_patience,
        batch_size=batch_size,
    )

    return frugal_flow, losses


def train_frugal_flow(
    key: jr.PRNGKey,
    y: ArrayLike,
    u_z: ArrayLike,  # impose discrete
    optimizer: optax.GradientTransformation | None = None,
    RQS_knots: int = 8,
    nn_depth: int = 1,
    nn_width: int = 50,
    flow_layers: int = 4,
    show_progress: bool = True,
    learning_rate: float = 5e-4,
    max_epochs: int = 100,
    max_patience: int = 5,
    batch_size: int = 100,
    condition: ArrayLike | None = None,
    mask_condition: bool = True,
    stop_grad_until_active: bool = False,
    causal_model="gaussian",
    causal_model_args: dict | None = None,
):
    valid_causal_models = [
        "gaussian",
        "flexible_discrete_output",
        "location_translation",
    ]
    if causal_model == "gaussian":
        frugal_flow, losses = train_frugal_flow_gaussian(
            key=key,
            y=y,
            u_z=u_z,  # impose discrete
            optimizer=optimizer,
            RQS_knots=RQS_knots,
            nn_depth=nn_depth,
            nn_width=nn_width,
            flow_layers=flow_layers,
            show_progress=show_progress,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            max_patience=max_patience,
            batch_size=batch_size,
            condition=condition,
            mask_condition=mask_condition,
            stop_grad_until_active=stop_grad_until_active,
            causal_model_args=causal_model_args,
        )

    elif causal_model == "flexible_discrete_output":
        frugal_flow, losses = train_frugal_flow_flexible_discrete(
            key=key,
            y=y,
            u_z=u_z,  # impose discrete
            optimizer=optimizer,
            RQS_knots=RQS_knots,
            nn_depth=nn_depth,
            nn_width=nn_width,
            flow_layers=flow_layers,
            show_progress=show_progress,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            max_patience=max_patience,
            batch_size=batch_size,
            condition=condition,
            mask_condition=mask_condition,
            stop_grad_until_active=stop_grad_until_active,
            causal_model_args=causal_model_args,
        )

    elif causal_model == "location_translation":
        frugal_flow, losses = train_frugal_flow_location_translation(
            key=key,
            y=y,
            u_z=u_z,
            optimizer=optimizer,
            RQS_knots=RQS_knots,
            nn_depth=nn_depth,
            nn_width=nn_width,
            flow_layers=flow_layers,
            show_progress=show_progress,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            max_patience=max_patience,
            batch_size=batch_size,
            condition=condition,
            mask_condition=mask_condition,
            stop_grad_until_active=stop_grad_until_active,
            causal_model_args=causal_model_args,
        )

    else:
        raise ValueError(f"Invalid choice. Please choose from: {valid_causal_models}")

    return frugal_flow, losses


def independent_continuous_marginal_flow(
    key: jr.PRNGKey,
    z_cont: ArrayLike,
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
    nvars = z_cont.shape[1]
    key, subkey = jr.split(key)

    base_dist = Uniform(-jnp.ones(nvars), jnp.ones(nvars))

    transformer = RationalQuadraticSpline(knots=RQS_knots, interval=1)

    flow = masked_independent_flow(  # masked_autoregressive_flow(
        key=subkey,
        base_dist=base_dist,
        transformer=transformer,
        flow_layers=flow_layers,
        nn_width=nn_width,
        nn_depth=nn_depth,
    )  # Support on [-1, 1]

    flow = Transformed(flow, Invert(Tanh(flow.shape)))  # Unbounded support

    flow = flow.merge_transforms()

    assert isinstance(flow.base_dist, _StandardUniform)

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


def univariate_discrete_cdf(
    key: jr.PRNGKey,
    z_discr: ArrayLike,
    max_unique_z_discr_size: int,
):
    if z_discr.ndim >= 2:
        _, dim = z_discr.shape
        if dim > 1:
            raise ValueError(
                "input must be 1D with shape (n_samples,) or 2D with shape (n_samples,1)"
            )

    if (z_discr.dtype != "int64") & (z_discr.dtype != "int32"):
        raise ValueError("type of input must be integer")

    n_samples = z_discr.shape[0]
    pmf_keys, pmf_vals = jnp.unique(
        z_discr, return_counts=True, size=max_unique_z_discr_size
    )
    # assert pmf_keys.all() == jnp.arange(max(pmf_keys)+1).all() # check increasing order is respected
    z_discr_empirical_pmf = pmf_vals / n_samples
    z_discr_empirical_cdf_long = z_discr_empirical_pmf.cumsum()

    def uniform_shift(standard_uniform, upper_index):
        # Function to handle the case where upper_index != 0
        def not_zero():
            lower = z_discr_empirical_cdf_long[upper_index - 1]
            upper = z_discr_empirical_cdf_long[upper_index]
            return standard_uniform * (upper - lower) + lower

        # Function to handle the case where upper_index == 0
        def zero():
            upper = z_discr_empirical_cdf_long[upper_index]
            return standard_uniform * upper

        # Using lax.cond to select which function to use
        return jax.lax.cond(upper_index != 0, not_zero, zero)

    uniforms = jr.uniform(key, (n_samples,))
    vmapped_uniform_shift = jax.vmap(uniform_shift)
    u_z_discr = vmapped_uniform_shift(uniforms, z_discr.flatten())

    return u_z_discr, z_discr_empirical_cdf_long


def get_independent_quantiles(
    key: jr.PRNGKey,
    z_discr: ArrayLike | None = None,  # impose discrete
    z_cont: ArrayLike | None = None,
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
    return_z_cont_flow=False,
):
    assert (z_discr is not None) | (z_cont is not None)
    res = {}

    key, subkey = jr.split(key)

    if z_cont is not None:
        partial_univariate_marginal_cdf = partial(
            univariate_marginal_cdf,
            key=subkey,
            optimizer=optimizer,
            RQS_knots=RQS_knots,
            flow_layers=flow_layers,
            nn_width=nn_width,
            nn_depth=nn_depth,
            show_progress=show_progress,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            max_patience=max_patience,
            batch_size=batch_size,
        )
        u_z_cont = []
        z_cont_flows = []
        for i in jnp.arange(z_cont.shape[1]):
            (
                u_z_cont_univariate,
                z_cont_flow_univariate,
            ) = partial_univariate_marginal_cdf(z_cont=z_cont[:, i])
            u_z_cont.append(u_z_cont_univariate)
            z_cont_flows.append(z_cont_flow_univariate)
        u_z_cont = jnp.hstack(u_z_cont)
        res["u_z_cont"] = u_z_cont

        if return_z_cont_flow:
            res["z_cont_flows"] = z_cont_flows

        # z_cont_flow, z_cont_losses = independent_continuous_marginal_flow(
        #     key=key,
        #     z_cont=z_cont,
        #     optimizer=optimizer,
        #     RQS_knots=RQS_knots,
        #     flow_layers=flow_layers,
        #     nn_width=nn_width,
        #     nn_depth=nn_depth,
        #     show_progress=show_progress,
        #     learning_rate=learning_rate,
        #     max_epochs=max_epochs,
        #     max_patience=max_patience,
        #     batch_size=batch_size,
        # )
        # z_cont_marginal_cdf = jax.vmap(z_cont_flow.bijection.inverse, in_axes=(0,))

        # u_z_cont = z_cont_marginal_cdf(z_cont)
        # res["u_z_cont"] = u_z_cont

        # if return_z_cont_flow:
        #     res["z_cont_flow"] = z_cont_flow

    def rankdata(z_disc):
        z_disc_ordered = []
        z_rank_mapping = {}
        for d in range(z_disc.shape[1]):
            z_disc_d = z_disc[:, d]
            unique_z_disc_d = jnp.unique(z_disc_d)
            rank_mapping = {
                k: v
                for k, v in zip(
                    np.array(unique_z_disc_d), np.arange(len(unique_z_disc_d))
                )
            }
            z_disc_new = jnp.array([rank_mapping[i] for i in np.array(z_disc_d)])
            z_disc_ordered.append(z_disc_new)
            z_rank_mapping[d] = rank_mapping
        return jnp.vstack(z_disc_ordered).T, z_rank_mapping

    if z_discr is not None:
        z_discr_ordered, z_discr_rank_mapping = rankdata(z_discr)
        n_discr_ordered = z_discr_ordered.shape[1]
        keys = jr.split(key, n_discr_ordered)
        vmapped_get_discrete_quantiles = jax.vmap(
            univariate_discrete_cdf, in_axes=(0, 1, None)
        )
        u_z_discr_T, z_discr_empirical_cdf_long = vmapped_get_discrete_quantiles(
            keys, z_discr_ordered, len(jnp.unique(z_discr))
        )

        res["z_discr_empirical_cdf_long"] = z_discr_empirical_cdf_long
        res["u_z_discr"] = u_z_discr_T.T
        res["z_discr_rank_mapping"] = z_discr_rank_mapping

    return res


def univariate_marginal_cdf(
    key: jr.PRNGKey,
    z_cont: ArrayLike,
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

    z_cont_flow, z_cont_losses = univariate_marginal_flow(
        key=key,
        z_cont=z_cont,
        optimizer=optimizer,
        RQS_knots=RQS_knots,
        flow_layers=flow_layers,
        nn_width=nn_width,
        nn_depth=nn_depth,
        show_progress=show_progress,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        max_patience=max_patience,
        batch_size=batch_size,
    )
    z_cont_marginal_cdf = jax.vmap(z_cont_flow.bijection.inverse, in_axes=(0,))
    u_z_cont = z_cont_marginal_cdf(z_cont)
    return u_z_cont, z_cont_flow
