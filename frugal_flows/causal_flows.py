import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from flowjax.bijections import (
    Affine,
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
    masked_autoregressive_flow_first_uniform,
    masked_independent_flow,
)
from frugal_flows.bijections import UnivariateNormalCDF


def train_copula_flow(
    key: jr.PRNGKey,
    u_z: ArrayLike,  # impose discrete
    optimizer: optax.GradientTransformation | None = None,
    RQS_knots: int = 8,
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
    stop_grad_until_active: bool = False,
):
    nvars = u_z.shape[1]
    key, subkey = jr.split(key)

    if condition is None:
        cond_dim = None
    else:
        cond_dim = condition.shape[1]
    list_bijections = [
        UnivariateNormalCDF(ate=5.0, scale=2.0, const=5.0, cond_dim=cond_dim)
    ] + [Identity(())] * nvars
    marginal_transform = Stack(list_bijections)

    base_dist = Uniform(-jnp.ones(nvars + 1), jnp.ones(nvars + 1))

    transformer = RationalQuadraticSpline(knots=RQS_knots, interval=1)
    if stop_grad_until_active:
        _, stop_grad_until = get_ravelled_pytree_constructor(transformer)
    else:
        stop_grad_until = None

    frugal_flow = (
        masked_autoregressive_flow_first_uniform(  # masked_autoregressive_flow(
            key=subkey,
            base_dist=base_dist,
            transformer=transformer,
            invert=True,
            cond_dim_mask=cond_dim,
            nn_depth=nn_depth,
            nn_width=nn_width,
            flow_layers=flow_layers
            stop_grad_until=stop_grad_until,
            # cond_dim_nomask=x.shape[1],
            # cond_dim=x.shape[1],
        )
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

    assert isinstance(frugal_flow.base_dist, _StandardUniform)

    frugal_flow = eqx.tree_at(
        where=lambda frugal_flow: frugal_flow.bijection.bijections[0],
        pytree=frugal_flow,
        replace_fn=NonTrainable,
    )

    frugal_flow = eqx.tree_at(
        where=lambda frugal_flow: frugal_flow.bijection.bijections[-2],
        pytree=frugal_flow,
        replace_fn=NonTrainable,
    )

    key, subkey = jr.split(key)

    # Train
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
        nn_depth=nn_depth
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
    u_z_discr = vmapped_uniform_shift(uniforms, z_discr)

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
        z_cont_flow, z_cont_losses = independent_continuous_marginal_flow(
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
        res["u_z_cont"] = u_z_cont

        if return_z_cont_flow:
            res["z_cont_flow"] = z_cont_flow

    if z_discr is not None:
        n_discr = z_discr.shape[1]
        keys = jr.split(key, n_discr)
        vmapped_get_discrete_quantiles = jax.vmap(
            univariate_discrete_cdf, in_axes=(0, 1, None)
        )
        u_z_discr_T, z_discr_empirical_cdf_long = vmapped_get_discrete_quantiles(
            keys, z_discr, len(jnp.unique(z_discr))
        )

        res["z_discr_empirical_cdf_long"] = z_discr_empirical_cdf_long
        res["u_z_discr"] = u_z_discr_T.T

    return res
