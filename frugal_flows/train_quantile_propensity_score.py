import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax
from flowjax.bijections import (
    Affine,
    Invert,
    RationalQuadraticSpline,
)
from flowjax.distributions import Transformed, Uniform, _StandardUniform
from flowjax.flows import masked_autoregressive_flow
from flowjax.train import fit_to_data
from flowjax.wrappers import NonTrainable
from jaxtyping import Array

from frugal_flows.causal_flows import univariate_discrete_cdf


def train_quantile_propensity_score(
    key: jr.PRNGKey,
    x: Array,
    condition: Array,
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
    return_x_quantiles: bool = False,
):
    nvars = condition.shape[1]

    key, subkey = jr.split(key)
    u_x, _ = univariate_discrete_cdf(
        key=subkey, z_discr=x, max_unique_z_discr_size=len(jnp.unique(x))
    )

    nvars = u_x[:, None].shape[1]
    base_dist = Uniform(-jnp.ones(nvars), jnp.ones(nvars))

    transformer = RationalQuadraticSpline(knots=RQS_knots, interval=1)

    key, subkey = jr.split(key)
    flow = masked_autoregressive_flow(
        key=subkey,
        base_dist=base_dist,
        transformer=transformer,
        flow_layers=flow_layers,
        nn_width=nn_width,
        nn_depth=nn_depth,
        cond_dim=condition.shape[1],
    )

    flow = Transformed(
        flow,
        Invert(Affine(loc=-jnp.ones(nvars), scale=jnp.ones(nvars) * 2)),
    )

    flow = flow.merge_transforms()

    assert isinstance(flow.base_dist, _StandardUniform)

    flow = eqx.tree_at(
        where=lambda flow: flow.bijection.bijections[0],
        pytree=flow,
        replace_fn=NonTrainable,
    )

    flow = eqx.tree_at(
        where=lambda flow: flow.bijection.bijections[-1],
        pytree=flow,
        replace_fn=NonTrainable,
    )

    # Train
    key, subkey = jr.split(key)
    flow, losses = fit_to_data(
        key=subkey,
        dist=flow,
        x=u_x[:, None],
        learning_rate=learning_rate,
        max_patience=max_patience,
        max_epochs=max_epochs,
        condition=condition,
        optimizer=optimizer,
        show_progress=show_progress,
        batch_size=batch_size,
    )

    if return_x_quantiles:
        return flow, losses, u_x
    else:
        return flow, losses
