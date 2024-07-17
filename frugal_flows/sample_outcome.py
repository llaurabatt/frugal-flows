import inspect
import warnings

import jax
import jax.numpy as jnp
import jax.random as jr
from flowjax.bijections import (
    Affine,
    Invert,
    Tanh,
)
from flowjax.bijections.bijection import AbstractBijection
from flowjax.bijections.utils import Identity
from flowjax.distributions import AbstractDistribution, _StandardUniform
from jaxtyping import ArrayLike

from frugal_flows.bijections import (
    LocCond,
    MaskedAutoregressiveFirstUniform,
    UnivariateNormalCDF,
)


def sample_outcome(
    key: jr.PRNGKey,
    n_samples: float,
    causal_model: str,
    causal_condition: ArrayLike | None = None,
    frugal_flow: AbstractDistribution | None = None,
    causal_cdf: AbstractBijection | None = UnivariateNormalCDF,
    u_yx: ArrayLike | None = None,
    **treatment_kwargs: dict,
):
    """
    Samples outcomes from a given causal model using frugal flows.

    Args:
        key: The PRNGKey for random number generation.
        n_samples: The number of outcome samples to generate.
        causal_model: The causal model to use for outcome generation. Must be one of ["logistic_regression", "causal_cdf", "location_translation"].
        causal_condition: The causal condition to use for outcome generation. Default is None.
        frugal_flow: The frugal flow object to use for outcome generation. Default is None. For causal model "location_translation", a frugal flow object is always required, and if u_yx is also provided, the u_yx quantiles will be used to sample from the flow object. For other causal models, either a frugal flow object or u_yx is required.
        causal_cdf: The causal CDF object to use for outcome generation. Default is UnivariateNormalCDF.
        u_yx: The input samples for the causal model. Default is None, in which case a frugal flow object is always required.
        **treatment_kwargs: Additional keyword arguments for the treatment model.

    Returns:
        outcome_samples: The generated outcome samples.

    Raises:
        ValueError: If the input arguments are invalid or missing.

    """

    valid_causal_models = ["logistic_regression", "causal_cdf", "location_translation"]

    if (u_yx is None) & (frugal_flow is None):
        raise ValueError("Either a frugal flow object or u_yx is required")

    if (
        (u_yx is not None)
        & (frugal_flow is not None)
        & (causal_model != "location_translation")
    ):
        raise ValueError(
            f"Only one between frugal flow object and u_yx can be provided for {causal_model} model"
        )

    if (
        (u_yx is not None)
        & (frugal_flow is not None)
        & (causal_model == "location_translation")
    ):
        # produce a flow_fake_condition even if u_yx is provided as it will be used to sample from the flow object
        flow_dim = frugal_flow.shape[0]
        if frugal_flow.cond_shape is None:
            flow_fake_condition = None
        else:
            flow_fake_condition = jnp.ones((n_samples, frugal_flow.cond_shape[0]))

        warnings.warn(
            f"Since both frugal flow object and u_yx are provided to {causal_model} model, u_yx quantiles will be used to sample from the flow object. If you want to fully sample from the flow object, please provide only the frugal flow object."
        )

    if (causal_model == "location_translation") & (frugal_flow is None):
        raise ValueError(
            f"A frugal flow object is required for simulating outcome with {causal_model} model"
        )
    if u_yx is not None:
        assert len(u_yx) == n_samples
        if (u_yx.min() < 0.0) | (u_yx.max() > 1.0):
            raise ValueError("u_yx input must be between 0. and 1.")
        if causal_model == "location_translation":
            # This model expects the input to be in (-1,1)
            corruni_standard = (u_yx * 2) - 1
        else:
            # This model expects the input to be in (0,1)
            corruni_standard = u_yx

    elif frugal_flow is not None:
        flow_dim = frugal_flow.shape[0]

        if frugal_flow.cond_shape is None:
            flow_fake_condition = None
        else:
            flow_fake_condition = jnp.ones((n_samples, frugal_flow.cond_shape[0]))

        # verify flow has a compatible structure

        assert isinstance(frugal_flow.base_dist, _StandardUniform)
        assert isinstance(frugal_flow.bijection.bijections[0].tree, Affine)
        assert isinstance(
            frugal_flow.bijection.bijections[1].bijection.bijection.bijections[0],
            MaskedAutoregressiveFirstUniform,
        )

        maf_dim = (
            frugal_flow.bijection.bijections[1]
            .bijection.bijection.bijections[0]
            .shape[0]
        )
        spline_n_params = int(
            frugal_flow.bijection.bijections[1]
            .bijection.bijection.bijections[0]
            .masked_autoregressive_mlp.layers[-1]
            .out_features
            / maf_dim
        )
        assert (
            frugal_flow.bijection.bijections[1]
            .bijection.bijection.bijections[0]
            .transformer_constructor(jnp.ones((spline_n_params)))
            .interval
            == 1
        )
        try:
            assert isinstance(
                frugal_flow.bijection.bijections[2].tree.bijections[0], Identity
            )
        except Exception:
            assert (isinstance(frugal_flow.bijection.bijections[2].tree, Invert)) & (
                isinstance(frugal_flow.bijection.bijections[2].tree.bijection, Affine)
            )

        # obtain u_y samples from flow
        uni_standard = jr.uniform(key, shape=(n_samples, flow_dim))
        uni_minus1_plus1 = jax.vmap(frugal_flow.bijection.bijections[0].tree.transform)(
            uni_standard
        )

        corruni_minus1_plus1 = jax.vmap(frugal_flow.bijection.bijections[1].transform)(
            uni_minus1_plus1, flow_fake_condition
        )
        corruni = jax.vmap(frugal_flow.bijection.bijections[2].tree.transform)(
            corruni_minus1_plus1, flow_fake_condition
        )
        corruni_y = corruni[:, 0]

        try:
            # in this case the flow expects the input to be in (-1,1)
            assert isinstance(
                frugal_flow.bijection.bijections[2].tree.bijections[0], Identity
            )
            corruni_standard = corruni_y
        except Exception:
            # in this case the flow expects the input to be in (0,1)
            assert (isinstance(frugal_flow.bijection.bijections[2].tree, Invert)) & (
                isinstance(frugal_flow.bijection.bijections[2].tree.bijection, Affine)
            )
            corruni_standard = (corruni_y / 2) + 0.5

    if causal_model == "logistic_regression":
        outcome_samples = logistic_outcome(
            u_y=corruni_standard,
            causal_condition=causal_condition,
            **treatment_kwargs,
        )

    elif causal_model == "causal_cdf":
        outcome_samples, _ = causal_cdf_outcome(
            u_y=corruni_standard,
            causal_condition=causal_condition,
            causal_cdf=causal_cdf,
            **treatment_kwargs,
        )

    elif causal_model == "location_translation":
        try:
            assert isinstance(
                frugal_flow.bijection.bijections[4].bijections[0].bijection, Tanh
            )
        except Exception:
            raise ValueError(
                f"{causal_model} causal_model requires a 'location_translation' pretrained frugal_flow"
            )

        outcome_samples = location_translation_outcome(
            u_y=corruni_standard,
            causal_condition=causal_condition,
            flow_condition=flow_fake_condition,
            frugal_flow=frugal_flow,
            **treatment_kwargs,
        )
    else:
        raise ValueError(
            f"Invalid causal_model choice. Please choose from: {valid_causal_models}"
        )

    return outcome_samples


def logistic_outcome(
    u_y: ArrayLike, ate: float, causal_condition: ArrayLike, const: float
):
    """
    Computes the logistic outcome based on the given inputs.

    Args:
        u_y: The input quantiles, of shape (n_samples,)
        ate: The average treatment effect. Float.
        causal_condition: The (univariate) causal condition. It is an Array with shape (n_samples, 1) or (n_samples,).
        const: The constant term. Float.

    Returns:
        The computed logistic outcome.

    """

    def get_y(u_y, ate, x, const):
        p = jax.nn.sigmoid(ate * x + const)
        return (u_y >= (1 - p)).astype(int)

    return jax.vmap(get_y, in_axes=(0, None, 0, None))(
        u_y, ate, causal_condition, const
    )


def causal_cdf_outcome(
    u_y: ArrayLike,
    causal_cdf: AbstractBijection,
    causal_condition: ArrayLike,
    **treatment_kwargs: dict,
):
<<<<<<< HEAD
    if causal_condition is not None:
        if causal_condition.ndim == 1:
            # Reshape one-dimensional array to two dimensions with second dim as 1
            causal_condition = causal_condition.reshape(-1, 1)

    if 'cond_dim' not in treatment_kwargs.keys():
        treatment_kwargs['cond_dim'] = causal_condition.shape[1]
        
=======
    """
    Compute the outcome samples and the causal CDF.

    Args:
        u_y (ArrayLike): The input quantiles, of shape (n_samples,)
        causal_cdf (AbstractBijection): The causal CDF.
        causal_condition (ArrayLike): The causal condition.
        **treatment_kwargs (dict): Additional keyword arguments for the causal CDF.

    Returns:
        tuple: A tuple containing the outcome samples and the causal CDF.
    """
>>>>>>> add comments to sample outcome and cleanup
    causal_cdf_init_params = [
        i
        for i in inspect.signature(causal_cdf.__init__).parameters.keys()
        if ((i != "self") and (i != "cond_dim"))
    ]
    for param in causal_cdf_init_params:
        if param not in treatment_kwargs.keys():
            treatment_kwargs[param] = None
            warnings.warn(
                f"The parameter {param} has not been provided and is therefore set to None."
            )

    causal_cdf_simulate = causal_cdf(**treatment_kwargs)
    samples = jax.vmap(causal_cdf_simulate.inverse)(u_y, causal_condition)
    return samples, causal_cdf_simulate


def location_translation_outcome(
    u_y: ArrayLike,
    frugal_flow: AbstractDistribution,
    causal_condition: ArrayLike,
    flow_condition: ArrayLike,
    **treatment_kwargs: dict,
):
    """
    Compute the outcome samples for the location_translation causal model.

    Args:
        u_y (ArrayLike): The input quantiles, of shape (n_samples,)
        frugal_flow (AbstractDistribution): The frugal flow object.
        causal_condition (ArrayLike): The causal condition.
        flow_condition (ArrayLike): The flow condition.
        **treatment_kwargs (dict): Additional keyword arguments for the treatment model.

    Returns:
        ArrayLike: The generated outcome samples.

    """

    causal_minus1_plus1 = jax.vmap(
        frugal_flow.bijection.bijections[3].bijections[0].transform
    )(u_y[:, None], flow_condition)
    causal_reals = jax.vmap(
        frugal_flow.bijection.bijections[4].bijections[0].transform
    )(causal_minus1_plus1.flatten(), flow_condition)

    loc_cond_cdf_simulate = LocCond(**treatment_kwargs)
    samples = jax.vmap(loc_cond_cdf_simulate.transform)(causal_reals, causal_condition)
    return samples
