import jax
import jax.numpy as jnp
import jax.random as jr
from flowjax.distributions import AbstractDistribution
from jax import Array


def from_quantiles_to_marginal_cont(
    key: jr.PRNGKey,
    flow: list | AbstractDistribution,
    n_samples: int,
    u_z: Array | None = None,
):
    """
    Convert quantiles to continuous marginal samples using the specified flow.

    Args:
        key (jr.PRNGKey): The random key for generating samples.
        flow (list | AbstractDistribution): The flow or list of flows to use for conversion.
        n_samples (int): The number of samples to generate.
        u_z (Array | None, optional): The quantiles to convert. Defaults to None, in which case independent standard uniforms are used.

    Returns:
        Array: The generated continuous marginal samples.

    Raises:
        ValueError: If an invalid flow object or list of flow objects is passed.
    """

    # Reshape one-dimensional array to two dimensions with second dim as 1
    if u_z is not None:
        if u_z.ndim == 1:
            u_z = u_z.reshape(-1, 1)

    if isinstance(flow, list):
        marginal_samples = []
        if u_z is not None:
            unis_standard = u_z
        else:
            nvars = len(flow)
            key, subkey = jr.split(key)
            unis_standard = jr.uniform(subkey, shape=(n_samples, nvars))

        for i, flow_i in enumerate(flow):
            flow_dim = 1
            uni_standard = unis_standard[:, i, None]
            uni_minus1_plus1 = jax.vmap(flow_i.bijection.bijections[0].tree.transform)(
                uni_standard
            )

            corruni_minus1_plus1 = jax.vmap(flow_i.bijection.bijections[1].transform)(
                uni_minus1_plus1
            )
            marginal_sample = jax.vmap(flow_i.bijection.bijections[2].transform)(
                corruni_minus1_plus1
            )
            marginal_samples.append(marginal_sample)
        marginal_samples = jnp.hstack(marginal_samples)
        return marginal_samples

    elif isinstance(flow, AbstractDistribution):
        flow_dim = flow.shape[0]
        if u_z is not None:
            uni_standard = u_z
        else:
            key, subkey = jr.split(key)
            uni_standard = jr.uniform(subkey, shape=(n_samples, flow_dim))
        uni_minus1_plus1 = jax.vmap(flow.bijection.bijections[0].tree.transform)(
            uni_standard
        )

        corruni_minus1_plus1 = jax.vmap(flow.bijection.bijections[1].transform)(
            uni_minus1_plus1
        )
        marginal_sample = jax.vmap(flow.bijection.bijections[2].transform)(
            corruni_minus1_plus1
        )
        return marginal_sample
    else:
        raise ValueError(
            "Pass valid flow object or list of valid univariate flow objects"
        )


def from_quantiles_to_marginal_discr(
    key: jr.PRNGKey,
    mappings: dict,
    nvars: int,
    empirical_cdfs: Array,
    n_samples: int,
    u_z: Array | None = None,
):
    """
    Convert quantiles to discrete marginal samples using the specified mappings.

    Args:
        key (jr.PRNGKey): The random key for generating samples.
        mappings (dict): Mapping variable categories to indices 0,1,...n_categories-1. It must be of the form {0:dict_0, 1:dict_1,...,nvars-1:dict_nvars-1} where dict_i maps the categories of the i-th variable to index values 0,1,.. n_categories-1.
        nvars (int): The number of variables.
        empirical_cdfs (Array): The empirical cumulative distribution functions. It must be of shape (n_vars, max_n_categories), where max_n_categories is the maximum number of categories among all variables. If a variable has fewer categories, a padding of 1.s should be added to the right.
        n_samples (int): The number of samples to generate.
        u_z (Array | None, optional): The quantiles to convert, with shape (n_samples, n_vars). Defaults to None, in which case independent standard uniforms are used.

    Returns:
        Array: The generated discrete marginal samples.

    Raises:
        ValueError: If an invalid mapping object is passed.
    """
    if u_z is not None:
        if u_z.ndim == 1:
            # Reshape one-dimensional array to two dimensions with second dim as 1
            u_z = u_z.reshape(-1, 1)

        unis_standard = u_z
    else:
        key, subkey = jr.split(key)
        unis_standard = jr.uniform(subkey, shape=(n_samples, nvars))

    keys = jr.split(key, nvars)
    try:
        z_discr_rank_mapping_array = jnp.vstack(
            [jnp.array(list(d.keys())) for d in mappings.values()]
        )
        vmapped_from_quantiles_to_marginal_discr = jax.vmap(
            univariate_from_quantiles_to_marginal_discr, in_axes=(0, 0, None, 0, 0)
        )

        marginal_samples_discr = vmapped_from_quantiles_to_marginal_discr(
            keys,
            empirical_cdfs,
            n_samples,
            z_discr_rank_mapping_array,
            unis_standard.T,
        )
    except Exception:
        marginal_samples_discr = []
        for d, mapping_d in enumerate(mappings.values()):
            z_discr_rank_mapping_array_d = jnp.array(list(mapping_d.keys()))

            marginal_samples_discr_d = univariate_from_quantiles_to_marginal_discr(
                keys[d],
                empirical_cdfs[d],
                n_samples,
                z_discr_rank_mapping_array_d,
                unis_standard.T[d],
            )
            marginal_samples_discr.append(marginal_samples_discr_d)
        marginal_samples_discr = jnp.vstack(marginal_samples_discr)
    return marginal_samples_discr.T


def univariate_from_quantiles_to_marginal_discr(
    cdf_levels: Array,
    key_mapping: Array,
    uni_standard,
):
    """
    Converts a univariate random variable from quantiles to a discrete marginal representation.

    Args:
        cdf_levels (Array): The cumulative distribution function (CDF) levels.
        key_mapping (Array): A mapping array of length (n_categories - 1 ) where the entries correspond to the variable categories. The order of the categories should match that in CDF_levels.
        uni_standard: The univariate random variable in standard uniform distribution.

    Returns:
        Array: The discrete marginal representation of the univariate random variable.
    """

    # uni_standard = jr.uniform(key, shape=(n_samples, 1))
    uni_standard = jnp.expand_dims(uni_standard, axis=1)
    comparisons = uni_standard > cdf_levels
    marginal_discr_sample = comparisons.sum(axis=1)

    def map_values(z_discr, mapping_array):
        # Use the values in z_discr as indices to access the mapping_array
        z_discr_mapped = mapping_array[z_discr]
        return z_discr_mapped

    marginal_discr_sample = map_values(marginal_discr_sample, key_mapping)
    return marginal_discr_sample
