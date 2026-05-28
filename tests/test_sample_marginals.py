"""Tests for ``sample_marginals`` (inverse PIT: quantiles -> marginal samples).

Discrete path: correctness of the empirical-CDF inverse, the vmapped fast
path (equal category counts across variables), and the loop fallback
(unequal category counts, when the rank-mapping arrays can't be stacked).

Continuous path: ``from_quantiles_to_marginal_cont`` builds samples by
walking a trained ``univariate_marginal_flow`` at three hardcoded chain
positions. Both call shapes (single flow, list of flows) are exercised
end-to-end on a small trained flow; outputs are asserted finite and
shape-correct.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from frugal_flows.basic_flows import univariate_marginal_flow
from frugal_flows.sample_marginals import (
    from_quantiles_to_marginal_cont,
    from_quantiles_to_marginal_discr,
    univariate_from_quantiles_to_marginal_discr,
)


def test_univariate_discrete_inverse_cdf_is_correct():
    """u > cdf_levels, summed, indexes the category. CDF [0.5,0.8,1.0],
    labels {10,20,30}: u=0.3->10, 0.6->20, 0.9->30, 0.99->30."""
    cdf_levels = jnp.array([0.5, 0.8, 1.0])
    key_mapping = jnp.array([10, 20, 30])
    u = jnp.array([0.3, 0.6, 0.9, 0.99])
    out = univariate_from_quantiles_to_marginal_discr(cdf_levels, key_mapping, u)
    assert jnp.array_equal(out, jnp.array([10, 20, 30, 30]))


def test_from_quantiles_to_marginal_discr_single_variable():
    """End-to-end discrete inverse PIT with one variable."""
    mappings = {0: {10: 0, 20: 1, 30: 2}}
    empirical_cdfs = jnp.array([[0.5, 0.8, 1.0]])
    u_z = jnp.array([[0.3], [0.6], [0.9]])
    out = from_quantiles_to_marginal_discr(
        key=jr.PRNGKey(0),
        mappings=mappings,
        nvars=1,
        empirical_cdfs=empirical_cdfs,
        n_samples=3,
        u_z=u_z,
    )
    assert out.shape == (3, 1)
    assert jnp.array_equal(out.ravel(), jnp.array([10, 20, 30]))


def test_fast_path_correct_on_equal_category_counts():
    """Two variables with the same number of categories: rank-mapping
    arrays stack cleanly, fast vmapped path runs, output matches the
    explicit per-variable evaluation."""
    mappings = {0: {10: 0, 20: 1, 30: 2}, 1: {100: 0, 200: 1, 300: 2}}
    empirical_cdfs = jnp.array([[0.5, 0.8, 1.0], [0.4, 0.7, 1.0]])
    u_z = jnp.array([[0.3, 0.5], [0.6, 0.6], [0.9, 0.95]])
    out = from_quantiles_to_marginal_discr(
        key=jr.PRNGKey(0),
        mappings=mappings,
        nvars=2,
        empirical_cdfs=empirical_cdfs,
        n_samples=3,
        u_z=u_z,
    )
    expected = jnp.stack(
        [
            univariate_from_quantiles_to_marginal_discr(
                empirical_cdfs[d],
                jnp.array(list(mapping_d.keys())),
                u_z.T[d],
            )
            for d, mapping_d in enumerate(mappings.values())
        ]
    ).T
    assert out.shape == (3, 2)
    assert jnp.array_equal(out, expected)


def test_loop_fallback_runs_on_unequal_category_counts():
    """Two variables with different category counts (2 and 3): the
    rank-mapping arrays can't be stacked, so the fast path raises
    ValueError and the Python-loop fallback runs. Output still matches
    the explicit per-variable evaluation."""
    mappings = {0: {10: 0, 20: 1}, 1: {100: 0, 200: 1, 300: 2}}
    # empirical_cdfs is padded to (n_vars, max_n_categories) with trailing 1s.
    empirical_cdfs = jnp.array([[0.5, 1.0, 1.0], [0.4, 0.7, 1.0]])
    u_z = jnp.array([[0.3, 0.5], [0.6, 0.6], [0.9, 0.95]])
    out = from_quantiles_to_marginal_discr(
        key=jr.PRNGKey(0),
        mappings=mappings,
        nvars=2,
        empirical_cdfs=empirical_cdfs,
        n_samples=3,
        u_z=u_z,
    )
    expected = jnp.stack(
        [
            univariate_from_quantiles_to_marginal_discr(
                empirical_cdfs[d],
                jnp.array(list(mapping_d.keys())),
                u_z.T[d],
            )
            for d, mapping_d in enumerate(mappings.values())
        ]
    ).T
    assert out.shape == (3, 2)
    assert jnp.array_equal(out, expected)


# --- continuous path ------------------------------------------------------


def _make_univariate_flow(key, z_cont):
    flow, _ = univariate_marginal_flow(
        key=key,
        z_cont=z_cont,
        RQS_knots=4,
        flow_layers=2,
        nn_width=4,
        nn_depth=1,
        max_epochs=1,
        max_patience=1,
        batch_size=16,
        show_progress=False,
    )
    return flow


def test_from_quantiles_to_marginal_cont_single_flow_runs():
    """``from_quantiles_to_marginal_cont`` on a single trained
    ``univariate_marginal_flow`` produces finite samples of the correct
    shape. Exercises the hardcoded ``bijections[0|1|2].transform`` chain
    walk on the AbstractDistribution branch."""
    k_train, k_u = jr.split(jr.PRNGKey(0))
    n = 16
    z_cont = jr.normal(k_train, (32,))
    flow = _make_univariate_flow(k_train, z_cont)
    u_z = jr.uniform(k_u, (n, 1))
    out = from_quantiles_to_marginal_cont(
        key=jr.PRNGKey(1), flow=flow, n_samples=n, u_z=u_z
    )
    assert out.shape == (n, 1)
    assert bool(jnp.isfinite(out).all())


def test_from_quantiles_to_marginal_cont_list_of_flows_runs():
    """List-of-flows branch: same trained flow reused twice. Output
    concatenates each per-flow sample column-wise into ``(n_samples,
    n_flows)``."""
    k_train, k_u = jr.split(jr.PRNGKey(0))
    n = 16
    z_cont = jr.normal(k_train, (32,))
    flow = _make_univariate_flow(k_train, z_cont)
    flows = [flow, flow]
    u_z = jr.uniform(k_u, (n, 2))
    out = from_quantiles_to_marginal_cont(
        key=jr.PRNGKey(1), flow=flows, n_samples=n, u_z=u_z
    )
    assert out.shape == (n, 2)
    assert bool(jnp.isfinite(out).all())
