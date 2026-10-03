"""Math-grounded tests for ``frugal_flows.bijections.LocCond``.

``LocCond`` is the bijection that injects the causal effect into the frugal
flow. Its contract (see the class docstring):

    forward:  y = x + ate * condition[0]
    inverse:  x = y - ate * condition[0]
    log|det J| = 0  in both directions (pure translation)

Only ``condition[0]`` is ever used. Nothing here is learned or random, so
these are exact-equality / closed-form checks, not statistical ones.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from frugal_flows.bijections import LocCond


def test_forward_is_exact_location_shift(key):
    """forward(x, c) - x == ate * c[0], exactly, for the load-bearing claim."""
    ate = 2.5
    bij = LocCond(ate=ate, cond_dim=1)
    x = jr.normal(key, ())
    condition = jnp.array([0.7])
    y = bij.transform(x, condition)
    assert y == pytest.approx(float(x) + ate * 0.7, rel=0, abs=1e-12)
    # The defining property, stated directly:
    assert (y - x) == pytest.approx(ate * condition[0], rel=0, abs=1e-12)


def test_round_trip_is_exact(key):
    """inverse(transform(x)) == x and transform(inverse(y)) == y, exactly."""
    bij = LocCond(ate=1.3, cond_dim=1)
    k1, k2 = jr.split(key)
    x = jr.normal(k1, ())
    condition = jr.normal(k2, (1,))

    x_rt = bij.inverse(bij.transform(x, condition), condition)
    assert x_rt == pytest.approx(float(x), rel=0, abs=1e-12)

    y = jr.normal(k1, ())
    y_rt = bij.transform(bij.inverse(y, condition), condition)
    assert y_rt == pytest.approx(float(y), rel=0, abs=1e-12)


def test_log_det_is_zero_both_directions(key):
    """LocCond is a pure translation, so log|det J| is exactly 0."""
    bij = LocCond(ate=3.14, cond_dim=1)
    k1, k2 = jr.split(key)
    x = jr.normal(k1, ())
    condition = jr.normal(k2, (1,))

    y_fwd, ld_fwd = bij.transform_and_log_det(x, condition)
    x_inv, ld_inv = bij.inverse_and_log_det(y_fwd, condition)

    assert ld_fwd == 0.0
    assert ld_inv == 0.0
    # *_and_log_det must agree with the plain transform/inverse.
    assert y_fwd == bij.transform(x, condition)
    assert x_inv == bij.inverse(y_fwd, condition)
    assert x_inv == pytest.approx(float(x), rel=0, abs=1e-12)


def test_only_first_condition_component_is_used(key):
    """Documented behaviour: extra conditioning dims are ignored."""
    bij = LocCond(ate=2.0, cond_dim=3)
    x = jr.normal(key, ())
    c_a = jnp.array([0.5, 1.0, -2.0])
    c_b = jnp.array([0.5, 99.0, 42.0])  # same [0], different tail
    assert bij.transform(x, c_a) == bij.transform(x, c_b)


def test_zero_ate_is_identity(key):
    """ate=0 (the default) leaves x untouched regardless of condition."""
    bij = LocCond()  # ate defaults to 0
    k1, k2 = jr.split(key)
    x = jr.normal(k1, ())
    condition = jr.normal(k2, (1,))
    assert bij.transform(x, condition) == x


def test_vmapped_batch_matches_sample_outcome_usage(key):
    """Mirror sample_outcome.location_translation_outcome's call pattern:

    jax.vmap(bij.transform)(causal_reals, causal_condition), where
    causal_condition is (n, k) and only column 0 (treatment) drives the shift.
    """
    ate = 1.75
    bij = LocCond(ate=ate, cond_dim=1)
    k1, k2 = jr.split(key)
    n = 256
    causal_reals = jr.normal(k1, (n,))
    treatment = jr.bernoulli(k2, 0.5, (n,)).astype(jnp.float64)
    causal_condition = treatment[:, None]  # (n, 1)

    samples = jax.vmap(bij.transform)(causal_reals, causal_condition)
    expected = causal_reals + ate * treatment
    assert jnp.allclose(samples, expected, rtol=0, atol=1e-12)
    # Empirically, the mean gap between treated and control equals `ate`.
    gap = samples[treatment == 1].mean() - samples[treatment == 0].mean()
    control_mean = causal_reals[treatment == 0].mean()
    treated_latent_mean = causal_reals[treatment == 1].mean()
    assert gap == pytest.approx(ate + (treated_latent_mean - control_mean), abs=1e-10)
