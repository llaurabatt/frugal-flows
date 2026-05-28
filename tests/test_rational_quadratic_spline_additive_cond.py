"""Math-grounded tests for ``RationalQuadraticSplineAdditiveCond``.

Contract:

    forward:  y = RQS(x) + ate * condition[0]   (identity tails, ate term
                                                   dropped when condition is None)
    inverse:  x = RQS⁻¹(y - ate * condition[0])
    forward log|det J| = log RQS'(x)             (ate term has zero dy/dx —
                                                   "Additive" in the class
                                                   name is load-bearing)
    inverse log|det J| = -log RQS'(x)            (x is the RQS-input preimage)

NOTE: an untrained spline initialises to the identity (all derivatives = 1),
so a non-identity spline is required to exercise the derivative-dependent
paths. ``_spline`` perturbs the trainable params so RQS' is non-constant.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from paramax import unwrap

from frugal_flows.bijections import RationalQuadraticSplineAdditiveCond


def _spline(ate=0.0, *, perturb=True, seed=1):
    """Non-identity RQS: perturb every trainable param, then unwrap."""
    bij = RationalQuadraticSplineAdditiveCond(knots=8, interval=4.0, ate=ate)
    if perturb:
        params, static = eqx.partition(bij, eqx.is_inexact_array)
        flat, treedef = jax.tree_util.tree_flatten(params)
        flat = [
            f + 0.7 * jr.normal(jr.fold_in(jr.PRNGKey(seed), i), f.shape)
            for i, f in enumerate(flat)
        ]
        bij = eqx.combine(jax.tree_util.tree_unflatten(treedef, flat), static)
    return unwrap(bij)


def test_spline_is_non_identity():
    """Guard: the perturbed spline really is non-identity (else other tests
    would silently not exercise the derivative paths)."""
    bij = _spline()
    xs = jnp.linspace(-3.5, 3.5, 25)
    ys = jax.vmap(bij.transform)(xs)
    assert not jnp.allclose(ys, xs, atol=1e-3)


def test_round_trip_no_condition(key):
    bij = _spline()
    xs = jr.uniform(key, (64,), minval=-3.5, maxval=3.5)
    rt = jax.vmap(lambda x: bij.inverse(bij.transform(x)))(xs)
    assert jnp.allclose(rt, xs, rtol=0, atol=1e-5)


def test_round_trip_with_condition(key):
    bij = _spline(ate=0.9)
    xs = jr.uniform(key, (64,), minval=-3.0, maxval=3.0)
    c = jnp.array([1.0])
    rt = jax.vmap(lambda x: bij.inverse(bij.transform(x, c), c))(xs)
    assert jnp.allclose(rt, xs, rtol=0, atol=1e-5)


def test_identity_tails():
    """Outside [-interval, interval] the spline is the identity (+ ate shift)."""
    bij = _spline(ate=0.5)
    x = jnp.array(9.0)  # well outside interval=4
    assert bij.transform(x) == x
    c = jnp.array([1.0])
    assert bij.transform(x, c) == x + 0.5 * 1.0


def test_forward_log_det_matches_autodiff(key):
    bij = _spline(ate=0.7)
    x = jr.uniform(key, (), minval=-3.0, maxval=3.0)
    c = jnp.array([1.0])
    y, log_det = bij.transform_and_log_det(x, c)
    dydx = jax.grad(lambda x_: bij.transform(x_, c))(x)
    assert y == bij.transform(x, c)
    assert log_det == pytest.approx(float(jnp.log(jnp.abs(dydx))), rel=0, abs=1e-5)


def test_inverse_log_det_correct_when_ate_zero(key):
    """With ate=0 the inverse log-det is correct: matches autodiff of inverse."""
    bij = _spline(ate=0.0)
    y = jr.uniform(key, (), minval=-3.0, maxval=3.0)
    _, ld_inv = bij.inverse_and_log_det(y)
    dxdy = jax.grad(bij.inverse)(y)
    assert ld_inv == pytest.approx(float(jnp.log(jnp.abs(dxdy))), rel=0, abs=1e-5)


def test_inverse_log_det_correct_with_nonzero_ate(key):
    """Ground truth = autodiff of inverse. With the bug-2 fix,
    ``inverse_and_log_det`` evaluates ``RQS'`` at the recovered RQS-input
    preimage (not at the wrongly-double-shifted point), so its log-det matches
    ``log|d inverse/dy|`` even when ``ate * condition[0] != 0``.
    """
    bij = _spline(ate=0.8)
    c = jnp.array([1.0])
    y = jr.uniform(key, (), minval=-3.0, maxval=3.0)
    _, ld_inv = bij.inverse_and_log_det(y, c)
    dxdy = jax.grad(lambda y_: bij.inverse(y_, c))(y)
    assert ld_inv == pytest.approx(float(jnp.log(jnp.abs(dxdy))), rel=0, abs=1e-5)


def test_inverse_log_det_matches_negative_forward_log_det(key):
    """Inverse-function theorem: at ``x = inverse(y, c)``,
    ``ld_inv(y, c) == -ld_fwd(x, c)``. Independent check of the bug-2 fix."""
    bij = _spline(ate=0.8)
    c = jnp.array([1.0])
    y = jr.uniform(key, (), minval=-3.0, maxval=3.0)
    x, ld_inv = bij.inverse_and_log_det(y, c)
    _, ld_fwd = bij.transform_and_log_det(x, c)
    assert ld_inv == pytest.approx(float(-ld_fwd), rel=0, abs=1e-5)
