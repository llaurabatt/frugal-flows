"""Math-grounded tests for ``frugal_flows.bijections.UnivariateNormalCDF``.

Contract (see the class docstring):

    loc = ate @ condition + const   (when cond_dim is not None)
    loc = const                     (when cond_dim is None)
    forward:  y = Φ((x - loc) / scale)
    inverse:  x = Φ⁻¹(y) * scale + loc
    forward log|det J| = log φ((x-loc)/scale) - log scale
    inverse log|det J| = log scale - log φ(Φ⁻¹(y))

We verify forward == Φ and inverse == Φ⁻¹ on a grid, exact round-trips, the
log-dets against autodiff (the gold standard), the ATE-injection semantics, and
the unconditional ``cond_dim=None`` path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from frugal_flows.bijections import UnivariateNormalCDF


def _make(ate=0.5, scale=1.3, const=0.2):
    """Conditional instance with ate.shape == (1,) so the assertion passes."""
    return UnivariateNormalCDF(
        ate=jnp.array([ate]), scale=scale, const=const, cond_dim=1
    )


def test_forward_is_the_gaussian_cdf_on_a_grid():
    """transform(x, c) == Φ((x - (ate·c + const)) / scale) over a grid."""
    ate, scale, const = 0.7, 1.4, -0.3
    bij = _make(ate, scale, const)
    condition = jnp.array([1.0])
    xs = jnp.linspace(-6.0, 6.0, 201)

    got = jax.vmap(lambda x: bij.transform(x, condition))(xs)
    loc = ate * 1.0 + const
    expected = jax.scipy.stats.norm.cdf(xs, loc=loc, scale=scale)
    assert jnp.allclose(got.squeeze(), expected, rtol=0, atol=1e-10)
    # Output is a valid CDF: in (0,1) and monotone increasing.
    assert jnp.all((got > 0) & (got < 1))
    assert jnp.all(jnp.diff(got.squeeze()) > 0)


def test_inverse_is_the_gaussian_quantile_on_a_grid():
    """inverse(y, c) == Φ⁻¹(y)·scale + (ate·c + const) over a grid in (0,1)."""
    ate, scale, const = -0.4, 0.9, 1.1
    bij = _make(ate, scale, const)
    condition = jnp.array([1.0])
    ys = jnp.linspace(1e-4, 1 - 1e-4, 201)

    got = jax.vmap(lambda y: bij.inverse(y, condition))(ys)
    loc = ate * 1.0 + const
    expected = jax.scipy.special.ndtri(ys) * scale + loc
    assert jnp.allclose(got.squeeze(), expected, rtol=0, atol=1e-10)


def test_round_trip_both_directions(key):
    """inverse(transform(x)) == x and transform(inverse(y)) == y."""
    bij = _make()
    k1, k2 = jr.split(key)
    condition = jr.normal(k1, (1,))
    xs = jr.normal(k2, (64,)) * 2.0

    rt_x = jax.vmap(lambda x: bij.inverse(bij.transform(x, condition), condition))(xs)
    assert jnp.allclose(rt_x.squeeze(), xs, rtol=0, atol=1e-8)

    ys = jnp.linspace(1e-3, 1 - 1e-3, 64)
    rt_y = jax.vmap(lambda y: bij.transform(bij.inverse(y, condition), condition))(ys)
    assert jnp.allclose(rt_y.squeeze(), ys, rtol=0, atol=1e-8)


def test_forward_log_det_matches_autodiff(key):
    """transform_and_log_det's log-det == log|d/dx transform| (autodiff)."""
    bij = _make()
    k1, k2 = jr.split(key)
    condition = jr.normal(k1, (1,))
    x = jr.normal(k2, ())

    y, log_det = bij.transform_and_log_det(x, condition)
    dydx = jax.grad(lambda x_: bij.transform(x_, condition).squeeze())(x)

    assert y == bij.transform(x, condition)
    assert log_det.squeeze() == pytest.approx(
        float(jnp.log(jnp.abs(dydx))), rel=0, abs=1e-9
    )


def test_inverse_log_det_matches_autodiff(key):
    """inverse_and_log_det's log-det == log|d/dy inverse| (autodiff)."""
    bij = _make()
    k1, _ = jr.split(key)
    condition = jr.normal(k1, (1,))
    y = jnp.array(0.37)

    x, log_det = bij.inverse_and_log_det(y, condition)
    dxdy = jax.grad(lambda y_: bij.inverse(y_, condition).squeeze())(y)

    assert x == bij.inverse(y, condition)
    assert log_det.squeeze() == pytest.approx(
        float(jnp.log(jnp.abs(dxdy))), rel=0, abs=1e-9
    )


def test_inverse_log_det_is_negative_forward_log_det(key):
    """Inverse-function theorem: ld_inv(y) == -ld_fwd(x) at x = inverse(y)."""
    bij = _make()
    condition = jr.normal(key, (1,))
    y = jnp.array(0.62)

    x, ld_inv = bij.inverse_and_log_det(y, condition)
    _, ld_fwd = bij.transform_and_log_det(x, condition)
    assert ld_inv.squeeze() == pytest.approx(float(-ld_fwd.squeeze()), rel=0, abs=1e-8)


def test_ate_injection_shifts_outcome_location():
    """The load-bearing causal claim: at sampling time (inverse), changing the
    treatment from 0 to 1 shifts the outcome by exactly `ate` for every quantile.
    """
    ate = 1.6
    bij = _make(ate=ate, scale=1.0, const=0.0)
    ys = jnp.linspace(1e-3, 1 - 1e-3, 50)

    x_control = jax.vmap(lambda y: bij.inverse(y, jnp.array([0.0])))(ys).squeeze()
    x_treated = jax.vmap(lambda y: bij.inverse(y, jnp.array([1.0])))(ys).squeeze()
    assert jnp.allclose(x_treated - x_control, ate, rtol=0, atol=1e-9)


def test_cond_dim_none_is_unconditional_gaussian_cdf():
    """With ``cond_dim=None`` the bijection is an unconditional Gaussian CDF:
    ``forward(x) = Φ((x - const) / scale)`` — the location is just ``const``,
    independent of any condition. Round-trip and the autodiff log-det must also
    agree under the conditional contract.
    """
    scale, const = 1.4, -0.3
    bij = UnivariateNormalCDF(scale=scale, const=const, cond_dim=None)
    assert bij.cond_shape is None

    xs = jnp.linspace(-6.0, 6.0, 201)
    got = jax.vmap(lambda x: bij.transform(x))(xs)
    expected = jax.scipy.stats.norm.cdf(xs, loc=const, scale=scale)
    assert jnp.allclose(got.squeeze(), expected, rtol=0, atol=1e-10)

    rt = jax.vmap(lambda x: bij.inverse(bij.transform(x)))(xs)
    assert jnp.allclose(rt.squeeze(), xs, rtol=0, atol=1e-8)

    x = jnp.array(0.7)
    _, log_det = bij.transform_and_log_det(x)
    dydx = jax.grad(lambda x_: bij.transform(x_).squeeze())(x)
    assert log_det.squeeze() == pytest.approx(
        float(jnp.log(jnp.abs(dydx))), rel=0, abs=1e-9
    )


def test_default_construction_does_not_crash():
    """``UnivariateNormalCDF()`` with all defaults (Python-scalar ``ate=0``)
    must construct without crashing — the constructor coerces ``ate`` to a JAX
    array. Regression test for the lost ``arraylike_to_array`` coercion in
    commit ``602c8b4``.
    """
    bij = UnivariateNormalCDF()
    assert bij.cond_shape is None
    assert bij.ate.shape == ()


def test_assertion_still_catches_real_shape_mismatch():
    """The protective assertion must still fire when ``cond_dim`` is supplied
    and ``ate.shape`` disagrees — this is a real user error.
    """
    with pytest.raises(AssertionError):
        UnivariateNormalCDF(ate=jnp.array([1.0]), cond_dim=2)
