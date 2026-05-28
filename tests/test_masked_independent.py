"""Math-grounded tests for ``MaskedIndependent``.

The bijection applies a per-coordinate elementwise ``transformer`` whose
parameters come from a fully first-layer-masked MLP, so the parameters are
input-independent. We verify the bijection mechanics regardless of that:
exact round-trips, log-det vs autodiff, and the documented input-independence.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from flowjax.bijections import Affine
from paramax import unwrap

from frugal_flows.bijections import MaskedIndependent


def _make(key, *, dim=3, cond_dim=None):
    bij = MaskedIndependent(
        key,
        transformer=Affine(),
        dim=dim,
        cond_dim=cond_dim,
        nn_width=16,
        nn_depth=1,
    )
    return unwrap(bij)


def test_round_trip_unconditional(key):
    bij = _make(key, dim=4)
    x = jr.normal(jr.split(key)[1], (4,))
    rt = bij.inverse(bij.transform(x))
    assert jnp.allclose(rt, x, rtol=0, atol=1e-6)


def test_round_trip_conditional(key):
    k1, k2, k3 = jr.split(key, 3)
    bij = _make(k1, dim=3, cond_dim=2)
    x = jr.normal(k2, (3,))
    c = jr.normal(k3, (2,))
    rt = bij.inverse(bij.transform(x, c), c)
    assert jnp.allclose(rt, x, rtol=0, atol=1e-6)


def test_forward_log_det_matches_autodiff(key):
    """log-det == log|det J| from autodiff (J is diagonal: elementwise map)."""
    k1, k2 = jr.split(key)
    bij = _make(k1, dim=4)
    x = jr.normal(k2, (4,))

    y, log_det = bij.transform_and_log_det(x)
    jac = jax.jacfwd(bij.transform)(x)
    _, logabsdet = jnp.linalg.slogdet(jac)

    assert jnp.allclose(y, bij.transform(x), rtol=0, atol=1e-12)
    assert log_det == pytest.approx(float(logabsdet), rel=0, abs=1e-6)


def test_inverse_and_log_det_is_negative_forward(key):
    k1, k2 = jr.split(key)
    bij = _make(k1, dim=3)
    y = jr.normal(k2, (3,))
    x, ld_inv = bij.inverse_and_log_det(y)
    _, ld_fwd = bij.transform_and_log_det(x)
    assert jnp.allclose(bij.transform(x), y, rtol=0, atol=1e-6)
    assert ld_inv == pytest.approx(float(-ld_fwd), rel=0, abs=1e-6)


def test_transformer_params_are_input_independent(key):
    """Documented behaviour: the first-layer mask zeros all input edges, so the
    transformer (hence its log-det) does not vary with x."""
    k1, k2, k3 = jr.split(key, 3)
    bij = _make(k1, dim=3)
    x_a = jr.normal(k2, (3,))
    x_b = jr.normal(k3, (3,)) * 5.0
    _, ld_a = bij.transform_and_log_det(x_a)
    _, ld_b = bij.transform_and_log_det(x_b)
    assert ld_a == pytest.approx(float(ld_b), rel=0, abs=1e-9)
