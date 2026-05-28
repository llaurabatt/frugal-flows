"""Math-grounded tests for the masked-autoregressive bijection family.

Covers:
- MaskedAutoregressiveFirstUniform: the first-coordinate-identity (uniform)
  invariant, round-trip, log-det vs autodiff.
- MaskedAutoregressiveHeterogeneous: identity slot at ``identity_idx``;
  the MLP-output row for the identity slot has zero gradient by construction
  (no explicit stop_gradient needed).
- MaskedAutoregressiveMaskedCond: standard MAF round-trip + log-det.
- MaskedAutoregressiveTransformerCond: round-trip + log-det.

A concrete unconditional ``Affine`` transformer is used throughout (its
``transform`` accepts and ignores ``condition``).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from flowjax.bijections import Affine
from paramax import unwrap

from frugal_flows.bijections import (
    MaskedAutoregressiveHeterogeneous,
    MaskedAutoregressiveMaskedCond,
    MaskedAutoregressiveTransformerCond,
    MaskedAutoregressiveFirstUniform,
)

NN = dict(nn_width=16, nn_depth=1)


def _logabsdet(f, x):
    return jnp.linalg.slogdet(jax.jacfwd(f)(x))[1]


# --------------------------- FirstUniform (module 5) ---------------------------

def test_first_uniform_holds_first_coordinate_fixed(key):
    k1, k2 = jr.split(key)
    bij = unwrap(
        MaskedAutoregressiveFirstUniform(k1, transformer=Affine(), dim=4, **NN)
    )
    xs = jr.uniform(k2, (200, 4), minval=-1.0, maxval=1.0)
    ys = jax.vmap(bij.transform)(xs)
    # Coordinate 0 is the identity, exactly, for every input.
    assert jnp.allclose(ys[:, 0], xs[:, 0], rtol=0, atol=1e-12)
    # The other coordinates are actually transformed (not identity).
    assert not jnp.allclose(ys[:, 1:], xs[:, 1:], atol=1e-3)


def test_first_uniform_round_trip_and_log_det(key):
    k1, k2 = jr.split(key)
    bij = unwrap(
        MaskedAutoregressiveFirstUniform(k1, transformer=Affine(), dim=4, **NN)
    )
    x = jr.normal(k2, (4,))
    y, ld = bij.transform_and_log_det(x)
    assert jnp.allclose(bij.inverse(y), x, rtol=0, atol=1e-5)
    assert ld == pytest.approx(float(_logabsdet(bij.transform, x)), rel=0, abs=1e-5)


# ------------------------- Heterogeneous (module 6) ---------------------------

@pytest.mark.parametrize("idx", [0, 2, 3])
def test_heterogeneous_identity_slot_position(key, idx):
    bij = unwrap(
        MaskedAutoregressiveHeterogeneous(
            key, transformer=Affine(), dim=4, identity_idx=idx, **NN
        )
    )
    x = jr.normal(jr.split(key)[1], (4,))
    y = bij.transform(x)
    assert y[idx] == pytest.approx(float(x[idx]), rel=0, abs=1e-12)
    assert jnp.allclose(bij.inverse(y), x, rtol=0, atol=1e-5)


def test_heterogeneous_identity_slot_mlp_row_is_unused(key):
    """The MLP output row producing identity-slot transformer params is
    dropped by `_flat_params_to_transformer` (Identity((1,)) takes its place),
    so the bijection output is invariant to perturbations of that row. By
    extension JAX autodiff zeros the gradient on it automatically -- no
    explicit stop_gradient is needed.
    """
    k1, k2 = jr.split(key)
    identity_idx = 2
    bij = MaskedAutoregressiveHeterogeneous(
        k1, transformer=Affine(), dim=4, identity_idx=identity_idx, **NN
    )
    x = jr.normal(k2, (4,))

    # MLP output is row-major in (dim, num_params).
    last_bias = bij.masked_autoregressive_mlp.layers[-1].bias
    num_params = last_bias.shape[0] // 4
    block = slice(identity_idx * num_params, (identity_idx + 1) * num_params)

    new_bias = last_bias.at[block].add(100.0)
    bij_perturbed = eqx.tree_at(
        lambda b: b.masked_autoregressive_mlp.layers[-1].bias, bij, new_bias
    )

    y_orig = unwrap(bij).transform(x)
    y_perturbed = unwrap(bij_perturbed).transform(x)
    assert jnp.allclose(y_orig, y_perturbed, atol=1e-12)


# --------------------------- MaskedCond (module 7) ----------------------------

def test_masked_cond_round_trip_and_log_det(key):
    k1, k2, k3 = jr.split(key, 3)
    bij = unwrap(
        MaskedAutoregressiveMaskedCond(
            k1, transformer=Affine(), dim=3, cond_dim_nomask=2, **NN
        )
    )
    x = jr.normal(k2, (3,))
    c = jr.normal(k3, (2,))
    y, ld = bij.transform_and_log_det(x, c)
    assert jnp.allclose(bij.inverse(y, c), x, rtol=0, atol=1e-5)
    assert ld == pytest.approx(
        float(_logabsdet(lambda z: bij.transform(z, c), x)), rel=0, abs=1e-5
    )


# ----------------------- TransformerCond (module 8) ---------------------------

def test_transformer_cond_round_trip_and_log_det(key):
    k1, k2, k3 = jr.split(key, 3)
    bij = unwrap(
        MaskedAutoregressiveTransformerCond(
            k1, transformer=Affine(), dim=3, cond_dim=2, **NN
        )
    )
    x = jr.normal(k2, (3,))
    c = jr.normal(k3, (2,))
    y, ld = bij.transform_and_log_det(x, c)
    assert jnp.allclose(bij.inverse(y, c), x, rtol=0, atol=1e-5)
    assert ld == pytest.approx(
        float(_logabsdet(lambda z: bij.transform(z, c), x)), rel=0, abs=1e-5
    )


