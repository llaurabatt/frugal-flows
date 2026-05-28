"""Smoke / invariant tests for the flow constructors in ``basic_flows``.

Bijection-level math is covered in the per-bijection test files. Here we only
check that the constructors assemble a valid invertible flow and that the
``first_uniform`` constructor preserves coordinate 0 through the whole stack
(the causal slot must stay untouched across layers + permutations).
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest
from flowjax.distributions import Uniform
from paramax import unwrap

from frugal_flows.basic_flows import (
    masked_autoregressive_flow_first_uniform,
    masked_independent_flow,
    univariate_marginal_flow,
)

DIM = 4
SMALL = dict(flow_layers=2, nn_width=8, nn_depth=1)


def _base(dim=DIM):
    return Uniform(-jnp.ones(dim), jnp.ones(dim))


def test_first_uniform_flow_round_trips_and_fixes_coord0(key):
    k1, k2 = jr.split(key)
    flow = masked_autoregressive_flow_first_uniform(
        k1, base_dist=_base(), **SMALL
    )
    bij = unwrap(flow.bijection)
    x = jr.uniform(k2, (DIM,), minval=-0.9, maxval=0.9)
    y = bij.transform(x)
    # Coordinate 0 is the causal slot: untouched through every layer/permute.
    assert y[0] == pytest.approx(float(x[0]), rel=0, abs=1e-10)
    assert jnp.allclose(bij.inverse(y), x, rtol=0, atol=1e-4)


def test_masked_independent_flow_round_trips(key):
    k1, k2 = jr.split(key)
    flow = masked_independent_flow(k1, base_dist=_base(), **SMALL)
    bij = unwrap(flow.bijection)
    x = jr.normal(k2, (DIM,))
    assert jnp.allclose(bij.inverse(bij.transform(x)), x, rtol=0, atol=1e-4)


def test_univariate_marginal_flow_builds_and_trains(key):
    """``univariate_marginal_flow`` constructs, freezes its base rescaler, and
    runs through one epoch of training without raising. Guards against latent
    breakage in the freezing path (this constructor is otherwise unexercised
    by the rest of the suite)."""
    k1, k2 = jr.split(key)
    z_cont = jr.normal(k2, (32,))
    flow, losses = univariate_marginal_flow(
        key=k1,
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
    assert flow is not None
    assert isinstance(losses, dict)
    assert "train" in losses and "val" in losses
    assert len(losses["train"]) >= 1
