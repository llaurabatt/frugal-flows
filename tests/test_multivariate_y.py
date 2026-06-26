"""Multivariate-outcome (K>1) tests for ``location_translation``.

These guard the Architecture-B extension of the frugal flow to a vector outcome
``Y`` of shape ``(n, K)``: a K-dimensional autoregressive causal-margin flow whose
first K coordinates are held marginally uniform (the Rosenblatt ranks), each
carrying its own additive treatment effect via a per-dim ``LocCond``. The scalar
(K=1) path is covered elsewhere; here K>1 is exercised end to end.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import paramax
import pytest
from flowjax.bijections import Affine, Concatenate, Invert, Stack

from frugal_flows.bijections import LocCond
from frugal_flows.causal_flows import (
    train_frugal_flow,
    train_frugal_flow_location_translation,
)

MAF = dict(nn_depth=1, nn_width=8, RQS_knots=4, flow_layers=2)
FIT = dict(max_patience=20, batch_size=64, show_progress=False, learning_rate=1e-3)
K = 3
NVARS = 2


def _make_data(seed, n=400, k=K, nvars=NVARS):
    """Synthetic vector outcome with per-dim ATE under Z-confounding."""
    key = jr.split(jr.PRNGKey(seed), 4)
    Z = jr.normal(key[0], (n, nvars))
    T = jr.bernoulli(key[1], jax.nn.sigmoid(Z[:, 0])).astype(float)
    eta = jr.normal(key[2], (n, k)) + Z.sum(1, keepdims=True)  # depends on Z -> confounded
    tau = jnp.linspace(-1.0, 1.0, k)
    Y = tau * T[:, None] + eta
    u_z = jr.uniform(key[3], (n, nvars))  # stand-in covariate quantiles in [0, 1]
    return Y, u_z, T[:, None], tau


@pytest.fixture(scope="module")
def mv():
    Y, u_z, cond, tau = _make_data(seed=0)
    flow, losses = train_frugal_flow_location_translation(
        key=jr.PRNGKey(0), y=Y, u_z=u_z, condition=cond,
        causal_model_args=MAF | {"ate": tau}, max_epochs=40, **MAF, **FIT,
    )
    return dict(flow=flow, losses=losses, Y=Y, u_z=u_z, cond=cond, tau=tau)


# --- builds / trains -------------------------------------------------------
def test_multivariate_trains_finite(mv):
    tr = jnp.asarray(mv["losses"]["train"])
    assert tr.shape[0] >= 1
    assert bool(jnp.isfinite(tr).all())


def test_dispatcher_multivariate_runs():
    Y, u_z, cond, tau = _make_data(seed=7)
    flow, losses = train_frugal_flow(
        key=jr.PRNGKey(7), y=Y, u_z=u_z, condition=cond,
        causal_model="location_translation", causal_model_args=MAF | {"ate": tau},
        max_epochs=1, **MAF, **FIT,
    )
    assert flow is not None
    assert len(losses["train"]) >= 1


# --- structure -------------------------------------------------------------
def test_chain_length(mv):
    assert len(mv["flow"].bijection.bijections) == 6


@pytest.mark.parametrize(
    "idx,expected",
    [(0, Affine), (1, Invert), (2, Concatenate), (3, Concatenate), (4, Stack), (5, Stack)],
)
def test_layout(mv, idx, expected):
    assert isinstance(mv["flow"].bijection.bijections[idx], expected)


def test_loccond_block_holds_K_dims(mv):
    block = mv["flow"].bijection.bijections[5]  # bijections_loccond (Stack)
    n_loccond = sum(isinstance(paramax.unwrap(b), LocCond) for b in block.bijections)
    assert n_loccond == K
    assert len(block.bijections) == K + NVARS


# --- per-dim ATE -----------------------------------------------------------
def test_ate_readoff_per_dim(mv):
    block = mv["flow"].bijection.bijections[5]
    ates = jnp.array([float(paramax.unwrap(block.bijections[k]).ate) for k in range(K)])
    assert ates.shape == (K,)
    assert bool(jnp.isfinite(ates).all())


def test_ate_scalar_broadcasts():
    Y, u_z, cond, _ = _make_data(seed=1)
    flow, _ = train_frugal_flow_location_translation(
        key=jr.PRNGKey(1), y=Y, u_z=u_z, condition=cond,
        causal_model_args=MAF | {"ate": 3.0}, max_epochs=1, **MAF, **FIT,
    )
    block = flow.bijection.bijections[5]
    assert sum(isinstance(paramax.unwrap(b), LocCond) for b in block.bijections) == K


def test_ate_length_mismatch_raises():
    Y, u_z, cond, _ = _make_data(seed=2)  # K = 3
    with pytest.raises(ValueError):
        train_frugal_flow_location_translation(
            key=jr.PRNGKey(2), y=Y, u_z=u_z, condition=cond,
            causal_model_args=MAF | {"ate": jnp.array([1.0, 2.0])},  # length 2 != 3
            max_epochs=1, **MAF, **FIT,
        )


# --- Rosenblatt property ---------------------------------------------------
def test_first_K_dims_marginally_uniform(mv):
    """Pushing data to the base, the first K coords are the Rosenblatt ranks:
    they sit in the StandardUniform[0,1] support and are spread, not collapsed."""
    flow = mv["flow"]
    data = jnp.hstack([mv["Y"], mv["u_z"]])
    base = jax.vmap(flow.bijection.inverse)(data, mv["cond"])
    ranks = base[:, :K]
    assert bool(jnp.isfinite(ranks).all())
    assert float(ranks.min()) >= -1e-6 and float(ranks.max()) <= 1 + 1e-6
    means = ranks.mean(0)
    stds = ranks.std(0)
    assert bool((means > 0.2).all() and (means < 0.8).all())  # centred ~0.5
    assert bool((stds > 0.15).all())                          # spread (Unif std ~0.29)
