"""Transformer-conditioner tests for the ``flexible_continuous`` causal margin.

These guard the ``causal_model_args={"conditioner": "transformer"}`` option:
the flexible (spline) margin built with a causal-transformer conditioner
(``bijections.transformer_autoregressive``) instead of the MADE-masked MLP.
The chain around the margin is conditioner-agnostic, so the structure, freeze
indices, graft site, and Rosenblatt property must all match the MLP arm; the
MLP default path is covered by ``test_multivariate_y_flexible.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest
from flowjax.bijections import Affine, Concatenate, Invert, Stack
from frugal_flows.bijections import TransformerAutoregressive
from frugal_flows.causal_flows import (
    pretrain_causal_margin,
    train_frugal_flow,
    train_frugal_flow_flexible_continuous,
)

MAF = dict(nn_depth=1, nn_width=8, RQS_knots=4, flow_layers=2)
TRF = MAF | {"conditioner": "transformer"}
FIT = dict(max_patience=20, batch_size=64, show_progress=False, learning_rate=1e-3)
K = 3
NVARS = 2


def _make_data(seed, n=400, k=K, nvars=NVARS):
    """Synthetic vector outcome with per-dim ATE under Z-confounding
    (same DGP as ``test_multivariate_y.py``)."""
    key = jr.split(jr.PRNGKey(seed), 4)
    Z = jr.normal(key[0], (n, nvars))
    T = jr.bernoulli(key[1], jax.nn.sigmoid(Z[:, 0])).astype(float)
    eta = jr.normal(key[2], (n, k)) + Z.sum(1, keepdims=True)
    tau = jnp.linspace(-1.0, 1.0, k)
    Y = tau * T[:, None] + eta
    u_z = jr.uniform(key[3], (n, nvars))
    return Y, u_z, T[:, None], tau


def _contains_transformer(tree) -> bool:
    leaves = jtu.tree_leaves(
        tree, is_leaf=lambda l: isinstance(l, TransformerAutoregressive)
    )
    return any(isinstance(l, TransformerAutoregressive) for l in leaves)


@pytest.fixture(scope="module")
def mv():
    Y, u_z, cond, tau = _make_data(seed=0)
    flow, losses = train_frugal_flow_flexible_continuous(
        key=jr.PRNGKey(0), y=Y, u_z=u_z, condition=cond,
        causal_model_args=TRF, max_epochs=40, **MAF, **FIT,
    )
    return dict(flow=flow, losses=losses, Y=Y, u_z=u_z, cond=cond, tau=tau)


# --- builds / trains -------------------------------------------------------
def test_transformer_trains_finite(mv):
    tr = jnp.asarray(mv["losses"]["train"])
    assert tr.shape[0] >= 1
    assert bool(jnp.isfinite(tr).all())


def test_dispatcher_transformer_runs():
    Y, u_z, cond, _ = _make_data(seed=7)
    flow, losses = train_frugal_flow(
        key=jr.PRNGKey(7), y=Y, u_z=u_z, condition=cond,
        causal_model="flexible_continuous", causal_model_args=TRF,
        max_epochs=1, **MAF, **FIT,
    )
    assert _contains_transformer(flow)
    assert len(losses["train"]) >= 1


def test_unknown_conditioner_raises():
    Y, u_z, cond, _ = _make_data(seed=8)
    with pytest.raises(ValueError, match="conditioner"):
        train_frugal_flow_flexible_continuous(
            key=jr.PRNGKey(8), y=Y, u_z=u_z, condition=cond,
            causal_model_args=MAF | {"conditioner": "nonsense"},
            max_epochs=1, **MAF, **FIT,
        )


# --- structure: identical chain to the MLP arm -----------------------------
def test_chain_length(mv):
    assert len(mv["flow"].bijection.bijections) == 5


@pytest.mark.parametrize(
    "idx,expected",
    [(0, Affine), (1, Invert), (2, Concatenate), (3, Concatenate), (4, Stack)],
)
def test_layout(mv, idx, expected):
    assert isinstance(mv["flow"].bijection.bijections[idx], expected)


def test_margin_slot_holds_K_wide_transformer(mv):
    block = mv["flow"].bijection.bijections[3]  # bijections_ate_maf (Concatenate)
    margin = block.bijections[0]
    assert margin.shape == (K,)
    assert margin.cond_shape == (1,)
    assert _contains_transformer(margin)


# --- Rosenblatt property ---------------------------------------------------
def test_first_K_dims_marginally_uniform(mv):
    flow = mv["flow"]
    data = jnp.hstack([mv["Y"], mv["u_z"]])
    base = jax.vmap(flow.bijection.inverse)(data, mv["cond"])
    ranks = base[:, :K]
    assert bool(jnp.isfinite(ranks).all())
    assert float(ranks.min()) >= -1e-6 and float(ranks.max()) <= 1 + 1e-6
    means = ranks.mean(0)
    stds = ranks.std(0)
    assert bool((means > 0.2).all() and (means < 0.8).all())
    assert bool((stds > 0.15).all())


# --- warm-start graft ------------------------------------------------------
def test_transformer_pretrain_grafts():
    Y, u_z, cond, _ = _make_data(seed=3)
    margin = pretrain_causal_margin(
        jr.PRNGKey(3), y=Y, condition=cond, causal_model_args=TRF,
        max_epochs=2, batch_size=64, show_progress=False,
    )
    assert margin.shape == (K,)
    assert _contains_transformer(margin)
    flow, losses = train_frugal_flow_flexible_continuous(
        key=jr.PRNGKey(3), y=Y, u_z=u_z, condition=cond,
        causal_model_args=TRF, pretrained_margin=margin, max_epochs=1, **MAF, **FIT,
    )
    assert flow is not None
    assert len(losses["train"]) >= 1


def test_cross_conditioner_graft_raises():
    """An MLP-pretrained margin cannot graft into a transformer-margin flow."""
    Y, u_z, cond, _ = _make_data(seed=4)
    margin_mlp = pretrain_causal_margin(
        jr.PRNGKey(4), y=Y, condition=cond, causal_model_args=MAF,
        max_epochs=1, batch_size=64, show_progress=False,
    )
    with pytest.raises(AssertionError):
        train_frugal_flow_flexible_continuous(
            key=jr.PRNGKey(4), y=Y, u_z=u_z, condition=cond,
            causal_model_args=TRF, pretrained_margin=margin_mlp,
            max_epochs=1, **MAF, **FIT,
        )
