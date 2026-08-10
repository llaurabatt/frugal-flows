"""Multivariate-outcome (K>1) tests for ``flexible_continuous``.

These guard the extension of the flexible (spline) causal margin to a vector
outcome ``Y`` of shape ``(n, K)``: a K-dimensional autoregressive spline margin
conditioned on treatment, a copula holding the first K coordinates marginally
uniform (the Rosenblatt ranks), and the per-dimension interventional read-out
that serves as the effect estimator for this nonparametric arm. The scalar
(K=1) path is covered in ``test_flow_structure.py`` / ``test_interventions.py``;
here K>1 is exercised end to end, mirroring ``test_multivariate_y.py`` for
``location_translation``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from flowjax.bijections import Affine, Concatenate, Invert, Stack
from frugal_flows.causal_flows import (
    pretrain_causal_margin,
    train_frugal_flow,
    train_frugal_flow_flexible_continuous,
)
from frugal_flows.interventions import interventional_samples, tau_curve

MAF = dict(nn_depth=1, nn_width=8, RQS_knots=4, flow_layers=2)
FIT = dict(max_patience=20, batch_size=64, show_progress=False, learning_rate=1e-3)
K = 3
NVARS = 2


def _make_data(seed, n=400, k=K, nvars=NVARS):
    """Synthetic vector outcome with per-dim ATE under Z-confounding
    (same DGP as ``test_multivariate_y.py``)."""
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
    flow, losses = train_frugal_flow_flexible_continuous(
        key=jr.PRNGKey(0), y=Y, u_z=u_z, condition=cond,
        causal_model_args=MAF, max_epochs=40, **MAF, **FIT,
    )
    return dict(flow=flow, losses=losses, Y=Y, u_z=u_z, cond=cond, tau=tau)


# --- builds / trains -------------------------------------------------------
def test_multivariate_trains_finite(mv):
    tr = jnp.asarray(mv["losses"]["train"])
    assert tr.shape[0] >= 1
    assert bool(jnp.isfinite(tr).all())


def test_dispatcher_multivariate_runs():
    Y, u_z, cond, _ = _make_data(seed=7)
    flow, losses = train_frugal_flow(
        key=jr.PRNGKey(7), y=Y, u_z=u_z, condition=cond,
        causal_model="flexible_continuous", causal_model_args=MAF,
        max_epochs=1, **MAF, **FIT,
    )
    assert flow is not None
    assert len(losses["train"]) >= 1


# --- structure -------------------------------------------------------------
def test_chain_length(mv):
    assert len(mv["flow"].bijection.bijections) == 5


@pytest.mark.parametrize(
    "idx,expected",
    [(0, Affine), (1, Invert), (2, Concatenate), (3, Concatenate), (4, Stack)],
)
def test_layout(mv, idx, expected):
    assert isinstance(mv["flow"].bijection.bijections[idx], expected)


def test_causal_margin_block_is_K_wide(mv):
    block = mv["flow"].bijection.bijections[3]  # bijections_ate_maf (Concatenate)
    margin = block.bijections[0]  # the treatment-conditioned spline margin
    assert margin.shape == (K,)
    assert margin.cond_shape == (1,)
    assert len(block.bijections) == 1 + NVARS  # margin + one Identity per covariate
    assert block.shape == (K + NVARS,)


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


# --- warm-start graft ------------------------------------------------------
def test_pretrain_margin_grafts_K_dim():
    Y, u_z, cond, _ = _make_data(seed=3)
    margin = pretrain_causal_margin(
        jr.PRNGKey(3), y=Y, condition=cond, causal_model_args=MAF,
        max_epochs=2, batch_size=64, show_progress=False,
    )
    assert margin.shape == (K,)
    assert margin.cond_shape == (1,)
    flow, losses = train_frugal_flow_flexible_continuous(
        key=jr.PRNGKey(3), y=Y, u_z=u_z, condition=cond,
        causal_model_args=MAF, pretrained_margin=margin, max_epochs=1, **MAF, **FIT,
    )
    assert flow is not None
    assert len(losses["train"]) >= 1


def test_graft_dim_mismatch_raises():
    """A margin pretrained at the wrong K cannot graft silently."""
    Y2, _, cond2, _ = _make_data(seed=4, k=2)
    margin_k2 = pretrain_causal_margin(
        jr.PRNGKey(4), y=Y2, condition=cond2, causal_model_args=MAF,
        max_epochs=1, batch_size=64, show_progress=False,
    )
    Y3, u_z, cond, _ = _make_data(seed=4, k=3)
    with pytest.raises(AssertionError):
        train_frugal_flow_flexible_continuous(
            key=jr.PRNGKey(4), y=Y3, u_z=u_z, condition=cond,
            causal_model_args=MAF, pretrained_margin=margin_k2,
            max_epochs=1, **MAF, **FIT,
        )


# --- per-dim interventional read-out --------------------------------------
N_MC = 2000


def test_interventional_readout_per_dim(mv):
    out = interventional_samples(jr.key(0), mv["flow"], cond_dim=1, n_mc=N_MC, dim_y=K)
    assert out["y0"].shape == (N_MC, K)
    assert out["y1"].shape == (N_MC, K)
    for stat in ("mean0", "mean1", "var0", "var1", "ate", "tau_sd", "frac_neg"):
        assert np.asarray(out[stat]).shape == (K,)
        assert bool(np.isfinite(out[stat]).all())
    assert isinstance(out["anynan"], bool)
    assert not out["anynan"]


def test_tau_curve_multivariate_shape(mv):
    out = interventional_samples(jr.key(1), mv["flow"], cond_dim=1, n_mc=N_MC, dim_y=K)
    u, curves = tau_curve(out["y0"], out["y1"], n_bins=10)
    assert u.shape == (10,)
    assert curves.shape == (10, K)
    assert bool(np.isfinite(curves).all())
