"""Structural assertions for trained flow objects.

For each flow constructor used in the package, this file asserts:
  - The total number of bijections in ``flow.bijection.bijections``.
  - The type of object at every position that consumer code (sample_marginals,
    sample_outcome, freezing calls in causal_flows / basic_flows /
    train_quantile_propensity_score) reaches into by hardcoded index.
  - The type at every deeper path that consumer code follows.

If a future flowjax upgrade reshuffles ``merge_transforms`` semantics, or
someone adds or removes a layer in a construction site without updating
the consumer code, one of these assertions will fail at the point of
divergence.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest
from flowjax.bijections import Affine, Concatenate, Invert, Stack

from frugal_flows.basic_flows import univariate_marginal_flow
from frugal_flows.bijections import (
    MaskedAutoregressiveFirstUniform,
    MaskedAutoregressiveHeterogeneous,
)
from frugal_flows.causal_flows import (
    train_copula_flow,
    train_frugal_flow_flexible_continuous,
    train_frugal_flow_flexible_discrete,
    train_frugal_flow_gaussian,
    train_frugal_flow_location_translation,
)
from frugal_flows.train_quantile_propensity_score import (
    train_quantile_propensity_score,
)

MAF = dict(nn_depth=1, nn_width=4, RQS_knots=4, flow_layers=2)
FIT = dict(max_epochs=1, max_patience=1, batch_size=16, show_progress=False)


@pytest.fixture(scope="module")
def synth():
    k = jr.PRNGKey(0)
    keys = jr.split(k, 5)
    n = 32
    return dict(
        key=k,
        y_cont=jr.uniform(keys[0], (n, 1)),
        y_disc=jr.randint(keys[0], (n, 1), 0, 2),
        u_z=jr.uniform(keys[1], (n, 2)),
        cond=jr.uniform(keys[2], (n, 1)),
        x_treat=jr.bernoulli(keys[3], 0.5, (n,)).astype(int),
        z_cont=jr.normal(keys[4], (n,)),
    )


# --- univariate_marginal_flow ---------------------------------------------


@pytest.fixture(scope="module")
def univariate_marginal(synth):
    flow, _ = univariate_marginal_flow(
        key=synth["key"], z_cont=synth["z_cont"], **MAF, **FIT,
    )
    return flow


def test_univariate_marginal_chain_length(univariate_marginal):
    assert len(univariate_marginal.bijection.bijections) == 3


@pytest.mark.parametrize("idx,expected,label", [
    (0, Affine, "base rescaler (from Uniform(-1, 1) flattened by merge_transforms)"),
    (1, Invert, "MAF block (Invert wraps the autoregressive flow)"),
    (2, Invert, "Invert(Tanh) for unbounded support"),
])
def test_univariate_marginal_layout(univariate_marginal, idx, expected, label):
    assert isinstance(univariate_marginal.bijection.bijections[idx], expected), label


# --- train_copula_flow ----------------------------------------------------


@pytest.fixture(scope="module")
def copula(synth):
    flow, _ = train_copula_flow(
        key=synth["key"], u_z=synth["u_z"], **MAF, **FIT,
    )
    return flow


def test_copula_chain_length(copula):
    assert len(copula.bijection.bijections) == 3


@pytest.mark.parametrize("idx,expected,label", [
    (0, Affine, "base rescaler (frozen)"),
    (1, Invert, "MAF block"),
    (2, Invert, "Invert(Affine) for unbounded support (frozen)"),
])
def test_copula_layout(copula, idx, expected, label):
    assert isinstance(copula.bijection.bijections[idx], expected), label


# --- train_quantile_propensity_score --------------------------------------


@pytest.fixture(scope="module")
def propensity(synth):
    flow, _ = train_quantile_propensity_score(
        key=synth["key"],
        x=synth["x_treat"],
        condition=synth["cond"],
        **MAF, **FIT,
        learning_rate=5e-4,
    )
    return flow


def test_propensity_chain_length(propensity):
    assert len(propensity.bijection.bijections) == 3


@pytest.mark.parametrize("idx,expected,label", [
    (0, Affine, "base rescaler (frozen)"),
    (1, Invert, "MAF block"),
    (2, Invert, "Invert(Affine) for unbounded support (frozen)"),
])
def test_propensity_layout(propensity, idx, expected, label):
    assert isinstance(propensity.bijection.bijections[idx], expected), label


# --- train_frugal_flow_location_translation -------------------------------


@pytest.fixture(scope="module")
def loc_translation(synth):
    flow, _ = train_frugal_flow_location_translation(
        key=synth["key"],
        y=synth["y_cont"],
        u_z=synth["u_z"],
        condition=synth["cond"],
        causal_model_args=MAF | {"ate": 0.0},
        **MAF, **FIT,
    )
    return flow


def test_loc_translation_chain_length(loc_translation):
    assert len(loc_translation.bijection.bijections) == 6


@pytest.mark.parametrize("idx,expected,label", [
    (0, Affine, "base rescaler (frozen)"),
    (1, Invert, "first_uniform MAF block"),
    (2, Concatenate, "bijections_affine (frozen, identity over first dim + Invert(Affine) over Z)"),
    (3, Concatenate, "bijections_ate_maf (MAF over causal dim + identity over Z)"),
    (4, Stack, "bijections_tanh (Invert(Tanh) over causal dim + identity over Z)"),
    (5, Stack, "bijections_loccond (LocCond over causal dim + identity over Z)"),
])
def test_loc_translation_layout(loc_translation, idx, expected, label):
    assert isinstance(loc_translation.bijection.bijections[idx], expected), label


# --- train_frugal_flow_flexible_continuous --------------------------------


@pytest.fixture(scope="module")
def flexible_continuous(synth):
    flow, _ = train_frugal_flow_flexible_continuous(
        key=synth["key"],
        y=synth["y_cont"],
        u_z=synth["u_z"],
        condition=synth["cond"],
        causal_model_args=MAF,
        **MAF, **FIT,
    )
    return flow


def test_flexible_continuous_chain_length(flexible_continuous):
    assert len(flexible_continuous.bijection.bijections) == 5


@pytest.mark.parametrize("idx,expected,label", [
    (0, Affine, "base rescaler (frozen)"),
    (1, Invert, "first_uniform MAF block"),
    (2, Concatenate, "bijections_affine (frozen)"),
    (3, Concatenate, "bijections_ate_maf"),
    (4, Stack, "bijections_tanh"),
])
def test_flexible_continuous_layout(flexible_continuous, idx, expected, label):
    assert isinstance(flexible_continuous.bijection.bijections[idx], expected), label


# --- train_frugal_flow_flexible_discrete ----------------------------------


@pytest.fixture(scope="module")
def flexible_discrete(synth):
    flow, _ = train_frugal_flow_flexible_discrete(
        key=synth["key"],
        y=synth["y_disc"],
        u_z=synth["u_z"],
        condition=synth["cond"],
        causal_model_args=MAF,
        **MAF, **FIT,
    )
    return flow


def test_flexible_discrete_chain_length(flexible_discrete):
    assert len(flexible_discrete.bijection.bijections) == 5


@pytest.mark.parametrize("idx,expected,label", [
    (0, Affine, "base rescaler (frozen)"),
    (1, Invert, "first_uniform MAF block"),
    (2, Concatenate, "bijections_affine (frozen)"),
    (3, Concatenate, "bijections_ate_maf"),
    (4, Concatenate, "bijections_affine_output (frozen)"),
])
def test_flexible_discrete_layout(flexible_discrete, idx, expected, label):
    assert isinstance(flexible_discrete.bijection.bijections[idx], expected), label


def test_flexible_discrete_inner_maf_is_first_uniform(flexible_discrete):
    """``sample_outcome.sample_outcome:115`` indexes 4 levels deep to confirm
    the inner MAF is the first-uniform variant."""
    inner = (
        flexible_discrete.bijection.bijections[1]
        .bijection.bijection.bijections[0]
    )
    assert isinstance(inner, MaskedAutoregressiveFirstUniform)


# --- train_frugal_flow_gaussian -------------------------------------------


@pytest.fixture(scope="module")
def gaussian(synth):
    flow, _ = train_frugal_flow_gaussian(
        key=synth["key"],
        y=synth["y_cont"],
        u_z=synth["u_z"],
        condition=synth["cond"],
        causal_model_args={
            "ate": jnp.zeros((1,)),
            "scale": jnp.array(1.0),
            "const": jnp.array(0.0),
        },
        **MAF, **FIT,
    )
    return flow


def test_gaussian_chain_length(gaussian):
    assert len(gaussian.bijection.bijections) == 4


@pytest.mark.parametrize("idx,expected,label", [
    (0, Affine, "base rescaler (frozen)"),
    (1, Invert, "heterogeneous MAF block"),
    (2, Invert, "Invert(Affine) for unbounded support (frozen)"),
    (3, Invert, "Invert(marginal_transform) (the causal-CDF stack)"),
])
def test_gaussian_layout(gaussian, idx, expected, label):
    assert isinstance(gaussian.bijection.bijections[idx], expected), label


def test_gaussian_inner_maf_is_heterogeneous(gaussian):
    """``sample_outcome.sample_outcome:115`` deep-indexed type assertion."""
    inner = gaussian.bijection.bijections[1].bijection.bijection.bijections[0]
    assert isinstance(inner, MaskedAutoregressiveHeterogeneous)
