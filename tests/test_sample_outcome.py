"""Tests for ``sample_outcome``.

``logistic_outcome``: inverse-CDF sampling of a Bernoulli: given uniform
``u_y`` and ``p = sigmoid(ate*x + const)``, return ``1`` iff
``u_y >= 1 - p``. So ``P(Y=1) = P(u_y >= 1-p) = p``. Both the exact
threshold and the empirical rate are checked.

``sample_outcome`` (the dispatcher): one functional test per
``causal_model`` branch — ``logistic_regression``, ``causal_cdf``,
``location_translation``. The first two are exercised through the
``u_yx``-only path (no flow needed). The third requires a trained
``train_frugal_flow_location_translation`` flow because
``location_translation_outcome`` reaches into the flow's chain at
depths 3 and 4 to recover the causal slot.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from frugal_flows.causal_flows import train_frugal_flow_location_translation
from frugal_flows.sample_outcome import logistic_outcome, sample_outcome


def test_logistic_outcome_exact_threshold():
    """p=0.5 (ate*x+const=0): u_y>=0.5 -> 1, else 0."""
    u_y = jnp.array([0.1, 0.49, 0.5, 0.9])
    x = jnp.zeros(4)
    out = logistic_outcome(u_y=u_y, ate=1.0, causal_condition=x, const=0.0)
    assert jnp.array_equal(out, jnp.array([0, 0, 1, 1]))


def test_logistic_outcome_empirical_rate_matches_sigmoid():
    """Empirical P(Y=1) ≈ sigmoid(ate*x + const)."""
    n = 40_000
    ate, const, x_val = 2.0, -0.3, 1.0
    u_y = jr.uniform(jr.PRNGKey(0), (n,))
    x = jnp.full((n,), x_val)
    out = logistic_outcome(u_y=u_y, ate=ate, causal_condition=x, const=const)

    p = 1.0 / (1.0 + jnp.exp(-(ate * x_val + const)))  # sigmoid(1.7) ≈ 0.8455
    assert out.mean() == pytest.approx(float(p), abs=0.01)


# --- sample_outcome dispatcher ------------------------------------------


MAF = dict(nn_depth=1, nn_width=4, RQS_knots=4, flow_layers=2)
FIT = dict(max_epochs=1, max_patience=1, batch_size=16, show_progress=False)


def test_sample_outcome_logistic_regression():
    """Dispatcher routes ``causal_model='logistic_regression'`` to
    ``logistic_outcome``. No flow required. Output is 0/1 ints of length
    ``n_samples``."""
    n = 64
    u_yx = jr.uniform(jr.PRNGKey(0), (n,))
    causal_condition = jnp.zeros((n, 1))
    out = sample_outcome(
        key=jr.PRNGKey(1),
        n_samples=n,
        causal_model="logistic_regression",
        causal_condition=causal_condition,
        u_yx=u_yx,
        ate=1.0,
        const=0.0,
    )
    assert out.shape == (n,)
    assert set(int(v) for v in out.tolist()).issubset({0, 1})


def test_sample_outcome_causal_cdf():
    """Dispatcher routes ``causal_model='causal_cdf'`` to
    ``causal_cdf_outcome`` (which inverts ``UnivariateNormalCDF`` at the
    given quantiles). No flow required. Output is finite floats of length
    ``n_samples``."""
    n = 64
    u_yx = jr.uniform(jr.PRNGKey(0), (n,))
    causal_condition = jnp.zeros((n, 1))
    out = sample_outcome(
        key=jr.PRNGKey(1),
        n_samples=n,
        causal_model="causal_cdf",
        causal_condition=causal_condition,
        u_yx=u_yx,
        ate=jnp.zeros((1,)),
        scale=jnp.array(1.0),
        const=jnp.array(0.0),
    )
    assert out.shape == (n,)
    assert bool(jnp.isfinite(out).all())


def test_sample_outcome_location_translation():
    """Dispatcher routes ``causal_model='location_translation'`` to
    ``location_translation_outcome``, which walks the trained
    location-translation frugal flow at ``bijections[3].bijections[0]``
    (the ate-MAF over the causal dim) and ``bijections[4].bijections[0]``
    (``Invert(Tanh)`` for unbounded support) before applying ``LocCond``.
    This is the deepest hardcoded indexing in the package."""
    n = 32
    key = jr.PRNGKey(0)
    k_y, k_uz, k_cond, k_u = jr.split(key, 4)
    y = jr.uniform(k_y, (n, 1))
    u_z = jr.uniform(k_uz, (n, 2))
    cond = jr.uniform(k_cond, (n, 1))

    flow, _ = train_frugal_flow_location_translation(
        key=key,
        y=y,
        u_z=u_z,
        condition=cond,
        causal_model_args=MAF | {"ate": 0.0},
        **MAF, **FIT,
    )

    u_yx = jr.uniform(k_u, (n,))
    causal_condition = jnp.zeros((n, 1))
    out = sample_outcome(
        key=jr.PRNGKey(1),
        n_samples=n,
        causal_model="location_translation",
        causal_condition=causal_condition,
        frugal_flow=flow,
        u_yx=u_yx,
        ate=0.5,
        cond_dim=1,
    )
    assert out.shape == (n,)
    assert bool(jnp.isfinite(out).all())
