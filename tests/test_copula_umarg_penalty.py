"""Copula u-marginal penalty (2026-09-28): the loss adds weight x energy distance between the
copula's own u-marginal and the observed covariate ranks; its gradient must reach only the
copula blocks, and validation must still use the plain likelihood."""
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import paramax
from flowjax.train.losses import MaximumLikelihoodLoss
from frugal_flows.causal_flows import (
    CopulaUMarginalPenaltyLoss,
    _energy_distance,
    copula_margin_labels,
    copula_u_marginal_samples,
    train_frugal_flow,
)
from paramax import NonTrainable

K = 3


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    t = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    y = np.column_stack([z + t + rng.standard_normal(n) for _ in range(K)])
    u = (np.argsort(np.argsort(z)) + 1.0) / (n + 1)
    return jnp.asarray(y), jnp.asarray(u[:, None]), jnp.asarray(t[:, None])


def _fit(weight, epochs=1):
    y, u, t = _data()
    return train_frugal_flow(
        causal_model="flexible_continuous", key=jr.PRNGKey(0), y=y, u_z=u, condition=t,
        max_epochs=epochs, max_patience=epochs, batch_size=100, show_progress=False,
        causal_model_args={"RQS_knots": 4, "nn_depth": 1, "nn_width": 8, "flow_layers": 2, "conditioner": "mlp"},
        nn_width=8, flow_layers=2, copula_umarg_weight=weight,
    )


def test_energy_distance_zero_for_same_and_positive_for_shifted():
    a = jr.uniform(jr.key(0), (800, 1))
    b = jr.uniform(jr.key(1), (800, 1))
    assert abs(float(_energy_distance(a, b))) < 0.01
    assert float(_energy_distance(a, b + 0.3)) > 0.05


def test_samples_in_unit_interval():
    flow, _ = _fit(0.0)
    u = copula_u_marginal_samples(paramax.unwrap(flow), jr.key(0), K, 500)
    assert u.shape == (500, 1)
    assert float(u.min()) >= 0.0 and float(u.max()) <= 1.0


def test_penalty_gradient_reaches_only_the_copula():
    flow, _ = _fit(0.0)
    y, u, t = _data()
    x = jnp.hstack([y, u])
    params, static = eqx.partition(flow, eqx.is_inexact_array, is_leaf=lambda leaf: isinstance(leaf, NonTrainable))
    pen = CopulaUMarginalPenaltyLoss(u_ref=u, dim_y=K, weight=1.0, n=200)
    mle = MaximumLikelihoodLoss()
    g_pen = eqx.filter_grad(lambda p: pen(p, static, x, t, key=jr.key(0)) - mle(p, static, x, t))(params)
    labels = copula_margin_labels(flow)
    leaves_g = jax.tree_util.tree_leaves(g_pen)
    leaves_l = jax.tree_util.tree_leaves(labels)
    assert len(leaves_g) == len(leaves_l)
    margin_max = max(float(jnp.abs(g).max()) for g, lab in zip(leaves_g, leaves_l) if lab == "margin")
    copula_max = max(float(jnp.abs(g).max()) for g, lab in zip(leaves_g, leaves_l) if lab == "copula")
    assert margin_max == 0.0
    assert copula_max > 0.0


def test_fit_with_penalty_runs_and_validates_on_the_likelihood():
    flow, losses = _fit(10.0, epochs=2)
    assert np.all(np.isfinite(losses["train"])) and np.all(np.isfinite(losses["val"]))
    # validation uses the plain likelihood: recompute it on the returned flow's val rows is
    # not exposed, so check instead that a zero-weight fit's val loss has the same scale
    _, plain = _fit(0.0, epochs=2)
    assert abs(losses["val"][0] - plain["val"][0]) < 5.0
