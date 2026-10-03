"""Reversed-copula arm (2026-09-30): structure, invertibility, counterfactuals, penalty gradient."""
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import paramax
from flowjax.train.losses import MaximumLikelihoodLoss
from frugal_flows.reversed_copula import (
    COPULA_BLOCK,
    MARGIN_BLOCK,
    RankUniformityLoss,
    build_reversed_flow,
    counterfactual_reversed,
    image_ranks,
    interventional_samples_reversed,
    train_frugal_flow_reversed,
)
from paramax import NonTrainable

K, D = 3, 1
ARGS = {"RQS_knots": 4, "nn_depth": 1, "nn_width": 8, "flow_layers": 2, "conditioner": "mlp"}


def _flow(seed=0):
    return paramax.unwrap(build_reversed_flow(jr.PRNGKey(seed), K, 1, D, ARGS, copula_nn_width=8,
                                              copula_flow_layers=2, copula_rqs_knots=4))


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    t = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    y = np.column_stack([z + t + rng.standard_normal(n) for _ in range(K)])
    u = (np.argsort(np.argsort(z)) + 1.0) / (n + 1)
    return jnp.asarray(y), jnp.asarray(u[:, None]), jnp.asarray(t[:, None])


def test_margin_ignores_u_and_copula_ignores_t():
    d = _flow()
    blocks = d.bijection.bijections
    x = jnp.array([0.1, -0.3, 0.5])
    a = blocks[MARGIN_BLOCK].transform(x, jnp.array([1.0, 0.2]))
    b = blocks[MARGIN_BLOCK].transform(x, jnp.array([1.0, 0.9]))
    assert jnp.allclose(a, b)
    a = blocks[COPULA_BLOCK].transform(x, jnp.array([0.0, 0.4]))
    b = blocks[COPULA_BLOCK].transform(x, jnp.array([1.0, 0.4]))
    assert jnp.allclose(a, b)


def test_inverts_and_log_prob_finite():
    d = _flow()
    w = jnp.array([[0.2, -0.5, 0.7], [-0.9, 0.1, 0.3]])
    c = jnp.array([[0.0, 0.3], [1.0, 0.8]])
    y = jax.vmap(d.bijection.transform)(w, c)
    back = jax.vmap(d.bijection.inverse)(y, c)
    assert jnp.allclose(back, w, atol=1e-5)
    assert np.all(np.isfinite(np.asarray(d.log_prob(y, c))))


def test_counterfactual_is_margin_transport_and_ignores_u():
    d = _flow()
    y, u, t = _data(20)
    same = counterfactual_reversed(d, y, t, t, D)
    assert np.allclose(same, np.asarray(y), atol=1e-4)          # t' = t returns the image
    cf = counterfactual_reversed(d, y, t, 1 - t, D)
    back = counterfactual_reversed(d, jnp.asarray(cf), 1 - t, t, D)
    assert np.allclose(back, np.asarray(y), atol=1e-4)          # round trip


def test_rank_penalty_gradient_reaches_only_the_margin():
    d = _flow()
    y, u, t = _data()
    cond = jnp.hstack([t, u])
    params, static = eqx.partition(d, eqx.is_inexact_array, is_leaf=lambda x: isinstance(x, NonTrainable))
    pen, mle = RankUniformityLoss(weight=1.0, n_ref=200), MaximumLikelihoodLoss()
    g = eqx.filter_grad(lambda p: pen(p, static, y, cond, key=jr.key(0)) - mle(p, static, y, cond))(params)
    g_cop = jax.tree_util.tree_leaves(g.bijection.bijections[COPULA_BLOCK])
    g_mar = jax.tree_util.tree_leaves(g.bijection.bijections[MARGIN_BLOCK])
    assert max(float(jnp.abs(a).max()) for a in g_cop) == 0.0
    assert max(float(jnp.abs(a).max()) for a in g_mar) > 0.0


def test_fit_and_readout_run():
    y, u, t = _data()
    flow, losses = train_frugal_flow_reversed(jr.PRNGKey(0), y, u, t, ARGS, nn_width=8, flow_layers=2,
                                              RQS_knots=4, max_epochs=2, max_patience=2,
                                              rank_penalty_weight=10.0)
    assert np.all(np.isfinite(losses["train"])) and np.all(np.isfinite(losses["val"]))
    out = interventional_samples_reversed(jr.key(1), flow, u, n_mc=200)
    for k in ("gformula_y0", "gformula_y1", "margin_y0", "margin_y1"):
        assert out[k].shape == (200, K)
    r = image_ranks(paramax.unwrap(flow), y, jnp.hstack([t, u]))
    assert float(jnp.abs(r).max()) <= 1.0 + 1e-6
