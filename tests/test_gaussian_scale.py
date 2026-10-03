"""Tests of the Gaussian-scale frugal flow (``frugal_flows.gaussian_scale``, 2026-10-01)."""
from __future__ import annotations

import hashlib
import json

import equinox as eqx
import frugal_flows.gaussian_scale as gs
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from flowjax.distributions import StandardNormal, Transformed
from frugal_flows.causal_flows import train_frugal_flow
from frugal_flows.interventions import interventional_samples
from jax.scipy.stats import norm

MARGIN_ARGS = dict(RQS_knots=4, nn_depth=1, nn_width=8, flow_layers=2)
COPULA = dict(RQS_knots=4, nn_depth=1, nn_width=8, flow_layers=2)


def _flow(margin, K=3, nvars=2, key=0, **margin_extra):
    return gs.build_gaussian_frugal_flow(jr.PRNGKey(key), K, nvars, 1, margin=margin,
                                         causal_model_args=MARGIN_ARGS | margin_extra, **COPULA)


def _data(n=64, K=3, nvars=2, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 2, (n, 1)).astype(float)
    y = rng.normal(size=(n, K)) + t
    g_z = rng.normal(size=(n, nvars))
    return jnp.asarray(y), jnp.asarray(g_z), jnp.asarray(t)


# (1) ------------------------------------------------------------------ independence at init
@pytest.mark.parametrize("margin", gs.MARGINS)
@pytest.mark.parametrize("nvars", [1, 2])
def test_init_copula_is_identity_and_density_factorises(margin, nvars):
    K = 3
    flow = _flow(margin, K, nvars)
    y, g_z, t = _data(K=K, nvars=nvars)
    x = jnp.hstack([y, g_z])
    cop = gs.copula_of(flow)
    out = jax.vmap(cop.transform)(x, t)
    assert np.array_equal(out[:, :K], x[:, :K])                       # identity on the first K
    if nvars == 1:                                                    # no permutation: full identity
        assert np.allclose(out, x, atol=1e-12)
        assert np.allclose(gs.copula_residuals(flow, y, g_z), g_z, atol=1e-12)
    margin_dist = Transformed(StandardNormal((K,)), gs.margin_of(flow))
    want = margin_dist.log_prob(y, t) + norm.logpdf(g_z).sum(1)
    assert np.allclose(flow.log_prob(x, t), want, atol=1e-10, rtol=0)


# (2) ------------------------------------------------------------------ frugal property
@pytest.mark.parametrize("margin", gs.MARGINS)
def test_samples_finite_and_outcome_ignores_copula_base(margin):
    K, nvars, n = 3, 2, 500
    flow = _flow(margin, K, nvars, ate=0.4)
    # perturb the copula so it is not the identity
    flow = eqx.tree_at(lambda f: f.bijection.bijections[0], flow,
                       replace_fn=lambda c: jax.tree_util.tree_map(
                           lambda a: a + 0.3 if eqx.is_inexact_array(a) else a, c))
    t = jnp.asarray(np.random.default_rng(0).integers(0, 2, (n, 1)).astype(float))
    s = flow.sample(jr.key(0), condition=t)
    assert s.shape == (n, K + nvars) and np.isfinite(np.asarray(s)).all()
    e = jr.normal(jr.PRNGKey(1), (n, K + nvars))
    e2 = e.at[:, K:].set(jr.normal(jr.PRNGKey(2), (n, nvars)) * 3.0)
    a = jax.vmap(flow.bijection.transform)(e, t)
    b = jax.vmap(flow.bijection.transform)(e2, t)
    assert np.array_equal(a[:, :K], b[:, :K])                         # outcome: first K base coords + T only
    assert not np.allclose(a[:, K:], b[:, K:])                        # the covariates do move
    r = interventional_samples(jr.key(3), flow, 1, 400, dim_y=K)      # Normal base: no clamping path
    assert r["n_clamped"] == 0 and not r["anynan"] and r["y0"].shape == (400, K)


# (3) ------------------------------------------------------------------ shift arm = ate vector
def test_shift_arm_crn_difference_is_the_ate_vector():
    ate = jnp.array([0.3, -0.7, 1.1])
    flow = _flow("shift", 3, 1, ate=ate)
    assert np.allclose(gs.shift_vector(flow), ate)
    r = interventional_samples(jr.key(0), flow, 1, 1000, dim_y=3)
    assert np.abs(r["y1"] - r["y0"] - np.asarray(ate)).max() < 1e-6
    # and after fitting: the sampled effect is the fitted shift vector
    y, g_z, t = _data(n=200, K=3, nvars=1)
    u = norm.cdf(g_z)
    fit, _ = train_frugal_flow(key=jr.PRNGKey(1), y=y, u_z=u, condition=t,
                               causal_model="location_translation_gaussian", causal_model_args=MARGIN_ARGS,
                               max_epochs=3, learning_rate=1e-2, show_progress=False, **COPULA)
    a = np.asarray(gs.shift_vector(fit))
    assert np.abs(a).max() > 0
    r = interventional_samples(jr.key(0), fit, 1, 1000, dim_y=3)
    assert np.abs(r["y1"] - r["y0"] - a).max() < 1e-6


# (4) ------------------------------------------------------------------ T not in the copula
@pytest.mark.parametrize("margin", gs.MARGINS)
def test_copula_has_zero_gradient_in_t(margin):
    flow = _flow(margin, 3, 2)
    cop = gs.copula_of(flow)
    cop = jax.tree_util.tree_map(lambda a: a + 0.2 if eqx.is_inexact_array(a) else a, cop)
    x = jr.normal(jr.PRNGKey(0), (5,))
    for direction in (cop.transform, cop.inverse):
        J = jax.jacfwd(direction, argnums=1)(x, jnp.array([0.7]))
        assert J.shape == (5, 1) and np.all(np.asarray(J) == 0.0)
        g = jax.grad(lambda c: direction(x, c).sum())(jnp.array([0.3]))
        assert np.all(np.asarray(g) == 0.0)


# (5) ------------------------------------------------------------------ planted Gaussian copula
def test_planted_gaussian_copula_recovered():
    """K=3, nvars=1, n=4000, RCT treatment. Latent e ~ N(0, I_3); g_Z = 0.4 * sum(e) + sqrt(0.52) * eps
    (corr(g_Z, e_k) = 0.4); y_k = mu_k + sd_k * e_k + ate_k * T, fitted after per-column
    standardisation (as the harness's P1). The shift arm must recover ate (data units) within 0.1
    (>= 3 sampling SEs at these sd) and the model-implied corr(g_Z, g_Y_k) within 0.1."""
    rng = np.random.default_rng(0)
    n, rho = 4000, 0.4
    ate, mu, sd = np.array([1.0, -0.5, 0.5]), np.array([0.0, 1.0, -1.0]), np.array([0.5, 1.0, 1.0])
    e = rng.normal(size=(n, 3))
    g_true = rho * e.sum(1, keepdims=True) + np.sqrt(1 - 3 * rho ** 2) * rng.normal(size=(n, 1))
    t = rng.integers(0, 2, (n, 1)).astype(float)
    y = mu + sd * e + ate * t
    m, s = y.mean(0), y.std(0)
    u_z = (np.argsort(np.argsort(g_true[:, 0])) + 1.0)[:, None] / (n + 1)     # ECDF midranks
    flow, losses = train_frugal_flow(
        key=jr.PRNGKey(0), y=jnp.asarray((y - m) / s), u_z=jnp.asarray(u_z), condition=jnp.asarray(t),
        causal_model="location_translation_gaussian",
        causal_model_args=dict(RQS_knots=8, nn_depth=1, nn_width=16, flow_layers=2),
        RQS_knots=8, nn_depth=1, nn_width=16, flow_layers=2, learning_rate=1e-2,
        max_epochs=150, max_patience=20, batch_size=200, show_progress=False)
    assert np.abs(np.asarray(gs.shift_vector(flow)) * s - ate).max() < 0.1
    r = interventional_samples(jr.key(1), flow, 1, 4000, dim_y=3)
    assert np.abs(np.asarray(r["ate"]) * s - ate).max() < 0.1
    S = np.asarray(flow.sample(jr.key(2), condition=jnp.zeros((20000, 1))))
    g_y = np.asarray(gs.outcome_scores(flow, S[:, :3], jnp.zeros((20000, 1))))
    implied = [np.corrcoef(g_y[:, k], S[:, 3])[0, 1] for k in range(3)]
    assert np.abs(np.array(implied) - rho).max() < 0.1, implied


# (6) ------------------------------------------------------------------ scores
def test_normal_scores_roundtrip_and_finite_at_bounds():
    u = jnp.linspace(1e-4, 1 - 1e-4, 101)
    assert np.allclose(gs.uniform_from_normal_scores(gs.normal_scores_from_uniform(u)), u, atol=1e-9)
    g = gs.normal_scores_from_uniform(jnp.array([0.0, 1.0, 0.5]))
    assert np.isfinite(np.asarray(g)).all() and g[0] < -4 and g[1] > 4 and g[2] == 0
    g32 = gs.normal_scores_from_uniform(jnp.array([0.0, 1.0], dtype=jnp.float32))
    assert np.isfinite(np.asarray(g32)).all()


# (7) ------------------------------------------------------------------ dispatch
HY = dict(nn_depth=1, nn_width=4, RQS_knots=4, flow_layers=2)

# Structure hashes (array leaf key paths + shapes, and the module class sequence) of each
# EXISTING arm after a 2-epoch fit, recorded on the unmodified code (diag/halo 395afd6) before
# the Gaussian-scale arm was added. A change here means an existing arm changed.
EXISTING_ARM_STRUCTURE = {
    "gaussian": "5e7efed202c1f8ac",
    "flexible_continuous": "2d5bd299f75b9d3a",
    "flexible_discrete_output": "09c22811f3237ca6",
    "location_translation": "1b3b9da851dbe9dd",
    "flexible_reversed": "9d3af5452633fe6f",
}
EXISTING_ARM_ARGS = {
    "gaussian": {"ate": jnp.zeros((1,)), "scale": jnp.array(1.0), "const": jnp.array(0.0)},
    "flexible_continuous": HY, "flexible_discrete_output": HY,
    "location_translation": HY | {"ate": 0.0}, "flexible_reversed": HY,
}


def _dispatch_data(discrete=False):
    ky, ku, kc = jr.split(jr.PRNGKey(0), 3)
    y = jr.randint(ky, (40, 1), 0, 2) if discrete else jr.uniform(ky, (40, 1))
    return y, jr.uniform(ku, (40, 2)), (jr.uniform(kc, (40, 1)) > 0.5).astype(float)


def _structure_hash(flow) -> str:
    leaves = jax.tree_util.tree_flatten_with_path(eqx.filter(flow, eqx.is_array))[0]
    struct = [(jax.tree_util.keystr(p), tuple(x.shape)) for p, x in leaves]
    mods = []
    jax.tree_util.tree_map(lambda x: None, flow, is_leaf=lambda x: mods.append(type(x).__name__) or False)
    return hashlib.sha256(json.dumps([struct, mods]).encode()).hexdigest()[:16]


@pytest.mark.parametrize("arm", list(EXISTING_ARM_STRUCTURE))
def test_existing_arms_dispatch_unchanged(arm):
    y, u, c = _dispatch_data(arm == "flexible_discrete_output")
    flow, losses = train_frugal_flow(key=jr.PRNGKey(1), y=y, u_z=u, condition=c, causal_model=arm,
                                     causal_model_args=EXISTING_ARM_ARGS[arm], max_epochs=2, max_patience=5,
                                     batch_size=16, show_progress=False, **HY)
    assert len(losses["train"]) == 2
    assert _structure_hash(flow) == EXISTING_ARM_STRUCTURE[arm]


@pytest.mark.parametrize("name,margin", list(gs.CAUSAL_MODELS.items()))
def test_new_arms_dispatch(name, margin):
    y, u, c = _dispatch_data()
    y = jnp.hstack([y, y ** 2])                                       # K = 2
    flow, losses = train_frugal_flow(key=jr.PRNGKey(1), y=y, u_z=u, condition=c, causal_model=name,
                                     causal_model_args=HY, max_epochs=2, max_patience=5, batch_size=16,
                                     show_progress=False, **HY)
    assert len(losses["train"]) == 2 and "val_idx" in losses["info"]
    assert isinstance(gs.copula_of(flow), gs.GaussianCopulaBlock)
    direct, _ = gs.train_frugal_flow_gaussian(jr.PRNGKey(1), y, u, c, margin=margin, causal_model_args=HY,
                                              max_epochs=2, max_patience=5, batch_size=16,
                                              show_progress=False, **HY)
    la = jax.tree_util.tree_leaves(eqx.filter(flow, eqx.is_array))
    lb = jax.tree_util.tree_leaves(eqx.filter(direct, eqx.is_array))
    assert all(np.array_equal(a, b) for a, b in zip(la, lb))        # dispatcher == direct call
    s = flow.sample(jr.key(0), condition=c)
    assert s.shape == (40, 4) and np.isfinite(np.asarray(s)).all()


def test_margin_matches_harness_builders():
    """The margins have the halo S6 layout: Invert(Scan(spread MAF layers)) with T unmasked, and the
    masked-T stack followed by LocCond (weight-for-weight equality with the harness builders is
    asserted in validation/morphomnist/diagnostics/halo/test_halo.py)."""
    key = jr.PRNGKey(5)
    a = MARGIN_ARGS | {"interval": 5.0}
    flex = gs.gaussian_margin_flexible(key, 4, jnp.zeros((1, 1)), a)
    shift = gs.gaussian_margin_shift(key, 4, jnp.zeros((1, 1)), a)
    from flowjax.bijections import Chain, Invert, Scan
    from frugal_flows.bijections import (
        LocCond,
        MaskedAutoregressiveMaskedCond,
        MaskedAutoregressiveSpread,
    )
    assert isinstance(flex, Invert) and isinstance(flex.bijection, Scan)
    assert isinstance(flex.bijection.bijection.bijections[0], MaskedAutoregressiveSpread)
    assert isinstance(shift, Chain) and isinstance(shift.bijections[1], LocCond)
    assert isinstance(shift.bijections[0].bijection.bijection.bijections[0], MaskedAutoregressiveMaskedCond)
    assert flex.cond_shape == shift.cond_shape == (1,)
