"""Tests for the interventional ATE read-out promoted into the package.

The do(0)/do(1) common-random-number read-out (``intervene`` in the old
diagnostics) now lives in ``frugal_flows.interventions`` and is exposed as
``FrugalFlowModel.estimate_ate`` / ``sample_do``. The correctness that matters:

1. **Paired CRN + transform inversion.** ``interventional_samples`` reads dim 0 of
   the fitted flow at T=0 vs T=1 under the SAME key and inverts the outcome
   transform BEFORE differencing -- so the ATE is on the original Y scale. A
   nonlinear transform is not estimand-preserving, so inverting-then-differencing
   must NOT equal differencing-on-the-transformed-scale.
2. **The model wires it.** ``estimate_ate`` delegates to the package function with
   the model's fitted flow + fitted outcome transform; ``sample_do`` inverts too;
   both refuse to run before a flow is fitted.
3. **The warm-start builder is self-contained.** ``pretrain_causal_margin`` returns
   a treatment-conditioned bijection (``cond_shape == (cond_dim,)``) ready to graft
   -- no diagnostics code needed.

The flow-dependent parts use a tiny deterministic fake flow so the tests are fast
and need no training; ``pretrain_causal_margin`` runs a 2-epoch real fit.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from frugal_flows.causal_flows import pretrain_causal_margin
from frugal_flows.interventions import interventional_samples, tau_curve
from frugal_flows.outcome_transforms import OutcomeTransform


class FakeFlow:
    """Deterministic stand-in for a fitted frugal flow.

    ``sample(key, condition)`` ignores the key (so common-random-numbers holds by
    construction) and returns dim 0 = ``z + shift * t``, where ``t`` is the
    condition level (0 for do(0), 1 for do(1)). So on the fitting scale the effect
    is a pure additive ``shift`` at every quantile.
    """

    def __init__(self, z, shift):
        self.z = np.asarray(z, dtype=float)
        self.shift = float(shift)

    def sample(self, key, condition):
        cond = np.asarray(condition)
        n = cond.shape[0]
        t = float(cond.reshape(n, -1)[0, 0])
        col0 = self.z[:n] + self.shift * t
        return jnp.asarray(np.stack([col0, np.zeros(n)], axis=1))


N_MC = 400
SHIFT = 0.7
Z = np.linspace(-1.0, 1.0, N_MC)  # base draws on the fitting scale


# --------------------------------------------------------------------------- #
# 1. interventional_samples: paired CRN + transform inversion
# --------------------------------------------------------------------------- #
def test_identity_ate_is_the_additive_shift():
    flow = FakeFlow(Z, SHIFT)
    out = interventional_samples(jr.key(0), flow, cond_dim=1, n_mc=N_MC)
    # pure location shift on the identity scale => ATE == shift, no heterogeneity
    assert out["ate"] == pytest.approx(SHIFT, abs=1e-9)
    assert out["tau_sd"] == pytest.approx(0.0, abs=1e-9)
    assert not out["anynan"]
    assert set(out) >= {"y0", "y1", "mean0", "mean1", "var0", "var1", "ate", "tau_sd"}


def test_nonlinear_transform_is_inverted_before_differencing():
    """log-scale additive shift => original-scale ATE = mean(exp z)*(e^shift - 1)."""
    flow = FakeFlow(Z, SHIFT)  # z, z+shift are on the LOG (fitting) scale
    t = OutcomeTransform("log", floor=0.0)  # inverse = exp; needs no fit
    out = interventional_samples(jr.key(0), flow, cond_dim=1, n_mc=N_MC, outcome_transform=t)

    expected = float(np.mean(np.exp(Z)) * (np.exp(SHIFT) - 1.0))
    assert out["ate"] == pytest.approx(expected, rel=1e-6)

    # estimand non-invariance: this must NOT equal the transformed-scale ATE (=shift)
    assert abs(out["ate"] - SHIFT) > 0.1


def test_floor_shifts_level_but_not_the_contrast():
    """b cancels in the contrast: exp(z)+b differenced drops b, ATE is floor-invariant."""
    flow = FakeFlow(Z, SHIFT)
    a = interventional_samples(jr.key(0), flow, 1, N_MC, outcome_transform=OutcomeTransform("log", floor=0.0))
    b = interventional_samples(jr.key(0), flow, 1, N_MC, outcome_transform=OutcomeTransform("log", floor=3.5))
    assert a["ate"] == pytest.approx(b["ate"], rel=1e-9)
    assert b["mean0"] == pytest.approx(a["mean0"] + 3.5, rel=1e-9)  # level shifts by b


def test_tau_curve_flat_for_location_shift():
    flow = FakeFlow(Z, SHIFT)
    out = interventional_samples(jr.key(0), flow, 1, N_MC)
    u, tau = tau_curve(out["y0"], out["y1"])
    assert u.shape == tau.shape
    assert np.allclose(tau, SHIFT, atol=1e-9)  # flat at the ATE, no spurious slope


# --------------------------------------------------------------------------- #
# 2. FrugalFlowModel wiring
# --------------------------------------------------------------------------- #
def _model_with_fake_flow(transform, shift=SHIFT):
    bench = pytest.importorskip("frugal_flows.benchmarking")  # imports wandb
    X = np.ones((N_MC, 1))
    Y = Z.reshape(N_MC, 1)
    model = bench.FrugalFlowModel(Y=Y, X=X, outcome_transform=transform)
    model.frugal_flow = FakeFlow(Z, shift)
    return model


def test_estimate_ate_matches_package_function_and_inverts_transform():
    t = OutcomeTransform("log", floor=0.0)
    model = _model_with_fake_flow(t)
    got = model.estimate_ate(jr.key(0), n_mc=N_MC)
    ref = interventional_samples(jr.key(0), model.frugal_flow, 1, N_MC, outcome_transform=model.outcome_transform)
    assert got["ate"] == pytest.approx(ref["ate"], rel=1e-9)
    # original-scale, not the transformed-scale shift
    assert got["ate"] == pytest.approx(float(np.mean(np.exp(Z)) * (np.exp(SHIFT) - 1.0)), rel=1e-6)


def test_sample_do_returns_inverted_samples():
    model = _model_with_fake_flow(OutcomeTransform("log", floor=0.0))
    y1 = model.sample_do(jr.key(0), t=1, n_mc=N_MC)
    assert np.allclose(y1, np.exp(Z + SHIFT), rtol=1e-6)


def test_estimate_ate_requires_a_fitted_flow():
    bench = pytest.importorskip("frugal_flows.benchmarking")
    model = bench.FrugalFlowModel(Y=Z.reshape(N_MC, 1), X=np.ones((N_MC, 1)))
    with pytest.raises(RuntimeError, match="requires a fitted flow"):
        model.estimate_ate(jr.key(0), n_mc=10)


# --------------------------------------------------------------------------- #
# 3. warm-start builder is self-contained
# --------------------------------------------------------------------------- #
def test_pretrain_causal_margin_returns_conditional_bijection():
    """A 2-epoch fit must yield the treatment-conditioned margin (cond_shape (1,)),
    i.e. the extraction picks the conditional bijection, not the base's Affine."""
    rng = np.random.default_rng(0)
    n = 200
    X = rng.integers(0, 2, size=(n, 1)).astype(float)
    y = (0.5 * X[:, 0] + 0.3 * rng.standard_normal(n)).reshape(n, 1)
    cargs = {"nn_depth": 4, "nn_width": 50, "RQS_knots": 8, "flow_layers": 4}
    margin = pretrain_causal_margin(
        jr.PRNGKey(0), jnp.asarray(y), jnp.asarray(X), cargs,
        learning_rate=5e-3, max_epochs=2, max_patience=2, batch_size=100,
    )
    assert margin.cond_shape == (1,)
