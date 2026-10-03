"""User-facing API of the Gaussian-scale frugal flow: one-call fit, defaults, save/load, examples."""
import os

import frugal_flows as ff
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from frugal_flows import gaussian_scale as gs

SMALL = dict(copula_args=dict(nn_width=8, flow_layers=2, RQS_knots=4),
             margin_args=dict(nn_width=8, flow_layers=2, RQS_knots=4))
EXAMPLE = os.path.join(os.path.dirname(__file__), "..", "examples", "morphomnist_8x8_n5000", "G-flex-std")


def _data(n=800, K=3, seed=0, shift=1.0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 2))
    t = (rng.random(n) < 1 / (1 + np.exp(-z[:, 0]))).astype(float)          # confounded by z[:, 0]
    y = z[:, :1] + shift * t[:, None] + 0.5 * rng.standard_normal((n, K)) + 3.0
    return y, z, t


def test_minimal_train_call_needs_no_margin_args():
    y, z, t = _data(n=120)
    flow, losses = ff.train_frugal_flow(jr.key(0), y=jnp.asarray(y), u_z=jnp.asarray(ff.ecdf_ranks(z)),
                                        condition=jnp.asarray(t[:, None]), causal_model="flexible_continuous_gaussian",
                                        max_epochs=1, show_progress=False)
    assert flow.sample(jr.key(1), condition=jnp.zeros((5, 1))).shape == (5, 3 + 2)


def test_one_call_fit_recovers_a_planted_shift():
    y, z, t = _data()
    flow, ot, info = ff.fit_gaussian_frugal_flow(jr.key(0), y, z, t, max_epochs=150, max_patience=20,
                                                 learning_rate=3e-3, **SMALL)
    s = ff.interventional_samples(jr.key(1), flow, 1, 4000, outcome_transform=ot, dim_y=3)
    assert np.all(np.abs(np.asarray(s["ate"]) - 1.0) < 0.3)                 # original Y scale
    cf = np.asarray(ot.inverse(ff.counterfactual_gaussian(flow, ot.forward(jnp.asarray(y)), t, 1 - t)))
    assert cf.shape == y.shape and np.isfinite(cf).all()
    assert set(info) >= {"losses", "build_kwargs", "u_z"}


def test_save_load_round_trip(tmp_path):
    y, z, t = _data(n=200)
    flow, ot, info = ff.fit_gaussian_frugal_flow(jr.key(0), y, z, t, max_epochs=2, **SMALL)
    ff.save_gaussian_flow(str(tmp_path), flow, info["build_kwargs"], outcome_transform=ot)
    flow2, ot2 = ff.load_gaussian_flow(str(tmp_path))
    x = jnp.hstack([ot.forward(jnp.asarray(y)), gs.normal_scores_from_uniform(info["u_z"])])
    c = jnp.asarray(t[:, None])
    assert np.allclose(flow.log_prob(x, c), flow2.log_prob(x, c))
    assert np.allclose(flow.sample(jr.key(3), condition=c), flow2.sample(jr.key(3), condition=c))
    assert np.allclose(ot.forward(jnp.asarray(y)), ot2.forward(jnp.asarray(y)))


def test_single_outcome_column():
    y, z, t = _data(n=150, K=1)
    flow, ot, _ = ff.fit_gaussian_frugal_flow(jr.key(0), y[:, 0], z, t, max_epochs=2, **SMALL)
    assert flow.sample(jr.key(1), condition=jnp.ones((4, 1))).shape == (4, 1 + 2)


def test_shift_margin_via_wrapper():
    y, z, t = _data(n=150)
    flow, _, _ = ff.fit_gaussian_frugal_flow(jr.key(0), y, z, t, margin="shift", max_epochs=2, **SMALL)
    assert gs.shift_vector(flow).shape == (3,)


@pytest.mark.skipif(not os.path.exists(os.path.join(EXAMPLE, "model_spec.json")), reason="example weights absent")
def test_example_weights_load():
    flow, ot = ff.load_gaussian_flow(EXAMPLE)
    s = ff.interventional_samples(jr.key(0), flow, 1, 2000, outcome_transform=ot, dim_y=64)
    tau = np.load(os.path.join(EXAMPLE, "arrays.npz"))["tau_hat"]
    assert np.abs(np.asarray(s["ate"]) - tau).mean() < 0.02     # MC read-out matches the saved map
