"""counterfactual_gaussian (2026-10-03): margin transport for the Gaussian-scale flexible arm."""
import frugal_flows.gaussian_scale as gs
import jax.numpy as jnp
import jax.random as jr
import numpy as np

K = 3
HY = dict(RQS_knots=4, nn_depth=1, nn_width=8, flow_layers=2)


def _fit(max_epochs=2, lr=5e-4):
    rng = np.random.default_rng(0)
    n = 1000
    z = rng.standard_normal(n)
    t = (rng.random(n) < 0.5).astype(float)
    y = np.column_stack([z + t + rng.standard_normal(n) for _ in range(K)])
    u = (np.argsort(np.argsort(z)) + 1.0) / (n + 1)
    flow, _ = gs.train_frugal_flow_gaussian(jr.PRNGKey(0), jnp.asarray(y), jnp.asarray(u[:, None]),
                                            jnp.asarray(t[:, None]), margin="flexible", causal_model_args=HY,
                                            max_epochs=max_epochs, max_patience=max_epochs, batch_size=100, learning_rate=lr,
                                            show_progress=False, **HY)
    return flow, y, t


def test_same_treatment_returns_the_image_and_round_trips():
    flow, y, t = _fit()
    y, t = y[:20], t[:20]
    assert np.allclose(gs.counterfactual_gaussian(flow, y, t, t), y, atol=1e-4)
    cf = gs.counterfactual_gaussian(flow, y, t, 1 - t)
    assert np.isfinite(cf).all() and cf.shape == y.shape
    assert np.allclose(gs.counterfactual_gaussian(flow, cf, 1 - t, t), y, atol=1e-4)
    assert not np.allclose(cf, y, atol=1e-3)


def test_pure_shift_counterfactual_moves_by_the_effect():
    flow, y, t = _fit(max_epochs=150, lr=5e-3)
    cf = gs.counterfactual_gaussian(flow, y, t, 1 - t)
    shift = np.where(t[:, None] == 0, cf - y, y - cf).mean()      # y(1) - y(0), true effect 1
    assert abs(shift - 1.0) < 0.35
