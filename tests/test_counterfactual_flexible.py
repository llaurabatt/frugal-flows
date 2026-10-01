"""counterfactual_flexible (2026-10-01): margin transport for the flexible-continuous arm."""
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from frugal_flows.causal_flows import train_frugal_flow
from frugal_flows.interventions import counterfactual_flexible

K = 3


def _fit():
    rng = np.random.default_rng(0)
    n = 300
    z = rng.standard_normal(n)
    t = (rng.random(n) < 0.5).astype(float)
    y = np.column_stack([z + t + rng.standard_normal(n) for _ in range(K)])
    u = (np.argsort(np.argsort(z)) + 1.0) / (n + 1)
    flow, _ = train_frugal_flow(
        causal_model="flexible_continuous", key=jr.PRNGKey(0), y=jnp.asarray(y), u_z=jnp.asarray(u[:, None]),
        condition=jnp.asarray(t[:, None]), max_epochs=2, max_patience=2, batch_size=100, show_progress=False,
        causal_model_args={"RQS_knots": 4, "nn_depth": 1, "nn_width": 8, "flow_layers": 2, "conditioner": "mlp"},
        nn_width=8, flow_layers=2)
    return flow, y[:20], u[:20, None], t[:20]


def test_same_treatment_returns_the_image_and_round_trips():
    flow, y, u, t = _fit()
    same = counterfactual_flexible(flow, y, u, t, t)
    assert np.allclose(same, y, atol=1e-4)
    cf = counterfactual_flexible(flow, y, u, t, 1 - t)
    back = counterfactual_flexible(flow, cf, u, 1 - t, t)
    assert np.allclose(back, y, atol=1e-4)
    assert not np.allclose(cf, y, atol=1e-3)          # switching treatment does move the image


def test_covariate_ranks_do_not_change_the_counterfactual():
    flow, y, u, t = _fit()
    a = counterfactual_flexible(flow, y, u, t, 1 - t)
    b = counterfactual_flexible(flow, y, np.full_like(u, 0.5), t, 1 - t)
    assert np.allclose(a, b)
