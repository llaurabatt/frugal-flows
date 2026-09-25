"""`sample_clamped` reproduces `flow.sample` for a uniform-base flow and falls back to
`flow.sample` for anything else (a Normal base, an object without `base_dist`)."""
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from flowjax.distributions import Normal, Uniform
from flowjax.flows import masked_autoregressive_flow
from frugal_flows.interventions import sample_clamped


def test_uniform_base_matches_flow_sample_when_nothing_is_clamped():
    flow = masked_autoregressive_flow(jr.PRNGKey(0), base_dist=Uniform(-jnp.ones(3), jnp.ones(3)),
                                      cond_dim=1, flow_layers=1, nn_width=4)
    cond = jnp.zeros((50, 1))
    key = jr.key(1)
    ref = np.asarray(flow.sample(key, condition=cond))
    got, n = sample_clamped(key, flow, 50, cond)
    assert n == 0 and np.array_equal(np.asarray(got), ref)
    # unconditional path
    flow_u = masked_autoregressive_flow(jr.PRNGKey(0), base_dist=Uniform(-jnp.ones(3), jnp.ones(3)),
                                        flow_layers=1, nn_width=4)
    ref = np.asarray(flow_u.sample(key, (50,)))
    got, n = sample_clamped(key, flow_u, 50)
    assert n == 0 and np.array_equal(np.asarray(got), ref)


def test_non_uniform_base_and_test_double_fall_back():
    flow = masked_autoregressive_flow(jr.PRNGKey(0), base_dist=Normal(jnp.zeros(3)), cond_dim=1,
                                      flow_layers=1, nn_width=4)
    cond = jnp.ones((20, 1))
    key = jr.key(2)
    got, n = sample_clamped(key, flow, 20, cond)
    assert n == 0 and np.array_equal(np.asarray(got), np.asarray(flow.sample(key, condition=cond)))

    class Fake:
        def sample(self, key, sample_shape=(), condition=None):
            return jnp.full((condition.shape[0], 2), 7.0)

    got, n = sample_clamped(key, Fake(), 4, jnp.zeros((4, 1)))
    assert n == 0 and got.shape == (4, 2) and float(got[0, 0]) == 7.0
