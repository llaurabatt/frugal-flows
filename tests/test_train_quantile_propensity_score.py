"""Smoke test for ``train_quantile_propensity_score``.

The MAF math is flowjax's own (trusted); the review concern here is wiring:
the discrete-PIT of the treatment, the conditional flow assembly, and that the
result is a usable conditional density. Trains 1 epoch on tiny data.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
from frugal_flows.train_quantile_propensity_score import (
    train_quantile_propensity_score,
)


def test_propensity_flow_builds_trains_and_is_conditional(key):
    k1, k2 = jr.split(key)
    n = 64
    x = jr.bernoulli(k1, 0.5, (n,)).astype(jnp.int32)  # discrete treatment
    condition = jr.normal(k2, (n, 2))  # confounders Z

    flow, losses, u_x = train_quantile_propensity_score(
        key=key,
        x=x,
        condition=condition,
        flow_layers=2,
        nn_width=8,
        nn_depth=1,
        max_epochs=1,
        batch_size=32,
        show_progress=False,
        return_x_quantiles=True,
    )

    # PIT output is a valid quantile in [0, 1].
    assert u_x.shape == (n,)
    assert jnp.all((u_x >= 0.0) & (u_x <= 1.0))
    # Training produced finite losses.
    assert jnp.isfinite(jnp.asarray(losses["train"][-1]))
    # The flow is a usable *conditional* density: log_prob needs the condition
    # and returns finite per-sample values.
    lp = flow.log_prob(u_x[:, None], condition)
    assert lp.shape == (n,)
    assert jnp.all(jnp.isfinite(lp))
