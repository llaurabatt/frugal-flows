"""Basic-call tests for the ``train_frugal_flow`` dispatcher.

Each test trains a tiny model for one epoch through the dispatcher, once
per ``causal_model`` variant. The assertion is that the call returns
without raising and produces a flow plus a populated losses dictionary.
The rest of the test suite does not exercise these dispatch branches,
so these tests are the only guard against API regressions inside them.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from frugal_flows.causal_flows import train_frugal_flow

DISPATCHER_KWARGS = dict(
    nn_depth=1,
    nn_width=4,
    RQS_knots=4,
    flow_layers=2,
    max_epochs=1,
    max_patience=1,
    batch_size=16,
    show_progress=False,
)

MAF_HYPERS = dict(
    nn_depth=1,
    nn_width=4,
    RQS_knots=4,
    flow_layers=2,
)


def _make_data(key, n=32, n_z=2, cond_dim=1, *, discrete_y=False):
    k_y, k_uz, k_cond = jr.split(key, 3)
    if discrete_y:
        y = jr.randint(k_y, (n, 1), 0, 2)
    else:
        y = jr.uniform(k_y, (n, 1))
    u_z = jr.uniform(k_uz, (n, n_z))
    condition = jr.uniform(k_cond, (n, cond_dim))
    return y, u_z, condition


@pytest.mark.parametrize(
    "causal_model,causal_model_args,discrete_y",
    [
        (
            "gaussian",
            {
                "ate": jnp.zeros((1,)),
                "scale": jnp.array(1.0),
                "const": jnp.array(0.0),
            },
            False,
        ),
        ("flexible_continuous", MAF_HYPERS, False),
        ("flexible_discrete_output", MAF_HYPERS, True),
        ("location_translation", MAF_HYPERS | {"ate": 0.0}, False),
    ],
    ids=["gaussian", "flexible_continuous", "flexible_discrete_output", "location_translation"],
)
def test_train_frugal_flow_dispatcher_runs(
    key,
    causal_model,
    causal_model_args,
    discrete_y,
):
    """The dispatcher branch trains for one epoch without raising."""
    data_key, train_key = jr.split(key)
    y, u_z, condition = _make_data(data_key, discrete_y=discrete_y)

    flow, losses = train_frugal_flow(
        key=train_key,
        y=y,
        u_z=u_z,
        condition=condition,
        causal_model=causal_model,
        causal_model_args=causal_model_args,
        **DISPATCHER_KWARGS,
    )

    assert flow is not None
    assert isinstance(losses, dict)
    assert "train" in losses and "val" in losses
    assert len(losses["train"]) >= 1
