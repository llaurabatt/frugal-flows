"""copula_lr_mult gives the flexible arm's copula its own learning rate and leaves the margin's
alone: labels land on the right blocks, and after exactly one optimisation step the margin's
parameters are identical with and without the multiplier while the copula's differ."""
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from frugal_flows.causal_flows import (
    COPULA_BLOCKS,
    copula_margin_labels,
    train_frugal_flow_flexible_continuous,
)
from paramax import NonTrainable


def _data(n=64, k=3, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.uniform(size=(n, 1))
    t = (rng.uniform(size=(n, 1)) < 0.5).astype(float)
    y = rng.normal(size=(n, k)) + t + z
    return jnp.asarray(y), jnp.asarray(z), jnp.asarray(t)


def _fit(mult, epochs=1):
    y, u_z, t = _data()
    # batch larger than the training set -> one batch per epoch -> one step per epoch
    return train_frugal_flow_flexible_continuous(
        key=jr.PRNGKey(0), y=y, u_z=u_z, condition=t, learning_rate=1e-2, max_epochs=epochs,
        max_patience=5, batch_size=10_000, nn_width=8, flow_layers=1, RQS_knots=4,
        causal_model_args={"nn_width": 8, "flow_layers": 1, "RQS_knots": 4, "nn_depth": 1},
        show_progress=False, copula_lr_mult=mult)


def _leaves_by_block(flow):
    params, _ = eqx.partition(flow, eqx.is_inexact_array, is_leaf=lambda leaf: isinstance(leaf, NonTrainable))
    labels = copula_margin_labels(flow)
    out = {"copula": [], "margin": []}
    for p, lab in zip(jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(labels), strict=True):
        out[lab].append(np.asarray(p))
    return out


def test_labels_cover_both_blocks():
    flow, _ = _fit(1.0, epochs=1)
    labels = jax.tree_util.tree_leaves(copula_margin_labels(flow))
    assert "copula" in labels and "margin" in labels
    assert COPULA_BLOCKS == (0, 1, 2) and len(flow.bijection.bijections) == 5


def test_one_step_moves_only_the_copula_differently():
    f1, l1 = _fit(1.0)
    f10, l10 = _fit(10.0)
    assert l1["info"]["n_epochs"] == 1
    a, b = _leaves_by_block(f1), _leaves_by_block(f10)
    for x, y in zip(a["margin"], b["margin"], strict=True):
        assert np.array_equal(x, y), "the margin's first step must not depend on the copula's rate"
    assert any(not np.array_equal(x, y) for x, y in zip(a["copula"], b["copula"], strict=True))
