"""The causal margin's pixel order (2026-10-02). Default: a random permutation after each layer.
``permute=False`` keeps one order in every layer, so the whole map is triangular in that order
(the Rosenblatt map); the default must be unchanged."""
import os
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from flowjax.bijections import Permute

from frugal_flows.basic_flows import masked_autoregressive_bijection
from frugal_flows.causal_flows import _build_flexible_margin

K = 6
ARGS = {"RQS_knots": 4, "nn_depth": 1, "nn_width": 8, "flow_layers": 3, "conditioner": "mlp"}


def _build(**extra):
    return _build_flexible_margin(jr.key(0), K, jnp.zeros((1, 1)), {**ARGS, **extra})


def _has_permute(bij):
    return any(isinstance(x, Permute) for x in jax.tree_util.tree_leaves(bij, is_leaf=lambda x: isinstance(x, Permute)))


def _jacobian(bij):
    x = jr.uniform(jr.key(1), (K,), minval=-0.8, maxval=0.8)
    return np.asarray(jax.jacfwd(lambda v: bij.transform(v, jnp.ones(1)))(x))


def test_default_is_unchanged_and_shuffles():
    default = _build()
    explicit = _build(margin_permute=True)
    assert _has_permute(default)
    for a, b in zip(jax.tree_util.tree_leaves(eqx.filter(default, eqx.is_array)),
                    jax.tree_util.tree_leaves(eqx.filter(explicit, eqx.is_array))):
        assert np.array_equal(a, b)
    J = _jacobian(default)
    assert np.abs(np.triu(J, 1)).max() > 1e-6 and np.abs(np.tril(J, -1)).max() > 1e-6


def test_fixed_order_is_triangular_with_the_same_layer_weights():
    fixed = _build(margin_permute=False)
    assert not _has_permute(fixed)
    J = _jacobian(fixed)
    assert np.abs(np.triu(J, 1)).max() < 1e-12          # coordinate k depends on coordinates <= k only
    assert np.all(np.diag(J) > 0)
    # same keys: the autoregressive layers start from the same weights as the default's
    first = lambda b: np.asarray(jax.tree_util.tree_leaves(eqx.filter(b, eqx.is_array))[0])
    assert np.array_equal(first(fixed), first(_build()))


def test_fixed_order_refused_for_transformer():
    with pytest.raises(ValueError, match="mlp"):
        _build_flexible_margin(jr.key(0), K, jnp.zeros((1, 1)),
                               {**ARGS, "conditioner": "transformer", "nn_width": 8, "margin_permute": False})


MM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "validation", "morphomnist")
sys.path.insert(0, os.path.abspath(MM))


def test_exp_ate_recovery_tag_and_args():
    E = pytest.importorskip("exp_ate_recovery")
    base = E.Config(preset="exp2_confounded_homogeneous", size=4, arm="flexible_continuous")
    assert "mfix" not in E.variant_tag(base) and E._margin_order_args(base) == {}
    fixed = E.Config(preset="exp2_confounded_homogeneous", size=4, arm="flexible_continuous", margin_order="fixed")
    assert E.variant_tag(fixed).endswith("mfix")
    assert E._margin_order_args(fixed) == {"margin_permute": False}
    with pytest.raises(ValueError):
        E._margin_order_args(E.Config(preset="exp2_confounded_homogeneous", size=4, arm="flexible_continuous", margin_order="bogus"))
