"""A conditional bijection that sees only part of the condition vector (2026-09-30).

The reversed-copula arm conditions on ``(t, u)`` but its two blocks must each see one part:
the copula reads ``u`` and not ``t``, the causal margin reads ``t`` and not ``u``. Wrapping each
block in ``SelectCondition`` hands it ``condition[idx]`` and nothing else, so the block cannot
depend on the other part by construction (flowjax's ``Chain`` requires one shared ``cond_shape``).
"""
from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from flowjax.bijections import AbstractBijection


class SelectCondition(AbstractBijection):
    """``bijection`` applied with ``condition[idx]`` in place of the full condition.

    Args:
        bijection: a conditional bijection whose ``cond_shape`` is ``(len(idx),)``.
        idx: positions of the full condition vector this bijection may read.
        cond_dim: length of the full condition vector.
    """

    bijection: AbstractBijection
    idx: tuple[int, ...] = eqx.field(static=True)
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...]

    def __init__(self, bijection: AbstractBijection, idx: Sequence[int], cond_dim: int):
        idx = tuple(int(i) for i in idx)
        if bijection.cond_shape != (len(idx),):
            raise ValueError(f"inner cond_shape {bijection.cond_shape} != ({len(idx)},)")
        if not all(0 <= i < cond_dim for i in idx):
            raise ValueError(f"idx {idx} out of range for cond_dim {cond_dim}")
        self.bijection = bijection
        self.idx = idx
        self.shape = bijection.shape
        self.cond_shape = (cond_dim,)

    def _select(self, condition):
        return jnp.asarray(condition)[jnp.array(self.idx)]

    def transform_and_log_det(self, x, condition=None):
        return self.bijection.transform_and_log_det(x, self._select(condition))

    def inverse_and_log_det(self, y, condition=None):
        return self.bijection.inverse_and_log_det(y, self._select(condition))
