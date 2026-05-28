"""Affine bijections."""

from __future__ import annotations

import jax.numpy as jnp
from flowjax.bijections.bijection import AbstractBijection
from flowjax.utils import arraylike_to_array
from jax import Array
from jax.typing import ArrayLike


class LocCond(AbstractBijection):
    """Condition-driven location shift ``y = x + ate * condition[0]``.

    This is the bijection that injects the causal effect into the frugal flow as
    an explicit, trainable parameter. It applies a pure translation of ``x`` by
    ``ate * condition[0]``; there is no scale term, so the Jacobian is the
    identity and ``log|det J| = 0`` (volume-preserving) in both directions.

    Only the **first** component of ``condition`` is used (``condition[0]``) — in
    the frugal-flows pipeline this is the binary treatment ``X``, so the shift is
    ``ate * X``. Any further conditioning components are ignored. ``ate`` is the
    average treatment effect; it is a trainable array field, optimised when this
    bijection is part of a flow (see ``causal_flows.train_frugal_flow_location_translation``)
    and supplied directly at sampling time (see ``sample_outcome.location_translation_outcome``).

    Forward:  ``y = x + ate * condition[0]``
    Inverse:  ``x = y - ate * condition[0]``

    Note:
        ``transform``/``inverse`` index ``condition[0]`` unconditionally, so a
        non-None ``condition`` is always required; the ``condition=None``
        default in the method signatures is only for interface compatibility
        with ``flowjax.AbstractBijection``.

    Args:
        ate: Average treatment effect — the per-unit shift applied per unit of
            ``condition[0]``. Defaults to 0. Its shape sets the bijection
            ``shape``.
        cond_dim: Length of the conditioning vector. Defaults to 1 (the minimum
            — only ``condition[0]`` is read). Sets ``cond_shape = (cond_dim,)``.
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...]
    ate: Array

    def __init__(
        self,
        ate: ArrayLike = 0,
        cond_dim: int = 1,
    ):
        self.ate = arraylike_to_array(ate)
        self.shape = self.ate.shape
        self.cond_shape = (cond_dim,)

    def transform_and_log_det(self, x, condition=None):
        return x + self.ate * condition[0], jnp.zeros(())

    def inverse_and_log_det(self, y, condition=None):
        return (y - self.ate * condition[0]), jnp.zeros(())
