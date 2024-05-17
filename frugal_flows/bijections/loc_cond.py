"""Affine bijections."""

from __future__ import annotations

import jax.numpy as jnp
from flowjax.bijections.bijection import AbstractBijection
from flowjax.utils import arraylike_to_array
from jax import Array
from jax.typing import ArrayLike


class LocCond(AbstractBijection):
    """Elementwise affine transformation ``y = a*x + b``.

    ``loc`` and ``scale`` should broadcast to the desired shape of the bijection.
    By default, we constrain the scale parameter to be postive using ``SoftPlus``, but
    other parameterizations can be achieved by replacing the scale parameter after
    construction e.g. using ``eqx.tree_at``.

    Args:
        loc: Location parameter. Defaults to 0.
        scale: Scale parameter. Defaults to 1.
    """

    shape: tuple[int, ...]
    # cond_shape: ClassVar[None] = None
    cond_shape: int | None = None
    ate: Array

    def __init__(
        self,
        ate: ArrayLike = 0,
        cond_dim: ArrayLike = None,
    ):
        self.ate = arraylike_to_array(ate)
        self.shape = self.ate.shape
        if cond_dim is None:
            self.cond_shape = None
        else:
            self.cond_shape = (cond_dim,)

    def transform(self, x, condition=None):
        return x + self.ate * condition[0]

    def transform_and_log_det(self, x, condition=None):
        return x + self.ate * condition[0], jnp.zeros(())

    def inverse(self, y, condition=None):
        return y - self.ate * condition[0]

    def inverse_and_log_det(self, y, condition=None):
        return (y - self.ate * condition[0]), jnp.zeros(())
