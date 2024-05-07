"""Univariate Normal CDF bijection."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flowjax import wrappers
from flowjax.bijections.bijection import AbstractBijection
from flowjax.utils import arraylike_to_array
from jax import Array
from jax.typing import ArrayLike


class UnivariateNormalCDF(AbstractBijection):
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
    scale: Array | wrappers.AbstractUnwrappable[Array]
    const: Array

    def __init__(
        self,
        ate: ArrayLike = 0,
        scale: ArrayLike = 1,
        const: ArrayLike = 0,
        cond_dim: ArrayLike = None,
    ):
        self.ate, scale, self.const = jnp.broadcast_arrays(
            *(arraylike_to_array(a, dtype=float) for a in (ate, scale, const)),
        )
        self.shape = scale.shape  # (1,)
        self.scale = scale  # wrappers.BijectionReparam(scale, SoftPlus()) #why not constraining it to be inputted as positive??
        if cond_dim is None:
            self.cond_shape = None
        else:
            self.cond_shape = (cond_dim,)

    def transform(self, x, condition=None):
        if self.cond_shape is None:
            location_x = self.const
        else:
            location_x = self.ate * condition[0] + self.const
        return jax.scipy.stats.norm.cdf(x, loc=location_x, scale=self.scale)

    def transform_and_log_det(self, x, condition=None):
        if self.cond_shape is None:
            location_x = self.const
        else:
            location_x = self.ate * condition[0] + self.const
        transformed_x = jax.scipy.stats.norm.cdf(x, loc=location_x, scale=self.scale)
        log_det_x = jax.scipy.stats.norm.logpdf(x, loc=location_x, scale=self.scale)
        return transformed_x, log_det_x

    def inverse(self, y, condition=None):
        if self.cond_shape is None:
            location_y = self.const
        else:
            location_y = self.ate * condition[0] + self.const
        return jax.scipy.special.ndtri(y) * self.scale + location_y

    def inverse_and_log_det(self, y, condition=None):
        if self.cond_shape is None:
            location_y = self.const
        else:
            location_y = self.ate * condition[0] + self.const
        inverse_y = jax.scipy.special.ndtri(y) * self.scale + location_y
        log_det_y = -jax.scipy.stats.norm.logpdf(
            inverse_y, loc=location_y, scale=self.scale
        )
        return inverse_y, log_det_y
