"""Univariate Normal CDF bijection."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flowjax.bijections.bijection import AbstractBijection
from flowjax.utils import arraylike_to_array
from jax import Array
from jax.typing import ArrayLike


class UnivariateNormalCDF(AbstractBijection):
    """Conditional Gaussian CDF — the "causal CDF" used for continuous outcomes.

    Forward maps a real-valued causal latent to a uniform quantile via the
    Gaussian CDF, with the treatment shifting the Gaussian's location:

        loc = ate @ condition + const
        forward:  y = Φ((x - loc) / scale)
        inverse:  x = Φ⁻¹(y) * scale + loc

    At sampling time the causal model uses ``inverse`` (see
    ``sample_outcome.causal_cdf_outcome``): a uniform quantile ``y`` is mapped to
    an outcome whose mean is shifted by ``ate @ condition`` — i.e. ``ate`` is the
    average treatment effect, injected as the Gaussian location. As a flow layer
    it occupies the causal-effect slot (see
    ``causal_flows.train_frugal_flow_gaussian`` / ``_flexible_continuous``).

    Log-determinants (both exact):
        forward:  log|dy/dx| = log φ((x-loc)/scale) - log scale
                              = ``norm.logpdf(x, loc, scale)``
        inverse:  log|dx/dy| = log scale - log φ(Φ⁻¹(y))
                              = ``-norm.logpdf(inverse_y, loc, scale)``

    Warning:
        ``scale`` is a trainable field but is **not** constrained positive (the
        ``SoftPlus`` reparam is commented out in ``__init__``). If optimisation
        drives ``scale`` ≤ 0, ``Φ``/``Φ⁻¹`` produce NaNs. Callers must keep it
        positive.

    Args:
        ate: Average treatment effect, shape ``(cond_dim,)`` when a condition is
            supplied. Defaults to 0. Coerced to a JAX array.
        scale: Std-dev of the Gaussian. Defaults to 1. Must be > 0 (unenforced).
        const: Baseline location (intercept). Defaults to 0.
        cond_dim: Conditioning dimension; sets ``cond_shape = (cond_dim,)``. If
            ``None``, the bijection is unconditional and the location reduces to
            ``const``. When not ``None``, ``ate.shape`` must equal
            ``(cond_dim,)`` (asserted in ``__init__``).
    """

    shape: tuple[int, ...]
    # cond_shape: ClassVar[None] = None
    cond_shape: int | None = None
    ate: Array
    scale: Array
    const: Array

    def __init__(
        self,
        ate: ArrayLike = 0,
        scale: ArrayLike = 1,
        const: ArrayLike = 0,
        cond_dim: ArrayLike = None,
    ):
        self.ate = arraylike_to_array(ate, dtype=float)
        scale, self.const = jnp.broadcast_arrays(
            *(arraylike_to_array(a, dtype=float) for a in (scale, const)),
        )
        self.shape = scale.shape  # (1,)
        self.scale = scale  # Parameterize(lambda x: jnn.softplus(x) + 1e-2, scale) — uncomment to close bug #1b
        if cond_dim is None:
            self.cond_shape = None
        else:
            self.cond_shape = (cond_dim,)
            assert (
                self.cond_shape == self.ate.shape
            ), "ate and condition must have the same shape"

    def transform_and_log_det(self, x, condition=None):
        if self.cond_shape is None:
            location_x = self.const
        else:
            location_x = (self.ate @ condition) + self.const
        transformed_x = jax.scipy.stats.norm.cdf(x, loc=location_x, scale=self.scale)
        log_det_x = jax.scipy.stats.norm.logpdf(x, loc=location_x, scale=self.scale)
        return transformed_x, log_det_x

    def inverse_and_log_det(self, y, condition=None):
        if self.cond_shape is None:
            location_y = self.const
        else:
            location_y = (self.ate @ condition) + self.const
        inverse_y = jax.scipy.special.ndtri(y) * self.scale + location_y
        log_det_y = -jax.scipy.stats.norm.logpdf(
            inverse_y, loc=location_y, scale=self.scale
        )
        return inverse_y, log_det_y
