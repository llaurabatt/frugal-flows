"""Rational quadratic spline bijection with an additive condition shift."""

from typing import ClassVar

from flowjax.bijections.bijection import AbstractBijection
from flowjax.bijections.rational_quadratic_spline import RationalQuadraticSpline


class RationalQuadraticSplineAdditiveCond(AbstractBijection):
    """Rational-quadratic spline plus an additive condition-driven shift in y.

    Wraps a vanilla :class:`~flowjax.bijections.RationalQuadraticSpline`
    (Durkan et al. 2019, https://arxiv.org/abs/1906.04032) and adds a
    location term that scales the first conditioning variable by ``ate``:

        forward:  y = RQS(x) + ate * condition[0]
        inverse:  x = RQS^{-1}(y - ate * condition[0])

    The shift is additive in ``y``, so ``dy/dx = RQS'(x)``: the inner spline
    owns the log-determinant entirely and the condition contributes zero.

    When ``condition`` is ``None`` (the default call from flowjax, since
    ``cond_shape`` is a ``ClassVar = None``), the shift drops out and the
    bijection is exactly the inner RQS. Conditioning is therefore "soft":
    the shift applies only when a caller explicitly forwards a ``condition``.

    Args:
        knots: Number of inner knots in the spline.
        interval: Interval to transform. Scalar ``B`` is treated as
            ``(-B, B)``; a tuple ``(a, b)`` is used verbatim.
        min_derivative: Lower bound on the derivative at each knot. Defaults
            to ``1e-3``.
        min_width: Minimum bin width, enforced by the underlying knot
            reparameterisation. Defaults to ``1e-3``.
        ate: Average treatment effect; per-unit shift applied as
            ``ate * condition[0]`` when a ``condition`` is supplied. Defaults
            to ``0.0``.
    """

    spline: RationalQuadraticSpline
    ate: float | int
    shape: ClassVar[tuple] = ()
    cond_shape: ClassVar[None] = None

    def __init__(
        self,
        *,
        knots: int,
        interval: float | int | tuple[int | float, int | float],
        min_derivative: float = 1e-3,
        min_width: float | int = 1e-3,
        ate: float | int = 0.0,
    ):
        self.spline = RationalQuadraticSpline(
            knots=knots,
            interval=interval,
            min_derivative=min_derivative,
            min_width=min_width,
        )
        self.ate = ate

    def transform_and_log_det(self, x, condition=None):
        y, log_det = self.spline.transform_and_log_det(x)
        if condition is not None:
            y = y + self.ate * condition[0]
        return y, log_det

    def inverse_and_log_det(self, y, condition=None):
        if condition is not None:
            y = y - self.ate * condition[0]
        x, log_det = self.spline.inverse_and_log_det(y)
        return x, log_det
