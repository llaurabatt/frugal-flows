"""Invertible outcome transforms for the frugal-flow causal margin.

The ``flexible_continuous`` (spline) causal margin passes the outcome through a
fixed ``tanh`` into a rational-quadratic spline on ``[-1, 1]``. A skewed /
heavy-tailed outcome saturates that ``tanh`` -- its mass collapses into a thin
sliver of the spline's domain -- which biases and destabilises the fitted causal
margin. Transforming ``Y`` onto a friendlier scale before fitting fixes this
(empirically: on a Gamma outcome, ``log`` removes the level bias and halves the
restart-to-restart ATE noise; see ``validation/diagnostics/SPLINE_HP_FINDINGS.md``).

A nonlinear transform is **not** estimand-preserving --
``E[f(Y1)] - E[f(Y0)] != f(E[Y1]) - f(E[Y0])`` -- so the transform must be
inverted on the **sampled counterfactual outcomes** before any contrast is
taken. This module owns both directions, and ``FrugalFlowModel`` applies it
around fit/sample, so the ATE (and any quantile effect) is always computed on
the ORIGINAL ``Y`` scale.

Available kinds (``b = floor``, the artificial lower bound / "bottom skew min value"):

======================  ================================  ==============================
kind                    forward  ``Z = T(Y)``              inverse  ``Y = T^{-1}(Z)``
======================  ================================  ==============================
``identity`` (default)  ``Y``                             ``Z``
``log``                 ``log(Y - b)``   (requires Y>b)   ``exp(Z) + b``
``asinh``               ``asinh((Y - b) / s)``            ``sinh(Z) * s + b``
``standardize``         ``(Y - mean) / sd``               ``Z * sd + mean``
======================  ================================  ==============================

``b`` is a REQUIRED argument for ``log`` / ``asinh`` -- it is deliberately not
defaulted, so a wrong floor is never applied silently to floored data. Pass
``floor=0.0`` for strictly-positive data with no artificial floor (``log`` -> plain
``log Y``, ``asinh`` -> ``asinh(Y/s)``); pass the known lower bound for bottom-coded
/ detection-limit / clamped data. ``b`` cancels in any contrast but anchors where
the skew compression starts. ``asinh`` is the
signed / zero-safe ``log``: defined for all ``Y`` (no positivity requirement),
linear near ``b``, log-like in the tail, with the crossover set by the scale
``s = asinh_scale`` (``None`` -> a robust data-driven default ``median(|Y - b|)``).

CAVEAT -- a **point mass** at the floor (left-censoring / zero-inflation) is NOT
fixable by any smooth transform: the continuous spline cannot represent an atom.
That needs a mixed discrete-continuous margin, not a transform.

WARNING -- under a non-identity transform the flow's scalar causal parameters
(e.g. the ``gaussian`` arm's ``ate``) are on the TRANSFORMED scale. Recover the
original-scale effect by sampling and inverting (``FrugalFlowModel.generate_samples``
does this automatically); never read the parameter directly as an ATE.
"""

from __future__ import annotations

import jax.numpy as jnp

KINDS = ("identity", "log", "asinh", "standardize")


class OutcomeTransform:
    """A fitted, invertible outcome transform. See module docstring for the maths.

    Args:
        kind: one of ``KINDS`` (``"raw"`` and ``None`` are accepted aliases for
            ``"identity"``).
        floor: ``b``, the artificial lower bound subtracted before ``log`` / ``asinh``.
            REQUIRED for ``log`` / ``asinh`` -- deliberately not defaulted, so a wrong
            (or accidental zero) floor is never applied silently to floored data. Pass
            ``floor=0.0`` explicitly for strictly-positive data with no artificial floor.
            Ignored for ``identity`` / ``standardize``.
        asinh_scale: ``s`` for ``asinh``; ``None`` -> robust ``median(|Y - b|)`` at fit.
    """

    def __init__(self, kind: str = "identity", floor: float | None = None, asinh_scale: float | None = None,
                 post_standardize: bool = False):
        kind = "identity" if kind in ("identity", "raw", None) else kind
        if kind not in KINDS:
            raise ValueError(f"unknown outcome transform {kind!r}; choose from {KINDS}")
        if kind in ("log", "asinh") and floor is None:
            raise ValueError(
                f"{kind!r} transform requires an explicit `floor` (the outcome's lower "
                f"bound / 'bottom skew min value'). It is deliberately not defaulted so a "
                f"non-zero floor is never applied by accident. Pass floor=0.0 for "
                f"strictly-positive data with no artificial floor."
            )
        if post_standardize and kind == "standardize":
            raise ValueError("post_standardize is redundant with kind='standardize'")
        self.kind = kind
        self.floor = 0.0 if floor is None else float(floor)  # unused for identity/standardize
        self.asinh_scale = None if asinh_scale is None else float(asinh_scale)
        # post_standardize: after the base transform, additionally centre+scale the
        # TRANSFORMED values to mean 0 / sd 1. This is what lands e.g. log Y in the
        # spline margin's fixed-`tanh` linear region instead of hugging the +/-1
        # boundary (where atanh's exploding derivative amplifies small-n tail error).
        # b/scale/mean cancel in the inverse, so the estimand is unchanged.
        self.post_standardize = bool(post_standardize)
        self._mean: float | None = None
        self._sd: float | None = None
        self._post_mean: float | None = None
        self._post_sd: float | None = None
        self.fitted = False

    def fit(self, Y) -> "OutcomeTransform":
        """Learn data-dependent params (asinh scale, standardize mean/sd) and validate."""
        Y = jnp.asarray(Y, dtype=float)
        b = self.floor
        if self.kind == "log":
            ymin = float(jnp.min(Y))
            if not ymin > b:
                raise ValueError(f"log transform requires Y > floor (b={b}); min(Y)={ymin:.6g}")
        elif self.kind == "asinh":
            if self.asinh_scale is None:
                s = float(jnp.median(jnp.abs(Y - b)))
                self.asinh_scale = s if s > 0 else 1.0
        elif self.kind == "standardize":
            self._mean = float(jnp.mean(Y))
            sd = float(jnp.std(Y))
            self._sd = sd if sd > 0 else 1.0
        # post-transform standardize params are fit on the base-transformed values
        if self.post_standardize:
            z = self._base_forward(Y)
            self._post_mean = float(jnp.mean(z))
            psd = float(jnp.std(z))
            self._post_sd = psd if psd > 0 else 1.0
        self.fitted = True
        return self

    def _base_forward(self, Y):
        """The base transform (before any post-standardize)."""
        Y = jnp.asarray(Y, dtype=float)
        b = self.floor
        if self.kind == "identity":
            return Y
        if self.kind == "log":
            return jnp.log(Y - b)
        if self.kind == "asinh":
            return jnp.arcsinh((Y - b) / self._require_scale())
        return (Y - self._require("_mean")) / self._require("_sd")  # standardize

    def _base_inverse(self, Z):
        """Inverse of the base transform (after any post-standardize has been undone)."""
        Z = jnp.asarray(Z, dtype=float)
        b = self.floor
        if self.kind == "identity":
            return Z
        if self.kind == "log":
            return jnp.exp(Z) + b
        if self.kind == "asinh":
            return jnp.sinh(Z) * self._require_scale() + b
        return Z * self._require("_sd") + self._require("_mean")  # standardize

    def forward(self, Y):
        """Map original-scale ``Y`` to the fitting scale ``Z`` (base transform, then post-standardize)."""
        z = self._base_forward(Y)
        if self.post_standardize:
            return (z - self._require("_post_mean")) / self._require("_post_sd")
        return z

    def inverse(self, Z):
        """Map fitting-scale samples ``Z`` back to the original ``Y`` scale (undo post-standardize, then base)."""
        Z = jnp.asarray(Z, dtype=float)
        if self.post_standardize:
            Z = Z * self._require("_post_sd") + self._require("_post_mean")
        return self._base_inverse(Z)

    def _require_scale(self) -> float:
        if self.asinh_scale is None:
            raise RuntimeError("asinh transform used before fit(); call fit(Y) first")
        return self.asinh_scale

    def _require(self, attr: str):
        val = getattr(self, attr)
        if val is None:
            raise RuntimeError(f"{self.kind} transform used before fit(); call fit(Y) first")
        return val


def as_outcome_transform(spec) -> OutcomeTransform:
    """Coerce ``None`` / a kind-string / an OutcomeTransform into an OutcomeTransform.

    ``None`` -> identity (a true no-op: nothing to invert). This is the low-level
    coercion; the *pipeline* default (what ``FrugalFlowModel`` does when no transform
    is chosen) is standardize, applied in ``FrugalFlowModel.__init__`` where the data
    is available to fit -- not here, because callers like ``interventional_samples``
    pass ``None`` to mean "the samples are already on the target scale, don't touch
    them". Bare ``"log"`` / ``"asinh"`` strings are rejected: they need an explicit
    ``floor``, so construct ``OutcomeTransform("log", floor=...)`` yourself.
    """
    if spec is None:
        return OutcomeTransform("identity")
    if isinstance(spec, OutcomeTransform):
        return spec
    if isinstance(spec, str):
        if spec in ("log", "asinh"):
            raise ValueError(
                f"{spec!r} needs an explicit floor; pass "
                f"OutcomeTransform({spec!r}, floor=...) instead of the bare string."
            )
        return OutcomeTransform(spec)  # identity / raw / standardize need no floor
    raise TypeError(f"outcome_transform must be None, str, or OutcomeTransform; got {type(spec)}")
