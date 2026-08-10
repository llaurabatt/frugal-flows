"""Invertible outcome transforms for the frugal-flow causal margin.

The ``flexible_continuous`` (spline) causal margin passes the outcome through a
fixed ``tanh`` into a rational-quadratic spline on ``[-1, 1]``. A skewed /
heavy-tailed outcome saturates that ``tanh`` -- its mass collapses into a thin
sliver of the spline's domain -- which biases and destabilises the fitted causal
margin. Transforming ``Y`` onto a friendlier scale before fitting fixes this
(empirically, on a Gamma outcome, ``log`` removes the tanh-saturation level bias).
Note the residual small-``n`` ATE bias is **confounding**, not saturation, and is
NOT fixable by any outcome transform -- it needs more data / better deconfounding.

A nonlinear transform is **not** estimand-preserving --
``E[f(Y1)] - E[f(Y0)] != f(E[Y1]) - f(E[Y0])`` -- so the transform must be
inverted on the **sampled counterfactual outcomes** before any contrast is
taken. This module owns both directions, and ``FrugalFlowModel`` applies it
around fit/sample, so the ATE (and any quantile effect) is always computed on
the ORIGINAL ``Y`` scale.

STANDARDIZATION POLICY -- the flow's fitting target is **always standardized**
(mean 0 / sd 1), because a well-conditioned target is uniformly safe (a 15-seed
Gamma study found it neither helps nor hurts recovery) and never a saturation risk.
All fitted statistics (mean/sd, asinh scale) are computed PER OUTCOME COLUMN, so
each dimension of a multivariate ``Y`` is standardized on its own scale:

  * no transform chosen (``FrugalFlowModel(outcome_transform=None)``) -> ``standardize``;
  * a base transform (``log`` / ``asinh``) -> the transformed values are standardized
    **after** the transform by default ("transform, then standardize"; controlled by
    ``post_standardize``, which defaults to ``True`` for ``log`` / ``asinh``);
  * ``"identity"`` / ``"raw"`` -> a true no-op (raw ``Y`` straight into the flow), the
    explicit escape hatch. Pass ``post_standardize=False`` for a raw base transform
    (e.g. plain ``log Y``).

Standardization is estimand-preserving: its inverse (``* sd + mean``) is undone on
the sampled outcomes -- and, for a ``log`` / ``asinh`` base, undone *before* the
base inverse -- so the contrast is always on the original ``Y`` scale.

Available kinds (``b = floor``, the artificial lower bound / "bottom skew min value";
``standardize`` here means the post-transform centre+scale of the ``Z`` values):

======================  ===================================  ==================================
kind                    forward  ``Z = T(Y)``                 inverse  ``Y = T^{-1}(Z)``
======================  ===================================  ==================================
``identity``            ``Y``                                ``Z``
``log``                 ``standardize(log(Y - b))``          ``exp(unstd(Z)) + b``
``asinh``               ``standardize(asinh((Y - b)/s))``    ``sinh(unstd(Z)) * s + b``
``standardize``         ``(Y - mean) / sd``                  ``Z * sd + mean``
======================  ===================================  ==================================

(with ``post_standardize=False`` the ``log`` / ``asinh`` rows drop the ``standardize``
/ ``unstd`` wrapper, giving plain ``log(Y - b)`` / ``asinh((Y - b)/s)``.)

``b`` is a REQUIRED argument for ``log`` / ``asinh`` -- it is deliberately not
defaulted, so a wrong floor is never applied silently to floored data. Pass
``floor=0.0`` for strictly-positive data with no artificial floor; pass the known
lower bound for bottom-coded / detection-limit / clamped data. ``b`` cancels in any
contrast but anchors where the skew compression starts. ``asinh`` is the
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
            A scalar applies to every outcome column; a length-K array gives each of a
            multivariate outcome's columns its own bound. Ignored for ``identity`` /
            ``standardize``.
        asinh_scale: ``s`` for ``asinh``, scalar or one value per outcome column;
            ``None`` -> robust per-column ``median(|Y - b|)`` at fit.
        post_standardize: after the base transform, additionally centre+scale the
            transformed values to mean 0 / sd 1 ("transform, then standardize"). Default
            ``None`` -> **True for log/asinh, False for identity/standardize**: the model
            always wants a well-conditioned fitting target, so a chosen base transform is
            standardized after by default. Pass ``False`` for the raw base transform
            (e.g. plain ``log Y``). Rejected for identity/standardize.
    """

    def __init__(self, kind: str = "identity", floor: float | None = None, asinh_scale: float | None = None,
                 post_standardize: bool | None = None):
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
        # Default policy: a base transform (log/asinh) is ALWAYS followed by a
        # standardize of the transformed values -- the model wants a well-conditioned
        # mean-0/sd-1 fitting target ("transform, then standardize, then fit"). Pass
        # post_standardize=False for the raw base transform (e.g. plain log). identity
        # (true no-op) and standardize (already standardizing) never post-standardize.
        if post_standardize is None:
            post_standardize = kind in ("log", "asinh")
        if post_standardize and kind in ("identity", "standardize"):
            raise ValueError(
                f"post_standardize does not apply to kind={kind!r} "
                f"(identity is a no-op; 'standardize' already standardizes)"
            )
        self.kind = kind
        # floor/asinh_scale: a scalar applies to every outcome column; a length-K
        # array gives each column its own value. Unused for identity/standardize.
        self.floor = 0.0 if floor is None else jnp.asarray(floor, dtype=float)
        self.asinh_scale = None if asinh_scale is None else jnp.asarray(asinh_scale, dtype=float)
        # post_standardize: after the base transform, additionally centre+scale the
        # TRANSFORMED values to mean 0 / sd 1. This is what lands e.g. log Y in the
        # spline margin's fixed-`tanh` linear region instead of hugging the +/-1
        # boundary (where atanh's exploding derivative amplifies small-n tail error).
        # b/scale/mean cancel in the inverse, so the estimand is unchanged.
        self.post_standardize = bool(post_standardize)
        # Fitted statistics: one value per outcome column (a scalar for 1-D Y).
        self._mean = None
        self._sd = None
        self._post_mean = None
        self._post_sd = None
        self.fitted = False

    def fit(self, Y) -> "OutcomeTransform":
        """Learn data-dependent params (asinh scale, standardize mean/sd) and validate."""
        Y = jnp.asarray(Y, dtype=float)
        b = self.floor
        # All statistics are per outcome column (axis 0), so a multivariate Y is
        # standardized / scaled on each dimension's own location and scale. For a
        # 1-D or single-column Y this reduces to the pooled statistics.
        if self.kind == "log":
            ymin = jnp.min(Y, axis=0)
            if not bool(jnp.all(ymin > b)):
                raise ValueError(f"log transform requires Y > floor per column (b={b}); min(Y)={ymin}")
        elif self.kind == "asinh":
            if self.asinh_scale is None:
                s = jnp.median(jnp.abs(Y - b), axis=0)
                self.asinh_scale = jnp.where(s > 0, s, 1.0)
        elif self.kind == "standardize":
            self._mean = jnp.mean(Y, axis=0)
            sd = jnp.std(Y, axis=0)
            self._sd = jnp.where(sd > 0, sd, 1.0)
        # post-transform standardize params are fit on the base-transformed values
        if self.post_standardize:
            z = self._base_forward(Y)
            self._post_mean = jnp.mean(z, axis=0)
            psd = jnp.std(z, axis=0)
            self._post_sd = jnp.where(psd > 0, psd, 1.0)
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

    def _require_scale(self):
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
