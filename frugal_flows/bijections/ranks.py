"""Hidden-unit ranks for the package's masked autoregressive networks.

In a MADE-style network a hidden unit of rank ``r`` reads inputs of rank ``<= r`` and
feeds outputs of rank ``> r``. Output ``d`` can therefore use exactly the inputs read by
some hidden unit of rank ``< d``; for it to be able to depend on ALL of inputs ``0..d-1``
there must be a hidden unit of rank ``d - 1``.

The package's layers used ``arange(nn_width) % dim``, flowjax's convention. It covers every
rank only when ``nn_width >= dim``. With fewer hidden units the highest ranks are never
assigned, so the inputs above ``nn_width - 1`` reach no output at all. In the copula, whose
first ``K`` coordinates (the outcome ranks) are held fixed and whose covariates come last,
that cut the covariates off from outcome ranks ``nn_width .. K-1`` (measured 2026-09-25: 14
of 64 pixels at 8x8 with width 50, 206 of 256 at 16x16). Ranks at ``dim - 1`` were also
assigned although they feed no output.
"""
from __future__ import annotations

import jax.numpy as jnp


def autoregressive_hidden_ranks(nn_width: int, dim: int, lo: int = 0):
    """Ranks for ``nn_width`` hidden units covering ``lo .. dim - 2``.

    ``lo`` is the lowest rank worth assigning: the first output that is actually used
    is ``lo + 1`` (for a layer whose first ``c`` coordinates are held fixed, ``lo = c - 1``).
    With at least as many units as ranks, the ranks cycle (``lo, lo+1, ..., dim-2, lo,
    ...``). With fewer units they are spread evenly and always include ``dim - 2``, so
    every input still reaches the last output, and every used output can use all inputs
    up to the highest rank below it.
    """
    hi = dim - 2
    if dim < 2 or hi < lo:
        return jnp.zeros(nn_width, dtype=jnp.int32) + max(lo, 0)
    n = hi - lo + 1
    if nn_width >= n:
        return lo + jnp.arange(nn_width) % n
    return jnp.round(jnp.linspace(lo, hi, nn_width)).astype(jnp.int32)
