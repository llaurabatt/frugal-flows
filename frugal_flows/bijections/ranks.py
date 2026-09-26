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

    ``lo`` is the lowest rank worth assigning, ``lo = (first output that is used) - 1``:
    ``-1`` for a layer with an unmasked condition (conditioning inputs have rank -1, and
    output 0 can only depend on them through units of rank -1, as in flowjax's own
    conditional layer); ``0`` for an unconditional or fully masked-condition layer; and
    ``c - 1`` for a layer whose first ``c`` coordinates are held fixed and discarded.
    With at least as many units as ranks, the ranks cycle (``lo, lo+1, ..., dim-2, lo,
    ...``); with ``nn_width >= dim`` and ``lo = -1`` that is exactly flowjax's conditional
    rule, and with ``lo = 0`` its unconditional rule. With fewer units the ranks are spread
    evenly and always include ``lo`` and ``dim - 2``, so every input still reaches the last
    output and every used output can use all inputs up to the highest rank below it.
    """
    hi = dim - 2
    if hi < lo:                      # dim 1 without condition: no output depends on inputs
        return jnp.full(nn_width, lo, dtype=jnp.int32)
    n = hi - lo + 1
    if nn_width >= n:
        return lo + jnp.arange(nn_width) % n
    return jnp.round(jnp.linspace(lo, hi, nn_width)).astype(jnp.int32)
