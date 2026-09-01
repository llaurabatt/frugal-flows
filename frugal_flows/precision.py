"""Package-level control of JAX's 64-bit precision.

The flows do heavy ``log`` / ``exp`` / CDF work and were developed, tuned and
validated in **float64**, so that stays the package default. What changes here is
that the default is now *overridable* instead of forced as an import side effect.

Why it needed changing: ``jax_enable_x64`` is a single global interpreter flag.
A module that flips it at import time changes the numerics of every other module
in the process -- including ones imported earlier -- so precision depended on
which frugal-flows modules a caller happened to import, and in what order. It
also could not be turned off without editing package source, which made the
package unusable on hardware that has no float64 at all (TPUs).

How to choose precision, in order of precedence:

1. **Environment** -- ``JAX_ENABLE_X64=0`` (or ``=1``). JAX parses this itself at
   import, and this module then leaves the flag alone. This is the right knob for
   "run this whole job in single precision", including on TPU::

       JAX_ENABLE_X64=0 python my_script.py

2. **Explicit call** -- ``frugal_flows.set_x64(False)``. Call it *before* creating
   any arrays; JAX bakes the dtype in at array-creation time, so flipping the flag
   afterwards leaves already-created arrays at their original precision.

3. **Default** -- float64, applied by ``apply_default_precision()`` only when the
   caller has expressed no preference at all.

Float32 caveat: it is not a free swap. Single precision makes the known
saturation failure modes worse (an ``Invert(Tanh)`` margin saturating to ``±inf``
rather than merely losing accuracy). It has been exercised successfully for
multivariate ``location_translation`` / ``flexible_continuous`` training and
interventional read-out; the ``gaussian`` arm has not been checked in float32.
"""

from __future__ import annotations

import os

import jax

#: The standard JAX environment variable. If it is set, JAX has already parsed it
#: and the caller's choice wins -- this package will not override it.
X64_ENV_VAR = "JAX_ENABLE_X64"

#: What the package asks for when the caller has expressed no preference.
DEFAULT_X64 = True


def set_x64(enabled: bool = True) -> None:
    """Turn JAX's global 64-bit precision flag on or off.

    Call before creating any arrays: JAX fixes dtypes at array-creation time, so
    arrays made before the switch keep the precision they were made with.

    Args:
        enabled: ``True`` for float64 (the package default), ``False`` for float32.
    """
    jax.config.update("jax_enable_x64", bool(enabled))


def x64_enabled() -> bool:
    """Whether JAX is currently in 64-bit mode."""
    return bool(jax.config.jax_enable_x64)


def apply_default_precision() -> bool:
    """Apply the package's float64 default, unless the caller already chose.

    A caller who sets ``JAX_ENABLE_X64`` in the environment has expressed a
    preference, so this leaves the flag untouched. Otherwise it selects
    ``DEFAULT_X64``. Idempotent, and safe to call from more than one module.

    Returns:
        The resulting state of the flag.
    """
    if X64_ENV_VAR not in os.environ:
        set_x64(DEFAULT_X64)
    return x64_enabled()
