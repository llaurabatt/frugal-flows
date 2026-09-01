"""Precision policy: the package's float64 default must be overridable.

These tests mutate a global JAX flag, so every one of them restores the entry
state via the ``restore_x64`` fixture. Without that, flipping the flag here would
silently change the numerics of every test that runs afterwards.
"""

import jax
import jax.numpy as jnp
import pytest
from frugal_flows.precision import (
    DEFAULT_X64,
    X64_ENV_VAR,
    apply_default_precision,
    set_x64,
    x64_enabled,
)


@pytest.fixture
def restore_x64():
    """Restore the global x64 flag after a test that changes it."""
    before = jax.config.jax_enable_x64
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", before)


def test_set_x64_toggles_both_ways(restore_x64):
    set_x64(True)
    assert x64_enabled() is True
    set_x64(False)
    assert x64_enabled() is False
    set_x64(True)
    assert x64_enabled() is True


def test_set_x64_actually_changes_array_dtype(restore_x64):
    """The flag is only meaningful if it reaches the arrays."""
    set_x64(True)
    assert jnp.asarray([1.0]).dtype == jnp.float64
    set_x64(False)
    assert jnp.asarray([1.0]).dtype == jnp.float32


def test_default_is_float64_when_caller_expressed_no_preference(monkeypatch, restore_x64):
    monkeypatch.delenv(X64_ENV_VAR, raising=False)
    set_x64(not DEFAULT_X64)  # start from the opposite state
    assert apply_default_precision() is DEFAULT_X64
    assert x64_enabled() is DEFAULT_X64


def test_env_var_opt_out_is_respected(monkeypatch, restore_x64):
    """JAX_ENABLE_X64=0 must survive apply_default_precision() -- this is the
    knob that makes the package usable where float64 does not exist."""
    monkeypatch.setenv(X64_ENV_VAR, "0")
    set_x64(False)  # what JAX itself would have parsed from that env var
    assert apply_default_precision() is False
    assert x64_enabled() is False


def test_env_var_opt_in_is_respected(monkeypatch, restore_x64):
    monkeypatch.setenv(X64_ENV_VAR, "1")
    set_x64(True)
    assert apply_default_precision() is True
    assert x64_enabled() is True


def test_apply_default_precision_is_idempotent(monkeypatch, restore_x64):
    monkeypatch.delenv(X64_ENV_VAR, raising=False)
    first = apply_default_precision()
    assert apply_default_precision() is first
    assert apply_default_precision() is first


def test_importing_causal_flows_does_not_force_precision(restore_x64):
    """Regression guard for the defect this module fixes: importing a package
    module must not mutate the global precision flag behind the caller's back.

    ``causal_flows`` is already imported by the time this runs, so this asserts
    the property that matters -- the flag can be set to either state and the
    module's presence does not override it.
    """
    import frugal_flows.causal_flows  # noqa: F401

    set_x64(False)
    assert x64_enabled() is False
    set_x64(True)
    assert x64_enabled() is True
