"""Tests for ``benchmarking.FrugalFlowModel`` construction.

Covers the fix for bug #8: the constructor previously used ``Z_disc != None`` /
``Z_cont != None``, which broadcast elementwise over array confounders and
raised ``ValueError: ambiguous truth value`` — making the class unusable with
its primary input (real multi-element array confounders). The idiom is now
``is None`` / ``is not None``, which is identity-check-on-the-Python-object
and never broadcasts.

``benchmarking`` imports ``wandb`` at module top (an unused hard dependency);
skip cleanly if it is not installed rather than error the suite.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

benchmarking = pytest.importorskip(
    "frugal_flows.benchmarking",
    reason="frugal_flows.benchmarking imports wandb (unused hard dep) — skip if absent",
)


def test_construct_with_array_z_disc():
    """Bug #8 fix: array-valued Z_disc must not crash construction."""
    Y = np.zeros((10, 1))
    X = np.zeros((10, 1))
    Z_disc = np.zeros((10, 2))
    model = benchmarking.FrugalFlowModel(Y=Y, X=X, Z_disc=Z_disc)
    assert model.conf_shape == 2
    assert model.Z_cont is None


def test_construct_with_array_z_cont():
    """Bug #8 fix: array-valued Z_cont must not crash construction."""
    Y = np.zeros((10, 1))
    X = np.zeros((10, 1))
    Z_cont = np.zeros((10, 3))
    model = benchmarking.FrugalFlowModel(Y=Y, X=X, Z_cont=Z_cont)
    assert model.conf_shape == 3
    assert model.Z_disc is None


def test_construct_with_both_z_disc_and_z_cont():
    """Bug #8 fix: both confounder blocks together must not crash."""
    Y = np.zeros((10, 1))
    X = np.zeros((10, 1))
    Z_disc = np.zeros((10, 2))
    Z_cont = np.zeros((10, 3))
    model = benchmarking.FrugalFlowModel(Y=Y, X=X, Z_disc=Z_disc, Z_cont=Z_cont)
    assert model.conf_shape == 5


def test_construct_with_no_confounders():
    """Degenerate path: both Z blocks None still works (was the only path
    that worked pre-fix; check it still works post-fix)."""
    Y = np.zeros((10, 1))
    X = np.zeros((10, 1))
    model = benchmarking.FrugalFlowModel(Y=Y, X=X)
    assert model.conf_shape == 0
    assert model.Z_disc is None
    assert model.Z_cont is None


# ---------------------------------------------------------------------------
# End-to-end pipeline coverage.
#
# The tests above only cover construction. The block below drives the full
# ``train_benchmark_model -> estimate_ate -> generate_samples`` pipeline on a
# tiny synthetic problem with BOTH continuous and discrete confounders, which is
# the configuration that exercises the propensity column-order path and the
# generate_samples key split. Kept fast with tiny hyperparameters (a few epochs,
# narrow flows) -- this is a wiring smoke test, not an accuracy benchmark.
# ---------------------------------------------------------------------------

N_TRAIN = 200
N_CONT = 2  # continuous confounder columns
N_DISC = 1  # discrete confounder columns


def _tiny_data(seed=0, dtype=np.float64):
    """Confounded binary treatment, continuous outcome, mixed confounders.

    ``X`` depends on both a continuous and the discrete confounder, so the
    propensity path sees both Z blocks. Returned dtypes are controlled by
    ``dtype`` (Y/X/Z_cont); Z_disc is always integer.
    """
    rng = np.random.default_rng(seed)
    Z_cont = rng.normal(size=(N_TRAIN, N_CONT))
    Z_disc = (rng.random((N_TRAIN, N_DISC)) < 0.5).astype(np.int32)
    lin = Z_cont[:, 0] + Z_disc[:, 0]
    X = (rng.random(N_TRAIN) < 1.0 / (1.0 + np.exp(-lin))).astype(dtype)[:, None]
    Y = (2.0 * X[:, 0] + Z_cont[:, 0] + 0.5 * rng.normal(size=N_TRAIN)).astype(dtype)[:, None]
    return (
        Y,
        X,
        Z_cont.astype(dtype),
        Z_disc,
    )


# Deliberately tiny: not the module ``hyperparam_dict`` (nn_width=200), which
# would make these tests slow. A few epochs is enough to exercise the wiring.
_MARGINAL_HP = {"max_epochs": 3, "max_patience": 2}
_FRUGAL_HP = {
    "nn_depth": 1,
    "nn_width": 8,
    "RQS_knots": 4,
    "flow_layers": 2,
    "learning_rate": 5e-3,
    "max_epochs": 3,
    "max_patience": 2,
}
_PROP_HP = {
    "nn_depth": 1,
    "nn_width": 8,
    "flow_layers": 2,
    "max_epochs": 3,
    "max_patience": 2,
}
# location_translation causal margin: architecture + the additive ATE parameter.
_CAUSAL_ARGS = {"nn_depth": 1, "nn_width": 8, "RQS_knots": 4, "flow_layers": 2, "ate": 0.0}
_GEN_OUTCOME_ARGS = {"ate": 2.0}  # LocCond shift used by generate_samples


def _fit_model(seed=0, dtype=np.float64):
    Y, X, Z_cont, Z_disc = _tiny_data(seed=seed, dtype=dtype)
    model = benchmarking.FrugalFlowModel(
        Y=Y, X=X, Z_cont=Z_cont, Z_disc=Z_disc, outcome_transform="standardize"
    )
    model.train_benchmark_model(
        jax.random.key(seed),
        marginal_hyperparam_dict=_MARGINAL_HP,
        frugal_hyperparam_dict=_FRUGAL_HP,
        causal_model="location_translation",
        causal_model_args=_CAUSAL_ARGS,
        prop_flow_hyperparam_dict=_PROP_HP,
    )
    return model


@pytest.fixture(scope="module")
def trained_model():
    """Fit the tiny pipeline once (float64, both Z blocks) and reuse it."""
    return _fit_model(seed=0, dtype=np.float64)


def test_estimate_ate_runs_and_is_finite(trained_model):
    """estimate_ate returns a populated dict with a finite ATE."""
    out = trained_model.estimate_ate(jax.random.key(1), n_mc=2000)
    assert "ate" in out
    assert np.isfinite(float(out["ate"]))


def test_generate_samples_shape_and_validity(trained_model):
    """generate_samples yields a well-formed synthetic dataset (both Z blocks)."""
    n = 300
    df = trained_model.generate_samples(
        jax.random.key(2),
        sampling_size=n,
        copula_param=0.5,
        outcome_causal_model="location_translation",
        outcome_causal_args=_GEN_OUTCOME_ARGS,
    )
    # Y, X, and N_CONT + N_DISC confounder columns.
    assert df.shape == (n, 2 + N_CONT + N_DISC)
    assert list(df.columns[:2]) == ["Y", "X"]
    assert not df.isnull().values.any()
    # X is a binary treatment.
    assert set(np.unique(df["X"].values)).issubset({0.0, 1.0})


def test_generate_samples_is_deterministic(trained_model):
    """Same key -> identical synthetic data.

    This locks the ``generate_samples`` PRNG plumbing: any change that reuses or
    drops a subkey (the key-split count) would surface here as a mismatch.
    """
    kwargs = dict(
        sampling_size=200,
        copula_param=0.5,
        outcome_causal_model="location_translation",
        outcome_causal_args=_GEN_OUTCOME_ARGS,
    )
    df1 = trained_model.generate_samples(jax.random.key(7), **kwargs)
    df2 = trained_model.generate_samples(jax.random.key(7), **kwargs)
    np.testing.assert_array_equal(df1.values, df2.values)


def test_propensity_condition_is_cont_then_disc(monkeypatch):
    """Bug B: the propensity flow must be conditioned on ``[Z_cont, Z_disc]``.

    ``generate_samples`` feeds the propensity flow ``full_Z_samples =
    hstack([Z_cont, Z_disc])``, so ``train_propensity_flow`` must fit on the same
    column order. Previously it fitted on ``hstack([Z_disc, Z_cont])`` -- silently
    transposing the condition columns when both Z blocks are present. Recorded via
    a stubbed trainer so no fitting is needed.
    """
    import jax.numpy as jnp

    Y, X, Z_cont, Z_disc = _tiny_data(seed=0)
    model = benchmarking.FrugalFlowModel(Y=Y, X=X, Z_cont=Z_cont, Z_disc=Z_disc)

    captured = {}

    class _DummyBijection:
        def transform(self, u, condition=None):
            return u

    class _DummyFlow:
        bijection = _DummyBijection()

    def _recorder(*, key, x, condition, **kwargs):
        captured["condition"] = condition
        return _DummyFlow(), None

    monkeypatch.setattr(benchmarking, "train_quantile_propensity_score", _recorder)
    model.train_propensity_flow(jax.random.key(0), _PROP_HP)

    expected = np.asarray(jnp.hstack([model.Z_cont, model.Z_disc]))
    transposed = np.asarray(jnp.hstack([model.Z_disc, model.Z_cont]))
    np.testing.assert_array_equal(np.asarray(captured["condition"]), expected)
    # Guard the assertion is meaningful: the two orders genuinely differ.
    assert not np.array_equal(expected, transposed)
