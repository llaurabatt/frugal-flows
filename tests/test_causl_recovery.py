"""End-to-end ATE-recovery tests on causl-simulated data (the paper's DGP).

The wiring tests in ``test_benchmarking.py`` prove the pipeline *runs*; these
prove it *recovers the right causal effect* -- which is what "does it actually
work" means for a causal method. Data is simulated by R's ``causl`` package and
cached as .npz fixtures (regenerate via ``tests/fixtures/_generate.py``):

- ``gaussian_known_ate`` -- 4 continuous (Gaussian) confounders, ATE = 1.
- ``mixed_known_ate``    -- 2 continuous (Gamma) + 2 discrete confounders, ATE = 1,
  with strong positive confounding (naive difference-in-means ~= 2.0).

Both fixtures are stored **float32**, and here they are fed to ``FrugalFlowModel``
*raw* (no manual cast) -- so these tests also exercise the x64 dtype fix (Bug A):
before it, a float32 ``Z_cont`` crashed ``train_marginal_cdfs`` with a lax.scan
carry-dtype error. The mixed fixture additionally drives the both-Z-blocks path
(continuous marginal flows + discrete empirical CDFs + the propensity flow).

The assertion is a genuine causal claim: the frugal flow must (a) land near the
true ATE and (b) remove most of the confounding bias that the naive
difference-in-means suffers. Deterministic (fixed seeds), so the across-seed
spread is the real estimator spread and the bands below are reproducible, not
flaky -- but they are still statistical, so each trains a real flow and the file
is gated behind RUN_SLOW_TESTS to keep the default suite fast.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_SLOW_TESTS"),
    reason="slow (~90s/case): trains real frugal flows. Set RUN_SLOW_TESTS=1 to run.",
)

benchmarking = pytest.importorskip("frugal_flows.benchmarking")

# Moderate hyperparameters -- enough to converge on N=2000, not the module
# defaults (nn_width=200) which would make each case minutes-long. Mirrors the
# calibrated recovery test in test_outcome_transforms.py.
_MHP = {"max_epochs": 120, "max_patience": 20}
_FHP = {
    "RQS_knots": 8, "nn_depth": 4, "nn_width": 50, "flow_layers": 4,
    "learning_rate": 5e-3, "max_epochs": 200, "max_patience": 25,
    "batch_size": 256, "show_progress": False,
}
_PHP = {
    "nn_depth": 4, "nn_width": 50, "flow_layers": 4,
    "max_epochs": 200, "max_patience": 25, "batch_size": 256, "show_progress": False,
}
_CARGS = {"nn_depth": 4, "nn_width": 50, "RQS_knots": 8, "flow_layers": 4}
_N_SEEDS = 3


def _naive_diff_in_means(Y, X):
    """The confounded estimator: E[Y|X=1] - E[Y|X=0], no adjustment."""
    x = np.asarray(X)[:, 0]
    y = np.asarray(Y)[:, 0]
    return float(y[x == 1].mean() - y[x == 0].mean())


def _recover_ate(data):
    """Fit FrugalFlowModel on the RAW (float32) fixture over several seeds and
    return (per-seed ATE array, naive diff-in-means, true ATE).

    ``Z_disc`` is passed only when the fixture has a discrete block. Feeding the
    arrays raw (no float64 cast) is deliberate -- it exercises Bug A.
    """
    import jax.random as jr

    true_ate = float(data["meta"]["ate"])
    Y, X, Zc, Zd = data["Y"], data["X"], data["Z_cont"], data["Z_disc"]
    naive = _naive_diff_in_means(Y, X)

    ates = []
    for seed in range(_N_SEEDS):
        kwargs = dict(Y=Y, X=X, Z_cont=Zc, outcome_transform="standardize")
        if Zd is not None:
            kwargs["Z_disc"] = Zd
        model = benchmarking.FrugalFlowModel(**kwargs)
        model.train_benchmark_model(
            jr.key(seed), _MHP, _FHP, "flexible_continuous", _CARGS, _PHP
        )
        ates.append(float(model.estimate_ate(jr.key(1000 + seed))["ate"]))
    return np.array(ates), naive, true_ate


def _assert_recovers(ates, naive, true_ate, band):
    """The frugal estimate must land within ``band`` of the truth AND remove most
    of the naive confounding bias (>= 40%)."""
    mean = float(ates.mean())
    assert np.all(np.isfinite(ates)), f"non-finite ATE estimates: {ates}"
    # (a) close to the truth
    assert abs(mean - true_ate) < band, (
        f"seed-mean ATE {mean:.3f} not within {band} of true {true_ate} "
        f"(per-seed {np.round(ates, 3).tolist()})"
    )
    # (b) genuine deconfounding: at least 40% of the naive bias removed. Guards
    # against a model that merely reproduces the (badly biased) naive estimator.
    naive_bias = abs(naive - true_ate)
    assert abs(mean - true_ate) < 0.6 * naive_bias, (
        f"insufficient deconfounding: |{mean:.3f} - {true_ate}| not < 0.6 * "
        f"|{naive:.3f} - {true_ate}| (naive bias {naive_bias:.3f})"
    )


def test_recovers_ate_gaussian_continuous_confounders(causl_gaussian_known_ate):
    """4 continuous Gaussian confounders, true ATE = 1, naive ~= 2.

    Calibrated: seed-mean ~0.98 (sd ~0.07), essentially full deconfounding.
    """
    ates, naive, true_ate = _recover_ate(causl_gaussian_known_ate)
    assert naive > true_ate + 0.5, f"fixture should be confounded; naive={naive:.3f}"
    _assert_recovers(ates, naive, true_ate, band=0.3)


def test_recovers_ate_mixed_confounders(causl_mixed_known_ate):
    """MIXED confounders: 2 continuous (Gamma) + 2 discrete, true ATE = 1,
    strong positive confounding (naive ~= 2.0).

    This is the both-Z-blocks configuration -- it drives the continuous marginal
    flows, the discrete empirical CDFs, and the propensity flow together.
    Calibrated: seed-mean ~1.2 (sd ~0.26), most of the +1.0 bias removed.
    """
    ates, naive, true_ate = _recover_ate(causl_mixed_known_ate)
    assert naive > true_ate + 0.5, f"fixture should be confounded; naive={naive:.3f}"
    _assert_recovers(ates, naive, true_ate, band=0.6)
