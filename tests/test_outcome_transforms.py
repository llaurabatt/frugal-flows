"""Tests for ``frugal_flows.outcome_transforms`` and its wiring into FrugalFlowModel.

The outcome transform exists because the ``flexible_continuous`` causal margin
saturates on skewed / heavy-tailed outcomes (raw Y is squashed by ``tanh`` into a
sliver of the spline's ``[-1, 1]`` domain). Correctness has THREE independent
parts, and each is tested here:

1. **The maths round-trips.** ``inverse(forward(Y)) == Y`` for every kind, incl. a
   non-zero floor and signed/zero data (where ``log`` cannot go but ``asinh`` can).
2. **The API forces a conscious floor.** ``log``/``asinh`` require an explicit
   ``floor``; the bare kind-strings are rejected; ``identity`` is a true no-op
   (backward compatibility).
3. **The estimand round-trips.** A nonlinear transform is NOT estimand-preserving
   (``E[f(Y1)] - E[f(Y0)] != f(E[Y1]) - f(E[Y0])``), so the ATE is correct only if
   samples are inverted BEFORE differencing. We assert both that inverting-then-
   differencing recovers the true ATE and that the wrong order does not — then that
   ``FrugalFlowModel`` actually wires ``forward`` at fit and ``inverse`` at sample.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from frugal_flows.outcome_transforms import OutcomeTransform, as_outcome_transform

# A strictly-positive, skewed outcome (all > 0.4, so floor=0.4 is also valid).
Y_POS = np.array([[0.5], [1.0], [2.0], [5.0], [20.0]])


# --------------------------------------------------------------------------- #
# 1. the transform maths round-trips
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kind,floor",
    [("identity", None), ("log", 0.0), ("asinh", 0.0), ("standardize", None),
     ("log", 0.4), ("asinh", 0.4)],
)
def test_roundtrip_recovers_Y(kind, floor):
    kw = {} if floor is None else {"floor": floor}
    t = OutcomeTransform(kind, **kw).fit(Y_POS)
    back = np.asarray(t.inverse(t.forward(Y_POS)))
    assert np.allclose(back, Y_POS), f"{kind} (floor={floor}) failed to round-trip"


def test_asinh_handles_signed_and_zero_data():
    """asinh is the zero-safe log: defined for Y <= 0, still round-trips."""
    Y = np.array([[-5.0], [-0.1], [0.0], [0.1], [5.0]])
    t = OutcomeTransform("asinh", floor=0.0).fit(Y)
    z = np.asarray(t.forward(Y))
    assert not np.isnan(z).any(), "asinh produced NaN on signed/zero data"
    assert np.allclose(np.asarray(t.inverse(z)), Y)


def test_fit_learns_data_dependent_params():
    Y = np.array([[1.0], [3.0], [5.0], [7.0]])
    ta = OutcomeTransform("asinh", floor=0.0).fit(Y)
    assert np.isclose(ta.asinh_scale, float(np.median(np.abs(Y - 0.0))))  # median(|Y|) = 4
    ts = OutcomeTransform("standardize").fit(Y)
    assert np.isclose(ts._mean, float(Y.mean())) and np.isclose(ts._sd, float(Y.std()))


def test_use_before_fit_raises():
    with pytest.raises(RuntimeError):
        OutcomeTransform("standardize").forward(np.array([1.0, 2.0]))     # needs mean/sd
    with pytest.raises(RuntimeError):
        OutcomeTransform("asinh", floor=0.0).forward(np.array([1.0, 2.0]))  # needs scale
    # an explicit scale means asinh needs no fit:
    OutcomeTransform("asinh", floor=0.0, asinh_scale=2.0).forward(np.array([1.0, 2.0]))


# --------------------------------------------------------------------------- #
# 2. the API forces a conscious floor / stays backward compatible
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["log", "asinh"])
def test_floor_is_required_for_log_and_asinh(kind):
    with pytest.raises(ValueError, match="floor"):
        OutcomeTransform(kind)  # no floor -> must raise


def test_bare_log_asinh_strings_rejected_others_ok():
    for s in ("log", "asinh"):
        with pytest.raises(ValueError, match="floor"):
            as_outcome_transform(s)  # needs an explicit floor -> construct object
    assert as_outcome_transform(None).kind == "identity"       # backward-compatible default
    assert as_outcome_transform("raw").kind == "identity"
    assert as_outcome_transform("standardize").kind == "standardize"


def test_log_floor_assertion_fires_when_data_below_floor():
    with pytest.raises(ValueError, match="requires Y > floor"):
        OutcomeTransform("log", floor=0.4).fit(np.array([[0.1], [0.5]]))  # min 0.1 <= 0.4


def test_identity_forward_is_a_true_noop():
    t = as_outcome_transform(None).fit(Y_POS)
    assert t.kind == "identity"
    assert np.array_equal(np.asarray(t.forward(Y_POS)), Y_POS.astype(float))


# --------------------------------------------------------------------------- #
# 2b. floor active / inactive / incorrectly specified
# --------------------------------------------------------------------------- #
FLOOR_TRUE = 5.0
Y_FLOORED = FLOOR_TRUE + np.array([[0.5], [1.0], [2.0], [5.0], [20.0]])  # support in (5, 25]


@pytest.mark.parametrize("kind", ["log", "asinh"])
def test_floor_active_roundtrips_on_shifted_support(kind):
    """Floor ACTIVE: a correct floor at the true lower bound is valid and exact."""
    t = OutcomeTransform(kind, floor=FLOOR_TRUE).fit(Y_FLOORED)
    assert np.allclose(np.asarray(t.inverse(t.forward(Y_FLOORED))), Y_FLOORED)


@pytest.mark.parametrize("kind", ["log", "asinh"])
def test_floor_inactive_still_valid_when_data_stays_positive(kind):
    """Floor INACTIVE (b=0): still valid here, because the shifted support is > 0."""
    t = OutcomeTransform(kind, floor=0.0).fit(Y_FLOORED)
    assert np.allclose(np.asarray(t.inverse(t.forward(Y_FLOORED))), Y_FLOORED)


@pytest.mark.parametrize("bad_floor", [5.5, 10.0, 30.0])  # >= min(Y_FLOORED) = 5.5
def test_incorrect_floor_breaks_log(bad_floor):
    """log(Y - b) is undefined once b reaches the data minimum -> fit must raise.

    This is the core 'breaks if the floor is wrong' guarantee: a floor set at or
    above the smallest observed outcome makes the transform ill-defined, and the
    package refuses it loudly rather than silently producing NaNs downstream.
    """
    with pytest.raises(ValueError, match="requires Y > floor"):
        OutcomeTransform("log", floor=bad_floor).fit(Y_FLOORED)


def test_asinh_tolerates_wrong_floor_but_still_roundtrips():
    """asinh is defined on all reals, so a (wrong) too-high floor never raises; the
    round-trip still holds because b cancels in the inverse. The cost of a wrong
    floor for asinh is a worse-ANCHORED fit, not a crash -- the deliberate contrast
    with log, which fails fast."""
    t = OutcomeTransform("asinh", floor=100.0).fit(Y_FLOORED)  # far above the data
    z = np.asarray(t.forward(Y_FLOORED))
    assert not np.isnan(z).any()
    assert np.allclose(np.asarray(t.inverse(z)), Y_FLOORED)


def test_floor_ignored_for_identity_and_standardize():
    """floor is meaningful only for log/asinh; identity/standardize accept but ignore it."""
    Y = np.array([[1.0], [2.0], [3.0]])
    ts = OutcomeTransform("standardize", floor=99.0).fit(Y)
    assert np.allclose(np.asarray(ts.inverse(ts.forward(Y))), Y)
    ti = OutcomeTransform("identity", floor=99.0).fit(Y)
    assert np.array_equal(np.asarray(ti.forward(Y)), Y.astype(float))


# --------------------------------------------------------------------------- #
# 3. the ESTIMAND round-trips (the reason the whole thing exists)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind,floor", [("log", 0.0), ("asinh", 0.0), ("standardize", None)])
def test_ate_recovered_by_inverting_samples_then_differencing(kind, floor):
    """Invert the counterfactual SAMPLES, then difference -> the original-scale ATE."""
    y0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y1 = np.array([2.5, 3.5, 4.5, 5.5, 6.5])           # constant +1.5 shift
    true_ate = float(np.mean(y1 - y0))
    kw = {} if floor is None else {"floor": floor}
    t = OutcomeTransform(kind, **kw).fit(np.concatenate([y0, y1])[:, None])
    r0 = np.asarray(t.inverse(t.forward(y0)))
    r1 = np.asarray(t.inverse(t.forward(y1)))
    assert np.isclose(float(np.mean(r1 - r0)), true_ate)


def test_transform_is_not_estimand_preserving():
    """Guard that the property above is NON-trivial: differencing on the
    transformed scale and mapping back does NOT give the ATE for a nonlinear map.
    This is exactly why FrugalFlowModel must invert the *samples*, not the effect.
    """
    y0 = np.array([1.0, 2.0, 3.0, 4.0])
    y1 = np.array([2.0, 4.0, 6.0, 8.0])                # multiplicative, so nonlinear matters
    true_ate = float(np.mean(y1 - y0))
    t = OutcomeTransform("log", floor=0.0).fit(np.concatenate([y0, y1])[:, None])
    z0, z1 = np.asarray(t.forward(y0)), np.asarray(t.forward(y1))
    wrong = float(np.exp(np.mean(z1)) - np.exp(np.mean(z0)))          # difference-then-map-back
    right = float(np.mean(np.asarray(t.inverse(z1)) - np.asarray(t.inverse(z0))))
    assert np.isclose(right, true_ate)
    assert not np.isclose(wrong, true_ate)


def test_ate_estimator_unbiased_over_iterations_within_sd_band():
    """Statistical, deterministic: counterfactual samples arrive on the TRANSFORMED
    (log) scale carrying finite-sample Monte-Carlo noise. Inverting-then-differencing
    must be an UNBIASED estimator of the original-scale ATE. Over several iterations
    the mean estimate lies within a few standard errors of the analytic truth.

    DGP is lognormal so both the true ATE and the transform are known in closed form:
        Y|do(t) = exp(N(mu + delta*t, sigma))  ->  E[Y|do(t)] = exp(mu + delta*t + sigma^2/2)
    The log transform is exactly the DGP's latent scale, so a correct inverse (exp)
    on the samples recovers E[Y|do(1)] - E[Y|do(0)] without bias; a wrong inverse
    would show up as a systematic offset that no widening of the band absorbs.
    """
    mu, sigma, delta = 0.5, 0.7, 0.8
    true_ate = float(np.exp(mu + delta + 0.5 * sigma**2) - np.exp(mu + 0.5 * sigma**2))
    t = OutcomeTransform("log", floor=0.0)  # inverse = exp(.), no fit needed
    n_mc = 20_000
    ates = []
    for seed in range(10):  # a few iterations
        rng = np.random.default_rng(seed)
        z0 = rng.normal(mu, sigma, n_mc)            # log-scale control samples
        z1 = rng.normal(mu + delta, sigma, n_mc)    # log-scale treated samples
        y0 = np.asarray(t.inverse(z0))
        y1 = np.asarray(t.inverse(z1))
        ates.append(float(np.mean(y1 - y0)))
    ates = np.array(ates)
    se = ates.std(ddof=1) / np.sqrt(len(ates))
    assert se > 0                                     # real sampling noise, not degenerate
    assert abs(ates.mean() - true_ate) <= 4.0 * se   # unbiased within ~4 standard errors


# --------------------------------------------------------------------------- #
# 4. FrugalFlowModel wiring: forward at fit, inverse at sample (monkeypatched)
# --------------------------------------------------------------------------- #
def test_model_default_transform_is_identity():
    benchmarking = pytest.importorskip("frugal_flows.benchmarking")
    model = benchmarking.FrugalFlowModel(Y=np.zeros((4, 1)), X=np.zeros((4, 1)))
    assert model.outcome_transform.kind == "identity"


def test_fit_trains_on_forward_transformed_Y(monkeypatch):
    """train_frugal_flow must be handed forward(Y), not raw Y."""
    benchmarking = pytest.importorskip("frugal_flows.benchmarking")
    import jax.numpy as jnp
    import jax.random as jr

    Y = np.array([[1.0], [2.0], [4.0], [8.0]])
    X = np.array([[0.0], [1.0], [0.0], [1.0]])
    Zc = np.array([[0.1], [0.2], [0.3], [0.4]])
    model = benchmarking.FrugalFlowModel(
        Y=Y, X=X, Z_cont=Zc, outcome_transform=OutcomeTransform("log", floor=0.0)
    )
    model.res = {"u_z_cont": jnp.asarray(Zc), "u_z_discr": None}  # stub stage-1 output

    captured = {}

    class _FakeBij:
        def transform(self, x, condition=None):
            return x

    class _FakeFlow:
        bijection = _FakeBij()

    def _fake_train(**kwargs):
        captured["y"] = np.asarray(kwargs["y"])
        return _FakeFlow(), {"val": [0.0]}

    monkeypatch.setattr(benchmarking, "train_frugal_flow", _fake_train)
    model.train_frugal_flow(jr.PRNGKey(0), {}, "flexible_continuous", {})

    expected = np.asarray(OutcomeTransform("log", floor=0.0).fit(Y).forward(Y))
    assert np.allclose(captured["y"], expected)
    assert not np.allclose(captured["y"], Y)  # non-trivial: it is NOT raw Y
    assert model.outcome_transform.fitted


def test_generate_samples_inverts_outcome(monkeypatch):
    """generate_samples must return Y on the ORIGINAL scale (inverse applied)."""
    benchmarking = pytest.importorskip("frugal_flows.benchmarking")
    import jax.numpy as jnp
    import jax.random as jr

    n = 5
    Y = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    X = np.array([[0.0], [1.0], [0.0], [1.0], [0.0]])
    Zc = np.zeros((n, 1))
    t = OutcomeTransform("log", floor=0.0).fit(Y)
    model = benchmarking.FrugalFlowModel(Y=Y, X=X, Z_cont=Zc, outcome_transform=t)
    model.conf_shape = 1
    model.res = {"z_cont_flows": None}  # read as an arg into the stubbed marginal sampler

    # stub the sampling pipeline (Z_cont-only, no-confounding branch)
    model.confounding_copula = lambda key, N, rho: (np.zeros(N), np.zeros(N))
    model.vmap_frugal_flow = lambda x, condition: jnp.zeros((x.shape[0], 1 + 1))
    monkeypatch.setattr(benchmarking, "from_quantiles_to_marginal_cont", lambda **kw: jnp.zeros((n, 1)))

    # sample_outcome returns a KNOWN transformed (log) scale outcome:
    z_scale = np.log(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    monkeypatch.setattr(benchmarking, "sample_outcome", lambda **kw: jnp.asarray(z_scale))

    df = model.generate_samples(
        jr.PRNGKey(0), sampling_size=n, copula_param=0.0,
        outcome_causal_model="gaussian", outcome_causal_args={}, with_confounding=False,
    )
    # inverse of log-scale samples == exp(z) == [1, 2, 3, 4, 5]; if the inverse were
    # NOT applied, df["Y"] would equal z_scale (log values) instead.
    assert np.allclose(df["Y"].to_numpy(), np.exp(z_scale))
    assert not np.allclose(df["Y"].to_numpy(), z_scale)


# --------------------------------------------------------------------------- #
# 5. end-to-end statistical ATE recovery (slow: trains flows; env-gated)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not os.environ.get("RUN_SLOW_TESTS"),
    reason="slow (~90s): trains 3 frugal flows. Set RUN_SLOW_TESTS=1 to run.",
)
def test_pipeline_recovers_known_ate_within_sd_band():
    """End-to-end statistical check that a non-identity transform does NOT corrupt
    the estimand. Fit ``FrugalFlowModel`` (asinh transform) on the causl gaussian
    fixture (known ATE = 1) over a few seeds; estimate the interventional ATE by
    paired-CRN sampling of do(X=0) vs do(X=1), inverting the transform on the
    samples; assert the seed-mean recovers the truth within ~4 standard errors.

    Deterministic (fixed seeds), so the SD is the real across-seed estimator spread
    and the assertion is reproducible, not flaky. Calibrated: mean ~1.01, ~0.3 SE off.
    """
    benchmarking = pytest.importorskip("frugal_flows.benchmarking")
    import jax.numpy as jnp
    import jax.random as jr

    fx = Path(__file__).resolve().parent / "fixtures" / "gaussian_known_ate.npz"
    if not fx.exists():
        pytest.skip("gaussian_known_ate.npz fixture missing (run tests/fixtures/_generate.py)")
    raw = np.load(fx)
    true_ate = float(raw["meta_ate"])
    # fixture is float32; cast to float64 so the x64 flow samples without a dtype clash.
    Y = jnp.asarray(np.asarray(raw["Y"], np.float64))
    X = jnp.asarray(np.asarray(raw["X"], np.float64))
    Zc = jnp.asarray(np.asarray(raw["Z_cont"], np.float64))

    mhp = {"max_epochs": 120, "max_patience": 20}
    fhp = {"RQS_knots": 8, "nn_depth": 4, "nn_width": 50, "flow_layers": 4,
           "learning_rate": 5e-3, "max_epochs": 200, "max_patience": 25,
           "batch_size": 256, "show_progress": False}
    cargs = {"nn_depth": 4, "nn_width": 50, "RQS_knots": 8, "flow_layers": 4}

    def interventional_ate(model, cond_dim, n_mc, key):
        flow, tr = model.frugal_flow, model.outcome_transform
        # paired CRN: same key for do(0) and do(1); read dim-0 = Y (fit scale); invert.
        y0 = np.asarray(flow.sample(key, condition=jnp.zeros((n_mc, cond_dim)))[:, 0])
        y1 = np.asarray(flow.sample(key, condition=jnp.ones((n_mc, cond_dim)))[:, 0])
        y0, y1 = np.asarray(tr.inverse(y0)), np.asarray(tr.inverse(y1))
        finite = np.isfinite(y0) & np.isfinite(y1)
        return float(np.mean((y1 - y0)[finite]))

    ates = []
    for seed in range(3):  # a couple of iterations
        model = benchmarking.FrugalFlowModel(
            Y=Y, X=X, Z_cont=Zc, outcome_transform=OutcomeTransform("asinh", floor=0.0)
        )
        s = jr.split(jr.key(seed), 20)
        model.train_marginal_cdfs(s[0], mhp)
        model.train_frugal_flow(s[1], fhp, "flexible_continuous", cargs)
        ates.append(interventional_ate(model, cond_dim=1, n_mc=20_000, key=jr.key(1000 + seed)))

    ates = np.array(ates)
    se = ates.std(ddof=1) / np.sqrt(len(ates))
    # +0.05 guards the degenerate se->0 edge; the real signal is |mean - true| ~ 0.01.
    assert abs(ates.mean() - true_ate) <= 4.0 * se + 0.05, f"ATE recovery off: {ates} (true {true_ate})"
