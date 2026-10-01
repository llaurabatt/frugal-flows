"""Quick checks of the halo harness: ``JAX_ENABLE_X64=0 pytest test_halo.py -q``."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("JAX_ENABLE_X64", "0")
HALO = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HALO, os.path.abspath(os.path.join(HALO, "..", ".."))]

import halo_data as hd  # noqa: E402
import halo_driver as hdr  # noqa: E402
import halo_metrics as hmx  # noqa: E402


def test_float32():
    import jax
    assert not jax.config.jax_enable_x64


def test_ff_stack_matches_lauras_margin_dist():
    """A1s (spread) == exp_ate_recovery._margin_dist: same pytree structure AND weights for
    the same key, conditional and unconditional; A1 (modulo) differs only in the class."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import exp_ate_recovery as ear
    import halo_models as hm
    cfg = ear.Config(arm="flexible_continuous", model="margin")
    k = jr.PRNGKey(3)
    for cond, cd in ((jnp.zeros((5, 1)), 1), (None, None)):
        a, b = ear._margin_dist(cfg, k, 64, cond), hm.ff_stack(k, 64, cd, rank_mode="spread")
        assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
        la = jax.tree_util.tree_leaves(eqx.filter(a, eqx.is_array))
        lb = jax.tree_util.tree_leaves(eqx.filter(b, eqx.is_array))
        assert all(np.array_equal(x, y) for x, y in zip(la, lb))
    m = hm.ff_stack(k, 64, None, rank_mode="modulo")
    shapes = lambda d: [x.shape for x in jax.tree_util.tree_leaves(eqx.filter(d, eqx.is_array))]
    assert shapes(m) == shapes(b)


def test_quiet_bounds_closed_form():
    from prepare_data import dequantize_and_logit
    y = dequantize_and_logit(-np.ones((100_000, 1)), np.random.default_rng(0))
    assert hd.QUIET_LO <= y.min() and y.max() <= hd.QUIET_HI
    assert y.min() - hd.QUIET_LO < 1e-4 and hd.QUIET_HI - y.max() < 1e-4
    assert hd.QUIET_HI < hd.FLOOR_Y


def test_ring_matches_region_masks():
    from exp_ate_recovery import region_masks
    d, r, _ = region_masks(8, 2)
    assert np.array_equal(d, hd.disc_mask_geometric()) and np.array_equal(r, hd.ring_mask())


def test_pixel_classes_seed1():
    ref = hd.class_reference(1)
    c = hd.pixel_classes(ref["Y"], hd.disc_mask_geometric(), ref["RAW"])
    q = c["quiet"].reshape(8, 8)
    border = np.ones((8, 8), bool)
    border[1:-1, 1:-1] = False
    assert c["quiet"].sum() == 32 and q[border].all()
    assert c["exact_floor"].sum() == 15 and c["disc"].sum() == 12


def test_template_regression_recovers_planted():
    rng = np.random.default_rng(0)
    tpl = {n: hd._std(rng.normal(size=64)) for n in hd.TEMPLATE_NAMES}
    err = 0.1 + 0.05 * tpl["t_mix"] - 0.02 * tpl["t_imb"]
    r = hmx.template_regression(err, tpl)
    assert abs(r["b_mix"] - 0.05) < 1e-9 and abs(r["b_imb"] + 0.02) < 1e-9 and r["r2"] > 0.999


def test_leak_x_zero_for_perfect_model():
    rng = np.random.default_rng(1)
    ref = rng.normal(-3.6, 0.04, size=(4000, 64))
    m = hmx.maps(ref, ref, lo=hd.QUIET_LO, hi=hd.QUIET_HI, floor_thr=hd.FLOOR_Y)
    assert np.allclose(m["LEAK_X0"], 0) and np.allclose(m["E_mu0"], 0)
    m = hmx.maps(ref, ref, lo=hd.QUIET_LO, hi=hd.QUIET_HI, floor_thr=hd.FLOOR_Y, leak=False)
    assert np.isnan(m["LEAK_X0"]).all()


@pytest.mark.parametrize("stage,n", [("S1", 221), ("S2", 180), ("S3", 180), ("S4", 80), ("S5", 100), ("S6", 90)])
def test_cell_counts_and_identities(stage, n):
    cells = hdr.enumerate_cells(stage)
    assert len(cells) == n
    assert all(c["seed_fit"] != c["seed_data"] for c in cells)
    assert len({c["identity_sha"] for c in cells}) == n
    assert {c["seed_data"] for c in cells} == set(range(31, 41))
    assert all((c["seed_mc2"] is not None) == (c["seed_data"] == 31) for c in cells)
    assert all(c["n_mc"] == 5000 for c in cells)
    assert all(c["seed_fit"] == 41 for c in cells if not c["primary"])
    first_expl = next((i for i, c in enumerate(cells) if not c["primary"]), len(cells))
    assert all(not c["primary"] for c in cells[first_expl:])          # primary queued first


def test_s5_cells_all_primary():
    cells = hdr.enumerate_cells("S5")
    assert len(cells) == 100 and all(c["primary"] and c["preproc"] == "P5" for c in cells)
    assert all(c["seed_fit"] != c["seed_data"] for c in cells)
    assert {c["seed_fit"] for c in cells} == {41, 42}
    cfgs = {(c["arm"], c["corpus"], c["task"], c["preset"], c["base_shift"]) for c in cells}
    assert cfgs == {("A1s", "A", "uncond", "E1", 0.0), ("A1s", "B", "uncond", "E1", 0.0),
                    ("ff_cond", "A", "cond", "E1", 1.0), ("ff_full", "A", "cond", "E1", 1.0),
                    ("ff_full", "A", "cond", "E2", 1.0)}
    assert len(hdr.enumerate_cells("S5", smoke=True)) == 5


def _frengression_scale(Y, floor=0.25):
    """Verbatim from exp_frengression_recovery.prepare_inputs (y_scaling == "per_pixel")."""
    y_sd_global = float(Y.std())
    sd = Y.std(axis=0)
    y_mean = Y.mean(axis=0)
    y_scale = np.maximum(sd, floor * (y_sd_global or 1.0))
    return y_mean, y_scale


def test_p5_roundtrip_and_floor():
    rng = np.random.default_rng(0)
    Y = np.column_stack([rng.normal(-3.6, 0.04, 5000), rng.normal(0, 2.0, 5000), rng.normal(1, 1.5, 5000)])
    pre = hd.Preproc("P5").fit(Y)
    t = pre.transform
    m, sc = _frengression_scale(Y)
    assert np.array_equal(t.y_mean, m) and np.array_equal(t.y_scale, sc)
    assert t.floor_value == pytest.approx(0.25 * Y.std())
    assert t.y_scale[0] == t.floor_value and t.y_scale[1] == pytest.approx(Y[:, 1].std())
    assert pre.info()["n_floored"] == 1 and pre.info()["preproc"] == "P5"
    Z = pre.forward(Y)
    assert np.allclose(pre.inverse(Z), Y, atol=1e-12, rtol=0)
    assert np.allclose(Z.mean(0), 0, atol=1e-10) and np.allclose(Z[:, 1:].std(0), [1, 1])
    assert Z[:, 0].std() < 1                                        # floored column is shrunk, not unit
    z = rng.normal(size=(100, 3))
    assert np.allclose(pre.forward(pre.inverse(z)), z, atol=1e-12)


def test_p5_global_sd_zero_guard():
    Y = np.full((10, 4), 2.0)
    t = hd.FlooredStandardize().fit(Y)
    assert t.floor_value == 0.25 and np.all(t.y_scale == 0.25)    # `or 1.0` guard
    assert np.allclose(t.inverse(t.forward(Y)), Y)


# ------------------------------------------------------------------ S6 (Amendment A2)
def test_s6_cells():
    cells = hdr.enumerate_cells("S6")
    assert len(cells) == 90 and all(c["seed_fit"] != c["seed_data"] for c in cells)
    assert all(c["preproc"] == "P1" and c["preset"] == "E1" and c["corpus"] == "A" for c in cells)
    prim = [c for c in cells if c["primary"]]
    assert len(prim) == 60 and all(c["base_shift"] == 1.0 for c in prim)
    assert {c["seed_fit"] for c in prim} == {41, 42}
    expl = [c for c in cells if not c["primary"]]
    assert len(expl) == 30 and all(c["base_shift"] == 0.0 and c["seed_fit"] == 41 for c in expl)
    assert {c["arm"] for c in cells} == {"n_cond", "lt_n", "lt"}
    assert len(hdr.enumerate_cells("S6", smoke=True)) == 6


def test_n_cond_sample_shape_finite():
    """N on a tiny K=4 synthetic: build, a 2-epoch fit, CRN draws: shape, finite, no clamp."""
    import jax.random as jr
    import halo_models as hm
    from flowjax.bijections import Invert, Scan
    from flowjax.distributions import Normal
    from frugal_flows.bijections.masked_autoregressive_spread import MaskedAutoregressiveSpread
    rng = np.random.default_rng(0)
    t = rng.integers(0, 2, (300, 1)).astype(np.float32)
    Y = (rng.normal(size=(300, 4)) + t).astype(np.float32)
    cfg = dict(arm="n_cond", task="cond", seed_fit=41, width=8, depth=1, layers=2, knots=4,
               lr=1e-2, max_epochs=2, patience=5, batch=50)
    d0 = hm.build("n_cond", jr.PRNGKey(0), 4, 1, cfg)
    assert isinstance(d0.base_dist, Normal) and isinstance(d0.bijection, Invert)
    assert isinstance(d0.bijection.bijection, Scan) and d0.cond_shape == (1,)
    assert any(isinstance(x, MaskedAutoregressiveSpread) for x in
               __import__("jax").tree_util.tree_leaves(d0, is_leaf=lambda x: isinstance(x, MaskedAutoregressiveSpread)))
    dist, _ = hm.fit_arm(cfg, Y, t)
    y0, y1, c = hm.sample_arms(7, dist, 200, "cond")
    assert y0.shape == (200, 4) and y1.shape == (200, 4) and c == 0
    assert np.isfinite(y0).all() and np.isfinite(y1).all()


def test_lt_n_tau_hat_is_ate_times_p1_sd():
    """LT-N at P1: the metric tau_hat is the fitted LocCond ate times the per-pixel P1 sd."""
    import halo_fit as hf
    import halo_models as hm
    rng = np.random.default_rng(1)
    t = rng.integers(0, 2, (300, 1)).astype(np.float32)
    Yraw = (rng.normal(size=(300, 4)) * np.array([0.5, 1, 2, 3]) + 0.7 * t).astype(np.float32)
    pre = hd.Preproc("P1").fit(Yraw)
    cfg = dict(arm="lt_n", task="cond", seed_fit=41, width=8, depth=1, layers=2, knots=4,
               lr=1e-2, max_epochs=3, patience=5, batch=50)
    dist, _ = hm.fit_arm(cfg, pre.forward(Yraw), t)
    a = hm.loccond_ate(dist)
    assert np.abs(a).max() > 0                                         # it was fitted
    sd = np.asarray(pre.transform._sd, np.float64)
    assert np.allclose(hf.ate_to_data_scale(pre, a), a * sd, rtol=1e-5, atol=1e-6)
    # the maps use it as tau_hat (override of the CRN mean); maps need K=64, so tile
    G = rng.normal(size=(50, 64))
    tau = np.tile(hf.ate_to_data_scale(pre, a), 16)
    m = hmx.maps(G, G, G + 1, G + 1, np.zeros(64), tau, lo=hd.QUIET_LO, hi=hd.QUIET_HI, floor_thr=hd.FLOOR_Y)
    assert np.allclose(m["tau_hat"], np.tile(a * sd, 16), rtol=1e-5, atol=1e-6)
    assert np.allclose(m["E_tau"], m["tau_hat"]) and np.allclose(m["tau_hat_crn"], 1)


def test_lt_p1_unit_conversion_roundtrip():
    """P1 shift conversion round-trips: a logit-unit effect d standardised to d/sd and mapped
    back gives d; and P0 returns the vector unchanged bit for bit."""
    import halo_fit as hf
    rng = np.random.default_rng(2)
    Y = rng.normal(size=(1000, 64)) * rng.uniform(0.05, 2, 64) + rng.normal(size=64)
    pre = hd.Preproc("P1").fit(Y)
    sd = np.asarray(pre.transform._sd, np.float64)
    d = rng.normal(size=64)
    assert np.allclose(hf.ate_to_data_scale(pre, d / sd), d, rtol=1e-5, atol=1e-6)
    # the shift is what the transform implies: forward(y + d) - forward(y) == d / sd
    z = np.asarray(pre.forward(Y[:5] + d)) - np.asarray(pre.forward(Y[:5]))
    assert np.allclose(z, d / sd, rtol=1e-4, atol=1e-5)
    p0 = hd.Preproc("P0").fit(Y)
    assert np.array_equal(hf.ate_to_data_scale(p0, d), d)
