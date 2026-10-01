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


@pytest.mark.parametrize("stage,n", [("S1", 302), ("S2", 240), ("S3", 360), ("S4", 120)])
def test_cell_counts_and_identities(stage, n):
    cells = hdr.enumerate_cells(stage)
    assert len(cells) == n
    assert all(c["seed_fit"] != c["seed_data"] for c in cells)
    assert len({c["identity_sha"] for c in cells}) == n
    assert {c["seed_data"] for c in cells} == set(range(31, 41))
    assert all((c["seed_mc2"] is not None) == (c["seed_data"] == 31) for c in cells)
