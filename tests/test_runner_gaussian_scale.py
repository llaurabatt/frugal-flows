"""validation/morphomnist/exp_ate_recovery (2026-10-01): the Gaussian-scale arms, outcome scaling,
the shift start, the placebo covariate shuffle and the reload, through the runner itself.

Small and fast: digit 0, 4x4 images, n = 300, 2 epochs. Correctness of the fits is not tested here
(that is the grid's job); the wiring is."""
import json
import os
import sys
from dataclasses import asdict

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

MM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "validation", "morphomnist")
sys.path.insert(0, os.path.abspath(MM))

E = pytest.importorskip("exp_ate_recovery")

BASE = dict(preset="exp2_confounded_homogeneous", size=4, digit=0, n=300, seed_data=1, seed_assign=1,
            seed_fit=3, max_epochs=2, marginal_max_epochs=2, n_mc=200, copula_nn_width=16, nn_width=16,
            flow_layers=2, copula_flow_layers=2, wandb=False, save_model=True)


def _cfg(**kw):
    return E.Config(**{**BASE, **kw})


@pytest.fixture(scope="module")
def data():
    return E.build_data(_cfg())


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """One run_one per new configuration, shared by the tests below."""
    root = str(tmp_path_factory.mktemp("runs"))
    out = {}
    for name, kw in {
        "flexgauss_raw": dict(arm="flexible_continuous_gaussian"),
        "flexgauss_std": dict(arm="flexible_continuous_gaussian", y_scaling="standardize"),
        "ltgauss_std": dict(arm="location_translation_gaussian", y_scaling="standardize", shift_init="naive"),
        "ltgauss_raw": dict(arm="location_translation_gaussian", shift_init="scalar"),
        "flexcont_std": dict(arm="flexible_continuous", y_scaling="standardize"),
    }.items():
        row = E.run_one(_cfg(**kw), runs_root=root)
        out[name] = (row, os.path.join(root, row["run_id"]))
    return out


@pytest.mark.parametrize("name", ["flexgauss_raw", "flexgauss_std", "ltgauss_std", "ltgauss_raw", "flexcont_std"])
def test_run_writes_everything(runs, name):
    row, d = runs[name]
    for f in ("config.json", "metrics.json", "arrays.npz", "model.eqx", "log.txt", "wandb.json"):
        assert os.path.exists(os.path.join(d, f)), f
    for p in ("ate_maps", "loss_curves", "samples_gallery", "tau_curves"):
        assert os.path.exists(os.path.join(d, "plots", f"{p}.png")), p
    m = json.load(open(os.path.join(d, "metrics.json")))
    assert np.isfinite(m["ate_mae"]) and np.isfinite(m["slope_imb"]) and np.isfinite(m["rho_retained"])
    if "gauss" in name:
        for k in ("gz_implied_ks_max", "gz_implied_ks_mean", "gy_ks_max"):
            assert 0.0 <= m[k] <= 1.0, k
        assert "copula_term_verified" not in m        # uniform-chain diagnostics skipped cleanly
    a = np.load(os.path.join(d, "arrays.npz"))
    if name.endswith("_std"):
        assert "y_sd" in a.files and "y_mean" in a.files
        assert "_ystd_" in os.path.basename(d)
    else:
        assert "_ystd_" not in os.path.basename(d)


@pytest.mark.parametrize("name", ["ltgauss_std", "ltgauss_raw"])
def test_shift_readout_matches_sampled_difference(runs, name):
    """tau_hat = shift x sd is exact; the paired CRN difference must agree to float error."""
    _, d = runs[name]
    m = json.load(open(os.path.join(d, "metrics.json")))
    assert m["shift_vs_crn_maxabs"] < 1e-3


def test_naive_shift_init_is_the_fitting_scale_difference(data):
    cfg = _cfg(arm="location_translation_gaussian", y_scaling="standardize", shift_init="naive")
    ot = E.outcome_transform_for(cfg, data)
    y_fit = np.asarray(ot.forward(jnp.asarray(data["Y"])), np.float64)
    t = np.asarray(data["X"])[:, 0].astype(bool)
    init = E.shift_init_value(cfg, data, y_fit)
    np.testing.assert_allclose(init, y_fit[t].mean(0) - y_fit[~t].mean(0), rtol=1e-5, atol=1e-6)
    # zero-epoch fit: the shift starts exactly there
    flow, _, _ = E.fit_flow(E.Config(**{**asdict(cfg), "max_epochs": 0, "marginal_max_epochs": 0}), data)
    from frugal_flows.gaussian_scale import shift_vector
    np.testing.assert_allclose(np.asarray(shift_vector(flow)), init, rtol=1e-5, atol=1e-6)
    # the data-scale size of a unit fitting-scale shift is the per-pixel sd
    np.testing.assert_allclose(E.outcome_scale(cfg, data), np.asarray(data["Y"]).std(0), rtol=1e-4)


@pytest.mark.parametrize("name", ["flexgauss_std", "ltgauss_std", "flexcont_std"])
def test_reload_reproduces_the_readout(runs, name, data):
    """load_model + the same read-out (same n_mc, seed_mc) gives the saved tau_hat back."""
    _, d = runs[name]
    stored = json.load(open(os.path.join(d, "config.json")))["config"]
    cfg = E.Config(**{k: v for k, v in stored.items() if k in {f.name for f in E.fields(E.Config)}})
    flow = E.load_model(d)
    dd = E.build_data(cfg)
    K = cfg.size ** 2
    if cfg.arm == "location_translation_gaussian":
        from frugal_flows.gaussian_scale import shift_vector
        tau = np.asarray(shift_vector(flow), np.float64) * E.outcome_scale(cfg, dd)
    else:
        tau, _, _ = E._tau_hat_flexible_continuous(cfg, flow, dd, K, ot=E.outcome_transform_for(cfg, dd))
    np.testing.assert_allclose(tau, np.load(os.path.join(d, "arrays.npz"))["tau_hat"], rtol=1e-5, atol=1e-6)


def test_z_shuffle_permutes_rows_jointly(data):
    cfg0 = _cfg(arm="flexible_continuous_gaussian", max_epochs=0)
    cfg1 = _cfg(arm="flexible_continuous_gaussian", max_epochs=0, z_shuffle_seed=7)
    _, _, u0 = E.fit_flow(cfg0, data)
    _, _, u1 = E.fit_flow(cfg1, data)
    perm = np.random.default_rng(7).permutation(u0.shape[0])
    np.testing.assert_array_equal(u1, u0[perm])
    assert not np.array_equal(u1, u0)
    assert "zshuf7" in E.variant_tag(cfg1) and "zshuf" not in E.variant_tag(cfg0)


def test_gaussian_u_z_equals_uniform_arm_u_z(data):
    """Same seed_fit -> same stage-1 covariate ranks for the uniform and the Gaussian arm."""
    _, _, ug = E.fit_flow(_cfg(arm="flexible_continuous_gaussian", max_epochs=0), data)
    _, _, uu = E.fit_flow(_cfg(arm="flexible_continuous", max_epochs=0), data)
    np.testing.assert_array_equal(ug, uu)


def test_defaults_unchanged(data):
    """Defaults: no new tag in the name, no transform, and fit_flow for flexible_continuous is the
    call it made before the change (raw Y), leaf for leaf."""
    cfg = _cfg(arm="flexible_continuous", max_epochs=1)
    assert cfg.y_scaling == "none" and cfg.shift_init == "zero" and cfg.z_shuffle_seed is None
    assert cfg.shift_lr_mult == 1.0
    assert E.outcome_transform_for(cfg, data) is None
    tag = E.variant_tag(cfg)
    assert not any(t in tag for t in ("ystd", "shi", "zshuf", "shlr"))
    flow, _, u_z = E.fit_flow(cfg, data)
    # the pre-change call, verbatim in substance
    from frugal_flows.causal_flows import train_frugal_flow
    key = jr.PRNGKey(cfg.seed_fit)
    key, _ = jr.split(key)
    key, sub = jr.split(key)
    ref, _ = train_frugal_flow(
        causal_model="flexible_continuous", key=sub, y=jnp.asarray(data["Y"]), u_z=jnp.asarray(u_z),
        condition=jnp.asarray(data["X"]), learning_rate=cfg.learning_rate, max_epochs=cfg.max_epochs,
        max_patience=cfg.max_patience, batch_size=cfg.batch_size, fit_kwargs=E._fit_kwargs(cfg, data),
        copula_lr_mult=cfg.copula_lr_mult, copula_umarg_weight=cfg.copula_umarg_weight,
        copula_umarg_n=cfg.copula_umarg_n,
        causal_model_args={"RQS_knots": cfg.rqs_knots, "nn_depth": cfg.nn_depth, "nn_width": cfg.nn_width,
                           "flow_layers": cfg.flow_layers, "conditioner": cfg.conditioner},
        nn_width=cfg.copula_nn_width, flow_layers=cfg.copula_flow_layers, RQS_knots=cfg.copula_rqs_knots,
        nn_depth=cfg.copula_nn_depth)
    la = jax.tree_util.tree_leaves(eqx.filter(flow, eqx.is_inexact_array))
    lb = jax.tree_util.tree_leaves(eqx.filter(ref, eqx.is_inexact_array))
    assert len(la) == len(lb) and all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(la, lb))


def test_guards():
    with pytest.raises(ValueError):
        _cfg(arm="flexible_continuous", shift_init="naive")
    with pytest.raises(ValueError):
        _cfg(arm="flexible_continuous_gaussian", y_scaling="log")
    with pytest.raises(ValueError):
        _cfg(arm="flexible_continuous_gaussian", select_on="copula")
    with pytest.raises(ValueError):
        _cfg(arm="frengression", y_scaling="standardize")
    assert E.ARM_SHORT["flexible_continuous_gaussian"] == "flexgauss"
    assert E.ARM_SHORT["location_translation_gaussian"] == "loctransgauss"


def test_shift_lr_mult_moves_the_shift_faster(data):
    """shift_lr_mult labels exactly the LocCond leaves: one epoch moves the shift further."""
    from frugal_flows.gaussian_scale import shift_vector
    kw = dict(arm="location_translation_gaussian", y_scaling="standardize", max_epochs=1)
    f1, _, _ = E.fit_flow(_cfg(**kw), data)
    f10, _, _ = E.fit_flow(_cfg(**kw, shift_lr_mult=10.0), data)
    d1 = np.abs(np.asarray(shift_vector(f1))).mean()
    d10 = np.abs(np.asarray(shift_vector(f10))).mean()
    assert d10 > 3 * d1
    assert "shlr10" in E.variant_tag(_cfg(**kw, shift_lr_mult=10.0))
