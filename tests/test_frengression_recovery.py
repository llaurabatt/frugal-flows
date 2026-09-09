"""Tests for the frengression MorphoMNIST runner.

These are correctness tests, not recovery tests: a 20-iteration fit at K=16
estimates nothing, and asserting it did would be asserting noise. What they
pin down is the machinery that could go wrong SILENTLY -- producing a finite,
plausible-looking number that is quietly measuring the wrong thing:

  * the dataset differing from the frugal-flow runner's, so the comparison is
    between two different DGPs;
  * oracle knowledge reaching the training path;
  * the scaling inverse being forgotten or applied twice, which rescales every
    reported effect while leaving the spatial pattern (and `ate_corr`) intact;
  * common random numbers not actually pairing, which inflates the Monte-Carlo
    error to the size of the effect being estimated;
  * the sampling call materialising n * n_mc rows;
  * a score formula drifting from the existing baseline/FF metrics, making the
    comparison table wrong in a way no other test would notice;
  * the established FF tau_curve and tau_u metric contract drifting.

Everything is written under `tmp_path`, so the suite never leaves experiment
artefacts in the worktree.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
MORPHO = REPO / "validation" / "morphomnist"
sys.path.insert(0, str(MORPHO))

import baselines  # noqa: E402
import compare_frengression_ff as cmp_mod  # noqa: E402
import exp_ate_recovery as ff  # noqa: E402
import exp_frengression_recovery as fr  # noqa: E402
import frengression_sweep_agent as sweep_agent  # noqa: E402
import torch  # noqa: E402
from frugal_flows.interventions import tau_curve  # noqa: E402

# y_scaling and noise_dim are pinned rather than left to the Config defaults, so
# that a later evidence-driven change to those defaults cannot silently change
# what these tests are testing.
TINY = dict(preset="exp2_confounded_homogeneous", size=4, n=250, num_iters=20,
            n_mc=1500, seed_data=0, seed_fit=0, seed_mc=0, threads=1,
            print_every=1000, y_scaling="per_pixel", noise_dim=64)


@pytest.fixture(scope="module")
def cfg():
    return fr.Config(**TINY)


@pytest.fixture(scope="module")
def data(cfg):
    return fr.build_data(cfg)


@pytest.fixture(scope="module")
def inputs(cfg, data):
    return fr.prepare_inputs(fr.OracleGuard(data), cfg)


@pytest.fixture(scope="module")
def fitted(cfg, inputs):
    model = fr.build_model(cfg, inputs)
    losses, info = fr.fit(cfg, model, inputs)
    return model, losses, info


# --------------------------------------------------------------------------- #
# the dataset must be the frugal-flow runner's dataset
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("preset", ["exp1_rct_homogeneous", "exp2_confounded_homogeneous",
                                    "exp4_covariate_cate"])
def test_dataset_identical_to_ff_runner(preset):
    """If these two disagree, every comparison in the paper is between different
    data-generating processes rather than between estimators."""
    knobs = dict(size=4, n=250, seed_data=3)
    a = fr.build_data(fr.Config(preset=preset, **knobs))
    b = ff.build_data(ff.Config(preset=preset, **knobs))
    for key in ("Y", "X", "Z", "ATE", "ATT", "ATC", "PROPENSITY", "TAU_MARGINAL"):
        assert np.array_equal(np.asarray(a[key]), np.asarray(b[key])), f"{key} differs"


@pytest.mark.parametrize("size,want", [(4, 1), (8, 2), (16, 4)])
def test_effective_radius_matches_ff(size, want):
    assert fr.Config(size=size).effective_radius == want
    assert ff.Config(size=size).effective_radius == want


# --------------------------------------------------------------------------- #
# oracle guard
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", ["ATE", "Y0", "Y1", "ITE", "PROPENSITY", "TAU_MARGINAL",
                                 "THICKNESS", "MAP", "FACTOR"])
def test_oracle_guard_blocks(data, key):
    guard = fr.OracleGuard(data)
    with pytest.raises(KeyError):
        guard[key]


def test_oracle_guard_allows_observables(data):
    guard = fr.OracleGuard(data)
    for key in ("Y", "X", "Z", "z_cat_idx"):
        assert guard[key] is not None
    assert guard.touched == {"Y", "X", "Z", "z_cat_idx"}


def test_training_path_touches_only_observables(cfg, data):
    """Asserting the GUARD's own record would be circular -- it can only ever
    hold allowed keys, since anything else raises. So the training path is run
    against a PERMISSIVE recorder that hands back every key, and the assertion
    is over what the code actually asked for."""

    class Recorder(dict):
        def __init__(self, d):
            super().__init__(d)
            self.reads = set()

        def __getitem__(self, k):
            self.reads.add(k)
            return super().__getitem__(k)

    rec = Recorder(data)

    class Reached(Exception):
        pass

    def stop(*a, **k):
        raise Reached

    import unittest.mock as mock
    with mock.patch.object(fr, "build_data", lambda c: rec), \
         mock.patch.object(fr, "evaluate", stop):
        # run_one is the real entry point: it builds the data, wraps it and
        # fits. Stopping at evaluate() means everything recorded up to here was
        # asked for by the TRAINING path, where oracle keys must never appear.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(Reached):
                fr.run_one(cfg, run_dir=os.path.join(td, "r"), plots=False)

    assert rec.reads, "the recorder saw no reads at all -- the test proves nothing"
    leaked = rec.reads - set(fr.MODEL_INPUT_KEYS)
    assert not leaked, f"training path read oracle keys: {sorted(leaked)}"


# --------------------------------------------------------------------------- #
# preprocessing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["global", "per_pixel", "none"])
def test_scaling_roundtrip(data, mode):
    cfg = fr.Config(**{**TINY, "y_scaling": mode})
    inp = fr.prepare_inputs(fr.OracleGuard(data), cfg)
    Y = np.asarray(data["Y"], dtype=np.float64)
    back = inp.unscale_y((Y - inp.y_mean) / inp.y_scale)
    assert np.allclose(back, Y, atol=1e-8)


def test_global_scaling_uses_one_sd(data):
    """`global` must apply a single sd to every pixel; `per_pixel` must not.

    (Which of the two is the DEFAULT is an empirical question settled by tuning,
    not by this test -- it pins the behaviour of each mode.)"""
    g = fr.prepare_inputs(fr.OracleGuard(data),
                          fr.Config(**{**TINY, "y_scaling": "global"}))
    assert len(set(np.round(g.y_scale, 12))) == 1
    assert np.isclose(g.y_scale[0], np.asarray(data["Y"], dtype=np.float64).std())
    p = fr.prepare_inputs(fr.OracleGuard(data),
                          fr.Config(**{**TINY, "y_scaling": "per_pixel"}))
    assert len(set(np.round(p.y_scale, 12))) > 1


def test_per_pixel_scaling_floors_quiet_pixels(data):
    cfg = fr.Config(**{**TINY, "y_scaling": "per_pixel"})
    inp = fr.prepare_inputs(fr.OracleGuard(data), cfg)
    Y = np.asarray(data["Y"], dtype=np.float64)
    floor = cfg.y_sd_floor * Y.std()
    assert (inp.y_scale >= floor - 1e-12).all()
    assert np.isclose(inp.y_scale.min(), floor) or (Y.std(axis=0) >= floor).all()


def test_inputs_are_float32_even_under_x64(inputs):
    """conftest turns jax's x64 flag ON, so build_preset hands back float64;
    an uncast array would not feed float32 Linear layers."""
    assert inputs.x.dtype == torch.float32
    assert inputs.y.dtype == torch.float32
    assert inputs.z.dtype == torch.float32


def test_constant_treatment_is_refused(data, cfg):
    bad = dict(data)
    bad["X"] = np.zeros_like(np.asarray(data["X"]))
    with pytest.raises(ValueError, match="constant"):
        fr.prepare_inputs(fr.OracleGuard(bad), cfg)


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #
def test_loss_curve_has_one_row_per_iteration(cfg, fitted):
    _, losses, info = fitted
    assert info["n_iters_run"] == cfg.num_iters
    assert list(losses["iter"]) == list(range(1, cfg.num_iters + 1))
    assert np.isfinite(losses["loss"]).all()


def test_loss_parser_reads_the_real_format():
    """A silent parse failure would leave loss_curve.csv empty while the run
    still reported status ok."""
    cap = fr._LossCapture(sys.stdout, print_every=10 ** 9)
    cap.write("Epoch 1: loss 5.6563,\tloss_y 1.8523, 1.9677, 0.2307,"
              "\tloss_eta 3.8040, 3.9784, 0.3489\n")
    cap.write("Epoch 2: loss nan,\tloss_y nan, nan, nan,\tloss_eta nan, nan, nan\n")
    cap.write("Stopping at iter 7: |d|=1e-04\n")
    assert [r[0] for r in cap.rows] == [1, 2]
    assert cap.rows[0][1] == pytest.approx(5.6563)
    assert np.isnan(cap.rows[1][1])
    assert cap.stopped_at == 7


def test_fit_is_seed_deterministic(cfg, inputs):
    out = []
    for _ in range(2):
        model = fr.build_model(cfg, inputs)
        _, info = fr.fit(cfg, model, inputs)
        out.append(info["loss_final"])
    assert out[0] == out[1]
    other = fr.Config(**{**TINY, "seed_fit": 7})
    model = fr.build_model(other, inputs)
    _, info = fr.fit(other, model, inputs)
    assert info["loss_final"] != out[0]


def test_train_xz_is_never_called(cfg, inputs, monkeypatch):
    """model_xz models P(X,Z); it is not needed to sample the causal margin and
    training it would be scope the task explicitly excludes."""
    model = fr.build_model(cfg, inputs)
    called = []
    monkeypatch.setattr(model, "train_xz", lambda *a, **k: called.append(1))
    fr.fit(cfg, model, inputs)
    fr.sample_margins(cfg, model, inputs)
    assert called == []


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def test_sampling_call_shape(cfg, inputs, fitted, monkeypatch):
    """`sample_causal_margin(X, sample_size=n_mc)` reads naturally and is
    catastrophically wrong: engression repeats x sample_size times."""
    model, _, _ = fitted
    seen = []
    real = model.sample_causal_margin

    def spy(x, sample_size=100):
        seen.append((tuple(x.shape), sample_size))
        return real(x, sample_size=sample_size)

    monkeypatch.setattr(model, "sample_causal_margin", spy)
    fr.sample_margins(cfg, model, inputs)
    assert seen == [((cfg.n_mc, 1), 1), ((cfg.n_mc, 1), 1)]


def test_crn_pairs_the_draws(cfg, inputs, fitted):
    """Unpaired, the per-pixel MC error is the same size as the effect being
    estimated, so the comparison would measure Monte-Carlo noise."""
    model, _, _ = fitted
    y0, y1, _ = fr.sample_margins(cfg, model, inputs)
    paired = ((y1 - y0).std(axis=0) / np.sqrt(len(y0))).max()
    unpaired = np.sqrt(y0.var(axis=0) / len(y0) + y1.var(axis=0) / len(y1)).max()
    assert paired * 3 < unpaired

    off = fr.Config(**{**TINY, "crn": False})
    z0, z1, _ = fr.sample_margins(off, model, inputs)
    assert not np.array_equal(z1 - z0, y1 - y0)


def test_treatment_actually_moves_the_margin(cfg, inputs, fitted):
    model, _, _ = fitted
    y0, y1, diag = fr.sample_margins(cfg, model, inputs)
    assert diag["mc_n_used"] > 0
    assert np.isfinite(y0).all() and np.isfinite(y1).all()
    assert not np.allclose(y1, y0)


def test_sampling_returns_original_units(cfg, inputs, fitted, monkeypatch):
    """A forgotten inverse would scale every effect by the global sd and still
    produce a plausible map with a good ate_corr -- so it is pinned with a
    known offset rather than eyeballed."""
    model, _, _ = fitted
    K, offset = inputs.K, 0.75

    class Stub:
        def __init__(self):
            self.calls = 0

        def __call__(self, x, sample_size=1):
            self.calls += 1
            base = torch.zeros(len(x), K, 1)
            return base + (offset if self.calls == 2 else 0.0)

    monkeypatch.setattr(model, "sample_causal_margin", Stub())
    y0, y1, _ = fr.sample_margins(cfg, model, inputs)
    assert np.allclose((y1 - y0).mean(axis=0), offset * inputs.y_scale, atol=1e-6)


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def test_score_parity_with_baselines(cfg, data, fitted, inputs, monkeypatch):
    """Every family's ATE columns must be the SAME function, not a re-derivation
    that happens to agree today.

    Comparing the output against `baselines.score` alone would be nearly
    circular -- a re-derived formula would very likely match to floating point
    on one dataset. So the call itself is spied: `evaluate` must delegate, and
    a sentinel value injected through that call must appear in the metrics.
    """
    model, losses, info = fitted
    y0, y1, diag = fr.sample_margins(cfg, model, inputs)
    calls = []
    real = baselines.score

    def spy(tau_hat, d):
        calls.append(tau_hat)
        out = dict(real(tau_hat, d))
        out["ate_corr"] = -0.4242          # sentinel: cannot arise by chance
        return out

    monkeypatch.setattr(fr, "score_effect_map", spy)
    tau_hat, metrics, _ = fr.evaluate(cfg, y0, y1, data, losses, info, diag, {})
    assert len(calls) == 1, "evaluate did not delegate to the shared helper exactly once"
    assert np.array_equal(calls[0], tau_hat)
    assert metrics["ate_corr"] == -0.4242, "evaluate overwrote or recomputed a score key"
    monkeypatch.undo()
    _, clean, _ = fr.evaluate(cfg, y0, y1, data, losses, info, diag, {})
    for k, v in real(tau_hat, data).items():
        assert clean[k] == v


def test_metrics_contract(cfg, data, fitted, inputs):
    model, losses, info = fitted
    y0, y1, diag = fr.sample_margins(cfg, model, inputs)
    _, m, _ = fr.evaluate(cfg, y0, y1, data, losses, info, diag, {})
    required = [
        "status", "method", "preset", "size", "radius", "seed_data", "seed_fit",
        "seed_mc", "device", "threads", "n_units", "n_pixels", "z_dim",
        *baselines.score(np.zeros(cfg.size ** 2), data).keys(),
        "frac_pixels_on_support", "tau_hat_mean_on_support", "tau_hat_mean_off_support",
        "true_effect_on_support", "design_naive_bias_mae", "design_oracle_ipw_bias_maxabs",
        "design_att_minus_ate_maxabs", "ate_mae_vs_naive", "num_iters", "lr",
        "hidden_dim", "num_layer", "noise_dim", "y_scaling", "z_scaling",
        "n_iters_run", "early_stopped", "loss_first", "loss_final", "loss_min",
        "loss_nonfinite_iters", "diverged", "fit_s", "mc_n", "mc_n_used",
        "mc_frac_dropped", "mc_crn", "mc_se_max", "mc_se_mean", "mc_se_unpaired_max",
        "tau_u_rmse_vs_marginal", "tau_u_rmse_on_support",
        "tau_u_sd_on_support", "tau_u_sd_off_support",
        "true_tau_u_sd_on_support",
    ]
    missing = [k for k in required if k not in m]
    assert not missing, f"metrics.json is missing {missing}"
    assert np.isfinite(m["ate_mae"]) and np.isfinite(m["mc_se_max"])


def test_tau_curve_metrics_match_ff_contract(cfg, data, fitted, inputs):
    model, losses, info = fitted
    y0, y1, diag = fr.sample_margins(cfg, model, inputs)
    _, metrics, extras = fr.evaluate(cfg, y0, y1, data, losses, info, diag, {})
    u, curves = tau_curve(y0, y1)
    support = np.asarray(data["ATE"]) != 0
    true_marg = np.asarray(data["TAU_MARGINAL"])
    curve_err = np.asarray(curves) - true_marg
    flat = np.asarray(curves).std(axis=0)
    assert np.array_equal(extras["tau_u"], u)
    assert np.array_equal(extras["tau_curves"], curves)
    assert metrics["tau_u_rmse_vs_marginal"] == pytest.approx(
        np.sqrt((curve_err**2).mean())
    )
    assert metrics["tau_u_rmse_on_support"] == pytest.approx(
        np.sqrt((curve_err[:, support] ** 2).mean())
    )
    assert metrics["tau_u_sd_on_support"] == pytest.approx(flat[support].mean())
    assert metrics["tau_u_sd_off_support"] == pytest.approx(flat[~support].mean())


# --------------------------------------------------------------------------- #
# archive, resume, determinism
# --------------------------------------------------------------------------- #
def test_run_one_artefacts_and_determinism(tmp_path, cfg):
    a = fr.run_one(cfg, run_dir=str(tmp_path / "a"), overwrite=True, plots=False)
    b = fr.run_one(cfg, run_dir=str(tmp_path / "b"), overwrite=True, plots=False)
    for name in ("config.json", "metrics.json", "ate_hat.csv", "loss_curve.csv",
                 "arrays.npz", "log.txt"):
        assert (tmp_path / "a" / name).exists(), f"missing {name}"
    assert not (tmp_path / "a" / "samples.npz").exists()
    assert a["status"] == "ok"
    for name in ("ate_hat.csv", "loss_curve.csv"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    assert a["ate_mae"] == b["ate_mae"]


def test_ate_hat_csv_matches_metrics(tmp_path, cfg):
    m = fr.run_one(cfg, run_dir=str(tmp_path / "c"), overwrite=True, plots=False)
    rows = (tmp_path / "c" / "ate_hat.csv").read_text().strip().splitlines()
    tau = np.array([float(r.split(",")[1]) for r in rows[1:]])
    ate = np.array([float(r.split(",")[2]) for r in rows[1:]])
    assert np.isclose(np.abs(tau - ate).mean(), m["ate_mae"])


def test_skip_done_ignores_unsuccessful_runs(tmp_path, cfg):
    """The FF runner counts any run with a metrics.json as done, so a run that
    finished non-finite is skipped forever. This one re-runs it."""
    root = tmp_path / "runs"
    fr.run_one(cfg, runs_root=str(root), plots=False)
    assert fr.cell_identity(cfg) in fr.completed_cells(str(root))

    run_dir = next(root.iterdir())
    m = json.loads((run_dir / "metrics.json").read_text())
    m["status"] = "failed_nonfinite"
    (run_dir / "metrics.json").write_text(json.dumps(m))
    assert fr.cell_identity(cfg) not in fr.completed_cells(str(root))


def test_overwrite_refuses_anything_that_is_not_a_run_folder(tmp_path, cfg):
    """--overwrite deletes a tree. `--run-dir . --overwrite` must not be able to
    remove the working directory."""
    victim = tmp_path / "not_a_run"
    victim.mkdir()
    (victim / "precious.txt").write_text("keep me")
    with pytest.raises(ValueError, match="refusing to --overwrite"):
        fr.run_one(cfg, run_dir=str(victim), overwrite=True, plots=False)
    assert (victim / "precious.txt").exists()


def test_run_dir_is_relative_to_cwd(tmp_path, cfg):
    """The repro gate passes a repo-root-relative --run-dir; resolving it
    against the script's own directory would write outside the gitignored path
    and fail gate_diff."""
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        fr.main([*_flags(cfg), "--run-dir", "out/here", "--overwrite", "--no-plots"])
        assert (tmp_path / "out" / "here" / "metrics.json").exists()
    finally:
        os.chdir(cwd)


def _flags(cfg):
    out = []
    for k, v in TINY.items():
        out += [f"--{k.replace('_', '-')}", str(v)]
    return out


def test_flow_runner_keeps_the_existing_tau_curve_import():
    assert ff.tau_curve is tau_curve


# --------------------------------------------------------------------------- #
# comparison script
# --------------------------------------------------------------------------- #
def _scores(value):
    return {
        "ate_mae": value,
        "ate_rmse": value,
        "ate_max_abs_err": value,
        "ate_mae_on_support": value,
        "ate_mae_off_support": value,
        "ate_corr": 0.9,
        "att_mae": value,
        "atc_mae": value,
    }


def _fake_freng_run(root: Path, preset, seed, ate_mae, size=8, radius=2, n_units=5923):
    d = root / f"run_{preset}_{seed}"
    (d / "plots").mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(
        {"run_id": d.name, "effective_radius": radius,
         "config": {"preset": preset, "seed_data": seed, "seed_fit": seed,
                    "size": size, "digit": 0}}))
    (d / "metrics.json").write_text(json.dumps(
        {"status": "ok", "n_units": n_units, **_scores(ate_mae)}))


def _fake_ff_run(root: Path, preset, seed, ate_mae, size=8, radius=2, n_units=5923):
    d = root / f"ff_{preset}_{seed}"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(
        {"run_id": d.name, "effective_radius": radius,
         "config": {"preset": preset, "seed_data": seed, "size": size, "digit": 0,
                    "arm": "flexible_continuous", "conditioner": "mlp"}}))
    (d / "metrics.json").write_text(json.dumps(
        {"n_units": n_units, **_scores(ate_mae)}))


def _write_baseline(root: Path, preset: str, seeds=(1, 2), method="ols"):
    rows = []
    for seed in seeds:
        rows.append({
            "preset": preset, "seed": seed, "method": method,
            "basis": "poly3", "n_pixels": 64, "n_units": 5923,
            **_scores(0.01),
        })
    keys = list(rows[0])
    with open(root / "baselines.csv", "w", newline="") as handle:
        import csv
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def test_comparison_joins_and_averages(tmp_path):
    fr_root, ff_root, bl_root = (tmp_path / n for n in ("freng", "ffruns", "bl"))
    for r in (fr_root, ff_root, bl_root):
        r.mkdir()
    for seed, v in ((1, 0.02), (2, 0.04)):
        _fake_freng_run(fr_root, "exp2_confounded_homogeneous", seed, v)
        _fake_ff_run(ff_root, "exp2_confounded_homogeneous", seed, v + 0.01)
    _write_baseline(bl_root, "exp2_confounded_homogeneous")

    rows = (cmp_mod.load_frengression(str(fr_root)) + cmp_mod.load_ff(str(ff_root))
            + cmp_mod.load_baselines(str(bl_root)))
    cmp_mod.validate_grid(
        rows, ["exp2_confounded_homogeneous"], [1, 2],
        ["frengression", "ff_flexcont_mlp", "ols"],
    )
    agg = cmp_mod.aggregate(rows)
    got = {(a["method"], a["n_seeds"]): a["ate_mae_mean"] for a in agg}
    assert got[("frengression", 2)] == pytest.approx(0.03)
    assert got[("ff_flexcont_mlp", 2)] == pytest.approx(0.04)
    assert got[("ols", 2)] == pytest.approx(0.010)


def test_comparison_backfills_existing_tau_u_metrics(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    preset = "exp2_confounded_homogeneous"
    _fake_freng_run(root, preset, 1, 0.02, n_units=80)
    run = root / f"run_{preset}_1"

    y0_scalar = np.linspace(-1.0, 1.0, 80)
    y0 = np.column_stack([y0_scalar, y0_scalar ** 3])
    # Reversal makes paired differences non-constant, while the two marginal
    # quantile shifts remain exactly 0.5 and 0.2.
    y1 = y0[::-1] + np.array([0.5, 0.2])
    np.savez(run / "samples.npz", y0=y0, y1=y1)
    np.savez(
        run / "arrays.npz",
        ATE=np.array([1.0, 0.0]),
        TAU_MARGINAL=np.zeros((40, 2)),
    )

    _, curves = tau_curve(y0, y1)
    truth = np.zeros((40, 2))
    err = curves - truth
    row = cmp_mod.load_frengression(str(root))[0]
    assert row["tau_u_rmse_vs_marginal"] == pytest.approx(np.sqrt((err**2).mean()))
    assert row["tau_u_rmse_on_support"] == pytest.approx(
        np.sqrt((err[:, [True, False]] ** 2).mean())
    )


def test_comparison_rejects_missing_duplicate_and_mismatched_cells(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    _fake_freng_run(root, "exp2_confounded_homogeneous", 1, 0.02)
    rows = cmp_mod.load_frengression(str(root))
    with pytest.raises(ValueError, match="missing"):
        cmp_mod.validate_grid(
            rows, ["exp2_confounded_homogeneous"], [1, 2], ["frengression"]
        )

    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="duplicate"):
        cmp_mod.validate_grid(
            rows, ["exp2_confounded_homogeneous"], [1], ["frengression"]
        )

    fr_row = dict(rows[0])
    ff_row = {**fr_row, "method": "ff_flexcont_mlp", "radius": 1}
    with pytest.raises(ValueError, match="DGP mismatch"):
        cmp_mod.validate_grid(
            [fr_row, ff_row], ["exp2_confounded_homogeneous"], [1],
            ["frengression", "ff_flexcont_mlp"],
        )


def test_existing_baseline_csv_is_accepted_without_function_changes(tmp_path):
    bl = tmp_path / "bl"
    bl.mkdir()
    hdr = ("preset,seed,method,basis,n_pixels,n_units,ate_mae,ate_rmse,ate_max_abs_err,"
           "ate_mae_on_support,ate_mae_off_support,ate_corr,att_mae,atc_mae")
    (bl / "b.csv").write_text(
        hdr + "\nexp1_rct_homogeneous,1,ols,poly3,16,300,0.01,0.01,0.01,0.01,0.01,0.9,0.01,0.01\n")
    row = cmp_mod.load_baselines(str(bl))[0]
    assert row["seed_data"] == 1
    assert row["size"] == 4
    assert row["radius"] == 1
    assert row["digit"] == 0


def test_sweep_seed_maps_to_data_and_fit():
    cfg = sweep_agent.config_from_mapping({"seed": 17, "preset": "exp4_covariate_cate"})
    assert cfg.seed_data == 17
    assert cfg.seed_fit == 17
    assert cfg.wandb is True


def test_wandb_is_opt_in_and_logs_common_metrics(monkeypatch, tmp_path, data):
    import types

    class Run:
        def __init__(self):
            self.logged = []
            self.summary = {}
            self.finished = False

        def log(self, payload, step):
            self.logged.append((step, payload))

        def finish(self):
            self.finished = True

    created = Run()
    calls = []
    fake = types.SimpleNamespace(
        run=None,
        init=lambda **kwargs: calls.append(kwargs) or created,
        Image=lambda path: path,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake)

    assert fr._wandb_start(fr.Config(), "off") == (None, False)
    run, owned = fr._wandb_start(fr.Config(wandb=True), "cell")
    assert run is created and owned
    assert calls[0]["job_type"] == "frengression"

    losses = {
        "iter": np.array([1, 2]),
        "loss": np.array([2.0, 1.0]),
        "loss_y": np.array([1.0, 0.5]),
        "loss_eta": np.array([1.0, 0.5]),
    }
    fr._wandb_log(run, data, losses, {"ate_mae": 0.125}, str(tmp_path))
    assert created.logged[-1][0] == 3
    assert created.logged[-1][1]["ate_mae"] == 0.125
    assert created.summary["ate_mae"] == 0.125


# --------------------------------------------------------------------------- #
# config guards and the CLI
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kw", [{"preset": "nope"}, {"y_scaling": "nope"},
                                {"z_scaling": "nope"}, {"device": "tpu"},
                                {"n_mc": 1}, {"threads": 0}])
def test_config_guards(kw):
    with pytest.raises(ValueError):
        fr.Config(**kw)


def test_selftest_passes():
    """The gate the factory pipeline runs; also the fastest end-to-end check."""
    out = subprocess.run(
        [sys.executable, str(MORPHO / "exp_frengression_recovery.py"), "--selftest"],
        capture_output=True, text=True, timeout=1800, cwd=str(REPO))
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-2000:]
    assert "checks passed" in out.stdout
