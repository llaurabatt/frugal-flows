"""S11 (halo prereg Amendment A6): fit ONE arm on ONE causl dataset through the package pipeline.

Runs in env ``frugal-flows-halo`` with ``PYTHONPATH=<frugal-flows-gauss worktree>``:

  FrugalFlowModel(Y, X, Z_cont[, Z_disc], outcome_transform="standardize")
    .train_benchmark_model(jr.key(seed_fit), _MHP, _FHP, arm, cargs(arm), _PHP)
    .estimate_ate(jr.key(1000 + seed_fit))["ate"]

with _MHP/_FHP/_PHP/_CARGS copied from ``tests/test_causl_recovery.py`` (asserted equal at import
when the test module is importable). Writes ``<out>/result.json``; on any exception writes
``<out>/FAILED`` (traceback) and exits 1. Package precision default (float64) is used and recorded.

No package file is modified. To record epochs run / best epoch, ``train_frugal_flow`` is wrapped
inside ``frugal_flows.benchmarking``'s namespace for this process only (it passes every argument
through unchanged and keeps the losses it returns).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

# ---- hyperparameters: identical to tests/test_causl_recovery.py --------------------------------
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

ARMS = ("gaussian", "location_translation", "flexible_continuous",
        "location_translation_gaussian", "flexible_continuous_gaussian")
SHIFT_ARMS = ("gaussian", "location_translation", "location_translation_gaussian")


def cargs_for(arm: str) -> dict:
    """Per-arm causal_model_args, as the package's own tests build them
    (tests/test_train_frugal_flow.py, tests/test_gaussian_scale.py):
      gaussian                      {"ate": zeros(1), "scale": 1.0, "const": 0.0}  (the notebook's init)
      location_translation          _CARGS | {"ate": 0.0}
      flexible_continuous           _CARGS
      location_translation_gaussian _CARGS | {"ate": 0.0}   (interval default 5)
      flexible_continuous_gaussian  _CARGS                  (interval default 5)
    """
    import jax.numpy as jnp
    if arm == "gaussian":
        return {"ate": jnp.zeros((1,)), "scale": jnp.array(1.0), "const": jnp.array(0.0)}
    if arm in ("location_translation", "location_translation_gaussian"):
        return dict(_CARGS) | {"ate": 0.0}
    if arm in ("flexible_continuous", "flexible_continuous_gaussian"):
        return dict(_CARGS)
    raise ValueError(arm)


def cargs_json(arm: str) -> dict:
    return {k: (float(v.ravel()[0]) if hasattr(v, "ravel") else v) for k, v in cargs_for(arm).items()}


def fitted_shift_std(arm: str, flow):
    """The fitted shift parameter on the STANDARDISED outcome scale (None for non-shift arms).

    gaussian: UnivariateNormalCDF.ate (mean of the Normal margin moves by ate * T);
    location_translation: LocCond.ate in the uniform-scale flow (y = margin(e) + ate * T);
    location_translation_gaussian: gaussian_scale.shift_vector.
    """
    import jax
    import paramax
    from frugal_flows.bijections.loc_cond import LocCond
    from frugal_flows.bijections.univariate_normal_cdf import UnivariateNormalCDF

    if arm not in SHIFT_ARMS:
        return None
    if arm == "location_translation_gaussian":
        from frugal_flows import gaussian_scale
        return [float(v) for v in jax.numpy.ravel(paramax.unwrap(gaussian_scale.shift_vector(flow)))]
    cls = UnivariateNormalCDF if arm == "gaussian" else LocCond
    found = [x for x in jax.tree_util.tree_leaves(paramax.unwrap(flow), is_leaf=lambda x: isinstance(x, cls))
             if isinstance(x, cls)]
    if len(found) != 1:
        raise RuntimeError(f"expected one {cls.__name__} in the {arm} flow, found {len(found)}")
    return [float(v) for v in jax.numpy.ravel(found[0].ate)]


def to_data_scale(shift_std, transform):
    """Standardize is y_std = (y - mean) / sd, so a shift of a in y_std is a shift of a * sd in y."""
    import numpy as np
    if shift_std is None:
        return None
    sd = np.ravel(np.asarray(transform._sd))
    return [float(a * s) for a, s in zip(shift_std, np.broadcast_to(sd, (len(shift_std),)))]


def git_head():
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def run(a) -> dict:
    import jax
    import jax.random as jr
    import numpy as np

    import frugal_flows
    from frugal_flows import benchmarking
    from npz_io import load_npz

    ff_file = os.path.abspath(frugal_flows.__file__)
    assert ff_file.startswith(REPO + os.sep), f"frugal_flows imported from {ff_file}, not {REPO}"

    # record the losses train_frugal_flow returns (pass-through wrapper, this process only)
    captured = {}
    _orig = benchmarking.train_frugal_flow

    def _recording(*args, **kw):
        flow, losses = _orig(*args, **kw)
        captured["losses"] = losses
        return flow, losses

    benchmarking.train_frugal_flow = _recording

    d = load_npz(a.data)
    meta = d["meta"]
    Y, X, Zc, Zd = d["Y"], d["X"], d["Z_cont"], d["Z_disc"]
    kwargs = dict(Y=Y, X=X, Z_cont=Zc, outcome_transform="standardize")
    if Zd is not None:
        kwargs["Z_disc"] = Zd
    model = benchmarking.FrugalFlowModel(**kwargs)

    t0 = time.time()
    model.train_benchmark_model(jr.key(a.seed_fit), _MHP, a.fhp, a.arm, cargs_for(a.arm), a.php)
    t_fit = time.time() - t0
    t1 = time.time()
    est = model.estimate_ate(jr.key(1000 + a.seed_fit))
    t_ate = time.time() - t1

    y0, y1 = np.asarray(est["y0"]), np.asarray(est["y1"])
    fin = np.isfinite(y0) & np.isfinite(y1)
    losses = captured.get("losses", {})
    val = [float(v) for v in losses.get("val", [])]
    shift_std = fitted_shift_std(a.arm, model.frugal_flow)
    shift_data = to_data_scale(shift_std, model.outcome_transform)
    true_ate = float(meta["true_ate"])
    ate_hat = float(est["ate"])
    return dict(
        ate_hat=ate_hat, bias=ate_hat - true_ate, true_ate=true_ate, naive=float(meta["naive"]),
        ols=float(meta.get("ols", float("nan"))),
        ate_hat_finite_pairs=float(np.mean((y1 - y0)[fin])) if fin.any() else None,
        n_nonfinite_pairs=int((~fin).sum()), n_mc=int(y0.shape[0]), n_clamped=int(est["n_clamped"]),
        tau_sd=float(est["tau_sd"]), mean0=float(est["mean0"]), mean1=float(est["mean1"]),
        shift_std=shift_std, shift_data_scale=shift_data,
        shift_minus_ate_hat=(None if shift_data is None else shift_data[0] - ate_hat),
        outcome_sd=float(np.ravel(np.asarray(model.outcome_transform._sd))[0]),
        epochs_run=len(val), best_epoch=(int(np.argmin(val)) + 1 if val else None),
        best_val_loss=(float(np.min(val)) if val else None),
        wall_fit_s=t_fit, wall_ate_s=t_ate,
        arm=a.arm, model=meta["model"], generator=meta["generator"], causal_params=meta["causal_params"],
        n=int(meta["n"]), seed_data=int(meta["seed"]), seed_fit=a.seed_fit, ate_key=1000 + a.seed_fit,
        data=os.path.abspath(a.data),
        hp=dict(MHP=_MHP, FHP=a.fhp, PHP=a.php, cargs=cargs_json(a.arm)),
        precision=dict(jax_enable_x64=bool(jax.config.jax_enable_x64), y_fit_dtype=str(jax.numpy.asarray(
            model.outcome_transform.forward(jax.numpy.asarray(Y, dtype=jax.numpy.float64))).dtype),
            JAX_ENABLE_X64_env=os.environ.get("JAX_ENABLE_X64")),
        xla_flags=os.environ.get("XLA_FLAGS"),
        threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
        git_head=git_head(), frugal_flows_file=ff_file, jax_version=jax.__version__,
        overlap_frac_outside_05_95=meta.get("overlap_frac_outside_05_95"), frac_treated=meta.get("frac_treated"),
        complete=True,
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--seed-fit", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-epochs", type=int, default=None, help="smoke only: overrides FHP/PHP max_epochs")
    a = p.parse_args(argv)
    a.fhp, a.php = dict(_FHP), dict(_PHP)
    if a.max_epochs is not None:
        a.fhp["max_epochs"] = a.php["max_epochs"] = a.max_epochs
    os.makedirs(a.out, exist_ok=True)
    for f in ("result.json", "FAILED"):
        if os.path.exists(os.path.join(a.out, f)):
            os.remove(os.path.join(a.out, f))
    try:
        res = run(a)
        res["smoke_max_epochs"] = a.max_epochs
        tmp = os.path.join(a.out, "result.json.tmp")
        with open(tmp, "w") as fh:
            json.dump(res, fh, indent=1)
        os.replace(tmp, os.path.join(a.out, "result.json"))
        print(f"OK {a.arm} {res['model']} n={res['n']} seed={res['seed_data']}: ate_hat={res['ate_hat']:.4f} "
              f"true={res['true_ate']:.4f} naive={res['naive']:.4f} shift={res['shift_data_scale']} "
              f"epochs={res['epochs_run']} best={res['best_epoch']} fit={res['wall_fit_s']:.0f}s", flush=True)
        return 0
    except Exception:  # noqa: BLE001
        with open(os.path.join(a.out, "FAILED"), "w") as fh:
            fh.write(traceback.format_exc())
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
