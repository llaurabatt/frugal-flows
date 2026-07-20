"""Quick causl-backed sense check for frugal-flow causal models.

A fast (no-notebook) smoke test you can rerun per model and eyeball side by side.
Generates a small dataset from the R `causl` package (known true ATE + known
interventional moments), fits the frugal flow for each requested causal_model,
and runs three sense checks:

  1. TRAIN   — finite val loss; no NaNs in interventional samples.
  2. ATE     — post-hoc ATE within --tol (relative) of the true causl ATE.
  3. MOMENTS — sampled E[Y|do(T=0)] ~ const, E[Y|do(T=1)] ~ const+ate,
               Var[Y|do(T)] ~ phi  (absolute tol).

The ATE / moments are read out MODEL-AGNOSTICALLY: sample the full fitted flow
at a fixed treatment T and read dim 0 (Y). Valid because the copula masks T
(mask_condition=True), so the flow's Y-marginal at externally-fixed T equals
p(Y | do(T)). This avoids the gaussian-only margin extraction in frugal_fitting.

Usage (from repo root, in the frugal-flows-flowjax env):
  micromamba run -n frugal-flows-flowjax python validation/diagnostics/quick_sense_check.py
  micromamba run -n frugal-flows-flowjax python validation/diagnostics/quick_sense_check.py \
      --models gaussian,flexible_continuous --generator gaussian --n 2000 --epochs 400
"""

from __future__ import annotations

import argparse
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)  # must precede any jnp array creation

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402

# Make the causl generators + frugal_flows importable regardless of cwd.
_VALIDATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VALIDATION_DIR not in sys.path:
    sys.path.insert(0, _VALIDATION_DIR)

import data_processing_and_simulations.causl_sim_data_generation as causl_py  # noqa: E402
from frugal_flows.causal_flows import train_frugal_flow  # noqa: E402

GENERATORS = {
    "gaussian": causl_py.generate_gaussian_samples,
    "mixed": causl_py.generate_mixed_samples,
}


def base_hyperparams(epochs: int) -> dict:
    return dict(
        RQS_knots=8,
        nn_depth=4,
        nn_width=50,
        flow_layers=4,
        learning_rate=5e-3,
        max_epochs=epochs,
        max_patience=max(20, epochs // 10),
        batch_size=256,
        show_progress=False,
    )


def model_args(causal_model: str, cond_dim: int) -> dict:
    if causal_model == "gaussian":
        return {"ate": jnp.zeros(cond_dim), "scale": 1.0, "const": 0.0}
    if causal_model in ("flexible_continuous", "location_translation"):
        return {"nn_depth": 4, "nn_width": 50, "RQS_knots": 8, "flow_layers": 4}
    raise ValueError(f"Unsupported causal_model for sense check: {causal_model}")


def final_val_loss(losses) -> float:
    if isinstance(losses, dict):
        seq = losses.get("val") or losses.get("train") or next(iter(losses.values()))
    else:
        seq = losses
    return float(seq[-1])


def fit_model(key, Y, u_z, X, causal_model, epochs):
    cond_dim = X.shape[1]
    ff, losses = train_frugal_flow(
        key, Y, u_z, condition=X,
        causal_model=causal_model,
        causal_model_args=model_args(causal_model, cond_dim),
        **base_hyperparams(epochs),
    )
    return ff, final_val_loss(losses)


def intervene_moments(key, ff, cond_dim, n_mc):
    """Estimate p(Y|do(T=t)) and the ATE by sampling the full flow at fixed T, dim 0.

    Common random numbers: the SAME base draw (same `key`) is pushed through both
    interventions, so we get PAIRED outcomes (Q_1(u), Q_0(u)) sharing the latent u.
    ATE = mean of per-draw differences tau(u)=Q_1(u)-Q_0(u). For the mean this equals
    difference-of-means, but with the MC noise cancelled. std(tau) is the quantile-
    effect heterogeneity: ~0 for a pure location shift (gaussian/location), >0 for a
    treatment-conditioned spline. Pairing assumes a shared-rank counterfactual coupling.
    """
    s0 = ff.sample(key, condition=jnp.zeros((n_mc, cond_dim)))
    s1 = ff.sample(key, condition=jnp.ones((n_mc, cond_dim)))
    y0, y1 = s0[:, 0], s1[:, 0]
    tau = y1 - y0  # paired per-draw treatment effect tau(u)
    return {
        "mean0": float(jnp.mean(y0)), "mean1": float(jnp.mean(y1)),
        "var0": float(jnp.var(y0)), "var1": float(jnp.var(y1)),
        "ate": float(jnp.mean(tau)),
        "tau_sd": float(jnp.std(tau)),  # effect heterogeneity across quantiles
        "anynan": bool(jnp.any(jnp.isnan(y0)) | jnp.any(jnp.isnan(y1))),
    }


def run_one(key, causal_model, Y, u_z, X, args, true, hard):
    ff, val_loss = fit_model(jr.fold_in(key, 1), Y, u_z, X, causal_model, args.epochs)
    m = intervene_moments(jr.fold_in(key, 2), ff, X.shape[1], args.n_mc)

    # --- always-hard SMOKE checks (cheap, scale-independent) ---
    finite_loss = bool(jnp.isfinite(jnp.array(val_loss)).item())
    no_nan = not m["anynan"]
    vars_ok = all(v > 0 and jnp.isfinite(jnp.array(v)).item() for v in (m["var0"], m["var1"]))
    # interventional effect points the right way (or true ATE ~0 -> skip sign)
    ate = m["ate"]
    sign_ok = (abs(true["ate"]) < 1e-8) or (ate * true["ate"] > 0)
    t_smoke = finite_loss and no_nan and vars_ok and sign_ok

    # --- recovery metrics: HARD only with --full, else informational ---
    ate_relerr = abs(ate - true["ate"]) / max(abs(true["ate"]), 1e-8)
    moment_err = max(
        abs(m["mean0"] - true["const"]),
        abs(m["mean1"] - (true["const"] + true["ate"])),
    )
    var_err = max(abs(m["var0"] - true["phi"]), abs(m["var1"] - true["phi"]))
    t_ate = ate_relerr < args.tol
    t_mom = (moment_err < args.moment_tol) and (var_err < args.var_tol)

    passed = (t_smoke and t_ate and t_mom) if hard else t_smoke
    return {
        "model": causal_model, "val_loss": val_loss,
        "ate": ate, "ate_relerr": ate_relerr, "tau_sd": m["tau_sd"],
        "mean0": m["mean0"], "mean1": m["mean1"], "var0": m["var0"], "var1": m["var1"],
        "moment_err": moment_err, "var_err": var_err,
        "t_smoke": t_smoke, "t_ate": t_ate, "t_mom": t_mom, "pass": passed,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default="gaussian,flexible_continuous",
                   help="comma-separated causal_model names")
    p.add_argument("--generator", default="gaussian", choices=list(GENERATORS))
    p.add_argument("--full", action="store_true",
                   help="recovery tier: large n + HARD pass/fail on absolute ATE/moments "
                        "(default is fast smoke tier: ATE shown but not gated)")
    p.add_argument("--n", type=int, default=None,
                   help="training sample size (default 2000 smoke / 20000 --full)")
    p.add_argument("--epochs", type=int, default=None,
                   help="max epochs (default 400 smoke / 1500 --full; early-stops)")
    p.add_argument("--n-mc", type=int, default=20000, help="MC samples for readout")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ate", type=float, default=1.0, help="true causl ATE (causal_params[1])")
    p.add_argument("--const", type=float, default=0.0, help="true causl intercept (causal_params[0])")
    p.add_argument("--phi", type=float, default=1.0, help="true causl outcome dispersion (Var)")
    p.add_argument("--tol", type=float, default=0.15, help="relative tol for ATE check (--full)")
    p.add_argument("--moment-tol", type=float, default=0.3, help="abs tol for interventional means (--full)")
    p.add_argument("--var-tol", type=float, default=0.5, help="abs tol for interventional variances (--full)")
    args = p.parse_args()

    # tier defaults
    if args.n is None:
        args.n = 20000 if args.full else 2000
    if args.epochs is None:
        args.epochs = 1500 if args.full else 400

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    true = {"ate": args.ate, "const": args.const, "phi": args.phi}

    tier = "FULL (recovery: hard ATE/moment pass-fail)" if args.full \
        else "SMOKE (fast: runs+NaN+sane-moments gated; ATE informational)"
    print(f"tier={tier}")
    print(f"generator={args.generator}  n={args.n}  epochs={args.epochs}  seed={args.seed}")
    print(f"true: ATE={args.ate}  const={args.const}  phi(Var)={args.phi}\n")

    # one causl dataset, shared across models (apples-to-apples)
    data = GENERATORS[args.generator](args.n, causal_params=[args.const, args.ate], seed=args.seed)
    X, Y, Z_disc, Z_cont = data["X"], data["Y"], data["Z_disc"], data["Z_cont"]
    uz = causl_py.generate_uz_samples(Z_disc, Z_cont, False, args.seed, base_hyperparams(args.epochs))
    u_z = uz["uz_samples"]
    print(f"data: X{tuple(X.shape)} Y{tuple(Y.shape)} u_z{tuple(u_z.shape)}  treated_frac={float(X.mean()):.3f}\n")

    key = jr.key(args.seed)
    rows = [run_one(jr.fold_in(key, i), m, Y, u_z, X, args, true, hard=args.full)
            for i, m in enumerate(models)]

    # report. In SMOKE tier ate/mom are informational (parenthesised); only smoke gates.
    def mark(ok):
        return "OK" if ok else "X"

    hdr = (f"{'model':<22}{'ATE':>8}{'relerr':>9}{'tau_sd':>8}{'E[Y|0]':>9}{'E[Y|1]':>9}{'Var0':>7}{'Var1':>7}"
           f"  smoke {'ate/mom' if args.full else '(ate/mom info)'}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        am = f"{mark(r['t_ate'])}/{mark(r['t_mom'])}"
        am = am if args.full else f"({am})"
        verdict = "PASS" if r["pass"] else "FAIL"
        print(f"{r['model']:<22}{r['ate']:>8.3f}{r['ate_relerr']:>8.1%}{r['tau_sd']:>8.3f}{r['mean0']:>9.3f}"
              f"{r['mean1']:>9.3f}{r['var0']:>7.2f}{r['var1']:>7.2f}  {mark(r['t_smoke']):>5} {am:>11}  {verdict}")
    print()
    overall = all(r["pass"] for r in rows)
    if args.full:
        print(f"OVERALL: {'PASS' if overall else 'FAIL'}  "
              f"(gated: smoke + ATE rel<{args.tol} + mean abs<{args.moment_tol} + var abs<{args.var_tol})")
    else:
        print(f"OVERALL: {'PASS' if overall else 'FAIL'}  (gated: smoke only — runs/NaN/sane+ordered moments)")
        print("note: absolute ATE recovery is NOT gated in smoke tier (needs ~20k samples). "
              "Run with --full for a hard recovery check.")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
