"""Multi-restart stability of the SPLINE (`flexible_continuous`) causal margin.

Motivation: the same (dataset, fit-key) spline fit at n=2000 was observed to swing
its ATE readout by ~0.5 across thread-count settings on this machine, suggesting a
flat / multi-basin optimization landscape (the loss surface has several minima that
correspond to visibly different ATE readouts). If that is right, ANY single spline
fit at n~2000 is an unreliable point estimate, and the fix would be to run several
restarts and select on validation loss rather than trust a lone fit.

This script isolates that question from the everyday seed-to-seed sampling noise:
for a FIXED dataset (data seed held fixed) and a FIXED u_z, it re-fits the spline
arm `--restarts` times, varying ONLY the flow's own init/training key. It reports:

  1. restart-to-restart spread of the ATE readout (mean/sd/min/max) per (dgp, config);
  2. whether picking the restart with the BEST validation loss lands closer to the
     true ATE / to the restart-mean than an arbitrary single restart would (the
     "best-val ate" vs "mean ate" comparison is the headline readout);
  3. Pearson corr(val_loss, ate) across restarts — if the landscape is multi-basin
     but val-loss-linked, low-loss restarts should cluster in ATE; if uncorrelated,
     best-val selection would not help.

It also dumps per-fit tau_hat(u) quantile-effect curves (same binning convention as
`ate_extraction_suite.tau_curve`) to an .npz, for a separate heterogeneity analysis
of whether the restart-to-restart spread is a uniform shift in tau(u) or a
shape change.

Three optimizer configs probe whether the instability is a step-size / budget
artifact rather than a property of the loss surface itself:
  baseline  learning_rate=5e-3  epochs=600   (matches base_hyperparams at n=2000)
  lowlr     learning_rate=1e-3  epochs=1200  (smaller steps, more of them)
  long      learning_rate=5e-3  epochs=2400  (same steps, much longer budget)
`max_patience` is NOT held fixed across configs: it scales with epochs using the
same `max(20, epochs // 10)` rule `base_hyperparams` uses, so early stopping is
proportionally as patient at every budget.

Usage (from validation/, in the frugal-flows-flowjax env):
  micromamba run -n frugal-flows-flowjax python -m diagnostics.spline_stability \
      --dgps gamma_b0,gamma_b1 --restarts 8 --configs baseline,lowlr,long \
      --out outputs/spline_stability.csv --curves-out outputs/spline_stability_curves.npz
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)  # must precede any jnp use

import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402

_VALIDATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VALIDATION_DIR not in sys.path:
    sys.path.insert(0, _VALIDATION_DIR)

import data_processing_and_simulations.causl_sim_data_generation as causl_py  # noqa: E402
from diagnostics.ate_extraction_suite import intervene, tau_curve  # noqa: E402
from diagnostics.h1_matrix import qte_integrated_error, robust_moments  # noqa: E402
from diagnostics.outcome_families import FAMILIES  # noqa: E402
from diagnostics.quick_sense_check import final_val_loss, model_args  # noqa: E402
from frugal_flows.causal_flows import train_frugal_flow  # noqa: E402

CAUSAL_MODEL = "flexible_continuous"
TAU_CURVE_BINS = 40  # must match ate_extraction_suite.TAU_CURVE_BINS for stacking

# Optimizer configs to probe (learning_rate, epochs). max_patience is derived, not
# fixed, so it scales proportionally with the epoch budget (see module docstring).
CONFIGS = {
    "baseline": dict(learning_rate=5e-3, epochs=600),
    "lowlr": dict(learning_rate=1e-3, epochs=1200),
    "long": dict(learning_rate=5e-3, epochs=2400),
}

FIELDNAMES = ["dgp", "config", "restart", "n", "data_seed", "true_ate", "ate", "bias",
              "tau_sd", "qte_int_err", "val_loss", "val_best", "n_drop", "secs", "error"]


def _patience(epochs: int) -> int:
    """Same rule base_hyperparams uses: patience scales proportionally with epochs."""
    return max(20, epochs // 10)


def fit_model_override(key, Y, u_z, X, causal_model, learning_rate, epochs, patience):
    """Fit the frugal flow with overridable optimizer hyperparams.

    Mirrors `quick_sense_check.fit_model`'s body exactly (same fixed architecture
    knobs: RQS_knots/nn_depth/nn_width/flow_layers/batch_size), but allows
    learning_rate/max_epochs/max_patience to be swept per-config instead of being
    hard-coded via `base_hyperparams(epochs)`.
    """
    cond_dim = X.shape[1]
    ff, losses = train_frugal_flow(
        key, Y, u_z, condition=X,
        causal_model=causal_model,
        causal_model_args=model_args(causal_model, cond_dim),
        RQS_knots=8,
        nn_depth=4,
        nn_width=50,
        flow_layers=4,
        learning_rate=learning_rate,
        max_epochs=epochs,
        max_patience=patience,
        batch_size=256,
        show_progress=False,
    )
    val_seq = losses["val"] if isinstance(losses, dict) else losses
    # last-epoch val loss (what final_val_loss reports) AND the best (minimum) val
    # loss: train_frugal_flow returns the best-val checkpoint's params, so val_best
    # is the statistic that actually governs the returned model — use IT for
    # restart selection, not the post-optimum drifted last value.
    return ff, final_val_loss(losses), float(np.min(np.asarray(val_seq, dtype=float)))


def _pearsonr_safe(x, y):
    """Pearson corr with nan-guards: needs >=3 points and nonzero variance in both."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def run(args):
    dgps = [d.strip() for d in args.dgps.split(",") if d.strip()]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in configs:
        if c not in CONFIGS:
            raise SystemExit(f"unknown config {c!r}; known: {list(CONFIGS)}")
    for d in dgps:
        if d not in FAMILIES:
            raise SystemExit(f"unknown dgp {d!r}; known: {list(FAMILIES)}")
    cp = [args.const, args.ate]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.curves_out)) or ".", exist_ok=True)

    total = len(dgps) * len(configs) * args.restarts
    done = 0
    print(f"[spline_stability] dgps={dgps} configs={configs} restarts={args.restarts} "
          f"n={args.n} data_seed={args.data_seed} causal_params={cp}", flush=True)

    curves = {}
    u_grid = None
    # summaries[(dgp, config)] -> (list of ate, list of val_loss), successful restarts only
    summaries = {}

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        w.writeheader(); fh.flush()

        for dgp in dgps:
            fam = FAMILIES[dgp]
            true_ate = fam.true_ate(cp)

            # dataset + u_z generated ONCE per dgp: only the fit key varies below.
            data = fam.generate(args.n, causal_params=cp, seed=args.data_seed)
            X, Y = data["X"], data["Y"]
            Z_disc, Z_cont = data["Z_disc"], data["Z_cont"]
            # use_marginal_flow=False -> empirical CDF path; frugal_flow_hyperparams
            # is unused on that path, so passing {} here is safe.
            uz = causl_py.generate_uz_samples(Z_disc, Z_cont, False, args.data_seed, {})
            u_z = uz["uz_samples"]
            print(f"[{dgp}] data X{tuple(np.asarray(X).shape)} Y{tuple(np.asarray(Y).shape)} "
                  f"u_z{tuple(np.asarray(u_z).shape)}  true_ate={true_ate:+.4f}", flush=True)

            for config_name in configs:
                cfg = CONFIGS[config_name]
                learning_rate, epochs = cfg["learning_rate"], cfg["epochs"]
                patience = _patience(epochs)
                ates, val_losses = [], []

                for r in range(args.restarts):
                    fit_key = jr.key(70000 + 1000 * r)  # data fixed; ONLY this varies
                    t0 = time.time()
                    row = {k: "" for k in FIELDNAMES}
                    row.update(dgp=dgp, config=config_name, restart=r, n=args.n,
                               data_seed=args.data_seed, true_ate=true_ate)
                    try:
                        ff, val_loss, val_best = fit_model_override(
                            fit_key, Y, u_z, X, CAUSAL_MODEL, learning_rate, epochs, patience,
                        )
                        m = intervene(jr.fold_in(fit_key, 2), ff, X.shape[1], args.n_mc)
                        ate, tau_sd, n_drop, n_keep = robust_moments(m["y0"], m["y1"])
                        qte_err = qte_integrated_error(fam, cp, m["y0"], m["y1"])
                        u_centers, tau_of_u = tau_curve(m["y0"], m["y1"], n_bins=TAU_CURVE_BINS)
                        if u_grid is None:
                            u_grid = u_centers
                        curves[f"{dgp}|{config_name}|r{r}"] = tau_of_u

                        row.update(ate=ate, bias=ate - true_ate, tau_sd=tau_sd,
                                   qte_int_err=qte_err, val_loss=val_loss,
                                   val_best=val_best, n_drop=n_drop)
                        ates.append(ate); val_losses.append(val_best)
                        if n_drop:
                            print(f"  [drop] {dgp}/{config_name} r={r}: "
                                  f"{n_drop}/{n_drop + n_keep} non-finite MC samples filtered", flush=True)
                    except Exception as e:  # noqa: BLE001 -- keep the sweep running
                        row.update(error=repr(e))
                    row["secs"] = round(time.time() - t0, 1)
                    w.writerow(row); fh.flush()
                    done += 1
                    tag = ("ERR " + row["error"]) if row["error"] else (
                        f"ate={float(row['ate']):+.3f} (true {true_ate:+.3f}) "
                        f"bias={float(row['bias']):+.3f} val_loss={float(row['val_loss']):.4f}")
                    print(f"[{done}/{total}] {dgp}/{config_name} r={r} {tag} ({row['secs']}s)", flush=True)

                summaries[(dgp, config_name)] = (ates, val_losses)

            # ground-truth tau(u) curve on the same u-grid the fits used
            if u_grid is not None:
                curves[f"{dgp}|true"] = fam.true_tau_curve(cp, u_grid)

    if u_grid is not None:
        curves["u_grid"] = u_grid
    np.savez(args.curves_out, **curves)
    print(f"[spline_stability] DONE {done}/{total}. csv={args.out}  curves={args.curves_out}", flush=True)

    # ---- summary table: restart spread + best-val-loss selection ----
    print("\n=== restart stability summary (ATE across restarts, fixed dataset) ===")
    hdr = (f"{'dgp':<12}{'config':<10}{'k':>3}{'mean':>9}{'sd':>8}{'min':>9}{'max':>9}"
           f"{'best_val_ate':>13}{'corr(val,ate)':>15}")
    print(hdr); print("-" * len(hdr))
    for dgp in dgps:
        for config_name in configs:
            ates, val_losses = summaries.get((dgp, config_name), ([], []))
            k = len(ates)
            if k == 0:
                print(f"{dgp:<12}{config_name:<10}{0:>3}{'--':>9}{'--':>8}{'--':>9}{'--':>9}"
                      f"{'--':>13}{'--':>15}  (all restarts failed)")
                continue
            ates_arr = np.asarray(ates)
            val_arr = np.asarray(val_losses)
            best_idx = int(np.argmin(val_arr))
            best_val_ate = ates_arr[best_idx]
            corr = _pearsonr_safe(val_arr, ates_arr)
            print(f"{dgp:<12}{config_name:<10}{k:>3}{ates_arr.mean():>9.3f}{ates_arr.std():>8.3f}"
                  f"{ates_arr.min():>9.3f}{ates_arr.max():>9.3f}{best_val_ate:>13.3f}{corr:>15.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dgps", default="gamma_b0,gamma_b1")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--restarts", type=int, default=8)
    p.add_argument("--configs", default="baseline,lowlr,long")
    p.add_argument("--n-mc", type=int, default=20000)
    p.add_argument("--const", type=float, default=1.0, help="causal_params[0] (b0)")
    p.add_argument("--ate", type=float, default=0.5, help="causal_params[1] (b1)")
    p.add_argument("--out", required=True, help="per-fit CSV path")
    p.add_argument("--curves-out", required=True, help="per-fit tau(u) curves .npz path")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
