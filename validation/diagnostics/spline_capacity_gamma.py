"""Spline (`flexible_continuous`) capacity/hyperparameter grid on the MISSPECIFIED
gamma DGP at n=2000, WITH multiple optimizer restarts per capacity cell.

Motivation: `spline_stability.py` found that the flexible_continuous spline arm at
n=2000, on a FIXED dataset, has large restart-to-restart ATE noise (sd 0.13-0.33)
that is a LEVEL shift — the tau(u) SHAPE is stably recovered across restarts
(shape~true corr +0.7 to +0.9) but the overall ATE level wanders depending on which
optimization basin a restart lands in. That means a capacity grid with only ONE fit
per cell would be uninterpretable: any capacity effect would be confounded with
restart-level noise (a lucky/unlucky restart could masquerade as a capacity effect).

This script fixes that by fitting R restarts per (RQS_knots) capacity cell on the
SAME dataset (data seed and u_z held fixed throughout; only the fit key varies), and
reporting BOTH the restart-mean ATE and the restart-SD separately. That lets us ask,
per capacity level:
  (a) does more capacity (more knots) REDUCE the restart-to-restart level noise
      (sd(ate) shrinking with knots)?
  (b) does it SHIFT the mean bias (mean(ate) - true_ate moving with knots)?
  (c) does it IMPROVE shape recovery (mean qte_int_err, the integrated |tau_hat(u) -
      tau_true(u)| error, shrinking with knots)?

The DGP is `gamma_b1` (log-link Gamma outcome): a location-shift causal margin is
misspecified for it (the true effect tau(u) grows with u, not flat), which is why
the flexible spline margin is the arm of interest here rather than `gaussian`.

Reuses (see module docstrings there for full rationale):
  - `diagnostics.spline_stability._patience`, `_pearsonr_safe` (optimizer-schedule
    and restart-spread-vs-val-loss helpers).
  - `diagnostics.h1_matrix.robust_moments`, `qte_integrated_error` (finite-filtered
    paired ATE/tau_sd, and the integrated shape-error metric).
  - `diagnostics.ate_extraction_suite.intervene`, `tau_curve` (paired-CRN
    interventional readout and the quantile-resolved effect curve).
  - `diagnostics.outcome_families.FAMILIES` (DGP + analytic ground truth).
  - `diagnostics.quick_sense_check.model_args`, `final_val_loss` (default
    `causal_model_args` dict, then overridden with the swept capacity knobs; last
    logged val loss).

Usage (from validation/, in the frugal-flows-flowjax env):
  micromamba run -n frugal-flows-flowjax python -m diagnostics.spline_capacity_gamma \
      --knots 4,8,16 --restarts 5 --out outputs/spline_capacity_gamma.csv
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
from diagnostics.spline_stability import _patience, _pearsonr_safe  # noqa: E402
from frugal_flows.causal_flows import train_frugal_flow  # noqa: E402

CAUSAL_MODEL = "flexible_continuous"

# Capacity knobs held FIXED while RQS_knots is swept (matches spline_stability's
# / quick_sense_check's default architecture).
NN_DEPTH = 4
NN_WIDTH = 50
FLOW_LAYERS = 4

# Fixed optimizer config (matches spline_stability's "baseline" config, the one
# that showed the restart instability at n=2000).
LEARNING_RATE = 5e-3
EPOCHS = 600
BATCH_SIZE = 256

FIELDNAMES = ["dgp", "RQS_knots", "nn_depth", "nn_width", "flow_layers", "restart",
              "n", "true_ate", "ate", "bias", "tau_sd", "qte_int_err", "val_loss",
              "val_best", "n_drop", "secs", "error"]


def fit_model_capacity(key, Y, u_z, X, causal_model, RQS_knots, nn_depth, nn_width,
                       flow_layers, learning_rate, epochs, patience, batch_size):
    """Fit the frugal flow with sweepable architecture (capacity) knobs.

    Mirrors `spline_stability.fit_model_override`'s body, but RQS_knots/nn_depth/
    nn_width/flow_layers are passed through as args instead of hard-coded, so a
    capacity grid can vary them. `causal_model_args` starts from
    `quick_sense_check.model_args`'s default dict and is then overridden with the
    SAME swept values, since the causal margin's own autoregressive bijection
    (`causal_model_args["RQS_knots"]` etc.) and the copula/base-flow transformer
    (`RQS_knots=` top-level kwarg) must agree — see `spline_capacity.py`'s
    `_configs()` for the same pattern.
    """
    cond_dim = X.shape[1]
    cma = model_args(causal_model, cond_dim)
    cma.update(RQS_knots=RQS_knots, nn_depth=nn_depth, nn_width=nn_width,
               flow_layers=flow_layers)
    ff, losses = train_frugal_flow(
        key, Y, u_z, condition=X,
        causal_model=causal_model,
        causal_model_args=cma,
        RQS_knots=RQS_knots,
        nn_depth=nn_depth,
        nn_width=nn_width,
        flow_layers=flow_layers,
        learning_rate=learning_rate,
        max_epochs=epochs,
        max_patience=patience,
        batch_size=batch_size,
        show_progress=False,
    )
    val_seq = losses["val"] if isinstance(losses, dict) else losses
    return ff, final_val_loss(losses), float(np.min(np.asarray(val_seq, dtype=float)))


def run(args):
    knots_list = [int(k.strip()) for k in args.knots.split(",") if k.strip()]
    if args.dgp not in FAMILIES:
        raise SystemExit(f"unknown dgp {args.dgp!r}; known: {list(FAMILIES)}")
    fam = FAMILIES[args.dgp]
    cp = [args.const, args.ate]
    true_ate = fam.true_ate(cp)
    patience = _patience(EPOCHS)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    total = len(knots_list) * args.restarts
    done = 0
    print(f"[spline_capacity_gamma] dgp={args.dgp} n={args.n} data_seed={args.data_seed} "
          f"causal_params={cp} knots={knots_list} restarts={args.restarts} "
          f"(nn_depth={NN_DEPTH} nn_width={NN_WIDTH} flow_layers={FLOW_LAYERS} "
          f"lr={LEARNING_RATE} epochs={EPOCHS} patience={patience}) true_ate={true_ate:+.4f}",
          flush=True)

    # Dataset + u_z generated ONCE for the whole grid: data seed and Z are fixed so
    # every cell's capacity effect is isolated from data-sampling noise; only the
    # fit key (restart) and RQS_knots (capacity) vary below.
    data = fam.generate(args.n, causal_params=cp, seed=args.data_seed)
    X, Y = data["X"], data["Y"]
    Z_disc, Z_cont = data["Z_disc"], data["Z_cont"]
    uz = causl_py.generate_uz_samples(Z_disc, Z_cont, False, args.data_seed, {})
    u_z = uz["uz_samples"]
    print(f"[{args.dgp}] data X{tuple(np.asarray(X).shape)} Y{tuple(np.asarray(Y).shape)} "
          f"u_z{tuple(np.asarray(u_z).shape)}", flush=True)

    # summaries[knots] -> dict(ates=[...], val_bests=[...], qte_errs=[...], tau_sds=[...])
    summaries = {}

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        w.writeheader(); fh.flush()

        for knots in knots_list:
            cell = {"ates": [], "val_bests": [], "qte_errs": [], "tau_sds": []}

            for r in range(args.restarts):
                fit_key = jr.key(80000 + 1000 * r)  # data fixed; ONLY this varies
                t0 = time.time()
                row = {k: "" for k in FIELDNAMES}
                row.update(dgp=args.dgp, RQS_knots=knots, nn_depth=NN_DEPTH,
                           nn_width=NN_WIDTH, flow_layers=FLOW_LAYERS, restart=r,
                           n=args.n, true_ate=true_ate)
                try:
                    ff, val_loss, val_best = fit_model_capacity(
                        fit_key, Y, u_z, X, CAUSAL_MODEL, knots, NN_DEPTH, NN_WIDTH,
                        FLOW_LAYERS, LEARNING_RATE, EPOCHS, patience, BATCH_SIZE,
                    )
                    m = intervene(jr.fold_in(fit_key, 2), ff, X.shape[1], args.n_mc)
                    ate, tau_sd, n_drop, n_keep = robust_moments(m["y0"], m["y1"])
                    qte_err = qte_integrated_error(fam, cp, m["y0"], m["y1"])

                    row.update(ate=ate, bias=ate - true_ate, tau_sd=tau_sd,
                               qte_int_err=qte_err, val_loss=val_loss,
                               val_best=val_best, n_drop=n_drop)
                    cell["ates"].append(ate)
                    cell["val_bests"].append(val_best)
                    cell["qte_errs"].append(qte_err)
                    cell["tau_sds"].append(tau_sd)
                    if n_drop:
                        print(f"  [drop] knots={knots} r={r}: {n_drop}/{n_drop + n_keep} "
                              f"non-finite MC samples filtered", flush=True)
                except Exception as e:  # noqa: BLE001 -- keep the grid running
                    row.update(error=repr(e))
                row["secs"] = round(time.time() - t0, 1)
                w.writerow(row); fh.flush()
                done += 1
                tag = ("ERR " + row["error"]) if row["error"] else (
                    f"ate={float(row['ate']):+.3f} (true {true_ate:+.3f}) "
                    f"bias={float(row['bias']):+.3f} qte_int_err={float(row['qte_int_err']):.3f} "
                    f"val_loss={float(row['val_loss']):.4f}")
                print(f"[{done}/{total}] knots={knots} r={r} {tag} ({row['secs']}s)", flush=True)

            summaries[knots] = cell

    print(f"[spline_capacity_gamma] DONE {done}/{total}. csv={args.out}", flush=True)

    # ---- summary table: per-knots restart-mean/-sd + capacity trend ----
    print(f"\n=== capacity grid summary (dgp={args.dgp}, n={args.n}, true_ate={true_ate:+.4f}) ===")
    hdr = (f"{'knots':>6}{'R':>4}{'mean_ate':>10}{'sd_ate':>9}{'min':>9}{'max':>9}"
           f"{'mean_qte':>11}{'mean_tau_sd':>13}{'corr(val,ate)':>15}")
    print(hdr); print("-" * len(hdr))
    trend_rows = []
    for knots in knots_list:
        cell = summaries.get(knots, {"ates": [], "val_bests": [], "qte_errs": [], "tau_sds": []})
        ates = cell["ates"]
        R = len(ates)
        if R == 0:
            print(f"{knots:>6}{0:>4}{'--':>10}{'--':>9}{'--':>9}{'--':>9}"
                  f"{'--':>11}{'--':>13}{'--':>15}  (all restarts failed)")
            continue
        ates_arr = np.asarray(ates)
        qte_arr = np.asarray(cell["qte_errs"])
        tau_sd_arr = np.asarray(cell["tau_sds"])
        corr = _pearsonr_safe(cell["val_bests"], ates)
        mean_ate, sd_ate = ates_arr.mean(), ates_arr.std()
        mean_qte, mean_tau_sd = qte_arr.mean(), tau_sd_arr.mean()
        trend_rows.append((knots, mean_ate, sd_ate, mean_qte, mean_tau_sd))
        print(f"{knots:>6}{R:>4}{mean_ate:>10.3f}{sd_ate:>9.3f}{ates_arr.min():>9.3f}"
              f"{ates_arr.max():>9.3f}{mean_qte:>11.3f}{mean_tau_sd:>13.3f}{corr:>15.3f}")

    if len(trend_rows) >= 2:
        first_k, _, first_sd, first_qte, _ = trend_rows[0]
        last_k, _, last_sd, last_qte, _ = trend_rows[-1]
        sd_dir = "shrinks" if last_sd < first_sd else ("grows" if last_sd > first_sd else "is flat")
        qte_dir = ("improves" if last_qte < first_qte else
                   ("worsens" if last_qte > first_qte else "is flat"))
        print(f"\nheadline: restart-level noise sd(ate) {sd_dir} from knots={first_k} "
              f"({first_sd:.3f}) to knots={last_k} ({last_sd:.3f}); mean shape error "
              f"(qte_int_err) {qte_dir} from {first_qte:.3f} to {last_qte:.3f}.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dgp", default="gamma_b1")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--knots", default="4,8,16", help="comma-separated RQS_knots sweep values")
    p.add_argument("--restarts", type=int, default=5)
    p.add_argument("--n-mc", type=int, default=20000)
    p.add_argument("--const", type=float, default=1.0, help="causal_params[0] (b0)")
    p.add_argument("--ate", type=float, default=0.5, help="causal_params[1] (b1)")
    p.add_argument("--out", required=True, help="per-fit CSV path")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
