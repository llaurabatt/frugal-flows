"""Pre-registered hyperparameter battery for the SPLINE (`flexible_continuous`)
causal margin: does (a) transforming Y before the flow, or (b) a cosine LR
schedule, reduce the restart-to-restart ATE level noise found at n=2000 by
`spline_stability.py`?

Motivation: `spline_stability.py` showed the spline arm's ATE readout swings
several tenths restart-to-restart at n=2000 under a FIXED dataset (only the
fit key varies), consistent with a flat / multi-basin optimisation landscape.
This script asks whether that noise is a scale/step-size artefact rather than
an intrinsic property of the loss surface: does putting Y on a friendlier
numerical scale before the autoregressive spline (log or standardize), or
decaying the learning rate (cosine vs const), tighten the restart spread of
the ATE?

DESIGN (pre-registered — see module docstring in the launching task; implement
exactly, do not add/drop cells after seeing results):
  - ONE dataset: gamma_b1, n=2000, data seed 0, causal_params=[1.0, 0.5].
    Generated ONCE; u_z generated ONCE from Z alone (unaffected by Y transforms).
  - Grid: warm_start in {cold, warm} x transform in {raw, log, standardize}
    x lr_schedule in {const, cosine} x restart in 0..5
    (6 fit keys jr.key(70000 + 1000*(restart_base+r))) = 72 fits.
    warm cells additionally pretrain the causal margin alone (H2) then graft it
    into the joint fit; cold cells pass pretrained_margin=None (the pre-existing path).
    Restrict axes via --warm-starts / --transforms / --lr-schedules; shard by
    disjoint --restart-base.
  - Architecture fixed at spline defaults: RQS_knots=8, nn_depth=4, nn_width=50,
    flow_layers=4, batch_size=256, epochs=600, patience=60 (max(20, epochs//10)).
  - Transforms applied to Y BEFORE fitting; inverted on the do(0)/do(1) SAMPLES
    after `intervene`, so every downstream metric (ate, tau_sd, qte_int_err) is
    on the ORIGINAL Y scale, comparable across transforms:
      raw:        Z = Y                                inverse: y = z
      log:        Z = log(Y - b)  (Y > b asserted)      inverse: y = exp(z) + b
      asinh:      Z = asinh((Y - b) / s)                inverse: y = sinh(z)*s + b
      standardize Z = (Y - mean_train) / sd_train       inverse: y = z*sd + mean
        b = --floor (the artificial lower bound / "bottom skew min value"; default
        0, so log -> plain log(Y), asinh -> asinh(Y/s)). b cancels in any contrast
        (ATE, tau(u)) but anchors where the skew compression starts; use it when the
        outcome is floored at a non-zero value. asinh is the signed / zero-safe log:
        defined for all Y (no positivity requirement), linear near b, log-like in the
        tail, with crossover set by s = --asinh-scale (default robust median(|Y-b|)).
        mean/sd (standardize) and s (asinh default) computed on the TRAINING data,
        i.e. the one dataset generated once for the whole battery.
  - LR schedules:
      const:  learning_rate = 5e-3 flat (the existing spline baseline).
      cosine: optax.cosine_decay_schedule(init_value=1e-2, decay_steps=T, alpha=1e-2)
        i.e. 1e-2 decaying to 1e-4 over T optimizer steps. `optax.adam`'s
        `learning_rate` arg accepts a Callable schedule directly (verified against
        the installed optax==0.2.8 signature and flowjax.train.fit_to_data, which
        does `optimizer = optax.adam(learning_rate)` when no optimizer is passed) —
        train_frugal_flow forwards `learning_rate` straight through, so no fallback
        ladder was needed.
        T = epochs * steps_per_epoch, with steps_per_epoch = floor(train_n /
        batch_size) and train_n = int(n * (1 - val_prop)), val_prop=0.1 (both read
        from flowjax.train.loops.fit_to_data / train_val_split / get_batches: the
        val split is taken first, then each epoch batches the train split with the
        LAST PARTIAL BATCH DROPPED). This is the exact step count fit_to_data will
        run per epoch, not the approximation the task allowed (ceil(0.9n/batch));
        computed exactly here since it was cheap to read off.

  Y is Gamma so Y>0 always (asserted); log transform's exp(z) inverse can
  overflow for boundary spline artefacts (base draw at the +/-1 tanh boundary
  maps through atanh to +/-inf, then through log-inverse to inf). This is a
  known numerical-boundary artefact (see `h1_matrix.robust_moments` docstring),
  not a modelling failure: filter it via `robust_moments`'s finite-pair filter
  AFTER inverting to the original scale, and report n_drop per fit (never silent).

READOUTS per fit (CSV row, one per (warm_start, transform, lr_schedule, restart)):
  warm_start, transform, lr_schedule, restart, n, true_ate, ate, bias, tau_sd,
  qte_int_err, val_loss, val_best, pre_val_loss, graft_ok, n_drop, tau_curve,
  secs, error.
  pre_val_loss = final val loss of the warm-start margin pretrain ("" when cold);
  graft_ok = the in-package graft assert passed ("" when cold); tau_curve = the
  ";"-joined 40-bin tau(u) on the ORIGINAL Y scale (for restart-averaging + overlay).
  val_loss/val_best/pre_val_loss are log-likelihoods of Z (the transformed Y), NOT
  comparable across transforms (different data scale -> different Jacobian term) —
  do not use val_loss to compare cells; adjudicate on the Y-scale metrics only.

SUMMARY per (transform, lr_schedule) cell: k, mean bias, sd bias (PRIMARY
metric for the restart-noise question), min/max ate, mean qte_int_err, total
n_drop. Then a pre-registered adjudication vs control = (raw, const):
  WIN iff sd(ate) < control_sd(ate)  AND  |mean_bias| <= |control_mean_bias| + 0.05.
Pre-registered predictions (recorded before running, for the record only):
  "log expected to cut both |bias| and sd + zero boundary drops; standardize
   weaker; cosine modest sd reduction."

Usage (from validation/, in the frugal-flows-flowjax env):
  micromamba run -n frugal-flows-flowjax python -m diagnostics.spline_hp_battery \
      --out outputs/flexible_te/spline_hp_battery.csv
  # tiny probe (verify schedule + inverse-transform plumbing before the real run):
  micromamba run -n frugal-flows-flowjax python -m diagnostics.spline_hp_battery \
      --transforms log --lr-schedules cosine --restarts 1 --n 200 --epochs 50 \
      --out /tmp/probe.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)  # must precede any jnp use

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from flowjax.bijections import Invert, Tanh  # noqa: E402
from flowjax.distributions import Transformed, Uniform  # noqa: E402
from flowjax.train import fit_to_data  # noqa: E402

_VALIDATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VALIDATION_DIR not in sys.path:
    sys.path.insert(0, _VALIDATION_DIR)

import data_processing_and_simulations.causl_sim_data_generation as causl_py  # noqa: E402
from diagnostics.ate_extraction_suite import intervene, tau_curve  # noqa: E402
from diagnostics.h1_matrix import qte_integrated_error, robust_moments  # noqa: E402
from diagnostics.outcome_families import FAMILIES  # noqa: E402
from diagnostics.quick_sense_check import final_val_loss, model_args  # noqa: E402
from frugal_flows.basic_flows import masked_autoregressive_bijection  # noqa: E402
from frugal_flows.causal_flows import train_frugal_flow  # noqa: E402

CAUSAL_MODEL = "flexible_continuous"
DGP = "gamma_b1"
DATA_SEED = 0
CAUSAL_PARAMS = [1.0, 0.5]
N = 2000
EPOCHS = 600
PATIENCE = max(20, EPOCHS // 10)  # 60, same rule base_hyperparams uses
BATCH_SIZE = 256
VAL_PROP = 0.1  # flowjax.train.fit_to_data default; must match to size the schedule
CONST_LR = 5e-3
COSINE_INIT = 1e-2
COSINE_ALPHA = 1e-2  # decays to init*alpha = 1e-4

TRANSFORMS = ["raw", "log", "asinh", "standardize"]
LR_SCHEDULES = ["const", "cosine"]
WARM_STARTS = ["cold", "warm"]

FIELDNAMES = ["warm_start", "transform", "lr_schedule", "restart", "n", "true_ate", "ate", "bias",
              "tau_sd", "qte_int_err", "val_loss", "val_best", "pre_val_loss", "graft_ok",
              "n_drop", "tau_curve", "secs", "error"]


def steps_per_epoch(n: int, batch_size: int, val_prop: float = VAL_PROP) -> int:
    """Exact optimizer-step count fit_to_data will run in one epoch.

    Matches flowjax.train.train_utils.train_val_split (train gets the remainder
    after an int() split) + get_batches (last partial batch dropped).
    """
    train_n = n - int(n * val_prop)
    return train_n // batch_size


def make_optimizer(lr_schedule: str, epochs: int, n: int, batch_size: int):
    """Build the (learning_rate_or_schedule) to pass through to train_frugal_flow.

    train_frugal_flow -> train_frugal_flow_flexible_continuous -> fit_to_data,
    which does `optimizer = optax.adam(learning_rate)` when no optimizer is
    passed; optax.adam's learning_rate accepts either a float or a Callable
    schedule, so a schedule can be handed straight through as `learning_rate`.
    """
    if lr_schedule == "const":
        return CONST_LR
    if lr_schedule == "cosine":
        spe = steps_per_epoch(n, batch_size)
        total_steps = max(1, epochs * spe)
        return optax.cosine_decay_schedule(
            init_value=COSINE_INIT, decay_steps=total_steps, alpha=COSINE_ALPHA,
        )
    raise ValueError(f"unknown lr_schedule {lr_schedule!r}")


# ---- Y transforms: applied pre-fit, inverted post-sample -------------------
def fit_transform(transform: str, Y: np.ndarray, floor: float = 0.0, asinh_scale: float | None = None):
    """Return (Z, inverse_fn) where Z is Y transformed for fitting and
    inverse_fn maps samples on the Z scale back to the original Y scale.

    `floor` (b) is the artificial lower bound of the outcome — the "bottom skew
    min value". Both skew-compressing transforms subtract it so the compression
    is applied to the excess-over-floor Y - b, not the raw level:
      raw:         Z = Y                                inverse: y = z
      log:         Z = log(Y - b)  (Y > b asserted)      inverse: y = exp(z) + b
      asinh:       Z = asinh((Y - b) / s)                inverse: y = sinh(z)*s + b
      standardize: Z = (Y - mean) / sd                   inverse: y = z*sd + mean

    b defaults to 0 (log -> plain log(Y); asinh -> asinh(Y/s)). b cancels in any
    contrast (ATE, tau(u)) but sets where the skew compression is anchored.
    `asinh_scale` (s) sets asinh's linear->log crossover; None -> a robust
    data-driven default, median(|Y - b|) (deterministic on the fixed dataset).
    mean/sd for `standardize` are computed on THIS Y and closed over by inverse_fn.
    """
    Y = np.asarray(Y, dtype=float)
    b = float(floor)
    if transform == "raw":
        return Y, (lambda z: z)
    if transform == "log":
        assert np.min(Y) > b, f"log transform requires Y > floor (b={b}); min(Y)={np.min(Y):.4g}"
        return np.log(Y - b), (lambda z, _b=b: np.exp(z) + _b)
    if transform == "asinh":
        s = float(asinh_scale) if asinh_scale is not None else float(np.median(np.abs(Y - b)))
        if not s > 0:  # degenerate (e.g. all Y == b); fall back to a unit scale
            s = 1.0
        return np.arcsinh((Y - b) / s), (lambda z, _b=b, _s=s: np.sinh(z) * _s + _b)
    if transform == "standardize":
        mean, sd = float(Y.mean()), float(Y.std())
        return (Y - mean) / sd, (lambda z, _m=mean, _s=sd: z * _s + _m)
    raise ValueError(f"unknown transform {transform!r}")


def pretrain_margin(key, Z, X, learning_rate, epochs, patience):
    """Warm-start (H2): fit the causal margin ALONE on (Z, X), return it for grafting.

    Isolates the dim-0 path of the full flexible_continuous flow --
    ``Uniform[-1,1] -> causal_maf(RQS|T) -> atanh -> Y`` -- and fits it by maximum
    likelihood on the (transformed) outcome Z conditioned on treatment X. Because the
    margin term moment-matches by construction, this lands the ATE *level* at the
    identified point before the copula is introduced (attacks the margin/copula level
    non-identifiability). The returned bijection is built by the SAME
    ``masked_autoregressive_bijection`` call (identical hyperparameters) the full flow
    uses, so its pytree matches the graft site ``bijections[-2].bijections[0]`` exactly
    (verified by the in-package assert in train_frugal_flow_flexible_continuous).

    Returns (pretrained_margin_bijection, pre_val_loss).
    """
    cargs = model_args(CAUSAL_MODEL, X.shape[1])  # nn_depth/nn_width/RQS_knots/flow_layers
    key, subkey = jr.split(key)
    causal_maf = masked_autoregressive_bijection(
        key=subkey, dim=1, condition=X,
        nn_depth=cargs["nn_depth"], nn_width=cargs["nn_width"],
        RQS_knots=cargs["RQS_knots"], flow_layers=cargs["flow_layers"],
    )
    # base Uniform[-1,1] -> causal_maf ([-1,1]) -> Invert(Tanh)=atanh -> Y (Z scale)
    margin_flow = Transformed(Uniform(-jnp.ones(1), jnp.ones(1)), causal_maf)
    margin_flow = Transformed(margin_flow, Invert(Tanh((1,))))
    margin_flow = margin_flow.merge_transforms()
    key, subkey = jr.split(key)
    trained, losses = fit_to_data(
        key=subkey, dist=margin_flow, data=(Z, X),
        learning_rate=learning_rate, max_epochs=epochs,
        max_patience=patience, batch_size=BATCH_SIZE, show_progress=False,
    )
    # Extract the causal_maf by its cond_shape, NOT by position: flowjax's
    # Uniform(-1,1) is itself Transformed(_StandardUniform, NonTrainable(Affine)),
    # so after merge_transforms() the chain is [Affine, causal_maf, Invert(Tanh)]
    # and bijections[0] is the base's UNCONDITIONAL Affine -- grafting that
    # produces ate=0.0 exactly and ~50% non-finite samples (root cause of the
    # first warm-start smoke failure). Exactly one element is conditional.
    conditional = [b for b in trained.bijection.bijections if b.cond_shape is not None]
    assert len(conditional) == 1 and conditional[0].cond_shape == (X.shape[1],), (
        f"expected exactly one conditional bijection with cond_shape ({X.shape[1]},); "
        f"got {[b.cond_shape for b in trained.bijection.bijections]}"
    )
    pretrained = conditional[0]
    return pretrained, final_val_loss(losses)


def fit_model_override(key, Z, u_z, X, learning_rate, epochs, patience, pretrained_margin=None):
    """Fit the spline frugal flow on (possibly transformed) Y=Z.

    Mirrors spline_stability.fit_model_override exactly (same fixed
    architecture knobs), but `learning_rate` may be a float or an optax
    schedule (Callable[[step], float]), and `pretrained_margin` (if given) is
    grafted into the causal-margin slot for warm-start fine-tuning.
    """
    cond_dim = X.shape[1]
    ff, losses = train_frugal_flow(
        key, Z, u_z, condition=X,
        causal_model=CAUSAL_MODEL,
        causal_model_args=model_args(CAUSAL_MODEL, cond_dim),
        RQS_knots=8,
        nn_depth=4,
        nn_width=50,
        flow_layers=4,
        learning_rate=learning_rate,
        max_epochs=epochs,
        max_patience=patience,
        batch_size=BATCH_SIZE,
        show_progress=False,
        pretrained_margin=pretrained_margin,
    )
    val_seq = losses["val"] if isinstance(losses, dict) else losses
    return ff, final_val_loss(losses), float(np.min(np.asarray(val_seq, dtype=float)))


def run(args):
    warm_starts = [w.strip() for w in args.warm_starts.split(",") if w.strip()]
    transforms = [t.strip() for t in args.transforms.split(",") if t.strip()]
    lr_schedules = [s.strip() for s in args.lr_schedules.split(",") if s.strip()]
    for w in warm_starts:
        if w not in WARM_STARTS:
            raise SystemExit(f"unknown warm_start {w!r}; known: {WARM_STARTS}")
    for t in transforms:
        if t not in TRANSFORMS:
            raise SystemExit(f"unknown transform {t!r}; known: {TRANSFORMS}")
    for s in lr_schedules:
        if s not in LR_SCHEDULES:
            raise SystemExit(f"unknown lr_schedule {s!r}; known: {LR_SCHEDULES}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    fam = FAMILIES[DGP]
    true_ate = fam.true_ate(CAUSAL_PARAMS)

    # ONE dataset for the whole battery; u_z from Z alone (unaffected by Y transforms).
    data = fam.generate(args.n, causal_params=CAUSAL_PARAMS, seed=args.data_seed)
    X, Y = data["X"], data["Y"]
    Z_disc, Z_cont = data["Z_disc"], data["Z_cont"]
    uz = causl_py.generate_uz_samples(Z_disc, Z_cont, False, args.data_seed, {})
    u_z = uz["uz_samples"]
    print(f"[spline_hp_battery] dgp={DGP} n={args.n} data_seed={args.data_seed} "
          f"causal_params={CAUSAL_PARAMS} true_ate={true_ate:+.4f}", flush=True)
    print(f"[spline_hp_battery] X{tuple(np.asarray(X).shape)} Y{tuple(np.asarray(Y).shape)} "
          f"u_z{tuple(np.asarray(u_z).shape)}  Y range [{np.min(Y):.3f}, {np.max(Y):.3f}]", flush=True)
    spe = steps_per_epoch(args.n, BATCH_SIZE)
    print(f"[spline_hp_battery] epochs={args.epochs} patience={_patience(args.epochs)} "
          f"steps_per_epoch={spe} cosine_total_steps={args.epochs * spe}", flush=True)

    total = len(warm_starts) * len(transforms) * len(lr_schedules) * args.restarts
    done = 0
    # summaries[(warm_start, transform, lr_schedule)] -> dict of lists
    summaries = {}

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        w.writeheader(); fh.flush()

        for warm_start in warm_starts:
            for transform in transforms:
                Z_full, inverse_fn = fit_transform(transform, Y, args.floor, args.asinh_scale)

                for lr_schedule in lr_schedules:
                    learning_rate = make_optimizer(lr_schedule, args.epochs, args.n, BATCH_SIZE)
                    patience = _patience(args.epochs)
                    pre_patience = _patience(args.pretrain_epochs)
                    cell = dict(ate=[], bias=[], qte=[], n_drop=0)

                    for r in range(args.restarts):
                        # --restart-base offsets the key so disjoint shards use disjoint
                        # restarts (repo shard-by-disjoint-restarts convention).
                        r_global = args.restart_base + r
                        fit_key = jr.key(70000 + 1000 * r_global)
                        t0 = time.time()
                        row = {k: "" for k in FIELDNAMES}
                        row.update(warm_start=warm_start, transform=transform,
                                   lr_schedule=lr_schedule, restart=r_global,
                                   n=args.n, true_ate=true_ate)
                        try:
                            pretrained = None
                            if warm_start == "warm":
                                # pretrain the causal margin alone on the SAME transformed
                                # target Z_full; a distinct key branch (fold_in 7) from the
                                # joint fit (fit_key) and readout (fold_in 2).
                                pretrained, pre_vl = pretrain_margin(
                                    jr.fold_in(fit_key, 7), Z_full, X,
                                    learning_rate, args.pretrain_epochs, pre_patience)
                                row["pre_val_loss"] = pre_vl
                            ff, val_loss, val_best = fit_model_override(
                                fit_key, Z_full, u_z, X, learning_rate, args.epochs, patience,
                                pretrained_margin=pretrained,
                            )
                            if warm_start == "warm":
                                # graft_ok=True means the in-package PRE-graft asserts
                                # (cond_shape match + array-pytree structure match against
                                # the causal-margin slot) passed -- a wrong-object graft
                                # raises inside train_frugal_flow and lands in `error`.
                                row["graft_ok"] = True
                            m = intervene(jr.fold_in(fit_key, 2), ff, X.shape[1], args.n_mc)
                            # invert samples to the ORIGINAL Y scale BEFORE any metric.
                            y0 = inverse_fn(m["y0"])
                            y1 = inverse_fn(m["y1"])
                            ate, tau_sd, n_drop, n_keep = robust_moments(y0, y1)
                            qte_err = qte_integrated_error(fam, CAUSAL_PARAMS, y0, y1)
                            # Persist the per-restart tau(u) curve for restart-averaging (H3b)
                            # and the tau overlay plot. Filter to the SAME finite pairs
                            # robust_moments keeps, so the curve is on the original Y scale
                            # and free of atanh-boundary +/-inf artefacts.
                            finite = np.isfinite(y0) & np.isfinite(y1)
                            _, tau_of_u = tau_curve(y0[finite], y1[finite])
                            tau_curve_str = ";".join(f"{v:.6g}" for v in tau_of_u)

                            row.update(ate=ate, bias=ate - true_ate, tau_sd=tau_sd,
                                       qte_int_err=qte_err, val_loss=val_loss,
                                       val_best=val_best, n_drop=n_drop,
                                       tau_curve=tau_curve_str)
                            cell["ate"].append(ate); cell["bias"].append(ate - true_ate)
                            cell["qte"].append(qte_err); cell["n_drop"] += n_drop
                            if n_drop:
                                print(f"  [drop] {warm_start}/{transform}/{lr_schedule} r={r_global}: "
                                      f"{n_drop}/{n_drop + n_keep} non-finite MC pairs filtered "
                                      f"(post inverse-transform)", flush=True)
                        except Exception as e:  # noqa: BLE001 -- keep the sweep running
                            row.update(error=repr(e))
                        row["secs"] = round(time.time() - t0, 1)
                        w.writerow(row); fh.flush()
                        done += 1
                        tag = ("ERR " + row["error"]) if row["error"] else (
                            f"ate={float(row['ate']):+.3f} (true {true_ate:+.3f}) "
                            f"bias={float(row['bias']):+.3f} val_loss={float(row['val_loss']):.4f} "
                            f"n_drop={row['n_drop']}")
                        print(f"[{done}/{total}] {warm_start}/{transform}/{lr_schedule} "
                              f"r={r_global} {tag} ({row['secs']}s)", flush=True)

                    summaries[(warm_start, transform, lr_schedule)] = cell

    print(f"[spline_hp_battery] DONE {done}/{total}. csv={args.out}", flush=True)

    # ---- summary table ----
    # NB: this is a quick per-cell readout only. The CANONICAL cross-shard adjudication
    # (2*sd/sqrt(K) interval rule, control = cold/raw/const) is done by
    # diagnostics.spline_hp_findings_plot over the sharded CSVs.
    print("\n=== spline HP battery summary (n={}, dgp={}) ===".format(args.n, DGP))
    print("NOTE: val_loss/val_best/pre_val_loss are NOT comparable across transforms — they "
          "are log-likelihoods of the TRANSFORMED Y (different data scale => different "
          "Jacobian term in the density), so do not rank cells by val_loss.")
    hdr = (f"{'warm':<6}{'transform':<12}{'lr_sched':<9}{'k':>3}{'mean_bias':>11}"
           f"{'2sd/sqrtK':>11}{'unbiased':>10}{'sd_ate':>9}{'mean_qte':>10}{'tot_ndrop':>10}")
    print(hdr); print("-" * len(hdr))
    for warm_start in warm_starts:
        for transform in transforms:
            for lr_schedule in lr_schedules:
                key = (warm_start, transform, lr_schedule)
                cell = summaries.get(key, dict(ate=[], bias=[], qte=[], n_drop=0))
                k = len(cell["ate"])
                if k == 0:
                    print(f"{warm_start:<6}{transform:<12}{lr_schedule:<9}{0:>3}{'--':>11}"
                          f"{'--':>11}{'--':>10}{'--':>9}{'--':>10}{cell['n_drop']:>10}  (all failed)")
                    continue
                ate_arr = np.asarray(cell["ate"])
                mean_bias = float(np.asarray(cell["bias"]).mean())
                sd_ate = float(ate_arr.std())
                half_ci = 2.0 * sd_ate / np.sqrt(k)
                unbiased = "yes" if abs(mean_bias) <= half_ci else "NO"
                mean_qte = float(np.mean(cell["qte"])) if cell["qte"] else float("nan")
                print(f"{warm_start:<6}{transform:<12}{lr_schedule:<9}{k:>3}{mean_bias:>+11.4f}"
                      f"{half_ci:>11.4f}{unbiased:>10}{sd_ate:>9.4f}{mean_qte:>10.4f}"
                      f"{cell['n_drop']:>10}")

    print("\nRun `python -m diagnostics.spline_hp_findings_plot` over the shard CSVs for the "
          "full per-hypothesis adjudication + figures.")
    print("pre-registered predictions (for reference only): log cuts |bias|+sd & zeroes "
          "boundary drops; warm-start collapses sd_ate; cosine modest sd reduction.")


def _patience(epochs: int) -> int:
    return max(20, epochs // 10)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--warm-starts", default=",".join(WARM_STARTS),
                   help="comma list of {cold,warm}; warm pretrains the causal margin then grafts it")
    p.add_argument("--transforms", default=",".join(TRANSFORMS))
    p.add_argument("--floor", type=float, default=0.0,
                   help="artificial lower bound b subtracted before log/asinh (log uses log(Y-b), asinh uses asinh((Y-b)/s)); default 0")
    p.add_argument("--asinh-scale", type=float, default=None,
                   help="asinh linear->log crossover scale s; default None => robust median(|Y-b|)")
    p.add_argument("--lr-schedules", default=",".join(LR_SCHEDULES))
    p.add_argument("--restarts", type=int, default=6)
    p.add_argument("--restart-base", type=int, default=0,
                   help="offset for disjoint-shard restart keys: fit_key = jr.key(70000 + 1000*(restart_base+r))")
    p.add_argument("--n", type=int, default=N)
    p.add_argument("--data-seed", type=int, default=DATA_SEED)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--pretrain-epochs", type=int, default=None,
                   help="epochs for the warm-start margin pretrain (default: same as --epochs)")
    p.add_argument("--n-mc", type=int, default=20000)
    p.add_argument("--out", required=True, help="per-fit CSV path")
    args = p.parse_args()
    if args.pretrain_epochs is None:
        args.pretrain_epochs = args.epochs
    run(args)


if __name__ == "__main__":
    main()
