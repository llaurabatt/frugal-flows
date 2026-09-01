"""ATE-map recovery across the MorphoMNIST experiment ladder.

Fits a multivariate frugal flow to a dataset from ``prepare_morphomnist_exps``,
recovers the per-pixel ``tau_hat``, and scores it against a truth that is EXACT
by construction (see that module: the imposed ATE is the spatial map itself, not
a Monte-Carlo average).

This is the general-purpose successor to ``toy_ate_recovery.py``. That script
stays as-is: it is the frozen reproducer for the runs already archived under
``runs/toy_ate_recovery/``, and its ``--replot`` reads their ``config.json``
back into its own ``Config``, so its knobs must not move. Its DGP is exactly
this script's ``exp1_rct_homogeneous`` preset.

The experiment (``--preset``)
-----------------------------
    exp1_rct_homogeneous            randomised assignment, one fixed effect map
    exp2_confounded_homogeneous     same effect, assignment driven by thickness
    exp3_confounded_heterogeneous   same assignment, effect varies with the
                                    covariate AND the outcome quantile

Any generator knob can be overridden on top of a preset (``--a-cov``,
``--b-quant``, ``--ps-slope``, ``--base-shift``, ...), so the presets are
starting points rather than a closed set.

The estimator (``--arm``, ``--conditioner``)
--------------------------------------------
    location_translation   treatment is MASKED from the margin flow, so the
                           whole effect rides on the per-pixel ``LocCond``
                           shift. ``tau_hat`` is an exact model PARAMETER.
                           Correctly specified for E1/E2; MISSPECIFIED for E3,
                           whose quantile-varying component is not a pure shift.

    flexible_continuous    treatment enters the K-dimensional spline margin
                           itself. ``tau_hat`` is ESTIMATED by paired
                           common-random-number interventional sampling, and
                           the quantile-resolved ``tau(u)`` comes with it.
                           Correctly specified throughout. Runs on either
                           conditioner engine via ``--conditioner``:

                               mlp          MADE-masked MLP (default)
                               transformer  causal-transformer conditioner
                                            (TarFlow-style; requires
                                            ``--nn-width`` divisible by
                                            ``--nn-heads``)

Outcome dimensionality
----------------------
``--size s`` sets the image side, so ``K = s^2`` outcome dimensions: ``--size 4``
(K=16) for a fast smoke test, ``--size 8`` (K=64) for the default sweep,
``--size 16`` (K=256) for the full-resolution run. ``--radius`` defaults to
``round(s/4)``, which holds the effect map at a constant ~20% of pixels as ``s``
changes; pass it explicitly to break that coupling.

Mind the cost curve on the transformer conditioner: its ``log_prob`` (training)
is one parallel pass, but SAMPLING solves one coordinate per ``lax.scan`` step
and runs a full attention pass over all K tokens at each, so the
``flexible_continuous`` read-out scales far worse in K than the MLP's. Probe
with a small ``--n-mc`` before committing to K=256.

Run folders
-----------
Every run writes a self-contained folder under ``runs/exp_ate_recovery/``:

    <UTC-stamp>_<preset-tag>_<arm-tag>_s<seed_fit>_k<K>_<suffix>/
        config.json    run_id + every knob + git commit/dirty + library versions
        log.txt        live training output (`tail -f` it)
        metrics.json   recovery scores vs ATE, ATT and ATC + timings
        arrays.npz     tau_hat, truth, tau(u) curves, losses (replottable)
        plots/         every figure as PNG

config.json and log.txt appear at launch, so a folder holding only those two
means the run is still training.

Quick start
-----------
Run everything from ``validation/morphomnist/``.

    # 0. does the whole pipeline still work? (~2 min, writes nothing permanent)
    python exp_ate_recovery.py --selftest

    # 1. one cell, default knobs
    python exp_ate_recovery.py --preset exp2_confounded_homogeneous

    # 2. the transformer conditioner (note the width: 50 is NOT divisible by 4)
    python exp_ate_recovery.py --preset exp3_confounded_heterogeneous \\
        --arm flexible_continuous --conditioner transformer --nn-width 48

    # 3. the whole 9-cell matrix
    python exp_ate_recovery.py --sweep --size 8

    # 4. a fast end-to-end cycle for debugging (recovers nothing; proves plumbing)
    python exp_ate_recovery.py --size 4 --n 400 --max-epochs 3 \\
        --marginal-max-epochs 3 --n-mc 200

    # 5. look at what you have
    python exp_ate_recovery.py --collect
    python exp_ate_recovery.py --replot runs/exp_ate_recovery/<run-id>


The five modes
--------------
``(default)``   Build one dataset, fit one flow, score it, archive the run.
``--sweep``     Every (preset, arm, conditioner) cell in turn: 3 presets x 3
                estimator combinations = 9 runs. The conditioner only applies
                to ``flexible_continuous``, so ``location_translation``
                contributes one cell per preset, not two. A cell that fails is
                logged and skipped so the rest of the matrix still completes.
``--collect``   Print one row per completed run in the archive. No fitting.
``--replot``    Rebuild every plot for an existing run from its ``arrays.npz``.
                No refit. Use after changing a plotting function.
``--selftest``  Run the pipeline end to end at tiny scale and assert the
                invariants that must hold (see "What the self-test checks").
                Uses a temporary run directory and removes it afterwards.


Flags: which experiment to generate
-----------------------------------
``--preset``        exp1_rct_homogeneous (default) | exp2_confounded_homogeneous
                    | exp3_confounded_heterogeneous
``--a-cov``         Covariate heterogeneity: how much the effect varies with
                    thickness. 0 in E1/E2, 0.5 in E3.
``--b-quant``       Quantile heterogeneity: how much the effect varies with the
                    outcome's own latent quantile. 0 in E1/E2, 0.4 in E3.
                    Keep ``a_cov + b_quant < 1``, or some units' effects flip
                    sign relative to the map -- legitimate, but a different DGP.
``--ps-slope``      Confounding strength in the logistic propensity. 0 = RCT
                    (E1), 1.2 in E2/E3.
``--ps-intercept``  Shifts the treated fraction. 0 (default) gives ~50/50.
``--base-shift``    Effect magnitude in logit space. Default 1.0.
``--effect``        Shape of the effect map: circle (default) | ring | const |
                    gradient. ``gradient`` is the hardest -- it has no flat
                    regions and no exact zeros for the estimator to lock onto.
``--seed-data``     RNG for the subset, the dequantisation noise and treatment
                    assignment. Change it for a different dataset draw.

The generator flags OVERRIDE the preset rather than replacing it, so
``--preset exp3_confounded_heterogeneous --b-quant 0`` gives E3 with only
covariate heterogeneity -- useful for isolating which kind of heterogeneity
breaks a given arm.


Flags: outcome dimensionality and sample
----------------------------------------
``--size``          Image side; K = size^2 outcome dimensions. 4 -> K=16 (fast
                    debugging), 8 -> K=64 (default), 16 -> K=256 (full res).
``--radius``        Effect-map radius. Defaults to ``round(size/4)``, which
                    holds the map at ~20% of pixels as K changes. Pass it
                    explicitly to break that coupling.
``--digit``         Which digit class to subset. Default 0 -> n=5923 and Z is
                    thickness alone (the discrete stage is bypassed).
``--all-digits``    Use all ten classes instead: n=60000, Z becomes 11-dim with
                    a digit one-hot, and the mixed continuous/discrete stage-one
                    transform gets exercised. A different, harder experiment --
                    not the ladder.
``--n``             Cap the sample size. Default: all of the subset.


Flags: the estimator
--------------------
``--arm``           location_translation (default) | flexible_continuous
``--conditioner``   mlp (default) | transformer. FLEXCONT ONLY -- passing it
                    with ``location_translation`` is an error, because that arm
                    always builds its own margin.
``--nn-heads``      Attention heads, transformer only. Default 4. Must divide
                    ``--nn-width``, and this is checked at launch rather than
                    deep inside the flow build.
``--ate-init``      Starting value for the LocCond shift parameters
                    (``location_translation`` only). ``tau_hat`` is genuinely
                    sensitive to it; it is recorded in config.json so a run
                    folder always documents the full invocation.


Flags: capacity and training
----------------------------
``--rqs-knots`` 8, ``--nn-depth`` 1, ``--nn-width`` 48, ``--flow-layers`` 4,
``--learning-rate`` 1e-2, ``--max-epochs`` 100, ``--max-patience`` 30,
``--batch-size`` 100, ``--seed-fit`` 34, ``--x64/--no-x64`` (default off).

The stage-one covariate margins have their own budget, separate from the
frugal flow's: ``--marginal-max-epochs`` 70, ``--marginal-max-patience`` 10.

``--nn-width`` defaults to 48 rather than 50 so it is divisible by
``--nn-heads``: the sweep then compares the mlp and transformer conditioners at
IDENTICAL capacity, instead of silently bumping the width for one of them.

THE LEARNING RATE IS NOT NEUTRAL BETWEEN ARMS. ``location_translation`` carries
an explicit additive parameter that has to travel from ``--ate-init`` to the
true effect, and it under-trains badly at small rates -- its shift parameters
stay bunched near their initial value. The flexible arms are far less sensitive,
and the transformer can diverge outright at larger rates. Tune per arm and say
so; a single shared rate quietly advantages whichever arm it happens to suit.


Flags: the interventional read-out (flexible_continuous only)
-------------------------------------------------------------
``--n-mc``          Paired common-random-number draws per treatment arm.
                    Default 5000.
``--seed-mc``       Read-out RNG, deliberately separate from the fit RNG.

A spline margin occasionally throws a draw into its tails and overflows to
+-inf. A plain mean would let ONE such draw among 5000 poison an entire pixel
(and it did, before this was handled). Non-finite draws are dropped before the
read-out, the discarded fraction is reported as ``mc_frac_dropped``, and the run
aborts only if every draw is bad. Watch that number: a handful is a numerical
artefact of the tails, a large fraction is a real pathology.

The transformer's read-out is much more expensive than the MLP's. Training cost
is comparable -- ``log_prob`` is one parallel pass for both -- but SAMPLING
solves one coordinate per ``lax.scan`` step and runs a full attention pass over
all K tokens at every step, so it scales far worse in K. If transformer cells
dominate your wall time, cut ``--n-mc`` before cutting epochs.


Flags: sweep control
--------------------
``--skip-done``     Skip cells already completed with matching knobs, so an
                    interrupted multi-hour sweep can resume. CAUTION: it keys on
                    the presence of ``metrics.json``, so a run that finished
                    with a non-finite score still counts as done and WILL be
                    skipped. Delete such folders before resuming.


Run folders
-----------
Every run writes a self-contained folder under ``runs/exp_ate_recovery/``:

    <UTC-stamp>_<preset-tag>_<arm-tag>_s<seed_fit>_k<K>_<suffix>/
        config.json    run_id + every knob + git commit/dirty + library versions
        log.txt        live training output (`tail -f` it)
        metrics.json   recovery scores vs ATE, ATT and ATC + timings
        arrays.npz     tau_hat, truth, tau(u) curves, losses (replottable)
        plots/         every figure as PNG

config.json and log.txt appear at launch, so a folder holding only those two
means the run is still training (or died). Only ``arrays.npz`` is gitignored, so
completed run folders can be committed at negligible size.


Reading metrics.json
--------------------
``ate_mae``               The headline: mean |tau_hat - ATE| over all K pixels.
``ate_mae_on_support``    ...restricted to pixels whose true effect is nonzero.
``ate_mae_off_support``   ...restricted to pixels whose true effect is EXACTLY
                          zero.
``ate_corr``              Spatial correlation: is the map in the right PLACE,
                          even if the magnitude is off?
``att_mae`` / ``atc_mae`` Same score against the ATT and the ATC.
``design_oracle_ipw_bias_maxabs``
                          The design's own sampling-noise floor.

READ THE SPLIT, NOT JUST THE TOTAL. A radius-2 disc covers 12 of 64 pixels, so
roughly 80% of the plain MAE's weight sits on pixels whose true effect is
exactly zero. A model that recovers the magnitude perfectly but smears a little
effect everywhere scores worse than one that does neither well. "Recovered the
magnitude" and "kept the zeros at zero" are different failures and the split
separates them.

JUDGE AGAINST THE FLOOR, NOT AGAINST ZERO. ``design_oracle_ipw_bias_maxabs`` is
what inverse-probability weighting by the TRUE propensity achieves -- an oracle
no real estimator has access to. It is the resolution limit at that sample size,
and it is the number recovery should be compared to.

ON E1 AND E2, ``ate_mae``, ``att_mae`` AND ``atc_mae`` ARE IDENTICAL BY
CONSTRUCTION -- the effect is homogeneous there, so all three estimands
coincide. They separate only on E3, where the comparison says WHICH estimand the
fit actually landed on.

``best_val_loss`` IS NOT A MODEL-SELECTION CRITERION HERE. Better density fits
have been observed to recover the ATE worse, by spending their extra capacity on
treatment-dependence in pixels that have none. Score against the truth.


What the self-test checks
-------------------------
``--selftest`` runs the whole path at K=16 with a 2-epoch fit and asserts:

  generator   the imposed ATE equals the effect map exactly; Y is the correct
              factual mixture of Y0/Y1; ITE == Y1 - Y0; ATT/ATC come from the
              right subsets; E1/E2 are homogeneous and E3 is not; E1 is
              unconfounded and E2/E3 are; the tau(u) curves integrate to the
              ATE; E2's curve is flat; generator overrides actually propagate;
              --digit toggles the discrete covariate block
  pipeline    K tracks --size; the radius auto-scales; stage-one u_z is uniform
              and correctly shaped; both arms build the expected bijection chain
              and return a length-K finite tau_hat (not the scalar that
              ``dim_y``'s default would silently produce); metrics carry every
              key the tables rely on
  archive     a run folder round-trips through save -> replot -> collect
  guards      invalid arm/conditioner/nn-width combinations are refused at
              launch rather than deep in the flow build

It is a correctness check, NOT a recovery check: a 2-epoch fit estimates
nothing, and the test asserts nothing about accuracy.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import secrets
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)  # prepare_morphomnist_exps / dataset are siblings

import equinox
import flowjax
import frugal_flows
import jax
import jax.numpy as jnp
import jax.random as jr
import paramax
from frugal_flows.causal_flows import get_independent_quantiles, train_frugal_flow
from frugal_flows.interventions import interventional_samples, tau_curve
from prepare_morphomnist_exps import PRESETS, build_preset, inverse_logit, summarise

ARMS = ("location_translation", "flexible_continuous")
CONDITIONERS = ("mlp", "transformer")
ARM_SHORT = {"location_translation": "loctrans", "flexible_continuous": "flexcont"}
PRESET_SHORT = {
    "exp1_rct_homogeneous": "e1rct",
    "exp2_confounded_homogeneous": "e2conf",
    "exp3_confounded_heterogeneous": "e3het",
}
# Expected bijection-chain length per arm, asserted before the parametric
# read-out so a future change to the chain layout fails loudly instead of
# silently reading the wrong block as the LocCond stack.
ARM_CHAIN_LEN = {"location_translation": 6, "flexible_continuous": 5}

# The sweep matrix. The conditioner is a flexible_continuous option, so
# location_translation contributes one cell per preset, not two.
SWEEP_CELLS = [("location_translation", "mlp"),
               ("flexible_continuous", "mlp"),
               ("flexible_continuous", "transformer")]


@dataclass
class Config:
    # ---- which experiment ----
    preset: str = "exp1_rct_homogeneous"
    # ---- data / outcome dimensionality ----
    size: int = 8              # image side; K = size^2 outcome dims
    radius: int | None = None  # effect-map radius; None -> round(size/4)
    digit: int | None = 0      # single digit class, or None for all ten
    n: int | None = None       # cap on sample size (None = all of the subset)
    seed_data: int = 0         # generator RNG (subset, dequantisation, assignment)
    # ---- generator overrides (None -> keep the preset's value) ----
    base_shift: float | None = None
    a_cov: float | None = None
    b_quant: float | None = None
    ps_slope: float | None = None
    ps_intercept: float | None = None
    effect: str | None = None
    # ---- model / training ----
    arm: str = "location_translation"
    conditioner: str = "mlp"   # flexible_continuous margin engine
    nn_heads: int = 4          # transformer conditioner only; must divide nn_width
    ate_init: float = 0.5      # location_translation only
    rqs_knots: int = 8
    nn_depth: int = 1
    nn_width: int = 48   # divisible by nn_heads, so --sweep compares the mlp and
                         # transformer conditioners at IDENTICAL capacity and the
                         # transformer cells need no special-casing
    flow_layers: int = 4
    learning_rate: float = 1e-2
    max_epochs: int = 100
    max_patience: int = 30
    batch_size: int = 100
    marginal_max_epochs: int = 70
    marginal_max_patience: int = 10
    seed_fit: int = 34
    x64: bool = False
    # ---- interventional read-out (flexible_continuous only) ----
    n_mc: int = 5000
    seed_mc: int = 0

    def __post_init__(self):
        if self.preset not in PRESETS:
            raise ValueError(f"unknown preset {self.preset!r}; choose from {list(PRESETS)}")
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r}; choose from {ARMS}")
        if self.conditioner not in CONDITIONERS:
            raise ValueError(
                f"unknown conditioner {self.conditioner!r}; choose from {CONDITIONERS}"
            )
        if self.conditioner != "mlp" and self.arm != "flexible_continuous":
            raise ValueError(
                "conditioner is a flexible_continuous option; "
                f"arm {self.arm!r} always uses its own margin"
            )
        if self.conditioner == "transformer" and self.nn_width % self.nn_heads:
            raise ValueError(
                f"transformer conditioner needs nn_width ({self.nn_width}) "
                f"divisible by nn_heads ({self.nn_heads})"
            )

    @property
    def effective_radius(self) -> int:
        """Hold the map at ~20% of pixels as K changes, unless overridden."""
        return self.radius if self.radius is not None else max(1, round(self.size / 4))


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
GENERATOR_OVERRIDE_KEYS = ("base_shift", "a_cov", "b_quant", "ps_slope",
                           "ps_intercept", "effect")


def build_data(cfg: Config) -> dict:
    """The chosen preset, with any explicit generator overrides applied."""
    overrides = {"size": cfg.size, "radius": cfg.effective_radius,
                 "digit": cfg.digit, "n": cfg.n, "seed": cfg.seed_data}
    overrides.update({k: getattr(cfg, k) for k in GENERATOR_OVERRIDE_KEYS
                      if getattr(cfg, k) is not None})
    return build_preset(cfg.preset, **overrides)


def fit_flow(cfg: Config, data: dict):
    """Stage-1 marginal quantiles for Z, then the frugal flow.

    Copied rather than imported from ``toy_ate_recovery`` on purpose: that
    version reaches into a toy-specific dict and duck-types its own ``Config``,
    so an import would break silently if either moved. It also has to differ
    here, to carry the discrete digit block when ``--digit`` is unset.
    """
    key = jr.PRNGKey(cfg.seed_fit)
    key, subkey = jr.split(key)

    z_cont = jnp.asarray(data["z_cont"])
    z_discr = np.asarray(data["z_discr"])
    quantile_args = {"z_cont": z_cont, "return_z_cont_flow": True}
    if z_discr.shape[1] > 0:  # all-digits variant; a single-digit subset has none
        quantile_args["z_discr"] = jnp.asarray(z_discr)

    z_res = get_independent_quantiles(
        key=subkey,
        max_epochs=cfg.marginal_max_epochs,
        max_patience=cfg.marginal_max_patience,
        **quantile_args,
    )
    u_z = np.asarray(z_res["u_z_cont"])
    if "u_z_discr" in z_res and z_discr.shape[1] > 0:
        u_z = np.hstack([u_z, np.asarray(z_res["u_z_discr"])])

    causal_model_args = {
        "RQS_knots": cfg.rqs_knots,
        "nn_depth": cfg.nn_depth,
        "nn_width": cfg.nn_width,
        "flow_layers": cfg.flow_layers,
    }
    if cfg.arm == "location_translation":
        causal_model_args["ate"] = cfg.ate_init
    if cfg.arm == "flexible_continuous":
        causal_model_args["conditioner"] = cfg.conditioner
        if cfg.conditioner == "transformer":
            causal_model_args["nn_heads"] = cfg.nn_heads

    key, subkey = jr.split(key)
    flow, losses = train_frugal_flow(
        causal_model=cfg.arm,
        key=subkey,
        y=jnp.asarray(data["Y"]),
        u_z=jnp.asarray(u_z),
        condition=jnp.asarray(data["X"]),
        learning_rate=cfg.learning_rate,
        max_epochs=cfg.max_epochs,
        max_patience=cfg.max_patience,
        batch_size=cfg.batch_size,
        causal_model_args=causal_model_args,
    )
    return flow, losses, u_z


def _tau_hat_location_translation(flow, K: int) -> np.ndarray:
    """Read the per-pixel ATE off the LocCond blocks (an exact parameter)."""
    loccond_block = flow.bijection.bijections[5]
    return np.array(
        [float(paramax.unwrap(loccond_block.bijections[k]).ate) for k in range(K)]
    )


def _tau_hat_flexible_continuous(cfg: Config, flow, data: dict, K: int):
    """Estimate the per-pixel ATE by paired common-random-number draws.

    ``interventional_samples`` needs a TYPED key (``jr.key``), and ``dim_y``
    must be passed explicitly: it defaults to 1, which would silently return
    pixel 0 only.
    """
    t0 = time.monotonic()
    readout = interventional_samples(
        jr.key(cfg.seed_mc),
        flow,
        cond_dim=int(np.asarray(data["X"]).shape[1]),
        n_mc=cfg.n_mc,
        dim_y=K,
    )
    readout_s = time.monotonic() - t0

    # A spline margin can throw the odd draw into its tails and overflow to
    # +-inf. `readout["ate"]` is a plain mean, so ONE such draw among n_mc
    # poisons that pixel's estimate (and `np.mean` propagates it to inf/nan),
    # discarding an otherwise good fit. Drop the offending draws and record how
    # many: a handful is a numerical artefact of the tails, but a large
    # fraction is a real pathology and must not be silently averaged away.
    y0, y1 = np.asarray(readout["y0"]), np.asarray(readout["y1"])
    keep = np.isfinite(y0).all(axis=1) & np.isfinite(y1).all(axis=1)
    frac_dropped = float(1.0 - keep.mean())
    if not keep.any():
        raise RuntimeError(
            f"every one of {cfg.n_mc} interventional draws was non-finite -- "
            "the fitted margin is degenerate, not merely heavy-tailed"
        )
    if frac_dropped:
        print(f"  WARNING: dropped {int((~keep).sum())}/{cfg.n_mc} non-finite draws "
              f"({frac_dropped:.3%}) before the read-out")
    y0, y1 = y0[keep], y1[keep]

    tau_hat = np.mean(y1 - y0, axis=0)
    u_grid, curves = tau_curve(y0, y1)

    extras = {
        "tau_u": np.asarray(u_grid),
        "tau_curves": np.asarray(curves),           # (n_bins, K)
        # Recomputed from the FILTERED draws: readout's own moments are taken
        # over the raw arrays, so a single non-finite draw leaves them inf/nan
        # even once tau_hat is clean.
        "mc_mean0": y0.mean(axis=0),
        "mc_mean1": y1.mean(axis=0),
        "mc_var0": y0.var(axis=0),
        "mc_var1": y1.var(axis=0),
        "mc_tau_sd": (y1 - y0).std(axis=0),
    }

    # The generator's TAU_MARGINAL is Q1(u) - Q0(u), which is what a flow with a
    # monotone causal margin can represent; TAU_PAIRED is the average ITE at
    # rank u and is NOT this arm's target (the two coincide only when the DGP is
    # rank-preserving, i.e. a_cov == 0).
    support = data["ATE"] != 0
    true_marg = np.asarray(data["TAU_MARGINAL"])
    curve_err = np.asarray(curves) - true_marg
    flat = np.asarray(curves).std(axis=0)
    arm_metrics = {
        "mc_n": int(cfg.n_mc),
        "mc_n_used": int(keep.sum()),
        "mc_frac_dropped": frac_dropped,   # >0 means the margin has heavy tails
        "mc_anynan": bool(readout["anynan"]),
        "readout_s": float(readout_s),
        "tau_u_rmse_vs_marginal": float(np.sqrt((curve_err**2).mean())),
        "tau_u_rmse_on_support": float(np.sqrt((curve_err[:, support] ** 2).mean())),
        "tau_u_sd_on_support": float(flat[support].mean()),
        "tau_u_sd_off_support": float(flat[~support].mean()),
        "true_tau_u_sd_on_support": float(true_marg[:, support].std(axis=0).mean()),
    }
    return tau_hat, arm_metrics, extras


def evaluate(cfg: Config, flow, data: dict, losses: dict, wall_time_s: float):
    """Per-pixel ``tau_hat`` for the configured arm, scored against the truth.

    Both arms return the same ``(K,)`` vector under the same score keys, so run
    folders are directly comparable across arms, conditioners and presets.

    ``tau_hat`` is also scored against ATT and ATC. Under E3 these differ from
    the ATE by a known amount, so the three MAEs together say WHICH estimand the
    fit actually landed on -- a distinction E1 and E2 cannot make, since there
    all three coincide.
    """
    K = cfg.size**2
    n_blocks = len(flow.bijection.bijections)
    assert n_blocks == ARM_CHAIN_LEN[cfg.arm], (
        f"expected a {ARM_CHAIN_LEN[cfg.arm]}-block chain for arm {cfg.arm!r}, "
        f"got {n_blocks} -- the read-out below indexes the chain by position"
    )

    extras: dict = {}
    arm_metrics: dict = {}
    if cfg.arm == "location_translation":
        tau_hat = _tau_hat_location_translation(flow, K)
    else:
        tau_hat, arm_metrics, extras = _tau_hat_flexible_continuous(cfg, flow, data, K)

    truth = np.asarray(data["ATE"])
    support = truth != 0
    err = tau_hat - truth
    design = summarise(data)
    metrics = {
        "preset": cfg.preset,
        "arm": cfg.arm,
        "conditioner": cfg.conditioner if cfg.arm == "flexible_continuous" else "n/a",
        # recovery against the primary estimand
        "ate_mae": float(np.abs(err).mean()),
        "ate_rmse": float(np.sqrt((err**2).mean())),
        "ate_corr": float(np.corrcoef(tau_hat, truth)[0, 1]),
        "ate_max_abs_err": float(np.abs(err).max()),
        # Most pixels carry a true effect of EXACTLY zero (a radius-2 disc is 12
        # of 64), so a plain MAE is dominated by off-support bleed and hides
        # whether the magnitude on support was recovered at all. Split it: these
        # two are different failure modes and a method can fail either alone.
        "ate_mae_on_support": float(np.abs(err[support]).mean()),
        "ate_mae_off_support": float(np.abs(err[~support]).mean()),
        "frac_pixels_on_support": float(support.mean()),
        # which estimand did it actually land on?
        "att_mae": float(np.abs(tau_hat - np.asarray(data["ATT"])).mean()),
        "atc_mae": float(np.abs(tau_hat - np.asarray(data["ATC"])).mean()),
        # is the map in the right place?
        "tau_hat_mean_on_support": float(tau_hat[support].mean()),
        "tau_hat_mean_off_support": float(tau_hat[~support].mean()),
        "true_effect_on_support": float(truth[support].mean()),
        # design context, so a run folder is readable without the generator
        "design_naive_bias_mae": float(abs(design["naive_bias_mean"])),
        "design_oracle_ipw_bias_maxabs": design["oracle_ipw_bias_maxabs"],
        "design_att_minus_ate_maxabs": design["att_minus_ate_maxabs"],
        # fit diagnostics
        "best_val_loss": float(np.min(losses["val"])),
        "n_epochs_run": int(len(losses["train"])),
        "n_units": int(np.asarray(data["Y"]).shape[0]),
        "n_pixels": int(K),
        "wall_time_s": float(wall_time_s),
        "s_per_epoch": float(wall_time_s / max(len(losses["train"]), 1)),
        **arm_metrics,
    }
    return tau_hat, metrics, extras


# --------------------------------------------------------------------------- #
# plots (all read from plain arrays, so --replot needs no refit)
# --------------------------------------------------------------------------- #
def make_plots(cfg: Config, data: dict, losses: dict, tau_hat: np.ndarray,
               u_z: np.ndarray, plots_dir: str, extras: dict | None = None):
    os.makedirs(plots_dir, exist_ok=True)
    size = cfg.size
    extras = extras or {}
    ate_true = np.asarray(data["ATE"])
    support = ate_true != 0

    def save(fig, name):
        fig.savefig(os.path.join(plots_dir, name), dpi=120, bbox_inches="tight")
        plt.close(fig)

    # 1. the headline: estimated vs true ATE map, and their difference
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    vmax = float(max(np.abs(tau_hat).max(), np.abs(ate_true).max()))
    im0 = axes[0].imshow(tau_hat.reshape(size, size), vmin=-vmax, vmax=vmax)
    axes[0].set_title(r"Estimated $\hat{\tau}$ per pixel")
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(ate_true.reshape(size, size), vmin=-vmax, vmax=vmax)
    axes[1].set_title("True ATE per pixel (exact)")
    fig.colorbar(im1, ax=axes[1])
    err = (tau_hat - ate_true).reshape(size, size)
    lim = float(np.abs(err).max()) or 1.0
    im2 = axes[2].imshow(err, cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[2].set_title(r"Error $\hat{\tau} -$ truth")
    fig.colorbar(im2, ax=axes[2])
    fig.suptitle(f"{cfg.preset}  |  {cfg.arm}"
                 + (f" / {cfg.conditioner}" if cfg.arm == "flexible_continuous" else ""))
    save(fig, "ate_maps.png")

    # 2. per-pixel scatter, and which estimand the fit landed on
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(ate_true, tau_hat, s=12, alpha=0.6)
    lo, hi = float(min(ate_true.min(), tau_hat.min())), float(max(ate_true.max(), tau_hat.max()))
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[0].set_xlabel("true ATE"); axes[0].set_ylabel(r"$\hat{\tau}$")
    axes[0].set_title("Per-pixel recovery")
    for lab, v in [("ATE", ate_true), ("ATT", np.asarray(data["ATT"])),
                   ("ATC", np.asarray(data["ATC"]))]:
        axes[1].bar(lab, float(np.abs(tau_hat - v).mean()))
    axes[1].set_ylabel("MAE vs estimand")
    axes[1].set_title("Which estimand did the fit land on?")
    save(fig, "recovery_scatter.png")

    # 3. spline arm only: estimated tau(u) against the exact marginal truth
    if "tau_curves" in extras:
        u = np.asarray(extras["tau_u"])
        curves = np.asarray(extras["tau_curves"])
        true_marg = np.asarray(data["TAU_MARGINAL"])
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
        for ax, mask, name in zip(axes, [support, ~support],
                                  ["on effect support", "off support"]):
            if not mask.any():
                ax.set_visible(False)
                continue
            ax.plot(u, curves[:, mask], color="C0", alpha=0.12, lw=0.8)
            ax.plot(u, curves[:, mask].mean(axis=1), color="C0", lw=2.5,
                    label=r"estimated $\hat{\tau}(u)$")
            ax.plot(u, true_marg[:, mask].mean(axis=1), color="k", ls="--", lw=2.0,
                    label=r"true $\tau^{marg}(u)$")
            ax.set_title(f"{name}  ({int(mask.sum())} pixels)")
            ax.set_xlabel("u")
            ax.legend(fontsize=8)
        axes[0].set_ylabel(r"$\tau(u)$")
        fig.suptitle(r"Quantile-resolved effect — estimated vs exact truth")
        save(fig, "tau_curves.png")

    # 4. training curves
    train, val = np.array(losses["train"]), np.array(losses["val"])
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = np.arange(1, len(train) + 1)
    ax.plot(epochs, train, label="train", marker="o", markersize=3)
    ax.plot(epochs, val, label="val", marker="o", markersize=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(); ax.grid(True)
    ax.set_title("Training / validation loss")
    save(fig, "loss_curves.png")

    # 5. design context: assignment balance and the stage-1 quantiles
    T = np.asarray(data["X"]).ravel().astype(bool)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    axes[0].hist([data["THICKNESS"][~T], data["THICKNESS"][T]], bins=40,
                 label=["T=0", "T=1"], density=True, histtype="step")
    axes[0].legend(); axes[0].set_title("Thickness by arm (confounding)")
    axes[1].hist(np.asarray(u_z)[:, 0], bins=30)
    axes[1].set_title(r"$U_{Z}$ stage-1 quantiles (should be flat)")
    mean_bright = inverse_logit(np.asarray(data["Y"])).mean(axis=1)
    axes[2].hist([mean_bright[~T], mean_bright[T]], bins=40,
                 label=["T=0", "T=1"], density=True, histtype="step")
    axes[2].legend(); axes[2].set_title("Mean brightness by arm")
    save(fig, "design_check.png")

    # 6. the imposed truth, for reference
    ITE = np.asarray(data["ITE"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    im = axes[0].imshow(ate_true.reshape(size, size)); fig.colorbar(im, ax=axes[0])
    axes[0].set_title("True per-pixel ATE")
    axes[1].hist(ITE.sum(axis=1), bins=50)
    axes[1].set_title("Per-unit total ITE (heterogeneity)")
    im = axes[2].imshow(ITE.std(axis=0).reshape(size, size)); fig.colorbar(im, ax=axes[2])
    axes[2].set_title("ITE sd across units")
    save(fig, "truth_panels.png")


# --------------------------------------------------------------------------- #
# run folder + metadata
# --------------------------------------------------------------------------- #
RUNS_ROOT = os.path.join(SCRIPT_DIR, "runs", "exp_ate_recovery")

# Arm-specific arrays in arrays.npz, listed explicitly so --replot can pick them
# out of an npz that may or may not contain them.
EXTRA_ARRAY_KEYS = ("tau_u", "tau_curves", "mc_mean0", "mc_mean1",
                    "mc_var0", "mc_var1", "mc_tau_sd")
# Truth arrays needed to rebuild every plot without regenerating the dataset.
TRUTH_ARRAY_KEYS = ("ATE", "ATT", "ATC", "TAU_U", "TAU_PAIRED", "TAU_MARGINAL",
                    "THICKNESS", "PROPENSITY")


def _git_info() -> dict:
    def run(*args):
        try:
            return subprocess.run(
                ["git", *args], cwd=SCRIPT_DIR, capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except Exception:
            return "unavailable"

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


class _Tee:
    """Duplicate a stream's writes into a file, so training progress shows on
    the terminal AND lands in the run folder's log.txt."""

    def __init__(self, stream, fileobj):
        self._stream = stream
        self._file = fileobj

    def write(self, s):
        self._stream.write(s)
        self._file.write(s)
        self._file.flush()
        return len(s)

    def flush(self):
        self._stream.flush()
        self._file.flush()


def run_id_for(cfg: Config) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    arm_tag = ARM_SHORT[cfg.arm] + ("-trf" if cfg.conditioner == "transformer" else "")
    return (f"{stamp}_{PRESET_SHORT[cfg.preset]}_{arm_tag}"
            f"_s{cfg.seed_fit}_k{cfg.size**2}_{secrets.token_hex(3)}")


def write_config(cfg: Config, run_id: str, run_dir: str):
    """Create the run folder and record the configuration at launch.

    Refuses an existing folder: a run-id collision must fail at launch rather
    than let two runs silently write into the same directory.
    """
    os.makedirs(run_dir, exist_ok=False)
    record = {
        "run_id": run_id,
        "config": asdict(cfg),
        "effective_radius": cfg.effective_radius,
        "git": _git_info(),
        "versions": {
            "jax": jax.__version__,
            "flowjax": flowjax.__version__,
            "equinox": equinox.__version__,
            "numpy": np.__version__,
            "frugal_flows": getattr(frugal_flows, "__version__", "unversioned"),
        },
        "x64_active": bool(jax.config.jax_enable_x64),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(record, f, indent=2)


def save_run(cfg: Config, data: dict, losses: dict, tau_hat: np.ndarray,
             u_z: np.ndarray, metrics: dict, extras: dict, run_dir: str):
    metrics = {"run_id": os.path.basename(run_dir), **metrics}
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez(
        os.path.join(run_dir, "arrays.npz"),
        tau_hat=tau_hat,
        u_z=np.asarray(u_z),
        X=np.asarray(data["X"]),
        Y=np.asarray(data["Y"]),
        ITE=np.asarray(data["ITE"]),
        loss_train=np.asarray(losses["train"]),
        loss_val=np.asarray(losses["val"]),
        **{k: np.asarray(data[k]) for k in TRUTH_ARRAY_KEYS},
        **{k: np.asarray(v) for k, v in (extras or {}).items()},
    )
    make_plots(cfg, data, losses, tau_hat, u_z, os.path.join(run_dir, "plots"), extras)


def replot(run_dir: str):
    """Regenerate every plot for an existing run from its saved arrays."""
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = Config(**json.load(f)["config"])
    a = np.load(os.path.join(run_dir, "arrays.npz"))
    data = {k: a[k] for k in TRUTH_ARRAY_KEYS}
    data.update({"X": a["X"], "Y": a["Y"], "ITE": a["ITE"]})
    losses = {"train": a["loss_train"], "val": a["loss_val"]}
    extras = {k: a[k] for k in EXTRA_ARRAY_KEYS if k in a}
    make_plots(cfg, data, losses, a["tau_hat"], a["u_z"],
               os.path.join(run_dir, "plots"), extras)
    print(f"replotted: {os.path.join(run_dir, 'plots')}")


def collect(runs_root: str = RUNS_ROOT) -> list[dict]:
    """One row per completed run in the archive, newest last."""
    rows = []
    for name in sorted(os.listdir(runs_root)) if os.path.isdir(runs_root) else []:
        path = os.path.join(runs_root, name, "metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                rows.append(json.load(f))
    return rows


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #
def run_one(cfg: Config, runs_root: str = None) -> dict:
    """Build, fit, score and archive one cell. Returns its metrics."""
    run_id = run_id_for(cfg)
    run_dir = os.path.join(runs_root or RUNS_ROOT, run_id)
    write_config(cfg, run_id, run_dir)
    print(f"run dir: {run_dir}")

    with open(os.path.join(run_dir, "log.txt"), "w") as lf:
        out, err = _Tee(sys.stdout, lf), _Tee(sys.stderr, lf)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            print(f"started; watch progress with: tail -f {run_dir}/log.txt")
            print(f"preset: {cfg.preset} | arm: {cfg.arm} | conditioner: {cfg.conditioner}")
            data = build_data(cfg)
            Y = np.asarray(data["Y"])
            print(f"data built: {Y.shape[0]} units x {Y.shape[1]} pixels "
                  f"(size={cfg.size}, radius={cfg.effective_radius})")
            t0 = time.monotonic()
            flow, losses, u_z = fit_flow(cfg, data)
            wall = time.monotonic() - t0
            n_ep = len(losses["train"])
            print(f"fit finished: {n_ep} epochs in {wall:.0f}s (~{wall / n_ep:.1f}s/epoch)")
            tau_hat, metrics, extras = evaluate(cfg, flow, data, losses, wall)
            save_run(cfg, data, losses, tau_hat, u_z, metrics, extras, run_dir)
            for k, v in metrics.items():
                print(f"  {k}: {v:.4g}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"run dir: {run_dir}")
    return {"run_id": run_id, **metrics}


# Config fields that identify a sweep cell, for --skip-done. A run with
# different capacity or a different seed is NOT the same cell, so the tuple
# covers the knobs that make two runs comparable rather than just the label.
CELL_IDENTITY = ("preset", "arm", "conditioner", "size", "radius", "digit", "n",
                 "seed_data", "seed_fit", "nn_width", "nn_depth", "flow_layers",
                 "max_epochs", "n_mc")


def completed_cells(runs_root: str = RUNS_ROOT) -> set[tuple]:
    """Identity tuples of every COMPLETED run in the archive.

    A folder without metrics.json is still training (or died), so it does not
    count as done and its cell will be re-run.
    """
    done = set()
    for name in sorted(os.listdir(runs_root)) if os.path.isdir(runs_root) else []:
        run_dir = os.path.join(runs_root, name)
        if not os.path.exists(os.path.join(run_dir, "metrics.json")):
            continue
        try:
            with open(os.path.join(run_dir, "config.json")) as f:
                c = json.load(f)["config"]
            done.add(tuple(c.get(k) for k in CELL_IDENTITY))
        except (OSError, KeyError, json.JSONDecodeError):
            continue  # an unreadable folder just means that cell re-runs
    return done


def run_sweep(base: Config, skip_done: bool = False) -> list[dict]:
    """Every (preset, arm, conditioner) cell, sequentially.

    A failing cell is reported and skipped rather than aborting the sweep: a
    partial matrix is more useful than none. The Config is built INSIDE the
    guard, so a cell whose knobs are invalid for its arm also just skips --
    otherwise one bad combination discards every cell still queued behind it.
    """
    rows = []
    done = completed_cells() if skip_done else set()
    cells = [(p, arm, cond) for p in PRESETS for arm, cond in SWEEP_CELLS]
    for i, (preset, arm, cond) in enumerate(cells, 1):
        print(f"\n=== [{i}/{len(cells)}] {preset} | {arm} | {cond} ===")
        try:
            cfg = Config(**{**asdict(base), "preset": preset, "arm": arm,
                            "conditioner": cond})
            if tuple(asdict(cfg).get(k) for k in CELL_IDENTITY) in done:
                print("  already completed, skipping (--skip-done)")
                continue
            rows.append(run_one(cfg))
        except Exception as exc:  # noqa: BLE001 - a bad cell must not kill the sweep
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            rows.append({"preset": preset, "arm": arm, "conditioner": cond,
                         "error": f"{type(exc).__name__}: {exc}"})
    return rows


def print_table(rows: list[dict]):
    cols = ["preset", "arm", "conditioner", "ate_mae", "ate_rmse", "ate_corr",
            "att_mae", "atc_mae", "wall_time_s"]
    widths = {c: max(len(c), *(len(_fmt(r.get(c))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        if "error" in r:
            print(f"{r['preset']:<{widths['preset']}}  {r['arm']:<{widths['arm']}}  "
                  f"{r['conditioner']:<{widths['conditioner']}}  -> {r['error']}")
            continue
        print("  ".join(_fmt(r.get(c)).ljust(widths[c]) for c in cols))


def _fmt(v) -> str:
    if v is None:
        return "-"
    return f"{v:.4g}" if isinstance(v, float) else str(v)


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def _selftest_generator(check):
    """Invariants of the DGP itself, independent of any flow."""
    exps = {p: build_preset(p, size=4, radius=1, n=400, seed=0) for p in PRESETS}

    for name, d in exps.items():
        tag = PRESET_SHORT[name]
        Y0, Y1 = np.asarray(d["Y0"]), np.asarray(d["Y1"])
        T = np.asarray(d["X"]).ravel().astype(bool)
        ITE, ATE = np.asarray(d["ITE"]), np.asarray(d["ATE"])

        check(f"{tag}: imposed ATE == effect map exactly",
              np.abs(ITE.mean(0) - np.asarray(d["MAP"])).max() < 1e-9)
        check(f"{tag}: ITE == Y1 - Y0",
              np.allclose(ITE, Y1 - Y0))
        check(f"{tag}: Y is the correct factual mixture",
              np.allclose(np.asarray(d["Y"]), np.where(T[:, None], Y1, Y0)))
        check(f"{tag}: ATT/ATC come from the right subsets",
              np.allclose(d["ATT"], ITE[T].mean(0))
              and np.allclose(d["ATC"], ITE[~T].mean(0)))
        # Both quantile curves must integrate back to the ATE. n need not divide
        # evenly into the bins, so allow one bin's worth of slack.
        for key in ("TAU_PAIRED", "TAU_MARGINAL"):
            check(f"{tag}: {key} integrates over u to the ATE",
                  np.abs(np.asarray(d[key]).mean(0) - ATE).max() < 1e-2)

    e1, e2, e3 = (exps[p] for p in ("exp1_rct_homogeneous",
                                    "exp2_confounded_homogeneous",
                                    "exp3_confounded_heterogeneous"))
    # Homogeneous rungs: every unit gets the identical effect, so all three
    # estimands coincide exactly and tau(u) is flat.
    for tag, d in (("e1rct", e1), ("e2conf", e2)):
        check(f"{tag}: effect is homogeneous across units",
              np.asarray(d["ITE"]).std(0).max() < 1e-12)
        check(f"{tag}: ATE == ATT == ATC",
              np.allclose(d["ATE"], d["ATT"]) and np.allclose(d["ATE"], d["ATC"]))
    check("e2conf: true tau(u) is flat in u",
          np.asarray(e2["TAU_MARGINAL"]).std(0).max() < 1e-9)
    check("e3het: effect is heterogeneous across units",
          np.asarray(e3["ITE"]).std(0).max() > 1e-3)
    check("e3het: ATT differs from ATE (estimands separate)",
          np.abs(np.asarray(e3["ATT"]) - np.asarray(e3["ATE"])).max() > 1e-3)

    # Assignment: E1 randomised, E2/E3 confounded through thickness.
    def corr(d):
        return abs(np.corrcoef(d["THICKNESS"],
                               np.asarray(d["X"]).ravel())[0, 1])
    check("e1rct: assignment is unconfounded", corr(e1) < 0.1)
    check("e2conf/e3het: assignment is confounded",
          corr(e2) > 0.3 and corr(e3) > 0.3)

    # Overrides must actually reach the generator, not be silently dropped.
    flat = build_preset("exp3_confounded_heterogeneous", size=4, radius=1, n=400,
                        seed=0, a_cov=0.0, b_quant=0.0)
    check("override: a_cov=b_quant=0 makes E3 homogeneous",
          np.asarray(flat["ITE"]).std(0).max() < 1e-12)
    strong = build_preset("exp1_rct_homogeneous", size=4, radius=1, n=400,
                          seed=0, base_shift=3.0)
    check("override: base_shift scales the imposed ATE",
          abs(float(np.asarray(strong["ATE"]).max()) - 3.0) < 1e-9)

    # --digit toggles the discrete covariate block.
    check("digit=0 -> no discrete covariates",
          np.asarray(e1["z_discr"]).shape[1] == 0)
    alld = build_preset("exp1_rct_homogeneous", size=4, radius=1, n=400,
                        seed=0, digit=None)
    check("digit=None -> 10 discrete covariates",
          np.asarray(alld["z_discr"]).shape[1] == 10
          and np.asarray(alld["Z"]).shape[1] == 11)


def _selftest_pipeline(check, runs_root):
    """A real (tiny) fit through each arm, plus the archive round-trip."""
    base = dict(size=4, n=400, max_epochs=2, marginal_max_epochs=2,
                marginal_max_patience=2, n_mc=200, batch_size=100)

    cfg = Config(preset="exp1_rct_homogeneous", **base)
    check("K tracks --size", cfg.size**2 == 16)
    check("radius auto-scales with size",
          Config(size=8, **{k: v for k, v in base.items() if k != 'size'}
                 ).effective_radius == 2
          and Config(size=16, **{k: v for k, v in base.items() if k != 'size'}
                     ).effective_radius == 4)

    data = build_data(cfg)
    check("build_data honours --size", np.asarray(data["Y"]).shape[1] == 16)

    for arm, cond in [("location_translation", "mlp"),
                      ("flexible_continuous", "mlp"),
                      ("flexible_continuous", "transformer")]:
        tag = f"{ARM_SHORT[arm]}/{cond}"
        c = Config(preset="exp3_confounded_heterogeneous", arm=arm,
                   conditioner=cond, **base)
        # The bijection-chain length is asserted inside evaluate(); catching
        # here turns a layout change into a reported FAIL for this arm rather
        # than a traceback that abandons every remaining check.
        try:
            m = run_one(c, runs_root=runs_root)
            ok, why = True, ""
        except Exception as exc:  # noqa: BLE001 - one broken arm must still report
            ok, why = False, f" ({type(exc).__name__}: {exc})"
        check(f"{tag}: fits and scores end to end (chain, read-out){why}", ok)
        if not ok:
            continue

        check(f"{tag}: metrics carry every key the tables use",
              all(k in m for k in ("ate_mae", "ate_mae_on_support",
                                   "ate_mae_off_support", "ate_corr",
                                   "att_mae", "atc_mae", "n_pixels")))
        check(f"{tag}: scored all K pixels, not just pixel 0",
              m["n_pixels"] == 16)
        check(f"{tag}: tau_hat is finite", np.isfinite(m["ate_mae"]))

        run_dir = os.path.join(runs_root, m["run_id"])
        tau = np.load(os.path.join(run_dir, "arrays.npz"))["tau_hat"]
        check(f"{tag}: tau_hat has one entry per pixel", tau.shape == (16,))
        if arm == "flexible_continuous":
            check(f"{tag}: read-out reports its dropped-draw fraction",
                  "mc_frac_dropped" in m and m["mc_frac_dropped"] < 1.0)

    # archive round-trip
    last = sorted(os.listdir(runs_root))[-1]
    replot(os.path.join(runs_root, last))
    check("replot regenerates plots without a refit",
          os.path.exists(os.path.join(runs_root, last, "plots", "ate_maps.png")))
    check("collect finds every completed run",
          len(collect(runs_root)) == 3)


def _selftest_guards(check):
    """Invalid combinations must be refused at launch, not deep in the build."""
    def raises(**kw):
        try:
            Config(**kw)
            return False
        except ValueError:
            return True

    check("guard: unknown preset refused", raises(preset="nope"))
    check("guard: unknown arm refused", raises(arm="nope"))
    check("guard: transformer + location_translation refused",
          raises(arm="location_translation", conditioner="transformer"))
    check("guard: nn_width not divisible by nn_heads refused",
          raises(arm="flexible_continuous", conditioner="transformer",
                 nn_width=50, nn_heads=4))
    check("guard: valid transformer config accepted",
          not raises(arm="flexible_continuous", conditioner="transformer",
                     nn_width=48, nn_heads=4))


def selftest() -> int:
    """End-to-end correctness check. Returns a process exit code."""
    import tempfile

    results: list[tuple[str, bool]] = []
    # Bound to the REAL stdout: the pipeline stage runs under a redirect to
    # silence training progress, and check output must survive it.
    report = sys.stdout

    def check(label, ok):
        results.append((label, bool(ok)))
        print(f"  {'PASS' if ok else 'FAIL'}  {label}", file=report)

    runs_root = tempfile.mkdtemp(prefix="exp_ate_selftest_")
    try:
        print("\n--- generator ---", file=report)
        _selftest_generator(check)
        print("\n--- guards ---", file=report)
        _selftest_guards(check)
        print("\n--- pipeline (tiny real fits; accuracy is NOT checked) ---",
              file=report)
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), \
                 contextlib.redirect_stderr(devnull):
                _selftest_pipeline(check, runs_root)
    finally:
        import shutil
        shutil.rmtree(runs_root, ignore_errors=True)

    failed = [lab for lab, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("FAILED:")
        for lab in failed:
            print(f"  - {lab}")
    return 1 if failed else 0


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    choices = {"arm": ARMS, "conditioner": CONDITIONERS, "preset": list(PRESETS)}
    for f in fields(Config):
        flag = "--" + f.name.replace("_", "-")
        kw = {"choices": choices[f.name]} if f.name in choices else {}
        if f.type == "bool":
            parser.add_argument(flag, action=argparse.BooleanOptionalAction,
                                default=f.default)
        elif f.type.endswith("| None"):
            # Take the DATACLASS default, not None: the generator-override
            # fields already default to None ("keep the preset's value"), but
            # --digit defaults to 0 and --radius to None, and hardcoding None
            # here would silently switch --digit to the all-classes variant.
            base = f.type.split("|")[0].strip()
            parser.add_argument(flag, type={"int": int, "float": float,
                                            "str": str}[base],
                                default=f.default, **kw)
        else:
            parser.add_argument(flag, type=type(f.default), default=f.default, **kw)
    parser.add_argument("--all-digits", action="store_true",
                        help="use all ten digit classes (digit=None)")
    parser.add_argument("--sweep", action="store_true",
                        help="run every (preset, arm, conditioner) cell")
    parser.add_argument("--skip-done", action="store_true",
                        help="sweep: skip cells already completed with these knobs")
    parser.add_argument("--replot", metavar="RUN_DIR", default=None,
                        help="regenerate plots for an existing run, no refit")
    parser.add_argument("--collect", action="store_true",
                        help="print a table of every completed run and exit")
    parser.add_argument("--selftest", action="store_true",
                        help="end-to-end correctness check; writes nothing permanent")
    args = parser.parse_args(argv)

    if args.selftest:
        raise SystemExit(selftest())
    if args.replot is not None:
        replot(args.replot)
        return
    if args.collect:
        rows = collect()
        print_table(rows) if rows else print(f"no completed runs under {RUNS_ROOT}")
        return

    cfg = Config(**{f.name: getattr(args, f.name) for f in fields(Config)})
    if args.all_digits:
        cfg = Config(**{**asdict(cfg), "digit": None})
    jax.config.update("jax_enable_x64", cfg.x64)
    os.makedirs(RUNS_ROOT, exist_ok=True)

    if args.sweep:
        rows = run_sweep(cfg, skip_done=args.skip_done)
        print(f"\n=== sweep complete: {len(rows)} cells ===")
        print_table(rows)
    else:
        run_one(cfg)


if __name__ == "__main__":
    main()
