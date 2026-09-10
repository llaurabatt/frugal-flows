"""Frengression on the MorphoMNIST ATE-recovery ladder.

A third estimator family alongside ``exp_ate_recovery.py`` (frugal flows) and
``baselines.py`` (naive / IPW / OLS / AIPW), on the SAME six E1-E6 presets with
the same generated datasets, the same truth and the same scores. Nothing about
the DGP or the other estimators changes.

What frengression is asked to do here, and nothing more: learn the two marginal
interventional laws

    P(Y | do(T=0))    and    P(Y | do(T=1))

and hence the per-pixel image ATE, ``ATE_k = E[Y_k(1)] - E[Y_k(0)]``. Not
invertibility, not abduction from a factual image, not individual
counterfactuals, not a CATE.

THE MODEL IS NOT REIMPLEMENTED HERE. This module constructs the official
``frengression.Frengression`` (pinned in environment-frengression.yaml) and
calls its own ``train_y``; everything around it -- scaling, seeding, sampling,
scoring, archiving -- is adapter code. The training loop lives upstream.

How the causal margin is sampled, since the call looks odd
---------------------------------------------------------
``model_y`` is built with input width ``x_dim + y_dim``: at training time it is
fed ``[x, eta]`` where ``eta = model_eta(x, z)`` carries the confounding.
``sample_causal_margin`` passes ONLY ``x``, and engression's first layer fills
the missing ``y_dim`` columns with fresh N(0, I) noise. That substitution is
exactly the paper's Eq. 5 sampling rule -- draw eta from its own marginal
rather than from its conditional given z -- so ``model_xz`` is never trained and
``train_xz`` is never called. The "covariate dimension does not aligned"
notice engression prints for that call is expected, and is silenced.

Defaults, and which of them are ours
------------------------------------
``lr``, ``hidden_dim`` and ``num_layer`` were tuned on seeds 101-103 over E2 and
E4 and the package's own values won, so they are unchanged. Two defaults DO
depart from upstream, both on measured evidence: ``noise_dim`` 10 -> 64, and
``y_scaling`` global -> per_pixel (see the field comments). Reporting seeds are
1-5, disjoint from the tuning seeds.

Two implementation choices that are load-bearing
------------------------------------------------
*Common random numbers.* ``torch.manual_seed(seed_mc)`` is re-applied
immediately before EACH of the two ``sample_causal_margin`` calls, so do(0) and
do(1) draw the same noise and ``y1 - y0`` is paired. Measured here: the paired
per-pixel Monte-Carlo standard error is around an order of magnitude below the
unpaired one, which matters because the unpaired error is the same size as the
whole quantity being estimated. Both are reported (``mc_se_max``,
``mc_se_unpaired_max``) so the pairing can be seen to be working.

*Sampling shape.* ``sample_causal_margin(torch.full((n_mc, 1), t),
sample_size=1)``, never the full ``X`` with ``sample_size=n_mc``: engression's
``sample_onebatch`` does ``x.repeat(sample_size, 1)``, so the natural-looking
call materialises ``n * n_mc`` rows -- tens of gigabytes -- and would silently
halve its batch size in a retry loop, changing the number of RNG draws and
breaking the pairing.

Seeds. ``--seed-data`` draws the dataset, ``--seed-fit`` initialises the
network, ``--seed-mc`` drives the interventional draws. ``--sweep --seeds S``
sets seed_data AND seed_fit to each S (one replicate = a fresh dataset and a
fresh initialisation), leaving seed_mc at its flag value; that is what makes a
row join to the baselines CSV's ``seed`` column.

Usage
-----
    python exp_frengression_recovery.py --selftest
    python exp_frengression_recovery.py --preset exp4_covariate_cate --size 8
    python exp_frengression_recovery.py --sweep --seeds 1 2 3 4 5 --size 8
    python exp_frengression_recovery.py --collect --csv runs/frengression/all.csv
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from importlib import metadata

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)  # prepare_morphomnist_exps / baselines are siblings

from baselines import score as score_effect_map
from frugal_flows.interventions import tau_curve
from prepare_morphomnist_exps import PRESETS, build_preset, summarise

# The only keys the training path may read. Everything else the generator
# returns is oracle knowledge and is available to `evaluate` alone.
MODEL_INPUT_KEYS = frozenset({"Y", "X", "Z", "z_cat_idx"})

PRESET_SHORT = {
    "exp1_rct_homogeneous": "e1rct",
    "exp2_confounded_homogeneous": "e2conf",
    "exp3_confounded_heterogeneous": "e3het",
    "exp4_covariate_cate": "e4cate",
    "exp5_quantile_effect": "e5quant",
    "exp6_spatial_cate": "e6spat",
}
Y_SCALINGS = ("global", "per_pixel", "none")
Z_SCALINGS = ("standardize", "none")
DEVICES = ("cpu", "mps")
RUNS_ROOT = os.path.join(SCRIPT_DIR, "runs", "frengression")

# frengression's own progress line, one per iteration at print_every_iter=1:
#   Epoch 7: loss 3.6235,\tloss_y 0.8152, 1.7142, 1.7980,\tloss_eta 2.8083, ...
_NUM = r"(-?(?:\d+\.?\d*(?:[eE][+-]?\d+)?|nan|inf))"
LOSS_RE = re.compile(
    rf"Epoch\s+(\d+):\s*loss\s+{_NUM},\s*loss_y\s+{_NUM},\s*{_NUM},\s*{_NUM},"
    rf"\s*loss_eta\s+{_NUM},\s*{_NUM},\s*{_NUM}"
)
STOP_RE = re.compile(r"Stopping at iter (\d+)")


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    """One frengression cell. Field names double as CLI flags."""

    preset: str = "exp1_rct_homogeneous"
    # ---- dataset (mirrors exp_ate_recovery.Config so the datasets match) ----
    size: int = 8
    radius: int | None = None
    digit: int | None = 0
    n: int | None = None
    seed_data: int = 0
    # ---- generator overrides; None keeps the preset's own value ----
    base_shift: float | None = None
    effect_mode: str | None = None
    a_cov: float | None = None
    a_bright: float | None = None
    a_inter: float | None = None
    h_shape: str | None = None
    g_shape: str | None = None
    b_quant: float | None = None
    ps_slope: float | None = None
    ps_intercept: float | None = None
    effect: str | None = None
    # ---- model capacity / optimisation (passed to the official train_y) ----
    num_iters: int = 5000
    lr: float = 1e-3
    hidden_dim: int = 100
    num_layer: int = 3
    # Raised from the package's 10 on the tuning evidence: 64 improved ate_mae.
    # lr, hidden_dim and num_layer were tuned too and the package defaults won,
    # so this and y_scaling are the only departures.
    noise_dim: int = 64
    # tol=0 disables frengression's early stop: its check is a strict `<`, and
    # a run that stops on a two-iteration loss plateau is not comparable with
    # one that ran the full budget.
    tol: float = 0.0
    # ---- preprocessing ----
    # This default was chosen the other way round on theory and CHANGED on
    # evidence. The argument for "global" (centre per pixel, divide by one
    # global sd) was that ~45% of pixels carry only dequantisation noise
    # (sd ~0.04) against ~1.0 on the effect's support, so per-pixel scaling
    # would let noise pixels dominate the multivariate training loss. Measured
    # on tuning seeds 101-103 over E2 and E4, per_pixel with the floor below
    # improved ATE recovery. The floor prevents near-constant background pixels
    # from being amplified excessively.
    y_scaling: str = "per_pixel"
    z_scaling: str = "standardize"
    y_sd_floor: float = 0.25
    # ---- fit / read-out ----
    seed_fit: int = 34
    n_mc: int = 50000
    seed_mc: int = 0
    crn: bool = True
    device: str = "cpu"
    threads: int = 4
    print_every: int = 100
    # ---- experiment tracking (off by default; local archives are authoritative) ----
    wandb: bool = False
    wandb_entity: str | None = None
    wandb_project: str = "morphomnist-ate"
    wandb_group: str | None = None
    wandb_tags: str | None = None

    def __post_init__(self):
        if self.preset not in PRESETS:
            raise ValueError(f"unknown preset {self.preset!r}; choose from {list(PRESETS)}")
        if self.y_scaling not in Y_SCALINGS:
            raise ValueError(f"y_scaling must be one of {Y_SCALINGS}, got {self.y_scaling!r}")
        if self.z_scaling not in Z_SCALINGS:
            raise ValueError(f"z_scaling must be one of {Z_SCALINGS}, got {self.z_scaling!r}")
        if self.device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}, got {self.device!r}")
        if self.n_mc < 2:
            raise ValueError("n_mc must be at least 2")
        if self.threads < 1:
            raise ValueError("threads must be at least 1")

    @property
    def effective_radius(self) -> int:
        """Copied from exp_ate_recovery.Config: holds the effect map at ~20% of
        pixels as K changes. Must agree with the FF runner or the two families
        are not fitted to the same dataset."""
        return self.radius if self.radius is not None else max(1, round(self.size / 4))


# --------------------------------------------------------------------------- #
# data, behind an enforced oracle guard
# --------------------------------------------------------------------------- #
class OracleGuard:
    """A read-only view of the generator's output exposing ONLY the observables.

    The guard is the mechanism, not a convention: the fitting path is handed
    this object rather than the raw dict, so an attempt to condition on `ATE`,
    `Y0`, `PROPENSITY` or any other oracle key raises instead of quietly
    producing an impossibly good result. `touched` records what was read, so a
    test can assert the allowlist was respected over a whole real fit.
    """

    def __init__(self, data: dict, allowed=MODEL_INPUT_KEYS):
        self._data = data
        self._allowed = frozenset(allowed)
        self.touched: set[str] = set()

    def __getitem__(self, key):
        if key not in self._allowed:
            raise KeyError(
                f"{key!r} is oracle knowledge and must not reach the training path; "
                f"only {sorted(self._allowed)} are observable"
            )
        self.touched.add(key)
        return self._data[key]

    def __contains__(self, key):
        return key in self._allowed and key in self._data

    def keys(self):
        return [k for k in self._allowed if k in self._data]


def build_data(cfg: Config) -> dict:
    """The same generator call the FF runner makes, with the same knobs."""
    overrides = {
        "size": cfg.size,
        "radius": cfg.effective_radius,
        "digit": cfg.digit,
        "seed": cfg.seed_data,
    }
    if cfg.n is not None:
        overrides["n"] = cfg.n
    for name in ("base_shift", "effect_mode", "a_cov", "a_bright", "a_inter",
                 "h_shape", "g_shape", "b_quant", "ps_slope", "ps_intercept", "effect"):
        value = getattr(cfg, name)
        if value is not None:
            overrides[name] = value
    return build_preset(cfg.preset, **overrides)


@dataclass
class Inputs:
    """Model-ready tensors plus the exact inverse of the scaling applied."""

    x: torch.Tensor
    z: torch.Tensor
    y: torch.Tensor
    y_mean: np.ndarray
    y_scale: np.ndarray
    z_mean: np.ndarray
    z_scale: np.ndarray
    K: int
    z_dim: int
    n: int
    y_sd_global: float
    n_binary_z: int = 0

    def unscale_y(self, arr: np.ndarray) -> np.ndarray:
        """Scaled space -> original Y units. Applied to every draw before any
        metric, so tau_hat is on the generator's scale."""
        return np.asarray(arr, dtype=np.float64) * self.y_scale + self.y_mean


def prepare_inputs(data, cfg: Config) -> Inputs:
    """Observables -> float32 tensors, with scaling fitted on observables only.

    `data` may be the raw generator dict or an OracleGuard; either way only the
    allowlisted keys are read. The float32 cast is explicit because the jnp
    dtype follows jax's x64 flag (which the test suite's conftest turns ON),
    and a float64 array would not feed float32 Linear layers.
    """
    if not isinstance(data, OracleGuard):
        data = OracleGuard(data)
    Y = np.asarray(data["Y"], dtype=np.float64)
    X = np.asarray(data["X"], dtype=np.float64)
    Z = np.asarray(data["Z"], dtype=np.float64)
    z_cat_idx = np.asarray(data["z_cat_idx"], dtype=bool)

    if not np.isfinite(Y).all() or not np.isfinite(X).all() or not np.isfinite(Z).all():
        raise ValueError("observed data contains non-finite values")
    if X.std() == 0:
        raise ValueError("treatment is constant -- do(0) and do(1) are not identified")

    n, K = Y.shape
    y_sd_global = float(Y.std())
    if cfg.y_scaling == "global":
        y_mean, y_scale = Y.mean(axis=0), np.full(K, y_sd_global or 1.0)
    elif cfg.y_scaling == "per_pixel":
        # Floor the per-pixel sd: an all-but-constant background pixel would
        # otherwise be amplified until its dequantisation noise dominates.
        sd = Y.std(axis=0)
        y_mean = Y.mean(axis=0)
        y_scale = np.maximum(sd, cfg.y_sd_floor * (y_sd_global or 1.0))
    else:
        y_mean, y_scale = np.zeros(K), np.ones(K)

    if cfg.z_scaling == "standardize":
        z_mean = np.where(z_cat_idx, 0.0, Z.mean(axis=0))
        sd = Z.std(axis=0)
        z_scale = np.where(z_cat_idx | (sd == 0), 1.0, sd)
    else:
        z_mean, z_scale = np.zeros(Z.shape[1]), np.ones(Z.shape[1])

    dev = torch.device(cfg.device)
    to_t = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)
    return Inputs(
        x=to_t(X),
        z=to_t((Z - z_mean) / z_scale),
        y=to_t((Y - y_mean) / y_scale),
        y_mean=y_mean, y_scale=y_scale, z_mean=z_mean, z_scale=z_scale,
        K=K, z_dim=Z.shape[1], n=n, y_sd_global=y_sd_global,
        n_binary_z=int(z_cat_idx.sum()),
    )


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def build_model(cfg: Config, inputs: Inputs):
    """The official Frengression, seeded and with the padding notice silenced.

    `device` must be passed: the package's own default is cuda. The seed is set
    BEFORE construction because the Linear initialisers consume the generator.
    """
    from frengression import Frengression  # lazy: keeps import cost off `--collect`

    torch.set_num_threads(cfg.threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(cfg.seed_fit)
    model = Frengression(
        x_dim=1,
        y_dim=inputs.K,
        z_dim=inputs.z_dim,
        num_layer=cfg.num_layer,
        hidden_dim=cfg.hidden_dim,
        noise_dim=cfg.noise_dim,
        x_binary=True,
        z_binary_dims=inputs.n_binary_z,
        y_binary=False,
        device=torch.device(cfg.device),
    )
    # Expected, not an error: sample_causal_margin deliberately passes only x
    # and lets the first layer fill the eta slot with N(0, I) noise.
    for net in (model.model_y, model.model_eta):
        net.input_layer.verbose = False
    return model


class _LossCapture:
    """Stream filter around frengression's stdout.

    `train_y` returns None and prints; the loss curve exists only as text. This
    parses every line (the model is called with print_every_iter=1) while
    forwarding just one line in `print_every` to the real stream, so log.txt
    stays readable and loss_curve.csv keeps every iteration.
    """

    def __init__(self, stream, print_every: int):
        self._stream = stream
        self._print_every = max(1, print_every)
        self._buf = ""
        self.rows: list[tuple] = []
        self.stopped_at: int | None = None

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._handle(line)
        return len(s)

    def _handle(self, line):
        m = LOSS_RE.search(line)
        if m:
            it = int(m.group(1))
            vals = [float(m.group(i)) for i in range(2, 9)]
            self.rows.append((it, *vals))
            if it == 1 or it % self._print_every == 0:
                self._stream.write(line + "\n")
            return
        s = STOP_RE.search(line)
        if s:
            self.stopped_at = int(s.group(1))
        if line.strip():
            self._stream.write(line + "\n")

    def flush(self):
        if self._buf.strip():
            self._handle(self._buf)
            self._buf = ""
        self._stream.flush()


def fit(cfg: Config, model, inputs: Inputs, out_stream=None):
    """Call the official ``train_y`` and recover its loss curve.

    Argument order is ``(x, z, y)`` -- upstream's, not the conventional one.
    """
    stream = out_stream if out_stream is not None else sys.stdout
    cap = _LossCapture(stream, cfg.print_every)
    t0 = time.monotonic()
    with contextlib.redirect_stdout(cap):
        model.train_y(
            inputs.x, inputs.z, inputs.y,
            num_iters=cfg.num_iters, lr=cfg.lr, print_every_iter=1, tol=cfg.tol,
        )
    cap.flush()
    fit_s = time.monotonic() - t0

    if not cap.rows:
        raise RuntimeError(
            "no loss lines were parsed from train_y's output -- its print format "
            "has changed and the loss curve would be silently empty"
        )
    arr = np.asarray(cap.rows, dtype=np.float64)
    losses = {
        "iter": arr[:, 0].astype(int),
        "loss": arr[:, 1], "loss_y": arr[:, 2],
        "loss_y_1": arr[:, 3], "loss_y_2": arr[:, 4],
        "loss_eta": arr[:, 5], "loss_eta_1": arr[:, 6], "loss_eta_2": arr[:, 7],
    }
    finite = np.isfinite(losses["loss"])
    info = {
        "n_iters_run": int(len(arr)),
        "early_stopped": bool(cap.stopped_at is not None),
        "loss_first": float(losses["loss"][0]),
        "loss_final": float(losses["loss"][-1]),
        "loss_min": float(np.nanmin(losses["loss"])) if finite.any() else float("nan"),
        "loss_min_iter": int(losses["iter"][np.nanargmin(np.where(finite, losses["loss"], np.inf))])
        if finite.any() else -1,
        "loss_y_final": float(losses["loss_y"][-1]),
        "loss_eta_final": float(losses["loss_eta"][-1]),
        "loss_nonfinite_iters": int((~finite).sum()),
        "fit_s": float(fit_s),
    }
    tail = max(1, len(arr) // 10)
    head_ref = float(np.nanmax(np.abs(losses["loss"][:tail]))) or 1.0
    info["loss_rel_decrease_last10pct"] = float(
        (losses["loss"][-tail] - losses["loss"][-1]) / head_ref
    ) if len(arr) > 1 else 0.0
    # A run whose loss ends materially above its own minimum has come apart;
    # that is a different failure from producing non-finite draws.
    info["diverged"] = bool(
        info["loss_nonfinite_iters"] > 0
        or not np.isfinite(info["loss_final"])
        or (np.isfinite(info["loss_min"]) and info["loss_final"] > info["loss_min"] + 0.2 * abs(info["loss_min"]))
    )
    return losses, info


def sample_margins(cfg: Config, model, inputs: Inputs):
    """Paired draws from the two causal margins, in ORIGINAL Y units.

    Re-seeding before each call is what pairs them: identical first-layer noise
    for do(0) and do(1), so `y1 - y0` isolates the treatment.
    """
    t0 = time.monotonic()
    draws = []
    for i, t in enumerate((0.0, 1.0)):
        torch.manual_seed(cfg.seed_mc if cfg.crn else cfg.seed_mc + i)
        x = torch.full((cfg.n_mc, 1), float(t), dtype=torch.float32,
                       device=torch.device(cfg.device))
        s = model.sample_causal_margin(x, sample_size=1)
        draws.append(np.asarray(s.detach().cpu().numpy(), dtype=np.float64)[:, :, 0])
    y0, y1 = (inputs.unscale_y(d) for d in draws)

    # Pairwise filter: dropping rows independently would break the pairing.
    keep = np.isfinite(y0).all(axis=1) & np.isfinite(y1).all(axis=1)
    diag = {
        "mc_n": int(cfg.n_mc),
        "mc_n_used": int(keep.sum()),
        "mc_frac_dropped": float(1.0 - keep.mean()),
        "mc_anynan": bool((~keep).any()),
        "mc_crn": bool(cfg.crn),
        "readout_s": float(time.monotonic() - t0),
    }
    return y0[keep], y1[keep], diag


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def evaluate(cfg: Config, y0: np.ndarray, y1: np.ndarray, data: dict,
             losses: dict, fit_info: dict, mc_diag: dict, timings: dict):
    """tau_hat and every reported metric. Oracle keys are read HERE and only here."""
    tau = y1 - y0
    tau_hat = tau.mean(axis=0)
    truth = np.asarray(data["ATE"])
    support = truth != 0
    n_used = max(1, len(tau))

    design = summarise(data)
    naive = None
    Y, X = np.asarray(data["Y"]), np.asarray(data["X"])[:, 0].astype(bool)
    if X.any() and (~X).any():
        naive = Y[X].mean(axis=0) - Y[~X].mean(axis=0)

    metrics = {
        "status": "ok",
        "method": "frengression",
        "arm": "frengression",
        "conditioner": "n/a",
        "preset": cfg.preset,
        "size": int(cfg.size),
        "radius": int(cfg.effective_radius),
        "digit": cfg.digit,
        "seed_data": int(cfg.seed_data),
        "seed_fit": int(cfg.seed_fit),
        "seed_mc": int(cfg.seed_mc),
        "device": cfg.device,
        "threads": int(cfg.threads),
        "n_units": int(np.asarray(data["Y"]).shape[0]),
        "n_pixels": int(len(truth)),
        "z_dim": int(np.asarray(data["Z"]).shape[1]),
        # ---- recovery: the SAME function every estimator family uses ----
        **score_effect_map(tau_hat, data),
        "frac_pixels_on_support": float(support.mean()),
        "tau_hat_mean_on_support": float(tau_hat[support].mean()) if support.any() else float("nan"),
        "tau_hat_mean_off_support": float(tau_hat[~support].mean()) if (~support).any() else float("nan"),
        "true_effect_on_support": float(truth[support].mean()) if support.any() else float("nan"),
        # ---- design context ----
        "design_naive_bias_mae": float(abs(design["naive_bias_mean"])),
        "design_oracle_ipw_bias_maxabs": design["oracle_ipw_bias_maxabs"],
        "design_att_minus_ate_maxabs": design["att_minus_ate_maxabs"],
        "ate_mae_vs_naive": float(np.abs(naive - truth).mean()) if naive is not None else float("nan"),
        # ---- knobs, so a run folder is readable without the driver ----
        "num_iters": int(cfg.num_iters), "lr": float(cfg.lr),
        "hidden_dim": int(cfg.hidden_dim), "num_layer": int(cfg.num_layer),
        "noise_dim": int(cfg.noise_dim), "y_scaling": cfg.y_scaling,
        "z_scaling": cfg.z_scaling, "y_sd_floor": float(cfg.y_sd_floor),
        **fit_info,
        **mc_diag,
        # Paired vs unpaired side by side: the gap IS the value of the pairing,
        # and an unpaired error of this size would swamp the effect itself.
        "mc_se_max": float((tau.std(axis=0) / np.sqrt(n_used)).max()),
        "mc_se_mean": float((tau.std(axis=0) / np.sqrt(n_used)).mean()),
        "mc_se_unpaired_max": float(
            np.sqrt(y0.var(axis=0) / n_used + y1.var(axis=0) / n_used).max()
        ),
        "mc_tau_sd_on_support": float(tau.std(axis=0)[support].mean()) if support.any() else float("nan"),
        "mc_tau_sd_off_support": float(tau.std(axis=0)[~support].mean()) if (~support).any() else float("nan"),
    }

    u_grid, curves = tau_curve(y0, y1)
    support = data["ATE"] != 0
    true_marg = np.asarray(data["TAU_MARGINAL"])
    curve_err = np.asarray(curves) - true_marg
    flat = np.asarray(curves).std(axis=0)
    metrics.update({
        "tau_u_rmse_vs_marginal": float(np.sqrt((curve_err**2).mean())),
        "tau_u_rmse_on_support": float(np.sqrt((curve_err[:, support] ** 2).mean())),
        "tau_u_sd_on_support": float(flat[support].mean()),
        "tau_u_sd_off_support": float(flat[~support].mean()),
        "true_tau_u_sd_on_support": float(true_marg[:, support].std(axis=0).mean()),
    })

    metrics.update({k: float(v) for k, v in timings.items()})
    if fit_info.get("diverged"):
        metrics["status"] = "diverged"
    extras = {
        "tau_u": np.asarray(u_grid),
        "tau_curves": np.asarray(curves),
        "mc_mean0": y0.mean(axis=0), "mc_mean1": y1.mean(axis=0),
        "mc_var0": y0.var(axis=0), "mc_var1": y1.var(axis=0),
        "mc_tau_sd": tau.std(axis=0),
    }
    return tau_hat, metrics, extras


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def make_plots(cfg: Config, data: dict, losses: dict, tau_hat: np.ndarray,
               plots_dir: str, extras: dict | None = None):
    """Figures named to match the Frugal Flow runner."""
    os.makedirs(plots_dir, exist_ok=True)
    size = cfg.size
    extras = extras or {}
    truth = np.asarray(data["ATE"])
    support = truth != 0

    def sq(v):
        return np.asarray(v).reshape(size, size)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, name), dpi=110)
        plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    vmax = max(np.abs(truth).max(), np.abs(tau_hat).max()) or 1.0
    panels = [(tau_hat, "estimated ATE"), (truth, "true ATE"), (tau_hat - truth, "error")]
    for a, (img, title) in zip(ax, panels):
        im = a.imshow(sq(img), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a.set_title(title, fontsize=10)
        a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle(f"{cfg.preset} | frengression | MAE "
                 f"{np.abs(tau_hat - truth).mean():.4f}", fontsize=10)
    save(fig, "ate_maps.png")

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].scatter(truth, tau_hat, s=18, alpha=0.7)
    lo = float(min(truth.min(), tau_hat.min()))
    hi = float(max(truth.max(), tau_hat.max()))
    ax[0].plot([lo, hi], [lo, hi], "k--", lw=1)
    ax[0].set_xlabel("true ATE")
    ax[0].set_ylabel("estimated")
    ax[0].set_title("per-pixel recovery", fontsize=10)
    names = ["ATE", "ATT", "ATC"]
    vals = [np.abs(tau_hat - np.asarray(data[k])).mean() for k in names]
    ax[1].bar(names, vals, color=["#3b6ea5", "#a5643b", "#5aa53b"])
    ax[1].set_title("MAE against each estimand", fontsize=10)
    save(fig, "recovery_scatter.png")

    if "tau_curves" in extras:
        u = np.asarray(extras["tau_u"])
        curves = np.asarray(extras["tau_curves"])
        true_marg = np.asarray(data["TAU_MARGINAL"])
        fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
        for a, mask, title in ((ax[0], support, "on support"),
                               (ax[1], ~support, "off support")):
            if mask.any():
                a.plot(u, curves[:, mask], color="#3b6ea5", alpha=0.25, lw=0.8)
                a.plot(u, curves[:, mask].mean(axis=1), color="#1b3f66", lw=2,
                       label="estimated (mean)")
                a.plot(u, true_marg[:, mask].mean(axis=1), "k--", lw=2, label="truth (mean)")
            a.set_title(f"tau(u), {title}", fontsize=10)
            a.set_xlabel("u")
        ax[0].legend(fontsize=8)
        save(fig, "tau_curves.png")

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(losses["iter"], losses["loss"], label="total", lw=1)
    ax.plot(losses["iter"], losses["loss_y"], label="loss_y", lw=1)
    ax.plot(losses["iter"], losses["loss_eta"], label="loss_eta", lw=1)
    ax.set_xlabel("iteration")
    ax.set_ylabel("energy loss")
    ax.legend(fontsize=8)
    ax.set_title("training loss (upstream keeps no held-out split)", fontsize=10)
    save(fig, "loss_curves.png")

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    T = np.asarray(data["X"])[:, 0].astype(bool)
    thick = np.asarray(data["THICKNESS"])
    ax[0].hist([thick[T], thick[~T]], bins=25, label=["treated", "control"])
    ax[0].set_title("thickness by arm (confounding)", fontsize=10)
    ax[0].legend(fontsize=8)
    ax[1].hist(np.asarray(data["PROPENSITY"]), bins=25, color="#777")
    ax[1].set_title("true propensity", fontsize=10)
    Ymean = np.asarray(data["Y"]).mean(axis=1)
    ax[2].hist([Ymean[T], Ymean[~T]], bins=25, label=["treated", "control"])
    ax[2].set_title("mean outcome by arm", fontsize=10)
    ax[2].legend(fontsize=8)
    save(fig, "design_check.png")

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    im = ax[0].imshow(sq(truth), cmap="RdBu_r")
    ax[0].set_title("true ATE map", fontsize=10)
    ax[0].axis("off")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    ax[1].hist(np.asarray(data["ITE"]).sum(axis=1), bins=30, color="#777")
    ax[1].set_title("total ITE per unit", fontsize=10)
    im = ax[2].imshow(sq(np.asarray(data["ITE"]).std(axis=0)), cmap="viridis")
    ax[2].set_title("ITE sd across units", fontsize=10)
    ax[2].axis("off")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    save(fig, "truth_panels.png")

# --------------------------------------------------------------------------- #
# run folder
# --------------------------------------------------------------------------- #
def _git_info() -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=SCRIPT_DIR, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return "unavailable"
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def _versions() -> dict:
    def ver(name):
        try:
            return metadata.version(name)
        except Exception:
            return "unavailable"

    commit = "unavailable"
    try:  # the pinned frengression commit lives in its dist-info, not in the module
        for dist in metadata.distributions():
            if (dist.metadata["Name"] or "").lower() == "frengression":
                raw = dist.read_text("direct_url.json")
                if raw:
                    commit = json.loads(raw).get("vcs_info", {}).get("commit_id", "unavailable")
                break
    except Exception:
        pass
    return {"torch": torch.__version__, "numpy": np.__version__,
            "engression": ver("engression"), "frengression": ver("frengression"),
            "frengression_commit": commit}


class _Tee:
    """Duplicate a stream's writes into a file (progress on screen AND in log.txt)."""

    def __init__(self, stream, fileobj):
        self._stream, self._file = stream, fileobj

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
    return (f"{stamp}_{PRESET_SHORT[cfg.preset]}_freng"
            f"_s{cfg.seed_fit}_k{cfg.size ** 2}_{secrets.token_hex(3)}")


def write_config(cfg: Config, run_id: str, run_dir: str, overwrite: bool = False):
    os.makedirs(run_dir, exist_ok=overwrite)
    record = {
        "run_id": run_id, "config": asdict(cfg), "effective_radius": cfg.effective_radius,
        "git": _git_info(), "versions": _versions(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)


def _write_csv(path: str, header: list[str], rows) -> None:
    """Fixed-format CSV: the repro gate compares these byte for byte, so no
    timestamps, no timings, no locale-dependent formatting."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(
                f"{v:.12e}" if isinstance(v, (float, np.floating)) else str(v)
                for v in row) + "\n")


def save_run(cfg: Config, data: dict, losses: dict, tau_hat: np.ndarray,
             metrics: dict, extras: dict,
             run_dir: str, plots: bool = True):
    metrics = {"run_id": os.path.basename(run_dir), **metrics}
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    _write_csv(
        os.path.join(run_dir, "ate_hat.csv"),
        ["pixel", "tau_hat", "ate", "att", "atc"],
        [(k, float(tau_hat[k]), float(np.asarray(data["ATE"])[k]),
          float(np.asarray(data["ATT"])[k]), float(np.asarray(data["ATC"])[k]))
         for k in range(len(tau_hat))],
    )
    _write_csv(
        os.path.join(run_dir, "loss_curve.csv"),
        ["iter", "loss", "loss_y", "loss_eta"],
        [(int(losses["iter"][i]), float(losses["loss"][i]),
          float(losses["loss_y"][i]), float(losses["loss_eta"][i]))
         for i in range(len(losses["iter"]))],
    )
    np.savez(
        os.path.join(run_dir, "arrays.npz"),
        tau_hat=tau_hat, ATE=np.asarray(data["ATE"]), ATT=np.asarray(data["ATT"]),
        ATC=np.asarray(data["ATC"]), TAU_U=np.asarray(data["TAU_U"]),
        TAU_MARGINAL=np.asarray(data["TAU_MARGINAL"]),
        THICKNESS=np.asarray(data["THICKNESS"]), PROPENSITY=np.asarray(data["PROPENSITY"]),
        X=np.asarray(data["X"]), ITE=np.asarray(data["ITE"]),
        # Kept so --replot needs no refit and no regenerated dataset.
        Y=np.asarray(data["Y"]),
        **{k: np.asarray(v) for k, v in losses.items()},
        **{k: np.asarray(v) for k, v in (extras or {}).items()},
    )
    if plots:
        make_plots(cfg, data, losses, tau_hat, os.path.join(run_dir, "plots"), extras)


def replot(run_dir: str):
    """Regenerate the plots for an existing run, no refit."""
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        cfg = Config(**json.load(f)["config"])
    a = np.load(os.path.join(run_dir, "arrays.npz"))
    data = {k: a[k] for k in ("ATE", "ATT", "ATC", "TAU_U", "TAU_MARGINAL",
                              "THICKNESS", "PROPENSITY", "X", "ITE", "Y")}
    losses = {k: a[k] for k in ("iter", "loss", "loss_y", "loss_eta")}
    extras = {k: a[k] for k in ("tau_u", "tau_curves") if k in a}
    make_plots(cfg, data, losses, a["tau_hat"], os.path.join(run_dir, "plots"), extras)
    print(f"replotted: {os.path.join(run_dir, 'plots')}")


def collect(runs_root: str = RUNS_ROOT, csv_path: str | None = None) -> list[dict]:
    rows = []
    for name in sorted(os.listdir(runs_root)) if os.path.isdir(runs_root) else []:
        path = os.path.join(runs_root, name, "metrics.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                rows.append(json.load(f))
    if csv_path and rows:
        keys = sorted({k for r in rows for k in r})
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
        print(f"wrote {csv_path} ({len(rows)} rows)")
    return rows


# Knobs that make two runs the same cell. Presentation-only fields (threads,
# print_every, plots) are excluded; everything that moves a number is in.
CELL_IDENTITY = ("preset", "size", "radius", "digit", "n", "seed_data", "seed_fit",
                 "seed_mc", "num_iters", "lr", "hidden_dim", "num_layer", "noise_dim",
                 "y_scaling", "z_scaling", "n_mc", "crn")


def completed_cells(runs_root: str = RUNS_ROOT) -> set[tuple]:
    """Identity tuples of runs that finished SUCCESSFULLY.

    Unlike the FF runner, a run that finished with a non-finite or divergent
    result does not count as done: it re-runs instead of silently standing as a
    result nobody can use.
    """
    done = set()
    for name in sorted(os.listdir(runs_root)) if os.path.isdir(runs_root) else []:
        run_dir = os.path.join(runs_root, name)
        mpath = os.path.join(run_dir, "metrics.json")
        if not os.path.exists(mpath):
            continue
        try:
            with open(mpath, encoding="utf-8") as f:
                if json.load(f).get("status") != "ok":
                    continue
            with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
                record = json.load(f)
            c = dict(record["config"])
            # The stored config keeps radius as the user gave it (often None);
            # the identity uses the RESOLVED radius, so that a cell run with
            # radius=None matches itself on a later --skip-done pass.
            c["radius"] = record.get("effective_radius", c.get("radius"))
            done.add(tuple(c.get(k) for k in CELL_IDENTITY))
        except (OSError, KeyError, json.JSONDecodeError):
            continue
    return done


def cell_identity(cfg: Config) -> tuple:
    d = asdict(cfg)
    d["radius"] = cfg.effective_radius
    return tuple(d.get(k) for k in CELL_IDENTITY)


# --------------------------------------------------------------------------- #
# experiment tracking
# --------------------------------------------------------------------------- #
def _wandb_start(cfg: Config, run_id: str):
    """Open a W&B run, or attach to the run created by a sweep agent."""
    if not cfg.wandb:
        return None, False
    try:
        import wandb
    except ImportError:
        print("  NOTE: --wandb requested but wandb is not installed; continuing without it")
        return None, False

    if wandb.run is not None:
        return wandb.run, False

    extra = [tag.strip() for tag in (cfg.wandb_tags or "").split(",") if tag.strip()]
    run = wandb.init(
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        group=cfg.wandb_group or cfg.preset,
        job_type="frengression",
        name=run_id,
        tags=[cfg.preset, "frengression", f"k{cfg.size ** 2}"] + extra,
        config={
            **asdict(cfg),
            "effective_radius": cfg.effective_radius,
            "n_pixels": cfg.size ** 2,
            **{f"git_{key}": value for key, value in _git_info().items()},
        },
        reinit=True,
    )
    return run, True


def _numeric(values: dict) -> dict:
    return {
        key: value for key, value in values.items()
        if isinstance(value, (int, float, bool))
    }


def _wandb_log(run, data: dict, losses: dict, metrics: dict, run_dir: str):
    """Log method-specific losses and the final common scientific metrics."""
    import wandb

    for i in range(len(losses["iter"])):
        run.log(
            {
                "fit/loss": float(losses["loss"][i]),
                "fit/loss_y": float(losses["loss_y"][i]),
                "fit/loss_eta": float(losses["loss_eta"][i]),
            },
            step=int(losses["iter"][i]),
        )

    final_step = int(losses["iter"][-1]) + 1
    payload = _numeric(metrics)
    payload.update({f"design/{k}": v for k, v in _numeric(summarise(data)).items()})
    for name in ("ate_maps", "recovery_scatter", "tau_curves", "loss_curves",
                 "design_check", "truth_panels"):
        path = os.path.join(run_dir, "plots", f"{name}.png")
        if os.path.exists(path):
            payload[f"plots/{name}"] = wandb.Image(path)
    run.log(payload, step=final_step)
    run.summary.update(_numeric(metrics))


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #
def run_one(cfg: Config, runs_root: str | None = None, run_dir: str | None = None,
            overwrite: bool = False, plots: bool = True) -> dict:
    """Build, fit, sample, score and archive one cell."""
    if run_dir is not None:
        run_dir = os.path.abspath(run_dir)
        run_id = os.path.basename(os.path.normpath(run_dir))
        if os.path.exists(run_dir) and overwrite:
            # --overwrite deletes a directory tree, so it may only ever be
            # pointed at something this script wrote. Without this,
            # `--run-dir . --overwrite` would remove the working directory.
            if not os.path.exists(os.path.join(run_dir, "config.json")):
                raise ValueError(
                    f"refusing to --overwrite {run_dir}: it exists but holds no "
                    "config.json, so it is not a run folder this script created"
                )
            if os.path.commonpath([run_dir, os.getcwd()]) == run_dir:
                raise ValueError(f"refusing to --overwrite {run_dir}: it contains the CWD")
            shutil.rmtree(run_dir)
    else:
        run_id = run_id_for(cfg)
        run_dir = os.path.join(runs_root or RUNS_ROOT, run_id)
    write_config(cfg, run_id, run_dir, overwrite=overwrite)
    print(f"run dir: {run_dir}")
    wb, owned = _wandb_start(cfg, run_id)

    try:
        return _run_one_inner(cfg, run_id, run_dir, plots, wb)
    finally:
        if wb is not None and owned:
            wb.finish()


def _run_one_inner(cfg: Config, run_id: str, run_dir: str, plots: bool, wb) -> dict:
    """Run one prepared cell; split out so W&B ownership is always released."""

    with open(os.path.join(run_dir, "log.txt"), "w", encoding="utf-8") as lf:
        out, err = _Tee(sys.stdout, lf), _Tee(sys.stderr, lf)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            print(f"preset: {cfg.preset} | frengression | device {cfg.device} | "
                  f"threads {cfg.threads}")
            timings, t_start = {}, time.monotonic()
            t0 = time.monotonic()
            data = build_data(cfg)
            timings["build_data_s"] = time.monotonic() - t0
            guard = OracleGuard(data)
            inputs = prepare_inputs(guard, cfg)
            print(f"data: {inputs.n} units x {inputs.K} pixels (size={cfg.size}, "
                  f"radius={cfg.effective_radius}), z_dim={inputs.z_dim}, "
                  f"y sd(global)={inputs.y_sd_global:.4f}")

            model = build_model(cfg, inputs)
            losses, fit_info = fit(cfg, model, inputs, out_stream=out)
            timings["fit_s"] = fit_info["fit_s"]
            timings["s_per_iter"] = fit_info["fit_s"] / max(1, fit_info["n_iters_run"])
            print(f"fit: {fit_info['n_iters_run']} iters in {fit_info['fit_s']:.0f}s "
                  f"(loss {fit_info['loss_first']:.4f} -> {fit_info['loss_final']:.4f})")

            y0, y1, mc_diag = sample_margins(cfg, model, inputs)
            if len(y0) == 0:
                metrics = {"run_id": run_id, "status": "failed_nonfinite",
                           "method": "frengression", "arm": "frengression",
                           "conditioner": "n/a", "preset": cfg.preset,
                           "seed_data": cfg.seed_data, "seed_fit": cfg.seed_fit,
                           **fit_info, **mc_diag}
                with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)
                print("FAILED: every interventional draw was non-finite")
                return metrics
            if mc_diag["mc_frac_dropped"]:
                print(f"  WARNING: dropped {mc_diag['mc_frac_dropped']:.3%} non-finite "
                      f"draw pairs before the read-out")

            t0 = time.monotonic()
            tau_hat, metrics, extras = evaluate(cfg, y0, y1, data, losses, fit_info,
                                                mc_diag, timings)
            timings["eval_s"] = time.monotonic() - t0
            metrics["eval_s"] = float(timings["eval_s"])
            metrics["wall_time_s"] = float(time.monotonic() - t_start)
            save_run(cfg, data, losses, tau_hat, metrics, extras, run_dir, plots=plots)
            metrics["total_s"] = float(time.monotonic() - t_start)
            metrics = {"run_id": run_id, **metrics}
            with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            if wb is not None:
                _wandb_log(wb, data, losses, metrics, run_dir)
            for k in ("status", "ate_mae", "ate_mae_on_support", "ate_mae_off_support",
                      "ate_corr", "mc_se_max", "mc_se_unpaired_max",
                      "loss_final", "total_s"):
                v = metrics.get(k)
                print(f"  {k}: {v:.4g}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"run dir: {run_dir}")
    return metrics


def run_sweep(base: Config, presets=None, seeds=(0,), skip_done: bool = False,
              runs_root: str | None = None, plots: bool = True) -> list[dict]:
    """presets x seeds. A seed sets BOTH seed_data and seed_fit."""
    rows = []
    root = runs_root or RUNS_ROOT
    done = completed_cells(root) if skip_done else set()
    presets = list(presets or PRESETS)
    cells = [(p, s) for s in seeds for p in presets]
    for i, (preset, seed) in enumerate(cells, 1):
        print(f"\n=== [{i}/{len(cells)}] {preset} | seed {seed} ===")
        try:
            cfg = Config(**{**asdict(base), "preset": preset,
                            "seed_data": seed, "seed_fit": seed})
            if cell_identity(cfg) in done:
                print("  already completed with status ok, skipping (--skip-done)")
                continue
            rows.append(run_one(cfg, runs_root=root, plots=plots))
        except Exception as exc:  # noqa: BLE001 - one bad cell must not kill the sweep
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            rows.append({"preset": preset, "seed_data": seed, "status": "error",
                         "error": f"{type(exc).__name__}: {exc}"})
    return rows


def print_table(rows: list[dict]):
    cols = ["preset", "seed_data", "status", "ate_mae", "ate_mae_on_support",
            "ate_corr", "mc_se_max", "total_s"]
    widths = [max(len(c), 22 if c == "preset" else 12) for c in cols]
    print("\n" + "".join(c.ljust(w) if c == "preset" else c.rjust(w)
                         for c, w in zip(cols, widths)))
    for r in rows:
        line = ""
        for c, w in zip(cols, widths):
            v = r.get(c, "-")
            s = f"{v:.4f}" if isinstance(v, float) else str(v)
            line += s.ljust(w) if c == "preset" else s.rjust(w)
        print(line)


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #
def selftest() -> int:
    """Correctness checks, not a recovery check: a 20-iteration fit estimates
    nothing. Writes only into a temporary directory, which it removes."""
    checks, failed = [], []

    def check(label, ok, note=""):
        checks.append((label, bool(ok), note))
        if not ok:
            failed.append(label)

    tmp = tempfile.mkdtemp(prefix="frengselftest-")
    try:
        for bad, kw in (("preset", {"preset": "nope"}), ("y_scaling", {"y_scaling": "nope"}),
                        ("z_scaling", {"z_scaling": "nope"}), ("device", {"device": "tpu"}),
                        ("n_mc", {"n_mc": 1}), ("threads", {"threads": 0})):
            try:
                Config(**kw)
                check(f"guard:{bad}", False, "invalid config accepted")
            except ValueError:
                check(f"guard:{bad}", True)

        for size, want in ((4, 1), (8, 2), (16, 4)):
            check(f"effective_radius:size{size}",
                  Config(size=size).effective_radius == want,
                  f"got {Config(size=size).effective_radius}, want {want}")

        cfg = Config(preset="exp2_confounded_homogeneous", size=4, n=300, num_iters=20,
                     n_mc=800, seed_data=0, seed_fit=0, seed_mc=0, threads=1,
                     print_every=1000)
        data = build_data(cfg)
        check("dataset:radius", int(data["config"]["radius"]) == 1,
              f"radius {data['config']['radius']}")
        guard = OracleGuard(data)
        try:
            guard["ATE"]
            check("guard:oracle_blocked", False, "ATE was readable through the guard")
        except KeyError:
            check("guard:oracle_blocked", True)
        check("guard:observables_ok", guard["Y"] is not None and guard["X"] is not None)

        inputs = prepare_inputs(guard, cfg)
        check("guard:touched_allowlist", guard.touched <= set(MODEL_INPUT_KEYS),
              f"touched {sorted(guard.touched)}")
        Y = np.asarray(data["Y"], dtype=np.float64)
        back = inputs.unscale_y((Y - inputs.y_mean) / inputs.y_scale)
        check("scaling:roundtrip", np.allclose(back, Y, atol=1e-6),
              f"max err {np.abs(back - Y).max():.2e}")
        check("inputs:dtype", inputs.y.dtype == torch.float32)

        m = run_one(cfg, run_dir=os.path.join(tmp, "cell"), overwrite=True, plots=True)
        check("run:status_ok", m.get("status") == "ok", str(m.get("status")))
        for f in ("config.json", "metrics.json", "ate_hat.csv", "loss_curve.csv",
                  "arrays.npz", "log.txt"):
            check(f"artefact:{f}", os.path.exists(os.path.join(tmp, "cell", f)))
        check("artefact:plots", len(os.listdir(os.path.join(tmp, "cell", "plots"))) >= 6)
        check("metrics:score_keys",
              all(k in m for k in ("ate_mae", "ate_rmse", "ate_corr", "att_mae", "atc_mae")))
        check("metrics:mc_pairing", m["mc_se_max"] < m["mc_se_unpaired_max"],
              f"paired {m['mc_se_max']:.5f} vs unpaired {m['mc_se_unpaired_max']:.5f}")
        check("metrics:tau_u", np.isfinite(m["tau_u_rmse_vs_marginal"]))

        replot(os.path.join(tmp, "cell"))
        check("replot:ok", True)

        m2 = run_one(cfg, run_dir=os.path.join(tmp, "cell2"), overwrite=True, plots=False)
        a = open(os.path.join(tmp, "cell", "ate_hat.csv"), "rb").read()
        b = open(os.path.join(tmp, "cell2", "ate_hat.csv"), "rb").read()
        check("determinism:ate_hat_csv", a == b, "identical seeds gave different CSVs")
        check("determinism:metrics", m["ate_mae"] == m2["ate_mae"])

        rows = collect(tmp)
        check("collect:rows", len(rows) == 2, f"{len(rows)} rows")
        done = completed_cells(tmp)
        check("skipdone:identity", cell_identity(cfg) in done)

        cfgd = Config(**{**asdict(cfg), "seed_fit": 999})
        check("skipdone:other_cell", cell_identity(cfgd) not in done)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    for label, ok, note in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  -- {note}" if note and not ok else ""))
    print(f"\n{sum(1 for _, ok, _ in checks if ok)}/{len(checks)} checks passed")
    if failed:
        print("failed:")
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
    choices = {"preset": list(PRESETS), "y_scaling": list(Y_SCALINGS),
               "z_scaling": list(Z_SCALINGS), "device": list(DEVICES)}
    for f in fields(Config):
        flag = "--" + f.name.replace("_", "-")
        kw = {"choices": choices[f.name]} if f.name in choices else {}
        if f.type == "bool":
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        elif f.type.endswith("| None"):
            base = f.type.split("|")[0].strip()
            parser.add_argument(flag, type={"int": int, "float": float, "str": str}[base],
                                default=f.default, **kw)
        else:
            parser.add_argument(flag, type=type(f.default), default=f.default, **kw)
    parser.add_argument("--all-digits", action="store_true", help="use all ten digit classes")
    parser.add_argument("--sweep", action="store_true", help="run presets x seeds")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="sweep: each seed sets BOTH seed_data and seed_fit")
    parser.add_argument("--presets", nargs="+", default=None, choices=list(PRESETS))
    parser.add_argument("--skip-done", action="store_true",
                        help="sweep: skip cells already completed with status ok")
    parser.add_argument("--runs-root", default=None, help="archive root (default: runs/frengression)")
    parser.add_argument("--run-dir", default=None,
                        help="write this exact folder (relative to CWD); no stamp, no suffix")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing --run-dir")
    parser.add_argument("--no-plots", action="store_true", help="skip figure generation")
    parser.add_argument("--collect", action="store_true", help="tabulate completed runs and exit")
    parser.add_argument("--csv", default=None, help="collect: also write this CSV")
    parser.add_argument("--replot", metavar="RUN_DIR", default=None)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)

    if args.selftest:
        raise SystemExit(selftest())
    if args.replot is not None:
        replot(args.replot)
        return
    root = args.runs_root or RUNS_ROOT
    if args.collect:
        rows = collect(root, args.csv)
        print_table(rows) if rows else print(f"no completed runs under {root}")
        return

    cfg = Config(**{f.name: getattr(args, f.name) for f in fields(Config)})
    if args.all_digits:
        cfg = Config(**{**asdict(cfg), "digit": None})

    if args.sweep:
        rows = run_sweep(cfg, presets=args.presets, seeds=args.seeds or [cfg.seed_data],
                         skip_done=args.skip_done, runs_root=root, plots=not args.no_plots)
        print(f"\n=== sweep complete: {len(rows)} cells ===")
        print_table(rows)
    else:
        os.makedirs(root, exist_ok=True)
        run_one(cfg, runs_root=root, run_dir=args.run_dir, overwrite=args.overwrite,
                plots=not args.no_plots)


if __name__ == "__main__":
    main()
