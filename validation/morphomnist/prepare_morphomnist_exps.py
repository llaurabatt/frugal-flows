"""MorphoMNIST experiment generators with an EXACTLY known per-pixel ATE.

One parameterised DGP covers the whole experiment ladder; the three experiments
differ only in knob settings (see ``PRESETS``):

    exp1  RCT,        homogeneous     ps_slope=0   a_cov=0    b_quant=0
    exp2  confounded, homogeneous     ps_slope>0   a_cov=0    b_quant=0
    exp3  confounded, heterogeneous   ps_slope>0   a_cov>0    b_quant>0

so 1 -> 2 isolates confounding and 2 -> 3 isolates heterogeneity, with nothing
else moving. This module is standalone: it reuses only the pure transforms from
``prepare_data`` (downsampling, dequantise/logit) and does its own treatment
assignment and effect construction. ``prepare_data.prepare_flow_data`` is left
exactly as it is.


The effect
----------
Everything happens in LOGIT space (the space the flow is fitted in), where the
treatment effect is additive:

    tau[i, k] = m[k] * (1 + a * h[i] + b * g[i, k])
    Y1        = Y0 + tau

    m[k]    fixed spatial map (a circle by default) -- the effect a homogeneous
            unit would get. This is the quantity we are trying to recover.
    h[i]    COVARIATE modulation: the unit's thickness, mapped to its own rank
            and rescaled to (-1, 1). Makes CATE vary with Z.
    g[i, k] QUANTILE modulation: pixel k's own rank within unit i's untreated
            value Y0[:, k], rescaled to (-1, 1). Makes tau vary with the
            outcome's latent quantile u.
    a, b    the two heterogeneity strengths.

Both modulators are built from RANKS, so each has sample mean EXACTLY zero and
lives in (-1, 1). Two consequences:

  1. the per-pixel sample ATE is exactly the map,

         ATE[k] = mean_i tau[i, k] = m[k] * (1 + a*mean(h) + b*mean(g[:, k]))
                = m[k] * (1 + 0 + 0) = m[k]

     with no Monte-Carlo error -- it is a knob you SET, not a number you
     measure. ``build_experiment`` asserts this.

  2. for a + b < 1 the bracket stays strictly positive, so no unit's effect
     flips sign relative to the map. Above 1 it can, which is a legitimate DGP
     but a different one; the summary reports ``frac_factor_negative``.

Beware the estimand you are comparing against: tau_hat from the flow is in
LOGIT space, so score it against ``ATE`` here, never against pixel-space
differences (the inverse logit is nonlinear and not estimand-preserving).


What the truth block contains
-----------------------------
``ATE`` is the marginal estimand. Because treatment is confounded AND the
effect is heterogeneous, ``ATT`` and ``ATC`` genuinely differ from it, and both
are computed exactly -- exp3 is therefore the experiment that can tell an ATE
estimator apart from an ATT estimator, which exp1 and exp2 structurally cannot.

Two quantile-resolved truths are emitted, because ``frugal_flows.interventions.
tau_curve`` and the DGP are not the same functional once a > 0:

    TAU_PAIRED    bins the true paired difference tau[i, k] by the rank of
                  Y0[i, k] -- i.e. the same computation tau_curve performs, run
                  on the true potential outcomes. Reads as "average ITE at
                  outcome rank u".
    TAU_MARGINAL  Q1(u) - Q0(u), the two margins sorted independently. This is
                  what a flow whose causal margin is monotone in u can actually
                  represent, so it is the honest estimation target.

With a = 0 the DGP is rank-preserving and the two coincide. With a > 0 two units
at the same outcome rank get different shifts depending on thickness, so they
separate; ``tau_curve_gap`` in the summary measures that separation. Both curves
average over u to ``ATE`` (exactly when n divides evenly into the bins, else to
within one bin's worth -- ~1e-4 at n ~ 6000).


Usage
-----
    from prepare_morphomnist_exps import build_preset, summarise

    data = build_preset("exp3_confounded_heterogeneous")
    data["ATE"]                    # (K,) exact per-pixel truth
    summarise(data)                # dict of design diagnostics

    python prepare_morphomnist_exps.py --list
    python prepare_morphomnist_exps.py --preset exp2_confounded_homogeneous
    python prepare_morphomnist_exps.py --preset exp3_confounded_heterogeneous \
        --a-cov 0.7 --save exp3.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, fields

import jax.numpy as jnp
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)  # prepare_data / dataset are siblings, not a package

from dataset import MorphoMNIST
from prepare_data import dequantize_and_logit, downsample, inverse_logit  # noqa: F401

TAU_CURVE_BINS = 40  # matches frugal_flows.interventions.TAU_CURVE_BINS

EFFECT_MAPS = ("circle", "ring", "const", "gradient")


@dataclass
class ExpConfig:
    """Every knob of the DGP. Recorded verbatim in the returned ``config``."""

    # ---- data ----
    size: int = 8                 # image side; K = size^2 outcome dims
    digit: int | None = 0         # single digit class, or None for all ten
    n: int | None = 10000         # cap on sample size (None = use everything)
    seed: int = 0
    split: str = "train"
    data_dir: str = "data"        # relative to this file

    # ---- spatial effect map m[k] ----
    effect: str = "circle"
    radius: int = 2               # circle/ring radius in pixels, centred
    base_shift: float = 1.0       # effect magnitude in logit space

    # ---- heterogeneity ----
    a_cov: float = 0.0            # covariate (thickness) modulation strength
    b_quant: float = 0.0          # quantile (own outcome rank) modulation strength

    # ---- treatment assignment: p = sigmoid(intercept + slope * zscore(thickness)) ----
    ps_intercept: float = 0.0     # 0 -> ~50% treated
    ps_slope: float = 0.0         # 0 -> RCT; >0 -> confounded through thickness

    def __post_init__(self):
        if self.effect not in EFFECT_MAPS:
            raise ValueError(f"unknown effect {self.effect!r}; choose from {EFFECT_MAPS}")
        if self.digit is not None and not 0 <= self.digit <= 9:
            raise ValueError(f"digit must be in 0..9 or None, got {self.digit}")
        if self.a_cov < 0 or self.b_quant < 0:
            raise ValueError("a_cov and b_quant must be non-negative")


PRESETS: dict[str, ExpConfig] = {
    # The toy: random assignment, one fixed effect map for everyone.
    "exp1_rct_homogeneous": ExpConfig(ps_slope=0.0, a_cov=0.0, b_quant=0.0),
    # Same effect, thickness now drives assignment. Isolates confounding.
    "exp2_confounded_homogeneous": ExpConfig(ps_slope=1.2, a_cov=0.0, b_quant=0.0),
    # Same confounding, effect now varies with the covariate AND the quantile.
    # a + b = 0.9 < 1, so no unit's effect flips sign.
    "exp3_confounded_heterogeneous": ExpConfig(ps_slope=1.2, a_cov=0.5, b_quant=0.4),
}


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
def spatial_map(cfg: ExpConfig) -> np.ndarray:
    """The per-pixel effect map ``m``, shape ``(K,)`` -- the ATE we are imposing."""
    size = cfg.size
    xx, yy = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    c = (size - 1) / 2
    r2 = (xx - c) ** 2 + (yy - c) ** 2
    if cfg.effect == "circle":
        m = (r2 <= cfg.radius**2).astype(np.float64)
    elif cfg.effect == "ring":
        m = ((r2 <= cfg.radius**2) & (r2 > (cfg.radius - 1) ** 2)).astype(np.float64)
    elif cfg.effect == "const":
        m = np.ones_like(r2, dtype=np.float64)
    else:  # "gradient" -- a smoothly varying map, harder than a binary mask
        m = (xx / max(size - 1, 1)).astype(np.float64)
    return cfg.base_shift * m.reshape(-1)


def rank_uniform(a: np.ndarray, axis: int = 0) -> np.ndarray:
    """Ranks of ``a`` along ``axis``, mapped to ``(0, 1)`` with mean EXACTLY 0.5.

    Using ``(rank + 0.5) / n`` rather than an empirical CDF guarantees the mean
    is 0.5 for any n, which is what makes the imposed ATE exact rather than
    approximate. Ties are broken arbitrarily; the outcomes here are dequantised
    and logit-transformed, so exact ties are measure-zero anyway.
    """
    n = a.shape[axis]
    order = np.argsort(a, axis=axis, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(n).reshape(
        [-1 if i == axis else 1 for i in range(a.ndim)]
    ), axis=axis)
    return (ranks + 0.5) / n


def _centred(u: np.ndarray) -> np.ndarray:
    """Rank scores in (0,1) -> modulator in (-1,1) with sample mean exactly 0."""
    return 2.0 * u - 1.0


def true_tau_curves(Y0: np.ndarray, Y1: np.ndarray, n_bins: int = TAU_CURVE_BINS):
    """Quantile-resolved truth, both functionals. See the module docstring.

    Returns ``(u_centers, tau_paired, tau_marginal)`` with the two curve arrays
    shaped ``(n_bins, K)``. ``u_centers`` is the same grid ``tau_curve`` uses, so
    estimated and true curves overlay without interpolation.
    """
    n, K = Y0.shape
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    u_centers = (np.arange(n_bins) + 0.5) / n_bins

    # paired: the true ITE, binned by the rank of the unit's own control outcome
    order = np.argsort(Y0, axis=0, kind="stable")
    tau_sorted = np.take_along_axis(Y1 - Y0, order, axis=0)
    tau_paired = np.array([tau_sorted[a:b].mean(axis=0) for a, b in zip(edges[:-1], edges[1:])])

    # marginal: Q1(u) - Q0(u), each margin sorted on its own
    q0 = np.sort(Y0, axis=0)
    q1 = np.sort(Y1, axis=0)
    tau_marginal = np.array(
        [q1[a:b].mean(axis=0) - q0[a:b].mean(axis=0) for a, b in zip(edges[:-1], edges[1:])]
    )
    return u_centers, tau_paired, tau_marginal


# --------------------------------------------------------------------------- #
# the generator
# --------------------------------------------------------------------------- #
def build_experiment(cfg: ExpConfig) -> dict:
    """Build one experiment. Model inputs are jnp; the truth block is numpy.

    The split is deliberate: anything the model is allowed to touch is a jnp
    array under a short key (``Y``, ``X``, ``Z``, ``u_z`` inputs); everything
    under an UPPERCASE truth key is numpy and is for evaluation only.
    """
    rng = np.random.default_rng(cfg.seed)
    ds = MorphoMNIST(os.path.join(SCRIPT_DIR, cfg.data_dir), split=cfg.split)

    # ---- subset: single digit class (default) or all ten ----
    digit_onehot = ds.digit.numpy().astype(np.float64)
    labels = digit_onehot.argmax(axis=1)
    pool = np.arange(len(ds)) if cfg.digit is None else np.where(labels == cfg.digit)[0]
    pool = rng.permutation(pool)
    if cfg.n is not None:
        pool = pool[: cfg.n]
    n = len(pool)
    if n < 2 * TAU_CURVE_BINS:
        raise ValueError(f"only {n} units after subsetting -- too few for {TAU_CURVE_BINS} bins")

    # ---- untreated potential outcome, in logit space ----
    images = downsample(ds.images[pool], size=cfg.size)
    Y0 = dequantize_and_logit(images.reshape(n, -1).numpy().astype(np.float64), rng)
    K = Y0.shape[1]

    # ---- covariates ----
    thickness = ds.thickness[pool].numpy().astype(np.float64)  # already in [-1, 1]
    if cfg.digit is None:
        Z = np.hstack([thickness[:, None], digit_onehot[pool]])
        z_cat_idx = np.zeros(Z.shape[1], dtype=bool)
        z_cat_idx[1:] = True
    else:
        # a constant one-hot carries no information and breaks the discrete
        # quantile stage, so a single-digit subset gets thickness only
        Z = thickness[:, None]
        z_cat_idx = np.zeros(1, dtype=bool)

    # ---- the effect ----
    m = spatial_map(cfg)                                  # (K,)
    h = _centred(rank_uniform(thickness))[:, None]        # (n, 1)  covariate modulation
    g = _centred(rank_uniform(Y0, axis=0))                # (n, K)  quantile modulation
    factor = 1.0 + cfg.a_cov * h + cfg.b_quant * g        # (n, K)
    ITE = m[None, :] * factor
    Y1 = Y0 + ITE

    # The whole point of rank-based modulators: this holds to machine precision.
    ate = ITE.mean(axis=0)
    assert np.allclose(ate, m, atol=1e-9), (
        f"imposed ATE drifted from the map by {np.abs(ate - m).max():.3g} -- "
        "the rank modulators are no longer mean-zero"
    )

    # ---- treatment assignment ----
    z = (thickness - thickness.mean()) / thickness.std()
    p = 1.0 / (1.0 + np.exp(-(cfg.ps_intercept + cfg.ps_slope * z)))
    T = (rng.uniform(size=n) < p).astype(np.float64)[:, None]
    Y = np.where(T == 1, Y1, Y0)

    treated = T[:, 0].astype(bool)
    u_grid, tau_paired, tau_marginal = true_tau_curves(Y0, Y1)

    return {
        # ---- model inputs: the ONLY things training may see ----
        "Y": jnp.asarray(Y),
        "X": jnp.asarray(T),
        "Z": jnp.asarray(Z),
        "z_cont": jnp.asarray(Z[:, ~z_cat_idx]),
        "z_discr": jnp.asarray(Z[:, z_cat_idx]),
        "z_cat_idx": z_cat_idx,
        # ---- ground truth: evaluation only ----
        "ATE": m,                                   # (K,) exact, == the imposed map
        "ATT": ITE[treated].mean(axis=0),           # (K,) exact
        "ATC": ITE[~treated].mean(axis=0),          # (K,) exact
        "ITE": ITE,
        "Y0": Y0,
        "Y1": Y1,
        "MAP": m,
        "FACTOR": factor,
        "TAU_U": u_grid,
        "TAU_PAIRED": tau_paired,                   # (n_bins, K)
        "TAU_MARGINAL": tau_marginal,               # (n_bins, K)
        "PROPENSITY": p,
        "THICKNESS": thickness,
        # ---- provenance ----
        "config": asdict(cfg),
        "image_size": cfg.size,
        "n_units": n,
        "n_pixels": K,
    }


def build_preset(name: str, **overrides) -> dict:
    """``build_experiment`` on a named preset, with optional knob overrides."""
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {list(PRESETS)}")
    cfg = ExpConfig(**{**asdict(PRESETS[name]), **overrides})
    data = build_experiment(cfg)
    data["preset"] = name
    return data


# --------------------------------------------------------------------------- #
# design diagnostics
# --------------------------------------------------------------------------- #
def summarise(data: dict) -> dict:
    """Design diagnostics: is the confounding real, is the truth recoverable?

    ``naive_bias`` is what an unadjusted difference in means gets wrong, and
    ``oracle_ipw_bias`` is what remains after weighting by the TRUE propensity.
    The first should be large (otherwise the experiment has no confounding to
    undo) and the second ~0 (otherwise the design is not identified and no
    estimator could succeed).
    """
    Y = np.asarray(data["Y"], dtype=np.float64)
    T = np.asarray(data["X"], dtype=np.float64)[:, 0].astype(bool)
    ATE, ITE, p = data["ATE"], data["ITE"], data["PROPENSITY"]
    n = len(Y)

    naive = Y[T].mean(axis=0) - Y[~T].mean(axis=0)
    w = np.where(T, 1.0 / p, 1.0 / (1.0 - p))[:, None]
    ipw = (Y * w * T[:, None]).sum(0) / (w * T[:, None]).sum(0) - (
        Y * w * ~T[:, None]
    ).sum(0) / (w * ~T[:, None]).sum(0)

    gap = data["TAU_PAIRED"] - data["TAU_MARGINAL"]
    return {
        "preset": data.get("preset", "custom"),
        "n_units": n,
        "n_pixels": data["n_pixels"],
        "treated_frac": float(T.mean()),
        "propensity_min": float(p.min()),
        "propensity_max": float(p.max()),
        "corr_thickness_treatment": float(np.corrcoef(data["THICKNESS"], T)[0, 1]),
        # the imposed truth
        "ate_mean": float(ATE.mean()),
        "ate_max": float(ATE.max()),
        "ate_exactness": float(np.abs(ITE.mean(axis=0) - ATE).max()),  # ~1e-16
        "att_minus_ate_maxabs": float(np.abs(data["ATT"] - ATE).max()),
        "atc_minus_ate_maxabs": float(np.abs(data["ATC"] - ATE).max()),
        # heterogeneity
        "ite_sd_across_units": float(ITE.std(axis=0).mean()),
        "factor_min": float(data["FACTOR"].min()),
        "frac_factor_negative": float((data["FACTOR"] < 0).mean()),
        "tau_curve_gap": float(np.abs(gap).max()),  # 0 iff rank-preserving
        # is the design confounded, and is it identified?
        "naive_bias_mean": float((naive - ATE).mean()),
        "naive_bias_maxabs": float(np.abs(naive - ATE).max()),
        "oracle_ipw_bias_mean": float((ipw - ATE).mean()),
        "oracle_ipw_bias_maxabs": float(np.abs(ipw - ATE).max()),
    }


def save_npz(data: dict, path: str):
    """Write every array to ``path``; the config rides along as a JSON string."""
    arrays = {
        k: np.asarray(v)
        for k, v in data.items()
        if hasattr(v, "shape") or isinstance(v, (int, float))
    }
    np.savez(path, config_json=json.dumps(data["config"]),
             preset=data.get("preset", "custom"), **arrays)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--preset", default="exp1_rct_homogeneous", choices=list(PRESETS))
    parser.add_argument("--list", action="store_true", help="print the presets and exit")
    parser.add_argument("--save", metavar="PATH", default=None, help="write an .npz")
    parser.add_argument("--all-digits", action="store_true",
                        help="use all ten digit classes (digit=None), not a single class")
    for f in fields(ExpConfig):
        flag = "--" + f.name.replace("_", "-")
        if f.name in ("digit", "n"):
            parser.add_argument(flag, type=int, default=None)  # None => keep the preset's
        elif f.type == "str":
            parser.add_argument(flag, type=str, default=None)
        else:
            parser.add_argument(flag, type=type(f.default), default=None)
    args = parser.parse_args(argv)

    if args.list:
        for name, cfg in PRESETS.items():
            print(f"{name}:")
            for k, v in asdict(cfg).items():
                print(f"    {k}: {v}")
        return

    overrides = {
        f.name: getattr(args, f.name)
        for f in fields(ExpConfig)
        if getattr(args, f.name) is not None
    }
    if args.all_digits:
        overrides["digit"] = None
    data = build_preset(args.preset, **overrides)

    print(f"preset: {args.preset}")
    for k, v in data["config"].items():
        print(f"    {k}: {v}")
    print()
    for k, v in summarise(data).items():
        print(f"  {k}: {v:.6g}" if isinstance(v, float) else f"  {k}: {v}")

    if args.save:
        save_npz(data, args.save)
        print(f"\nsaved: {args.save}")


if __name__ == "__main__":
    main()
