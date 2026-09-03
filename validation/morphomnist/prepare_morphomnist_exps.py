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
    h[i]    COVARIATE modulation: the unit's thickness, mapped to its own rank,
            passed through ``h_shape`` and rescaled to [-1, 1]. Makes CATE vary
            with Z. ``h_shape`` controls HOW: "linear" gives a CATE linear in
            the thickness rank, while "cubic"/"quadratic"/"sine"/"step" give
            non-trivial and (for the last three) NON-MONOTONE CATE functions.
            The exact ATE survives all of them -- see below.
    g[i, k] QUANTILE modulation: pixel k's own rank within unit i's untreated
            value Y0[:, k], rescaled to (-1, 1). Makes tau vary with the
            outcome's latent quantile u.
    a, b    the two heterogeneity strengths.

Both modulators are built from RANKS and then have their EMPIRICAL mean
subtracted, so each has sample mean exactly zero and lives in [-1, 1].

The centering is what does the work, NOT the linearity: subtracting the observed
mean rather than an analytic one makes the mean exactly zero for any shaping
function whatsoever. That is why ``h_shape`` can be an arbitrary (even
discontinuous, even non-monotone) function of the rank without disturbing
anything below.

Two consequences:

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

# How the per-unit effect is constructed. They differ in WHAT the effect is
# allowed to depend on, which is the difference between "treatment effect
# heterogeneity" and "a coupling between the potential outcomes".
EFFECT_MODES = (
    "outcome_coupled",     # tau depends on covariates AND on the unit's own Y(0)
    "covariate_only",      # tau depends on OBSERVED COVARIATES ONLY -- no Y(0)
    "quantile_primitive",  # delta_k(u) is specified directly; Y1 is built from it
)

# Spatial shape of the covariate modulation: makes the covariate's influence on
# the effect vary across pixels, so units get differently-SHAPED effects rather
# than the same map rescaled. Works in every mode except quantile_primitive.
SPATIAL_BASES = ("none", "gradient_x", "gradient_y", "diagonal", "radial")

# Shape of the covariate modulation, applied to the rank score u in (0,1).
# The imposed ATE stays exact for ANY of these -- see `_modulator`. Defined here
# rather than beside `_modulator` because ``ExpConfig.__post_init__`` validates
# against it, and PRESETS instantiates ExpConfig at import time.
H_SHAPES = {
    "linear":    lambda u: 2.0 * u - 1.0,            # monotone, constant slope
    "cubic":     lambda u: (2.0 * u - 1.0) ** 3,     # monotone, flat middle, steep tails
    "quadratic": lambda u: (2.0 * u - 1.0) ** 2,     # NON-monotone, U-shaped
    "sine":      lambda u: np.sin(2.0 * np.pi * u),  # NON-monotone, oscillating
    "step":      lambda u: (u > 0.5).astype(float),  # discontinuous, two groups
}


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

    # ---- how the effect is built; see EFFECT_MODES and the module docstring ----
    effect_mode: str = "outcome_coupled"

    # ---- heterogeneity ----
    a_cov: float = 0.0            # covariate (thickness) modulation strength
    a_bright: float = 0.0         # brightness modulation   (covariate_only)
    a_inter: float = 0.0          # thickness x brightness  (covariate_only)
    spatial_basis: str = "none"   # spatial shape of the covariate modulation
    a_spatial: float = 0.0        # its strength; 0 leaves the effect unshaped
    g_shape: str = "linear"       # shape of the quantile term / of delta(u)
    h_shape: str = "linear"       # SHAPE of that modulation; see H_SHAPES.
                                  # "linear" makes the CATE linear in the
                                  # thickness rank; the others give a
                                  # non-trivial (and non-monotone) CATE
                                  # without disturbing the exact ATE.
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
        for name in ("h_shape", "g_shape"):
            if getattr(self, name) not in H_SHAPES:
                raise ValueError(
                    f"unknown {name} {getattr(self, name)!r}; "
                    f"choose from {list(H_SHAPES)}"
                )
        if self.effect_mode not in EFFECT_MODES:
            raise ValueError(
                f"unknown effect_mode {self.effect_mode!r}; choose from {EFFECT_MODES}"
            )
        if self.spatial_basis not in SPATIAL_BASES:
            raise ValueError(
                f"unknown spatial_basis {self.spatial_basis!r}; "
                f"choose from {SPATIAL_BASES}"
            )
        if self.effect_mode == "quantile_primitive" and (self.a_cov or self.a_spatial):
            raise ValueError(
                "quantile_primitive specifies delta(u) as the primitive, so the "
                "marginal quantile contrast equals it exactly. Any UNIT-level "
                "term -- a_cov or a_spatial -- makes two units at the same u "
                "receive different shifts and breaks that identity. Set them to "
                "0, or use outcome_coupled / covariate_only."
            )
        if self.effect_mode != "covariate_only" and (self.a_bright or self.a_inter):
            raise ValueError(
                "a_bright / a_inter are covariate_only knobs"
            )


PRESETS: dict[str, ExpConfig] = {
    # The toy: random assignment, one fixed effect map for everyone.
    "exp1_rct_homogeneous": ExpConfig(ps_slope=0.0, a_cov=0.0, b_quant=0.0),
    # Same effect, thickness now drives assignment. Isolates confounding.
    "exp2_confounded_homogeneous": ExpConfig(ps_slope=1.2, a_cov=0.0, b_quant=0.0),
    # Same confounding, effect now varies with the covariate AND the quantile.
    # a + b = 0.9 < 1, so no unit's effect flips sign.
    "exp3_confounded_heterogeneous": ExpConfig(ps_slope=1.2, a_cov=0.5, b_quant=0.4),
    # E3 with the Y(0) coupling removed: the effect is a function of OBSERVED
    # COVARIATES ONLY (thickness, brightness, and their interaction), so it is
    # unambiguously treatment-effect heterogeneity and nothing else. Coefficients
    # sum to 0.9 < 1, so again no sign flips.
    "exp4_covariate_cate": ExpConfig(ps_slope=1.2, effect_mode="covariate_only",
                                     a_cov=0.4, a_bright=0.3, a_inter=0.2),
    # The quantile-varying effect delta(u) as the PRIMITIVE: no covariate term,
    # so the marginal quantile contrast Q1(u) - Q0(u) is exactly what was
    # imposed, and TAU_ANALYTIC gives it in closed form.
    "exp5_quantile_effect": ExpConfig(ps_slope=1.2, effect_mode="quantile_primitive",
                                      b_quant=0.6, g_shape="linear"),
    # E4 with the effect SHAPED as well as scaled by the covariate: thick digits
    # respond more on one side of the image, thin digits on the other, so the
    # heterogeneity is spatially coherent rather than a uniform rescaling.
    # Still covariate-only, so still unambiguously CATE.
    "exp6_spatial_cate": ExpConfig(ps_slope=1.2, effect_mode="covariate_only",
                                   a_cov=0.3, a_bright=0.2, a_inter=0.1,
                                   spatial_basis="gradient_x", a_spatial=0.3),
}


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
def spatial_pattern(cfg: ExpConfig, m: np.ndarray) -> np.ndarray | None:
    """The spatial modulation pattern ``psi``, shape ``(K,)``, or None.

    ``psi`` makes the covariate's influence on the treatment effect vary ACROSS
    PIXELS, so different units get differently-shaped effects rather than the
    same map scaled up and down: with ``gradient_x``, thick digits respond more
    on one side of the image and thin digits on the other. That is the spatially
    coherent heterogeneity a real lesion would show.

    Centred and scaled over the effect's SUPPORT (where ``m != 0``), since that
    is the only region where it acts. Centring means the spatial term
    redistributes effect within the support rather than changing its average, so
    ``a_spatial`` is orthogonal to ``base_shift``.
    """
    if cfg.spatial_basis == "none":
        return None
    size = cfg.size
    xx, yy = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    c = (size - 1) / 2
    if cfg.spatial_basis == "gradient_x":
        psi = xx - c                                    # left  <-> right
    elif cfg.spatial_basis == "gradient_y":
        psi = yy - c                                    # top   <-> bottom
    elif cfg.spatial_basis == "diagonal":
        psi = (xx - c) + (yy - c)                       # corner <-> corner
    else:                                               # "radial"
        psi = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)    # centre <-> periphery
    psi = psi.reshape(-1).astype(np.float64)

    ref = m != 0
    if not ref.any():
        ref = np.ones_like(psi, dtype=bool)
    psi = psi - psi[ref].mean()
    scale = np.abs(psi[ref]).max()
    return psi / scale if scale > 0 else psi


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


def _modulator(u: np.ndarray, shape: str = "linear") -> np.ndarray:
    """Rank scores in (0,1) -> modulator in [-1, 1] with sample mean EXACTLY 0.

    The exactness of the imposed ATE does not depend on the modulation being
    linear in the rank -- it depends only on the modulator having zero sample
    mean, which is enforced here by subtracting the EMPIRICAL mean rather than
    an analytic one. Any ``shape`` therefore leaves ``ATE == m`` intact, which
    is what lets the CATE be an arbitrarily nasty function of the covariate.

    Rescaling by ``max|.|`` afterwards puts every shape on a common footing, so
    ``a_cov`` means the same thing whichever is chosen, and keeps the effect
    multiplier positive whenever the coefficients sum below 1.

    Operates along axis 0, so it handles both a per-unit ``(n,)`` covariate
    score and a per-unit-per-pixel ``(n, K)`` one (centred within each column).
    """
    if shape not in H_SHAPES:
        raise ValueError(f"unknown shape {shape!r}; choose from {list(H_SHAPES)}")
    phi = np.asarray(H_SHAPES[shape](u), dtype=np.float64)
    phi = phi - phi.mean(axis=0)                    # exact zero mean
    scale = np.abs(phi).max(axis=0)
    return np.divide(phi, scale, out=np.zeros_like(phi), where=scale > 0)


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
    brightness = ds.intensity[pool].numpy().astype(np.float64)  # ditto
    # Brightness joins Z exactly when the effect uses it: a covariate the effect
    # depends on MUST be observed, or the CATE is not identified.
    uses_brightness = cfg.effect_mode == "covariate_only"
    cont = ([thickness, brightness] if uses_brightness else [thickness])
    if cfg.digit is None:
        Z = np.hstack([np.column_stack(cont), digit_onehot[pool]])
        z_cat_idx = np.zeros(Z.shape[1], dtype=bool)
        z_cat_idx[len(cont):] = True
    else:
        # a constant one-hot carries no information and breaks the discrete
        # quantile stage, so a single-digit subset gets the continuous block only
        Z = np.column_stack(cont)
        z_cat_idx = np.zeros(len(cont), dtype=bool)

    # ---- the effect ----
    m = spatial_map(cfg)                                              # (K,)
    h = _modulator(rank_uniform(thickness), cfg.h_shape)[:, None]     # (n, 1)
    psi = spatial_pattern(cfg, m)                                     # (K,) or None
    delta = None

    # The covariate's influence on the effect, allowed to vary across pixels.
    # (a_cov + a_spatial*psi_k) is a per-PIXEL coefficient on the per-UNIT score
    # h_i, so units get differently-shaped effects, not just rescaled ones. The
    # imposed ATE is untouched: averaging over units still hits mean(h) == 0
    # whatever psi is.
    cov_coef = cfg.a_cov if psi is None else cfg.a_cov + cfg.a_spatial * psi[None, :]

    if cfg.effect_mode == "outcome_coupled":
        # The original construction. `g` is a function of the unit's OWN Y(0)
        # rank, so the b-term is a coupling between the potential outcomes, not
        # covariate heterogeneity -- see the module docstring.
        g = _modulator(rank_uniform(Y0, axis=0), cfg.g_shape)         # (n, K)
        factor = 1.0 + cov_coef * h + cfg.b_quant * g

    elif cfg.effect_mode == "covariate_only":
        # No dependence on Y(0) anywhere: the effect is a function of observed
        # covariates alone, with an interaction so the CATE surface is not
        # merely additive.
        hb = _modulator(rank_uniform(brightness), cfg.h_shape)[:, None]
        inter = _modulator(h * hb, "linear")   # re-centred: a product of two
                                               # mean-zero terms is not mean-zero
        factor = 1.0 + cov_coef * h + cfg.a_bright * hb + cfg.a_inter * inter

    else:  # "quantile_primitive"
        # The quantile-varying effect IS the primitive: delta_k(u) is specified
        # directly and Y1 is built from it, so the marginal quantile contrast
        # Q1(u) - Q0(u) equals delta by construction rather than emerging from a
        # coupling. No covariate term -- adding one would break that identity.
        u_y = rank_uniform(Y0, axis=0)                                # (n, K)
        psi = _modulator(u_y, cfg.g_shape)
        factor = 1.0 + cfg.b_quant * psi
        delta = m[None, :] * factor            # delta_k(u) at each unit's own u

    ITE = m[None, :] * factor
    Y1 = Y0 + ITE

    # The whole point of the centred modulators: this holds to machine precision.
    ate = ITE.mean(axis=0)
    assert np.allclose(ate, m, atol=1e-9), (
        f"imposed ATE drifted from the map by {np.abs(ate - m).max():.3g} -- "
        "the modulators are no longer mean-zero"
    )

    # ---- treatment assignment ----
    z = (thickness - thickness.mean()) / thickness.std()
    p = 1.0 / (1.0 + np.exp(-(cfg.ps_intercept + cfg.ps_slope * z)))
    T = (rng.uniform(size=n) < p).astype(np.float64)[:, None]
    Y = np.where(T == 1, Y1, Y0)

    treated = T[:, 0].astype(bool)
    u_grid, tau_paired, tau_marginal = true_tau_curves(Y0, Y1)

    # In quantile_primitive the marginal quantile contrast is supposed to BE the
    # imposed delta. That holds iff Q0(u) + delta(u) is still increasing in u --
    # a steeply falling delta could reorder the treated margin, at which point
    # the sorted contrast stops equalling delta. Check rather than assume.
    tau_analytic, rank_preserved = None, None
    if delta is not None:
        order = np.argsort(Y0, axis=0, kind="stable")
        y1_sorted = np.take_along_axis(Y1, order, axis=0)
        rank_preserved = bool(np.all(np.diff(y1_sorted, axis=0) >= 0))
        edges = np.linspace(0, n, TAU_CURVE_BINS + 1).astype(int)
        d_sorted = np.take_along_axis(delta, order, axis=0)
        tau_analytic = np.array([d_sorted[a:b].mean(axis=0)
                                 for a, b in zip(edges[:-1], edges[1:])])
        if rank_preserved and not np.allclose(tau_analytic, tau_marginal, atol=1e-9):
            raise AssertionError(
                "quantile_primitive: Q1(u) - Q0(u) does not match the imposed "
                "delta despite a monotone treated margin -- the construction is "
                "inconsistent"
            )

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
        "BRIGHTNESS": brightness,
        "PSI": psi,                                 # (K,) spatial pattern, or None
        # quantile_primitive only: the imposed delta(u) on the tau-curve grid,
        # and whether the treated margin stayed monotone (so TAU_MARGINAL == it)
        "TAU_ANALYTIC": tau_analytic,
        "RANK_PRESERVED": rank_preserved,
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
def _on_support(factor: np.ndarray, ate: np.ndarray) -> np.ndarray:
    """``factor`` restricted to pixels with a nonzero effect.

    ``factor`` is ``(n, 1)`` when the modulation is a per-unit scalar and
    ``(n, K)`` once it varies across pixels; only the latter needs masking.
    """
    factor = np.asarray(factor)
    if factor.ndim < 2 or factor.shape[1] == 1:
        return factor
    support = np.asarray(ate) != 0
    return factor[:, support] if support.any() else factor


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
        # Restricted to the effect's SUPPORT. Off support m_k == 0, so the
        # multiplier there is unobservable -- and with a spatial basis it is
        # routinely negative out at the edges, which would otherwise read as a
        # sign flip that no unit actually experiences.
        "factor_min": float(_on_support(data["FACTOR"], ATE).min()),
        "frac_factor_negative": float((_on_support(data["FACTOR"], ATE) < 0).mean()),
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
        elif f.name == "h_shape":
            parser.add_argument(flag, type=str, default=None, choices=list(H_SHAPES))
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
