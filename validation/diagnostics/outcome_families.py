"""Causl ground-truth generators parameterised by OUTCOME family.

The existing `causl_sim_data_generation.py` generators all use a Gaussian
*outcome* (`Y` family 1, identity link); their `3`/`5` family codes describe the
*confounders* `Z` and treatment `X`, not `Y`. This module adds generators that
vary the **outcome** family so we can ask the same question — *can the frugal
flow extract the ATE?* — across continuous outcome distributions (Gaussian,
Gamma, ...).

It is purely additive: it only builds R scripts and hands them to the existing
`causl_py.generate_data_samples`, reusing the same rpy2 plumbing and the same
`{Z_disc, Z_cont, X, Y}` output contract. Nothing in the core package or the
existing validation modules is modified.

Each family is described by an `OutcomeFamily` spec that ALSO carries the
*correct* ground truth, because the truth depends on the link:

    Gaussian (family 1, identity link):
        E[Y | do(X=t)] = const + ate * t
        true ATE       = ate
        Var[Y | do(X)] = phi

    Gamma (family 3, log link):
        E[Y | do(X=t)] = exp(const + ate * t)          (MULTIPLICATIVE effect)
        true ATE       = exp(const + ate) - exp(const)  (NON-additive in the mean)
        Var[Y | do(X)] = phi * E[Y | do(X)]^2

The Gamma case is therefore the decisive non-additive test: a location-shift
causal margin (`causal_model='gaussian'`) is misspecified for it, while the
treatment-conditioned spline (`flexible_continuous`) can adapt.

Link/parameterisation verified against the d3 causl-conventions table
(project memory `reference-causl-conventions`):
    family | code | link     | mean      | variance
    gauss  |  1   | identity | beta      | phi
    gamma  |  3   | log      | exp(beta) | phi * exp(beta)^2
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np

# Make the existing validation package importable from validation/diagnostics/.
_VALIDATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VALIDATION_DIR not in sys.path:
    sys.path.insert(0, _VALIDATION_DIR)

import data_processing_and_simulations.causl_sim_data_generation as causl_py  # noqa: E402


@dataclass(frozen=True)
class OutcomeFamily:
    """One outcome-family generator + its analytic ground truth.

    Attributes:
        name: short handle (e.g. "gaussian", "gamma").
        y_family: causl family code for Y (1 = Gaussian, 3 = Gamma).
        link: "identity" or "log" — how `const + ate*X` maps to E[Y|do(X)].
        phi: causl dispersion for Y.
        generate: callable (N, causal_params, seed) -> {Z_disc, Z_cont, X, Y}.
    """

    name: str
    y_family: int
    link: str
    phi: float
    generate: Callable

    # ---- analytic ground truth (depends on the link) -------------------------
    def mean_do(self, causal_params, t: int) -> float:
        const, ate = float(causal_params[0]), float(causal_params[1])
        lin = const + ate * t
        return lin if self.link == "identity" else math.exp(lin)

    def true_ate(self, causal_params) -> float:
        return self.mean_do(causal_params, 1) - self.mean_do(causal_params, 0)

    def var_do(self, causal_params, t: int) -> float:
        m = self.mean_do(causal_params, t)
        return self.phi if self.link == "identity" else self.phi * m * m

    def true_tau_curve(self, causal_params, u_grid) -> np.ndarray:
        """Analytic paired treatment effect tau(u) = Q_{do1}(u) - Q_{do0}(u).

        This is the GROUND-TRUTH per-quantile effect the paired-CRN readout
        estimates (same base draw pushed through do(0) and do(1)):

          identity link (Gaussian): both interventional margins are N(mean_t, phi),
              so Q_{do1}(u) - Q_{do0}(u) = mean1 - mean0 = ATE for every u
              => tau(u) is FLAT. Any spread the spline shows here is SPURIOUS.

          log link (Gamma): p(Y|do(t)) is Gamma(shape k=1/phi, scale theta_t=mean_t*phi),
              whose quantile is theta_t * G^{-1}(u; k). Hence
              tau(u) = (theta1 - theta0) * G^{-1}(u; k),
              which GROWS with u (a genuine, multiplicative effect heterogeneity).
              Here tau_sd>0 is CORRECT, so Gamma is the positive control.

        Needs scipy for the Gamma quantile; raises if unavailable on that path.
        """
        u = np.asarray(u_grid, dtype=float)
        if self.link == "identity":
            return np.full_like(u, self.true_ate(causal_params))
        # log link (Gamma)
        from scipy.stats import gamma as _gamma  # local import: only needed here
        k = 1.0 / self.phi
        theta0 = self.mean_do(causal_params, 0) * self.phi
        theta1 = self.mean_do(causal_params, 1) * self.phi
        g = _gamma.ppf(u, a=k)
        return (theta1 - theta0) * g

    def sample_truth(self, causal_params, t: int, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n iid samples from the TRUE interventional margin p(Y | do(X=t)).

        Used only to overlay the ground-truth shape on the figures. Valid because
        the frugal parametrisation specifies Y's causal margin directly, so
        p(Y | do(X=t)) is exactly this outcome family with the linked mean.
        """
        m = self.mean_do(causal_params, t)
        if self.link == "identity":
            return rng.normal(loc=m, scale=math.sqrt(self.phi), size=n)
        # Gamma, log link: shape k = 1/phi, scale theta = mean * phi (so mean = k*theta).
        k = 1.0 / self.phi
        theta = m * self.phi
        return rng.gamma(shape=k, scale=theta, size=n)


def _confound_coef(confounded: bool, confound_beta: float | None) -> float:
    """Resolve the Z->X propensity coefficient magnitude beta.

    Backward-compatible with the original boolean `confounded` toggle:
      confound_beta is None -> beta = 1.0 if confounded else 0.0  (the old on/off).
      confound_beta given    -> beta = confound_beta               (continuous knob).
    beta scales the coefficients on Zc1,Zc2,Zc3 in the X propensity equally, so
    beta=0 is X ⟂ Z (unconfounded), beta=1 reproduces the original confounded DGP,
    and beta>1 strengthens the backdoor.
    """
    if confound_beta is not None:
        return float(confound_beta)
    return 1.0 if confounded else 0.0


def _build_rscript(N: int, causal_params, seed: int, y_family: int, y_phi: float,
                   confounded: bool = True, confound_beta: float | None = None) -> str:
    """Causl R script: 4 Gaussian confounders, binary X, outcome family `y_family`.

    The confounder / treatment / copula structure is held FIXED across families
    (copied from `generate_gaussian_samples`) so that only the outcome family
    varies between specs — an apples-to-apples comparison of outcome distributions.

    The Z->X (propensity) dependence is set by a coefficient `beta` (see
    `_confound_coef`): X = c(0, beta, beta, beta) on (intercept, Zc1, Zc2, Zc3).
      beta = 0 -> treatment depends only on the intercept => X ⟂ Z (unconfounded).
      beta = 1 -> the original confounded DGP (Zc1..Zc3 each with coefficient 1).
      beta > 1 -> stronger backdoor.
    The Y->Z copula is left untouched; only the *treatment* arm of the confounding
    backdoor is scaled, which is what modulates the copula's overlap with the
    T-effect (the driver the bias diagnostics probe).
    """
    c0, a0 = causal_params[0], causal_params[1]
    beta = _confound_coef(confounded, confound_beta)
    x_beta = f"c(0,{beta},{beta},{beta})"
    return f"""
    library(causl)
    pars <- list(Zc1 = list(beta = 0, phi=1),
                 Zc2 = list(beta = c(1,1), phi=1),
                 Zc3 = list(beta = c(1,1), phi=1),
                 Zc4 = list(beta = c(0,1,1,1), phi=0.5),
                 X = list(beta = {x_beta}),
                 Y = list(beta = c({c0}, {a0}), phi={y_phi}),
                 cop = list(beta=matrix(c(2,1,0.5,1,1,1,1,1,1,1), nrow=1)))

    set.seed({seed})
    fams <- list(c(1,1,1,1), 5, {y_family}, 1)
    data_samples <- causalSamp({N}, formulas=list(list(Zc1~1, Zc2~Zc1, Zc3~Zc1, Zc4~Zc3+Zc2+Zc1), X~Zc1+Zc2+Zc3, Y~X, ~1), family=fams, pars=pars)
    """


def _make_generator(y_family: int, y_phi: float, confounded: bool = True,
                    confound_beta: float | None = None):
    def _gen(N, causal_params, seed=0):
        return causl_py.generate_data_samples(
            _build_rscript(N, causal_params, seed, y_family, y_phi,
                           confounded, confound_beta))
    return _gen


def make_gaussian_family(confound_beta: float) -> "OutcomeFamily":
    """Gaussian-outcome family with a chosen Z->X confounding strength `beta`.

    Same analytic ground truth as `FAMILIES['gaussian']` (identity link, phi=1),
    but the propensity backdoor coefficient is `confound_beta` instead of the
    default 1.0. Used by the confounding-strength sweep (E2) to trace how the
    spline arm's spurious effect heterogeneity scales with confounding.
    """
    return OutcomeFamily(
        name=f"gaussian_b{confound_beta:g}", y_family=1, link="identity", phi=1.0,
        generate=_make_generator(1, 1.0, confound_beta=confound_beta),
    )


# Registry of outcome families. Add new continuous families here (e.g. inverse
# Gaussian) and the suite picks them up automatically.
GAMMA_PHI = 0.5  # shape = 1/phi = 2: clearly right-skewed but stable to fit


def make_gamma_family(confound_beta: float) -> "OutcomeFamily":
    """Gamma-outcome family (log link, phi=GAMMA_PHI) with a chosen Z->X strength `beta`.

    Same analytic ground truth as `FAMILIES['gamma']` (log link => multiplicative
    effect, so `true_ate`/`true_tau_curve` are beta-INDEPENDENT — beta only touches
    the Z->X propensity backdoor, not the causal margin). Used by the overlap
    dose-response sweep (E-vii(b)): beta=0 is X ⟂ Z (perfect overlap), beta=1 is the
    original confounded gamma DGP, beta>1 worsens positivity. `gamma_b1` reproduces
    `FAMILIES['gamma']` exactly (a built-in consistency check).
    """
    return OutcomeFamily(
        name=f"gamma_b{confound_beta:g}", y_family=3, link="log", phi=GAMMA_PHI,
        generate=_make_generator(3, GAMMA_PHI, confound_beta=confound_beta),
    )


FAMILIES: dict[str, OutcomeFamily] = {
    "gaussian": OutcomeFamily(
        name="gaussian", y_family=1, link="identity", phi=1.0,
        generate=_make_generator(1, 1.0),
    ),
    "gamma": OutcomeFamily(
        name="gamma", y_family=3, link="log", phi=GAMMA_PHI,
        generate=_make_generator(3, GAMMA_PHI),
    ),
    # Unconfounded twin of `gaussian` (X ⟂ Z): same causal-margin truth, but no
    # Z->X backdoor. Used by the bias diagnostics to test whether the small-n ATE
    # attenuation is driven by the copula absorbing the confounded treatment effect.
    "gaussian_unconfounded": OutcomeFamily(
        name="gaussian_unconfounded", y_family=1, link="identity", phi=1.0,
        generate=_make_generator(1, 1.0, confounded=False),
    ),
    # Overlap dose-response sweep (E-vii(b)): the same Gamma DGP at increasing
    # Z->X confounding strength beta. beta=0 is unconfounded (perfect overlap);
    # beta=1.5 is the worst positivity. `gamma_b1` == `gamma` (consistency check).
    **{f"gamma_b{b:g}": make_gamma_family(b) for b in (0.0, 0.5, 1.0, 1.5)},
}


if __name__ == "__main__":
    # Cheap self-test: generate a small dataset per family and check the EMPIRICAL
    # treated/untreated outcome means line up with the analytic linked truth.
    cp = [0.0, 1.0]
    for name, fam in FAMILIES.items():
        d = fam.generate(2000, causal_params=cp, seed=0)
        Y = np.asarray(d["Y"]).ravel()
        X = np.asarray(d["X"]).ravel()
        emp0, emp1 = Y[X == 0].mean(), Y[X == 1].mean()
        # NB: empirical means are CONDITIONAL E[Y|X], not interventional, so they
        # carry confounding bias; this is a sanity check on order of magnitude,
        # not an identification claim.
        print(f"{name:9s} y_family={fam.y_family} link={fam.link} phi={fam.phi}")
        print(f"  true do-means: E[Y|do0]={fam.mean_do(cp,0):.3f}  E[Y|do1]={fam.mean_do(cp,1):.3f}"
              f"  true ATE={fam.true_ate(cp):+.3f}")
        print(f"  empir  E[Y|X=0]={emp0:.3f}  E[Y|X=1]={emp1:.3f}  (confounded; rough check)")
        print(f"  Y range: [{Y.min():.2f}, {Y.max():.2f}]  any<=0: {(Y<=0).any()}\n")
