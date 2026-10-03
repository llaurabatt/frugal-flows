"""Frugal flows: causal generative models that parametrise the causal margin p(y | do(t)) directly
and fit the rest of the joint (covariates, their dependence on the outcome) with a copula flow.

Main entry points:

* ``train_frugal_flow``         fit any arm (``causal_model=...``), scalar or multivariate Y
* ``fit_gaussian_frugal_flow``  one-call fit of the Gaussian-scale flow for multivariate Y
* ``interventional_samples``    paired draws of Y | do(T=0) and Y | do(T=1), and the ATE
* ``counterfactual_gaussian`` / ``counterfactual_flexible``   unit-level counterfactuals
* ``save_gaussian_flow`` / ``load_gaussian_flow``             persist a Gaussian-scale fit
* ``OutcomeTransform``          per-column outcome scaling used at fit time
"""
from .basic_flows import masked_independent_flow
from .causal_flows import train_frugal_flow
from .gaussian_scale import (
    PAPER_SETTINGS,
    counterfactual_gaussian,
    ecdf_ranks,
    fit_gaussian_frugal_flow,
    load_gaussian_flow,
    save_gaussian_flow,
)
from .interventions import counterfactual_flexible, interventional_samples
from .outcome_transforms import OutcomeTransform
from .precision import apply_default_precision, set_x64, x64_enabled

__version__ = "0.2.0"

__all__ = [
    "train_frugal_flow",
    "fit_gaussian_frugal_flow",
    "interventional_samples",
    "counterfactual_gaussian",
    "counterfactual_flexible",
    "save_gaussian_flow",
    "load_gaussian_flow",
    "ecdf_ranks",
    "PAPER_SETTINGS",
    "OutcomeTransform",
    "masked_independent_flow",
    "set_x64",
    "x64_enabled",
    "apply_default_precision",
    "__version__",
]
