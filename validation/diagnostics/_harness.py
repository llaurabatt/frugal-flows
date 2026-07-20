"""Shared, script-local plumbing for the identification diagnostics.

NOTHING here re-implements modelling logic. It is a thin wrapper around the
*existing* ``validation/data_processing_and_simulations/causl_sim_data_generation.py``
(generators + ``frugal_fitting``). The only reason this file exists is to give
the three diagnostic scripts a single, well-documented entry point and to make
the truth/parametrisation explicit (the notebook leaves it implicit and even
self-inconsistent — see ``true_params_from_causal_params``).

Key facts this harness encodes (verified against the source, 2026-06-10):

* The causl generators encode the ground truth as ``causal_params = [const, ate]``:
  the outcome formula is ``Y ~ X`` with linear predictor ``const + ate * X`` and a
  Gaussian family at ``phi = 1`` (so the causal margin's ``scale`` truth is 1.0).
* ``frugal_fitting`` already extracts the trained causal margin
  (``UnivariateNormalCDF``) for us as ``out['causal_margin']`` — we do NOT need to
  re-derive the brittle ``flow.bijection.bijections[-1].bijection.bijections[0]``
  index path.
* For ``causal_model='gaussian'`` the ``causal_model_args`` are *initial* values
  for the margin's ``(ate, const, scale)`` — the fit then moves them to recover
  the truth.

NB: the notebook's ``run_simulations`` hardcodes ``causal_model='gaussian'``,
prints intermediate results, and self-references ``causl_py.frugal_fitting``. We
deliberately bypass it and loop ``frugal_fitting`` ourselves — same finding, but
we also keep the fitted flow object that diagnostics d2/d3 need. (Not a fix to
the core code; we simply don't call that function.)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

# --- make the existing validation package importable from validation/diagnostics/ ---
# validation/diagnostics/_harness.py -> parent of parent is validation/
_VALIDATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VALIDATION_DIR not in sys.path:
    sys.path.insert(0, _VALIDATION_DIR)

# Importing this module triggers `importr('causl')` at load time (the try/except
# at the top of the module). That requires the frugal-flows-flowjax env.
import data_processing_and_simulations.causl_sim_data_generation as causl_py  # noqa: E402

# The spline-based causal margin is numerically delicate; the package assumes x64
# (benchmarking.py sets this globally). Mirror it here so a standalone script run
# behaves identically to the notebook.
jax.config.update("jax_enable_x64", True)


# Hyperparameters copied verbatim from Continous_Frugal_Flows.ipynb so the
# scripts reproduce the notebook fit. `show_progress` is added (flows straight
# through frugal_fitting -> train_frugal_flow) so loops can be quiet.
DEFAULT_HYPERPARAMS: dict = {
    "learning_rate": 5e-3,
    "RQS_knots": 8,
    "flow_layers": 5,
    "nn_width": 50,
    "nn_depth": 4,
    "max_patience": 100,
    "max_epochs": 10000,
    "show_progress": False,
}

# A deliberately small/fast config for the feedback loop (NOT the notebook scale).
SMOKE_HYPERPARAMS: dict = {
    **DEFAULT_HYPERPARAMS,
    "max_patience": 20,
    "max_epochs": 400,
}

# The causl ground-truth generators, by short name. These are the same callables
# the notebook uses; we only give them friendly handles.
GENERATORS: dict = {
    "gaussian": causl_py.generate_gaussian_samples,
    "mixed": causl_py.generate_mixed_samples,
    "discrete": causl_py.generate_discrete_samples,
    "many_discrete": causl_py.generate_many_discrete_samples,
}

PARAM_NAMES = ("ate", "const", "scale")


def true_params_from_causal_params(causal_params) -> dict:
    """Map a generator's ``causal_params = [const, ate]`` to the causal-margin truth.

    Returns ``{'ate', 'const', 'scale'}``. ``scale`` is 1.0 because the causl
    outcome family is Gaussian with ``phi = 1``.

    This is the correct truth to compare recovered params against. (The notebook
    sets ``TRUE_PARAMS = {ate:1, const:0, scale:1}`` while generating with
    ``CAUSAL_PARAMS = [2, 5]`` — an internal inconsistency we avoid by deriving
    the truth here.)
    """
    const, ate = float(causal_params[0]), float(causal_params[1])
    return {"ate": ate, "const": const, "scale": 1.0}


def _scalar(x) -> float:
    """Coerce a size-1 jax/np array (or python scalar) to a float."""
    return float(np.ravel(np.asarray(x))[0])


@dataclass
class FitResult:
    """One frugal-flow fit on one ground-truth dataset.

    Holds enough for all three diagnostics: the recovered margin params (d1), the
    fitted margin object + flow (d2), and the input data arrays (d3).
    """

    seed: int
    recovered: dict           # {'ate', 'const', 'scale'} learned by the margin
    true_params: dict         # {'ate', 'const', 'scale'} from the generator
    causal_margin: object     # UnivariateNormalCDF
    frugal_flow: object
    val_loss: float
    data: dict = field(repr=False, default_factory=dict)  # {'X','Y','Z_cont','Z_disc'}


def generate_dataset(generator_name: str, causal_params, n_samples: int, seed: int) -> dict:
    """Draw one ground-truth dataset from a named causl generator.

    Returns ``{'Z_disc', 'Z_cont', 'X', 'Y'}`` (the generators' native dict).
    """
    gen = GENERATORS[generator_name]
    return gen(n_samples, causal_params=causal_params, seed=seed)


def fit_once(
    generator_name: str = "mixed",
    causal_params=(1.0, 1.0),
    n_samples: int = 2000,
    seed: int = 0,
    hyperparams: dict | None = None,
    causal_model: str = "gaussian",
    init_args: dict | None = None,
) -> FitResult:
    """Generate one dataset and fit one frugal flow; return recovered params + flow.

    This is the single shared step under all three diagnostics. It calls the
    existing ``causl_py.frugal_fitting`` unchanged.

    Args:
        generator_name: key into ``GENERATORS`` (e.g. ``'mixed'``).
        causal_params: ``[const, ate]`` passed to the generator.
        n_samples: dataset size.
        seed: seed for both data generation and the fit.
        hyperparams: frugal-flow hyperparameters; defaults to ``DEFAULT_HYPERPARAMS``.
        causal_model: causal-margin parametrisation (``'gaussian'`` here).
        init_args: initial ``causal_model_args`` for the margin. Defaults to the
            notebook's ``{'ate': [0.], 'const': 0., 'scale': 1.}``.
    """
    hyperparams = dict(DEFAULT_HYPERPARAMS if hyperparams is None else hyperparams)
    if init_args is None:
        init_args = {"ate": jnp.array([0.0]), "const": 0.0, "scale": 1.0}

    data = generate_dataset(generator_name, causal_params, n_samples, seed)
    Z_disc, Z_cont, X, Y = data["Z_disc"], data["Z_cont"], data["X"], data["Y"]

    out, min_loss = causl_py.frugal_fitting(
        X,
        Y,
        Z_cont=Z_cont,
        Z_disc=Z_disc,
        seed=seed,
        frugal_flow_hyperparams=hyperparams,
        causal_model=causal_model,
        causal_model_args=init_args,
    )

    margin = out["causal_margin"]  # UnivariateNormalCDF, already extracted upstream
    recovered = {
        "ate": _scalar(margin.ate),
        "const": _scalar(margin.const),
        "scale": _scalar(margin.scale),
    }

    return FitResult(
        seed=seed,
        recovered=recovered,
        true_params=true_params_from_causal_params(causal_params),
        causal_margin=margin,
        frugal_flow=out["frugal_flow"],
        val_loss=float(min_loss),
        data={"X": X, "Y": Y, "Z_cont": Z_cont, "Z_disc": Z_disc},
    )


def outputs_dir() -> str:
    """Absolute path to ``validation/diagnostics/outputs`` (created if missing)."""
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(d, exist_ok=True)
    return d
