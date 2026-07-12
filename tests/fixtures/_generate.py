"""Regenerate cached test fixtures by calling R's causl simulator.

Run once after a fresh checkout:

    python tests/fixtures/_generate.py

Outputs are .npz files alongside this script. They are deterministic given the
seeds set below; only re-run if you intentionally want to change a fixture.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from rpy2.robjects import default_converter, pandas2ri
import rpy2.robjects as ro

ro.conversion.set_conversion(default_converter + pandas2ri.converter)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "validation"))
from data_processing_and_simulations import causl_sim_data_generation as causl_py  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent


def _save(name: str, data: dict, **meta) -> None:
    path = FIXTURES_DIR / f"{name}.npz"
    arrays = {k: np.asarray(v) for k, v in data.items() if v is not None}
    missing = [k for k, v in data.items() if v is None]
    np.savez(path, _missing=np.array(missing), **arrays, **{f"meta_{k}": np.array(v) for k, v in meta.items()})
    print(f"wrote {path} ({', '.join(f'{k}={v.shape}' for k, v in arrays.items())})")


def gaussian_known_ate():
    """Gaussian Z, continuous X, continuous Y with known ATE=1.

    Y = ate*X + const + ε,  ate=1, const=1, ε ~ N(0, 1).
    """
    data = causl_py.generate_gaussian_samples(N=2000, causal_params=[1.0, 1.0], seed=0)
    _save("gaussian_known_ate", data, ate=1.0, const=1.0, scale=1.0, n=2000, seed=0)


def mixed_known_ate():
    """MIXED confounders (2 continuous Zc + 2 discrete Zd), binary X, Gaussian Y,
    known ATE=1.

    Y = ate*X + const + ε,  ate=1, const=1, ε ~ N(0, 1). The frugal parametrisation
    fixes the interventional margin p(Y|do(X)) at these betas regardless of the
    confounding copula, so the true ATE is exactly causal_params[1] = 1.

    This is the configuration the propensity column-order bug (Bug B) corrupts --
    both Z blocks present -- so it is the natural fixture for a mixed-confounder
    recovery + synthetic-data round-trip test.
    """
    data = causl_py.generate_discrete_samples(N=2000, causal_params=[1.0, 1.0], seed=0)
    _save("mixed_known_ate", data, ate=1.0, const=1.0, scale=1.0, n=2000, seed=0)


if __name__ == "__main__":
    gaussian_known_ate()
    mixed_known_ate()
