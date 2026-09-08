"""Shared scientific metrics for the MorphoMNIST validation runners."""

from __future__ import annotations

import numpy as np

EFFECT_SCORE_KEYS = (
    "ate_mae",
    "ate_rmse",
    "ate_max_abs_err",
    "ate_mae_on_support",
    "ate_mae_off_support",
    "ate_corr",
    "att_mae",
    "atc_mae",
)

MARGINAL_QTE_BINS = 40


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(values[mask])) if np.any(mask) else float("nan")


def score_effect_map(tau_hat: np.ndarray, data: dict) -> dict:
    """Score one estimated per-pixel effect map against the common truths."""
    tau_hat = np.asarray(tau_hat, dtype=np.float64)
    ate = np.asarray(data["ATE"], dtype=np.float64)
    if tau_hat.shape != ate.shape:
        raise ValueError(f"tau_hat has shape {tau_hat.shape}; expected {ate.shape}")
    if not np.isfinite(tau_hat).all():
        raise ValueError("tau_hat contains non-finite values")

    support = ate != 0
    abs_err = np.abs(tau_hat - ate)
    return {
        "ate_mae": float(abs_err.mean()),
        "ate_rmse": float(np.sqrt(np.mean((tau_hat - ate) ** 2))),
        "ate_max_abs_err": float(abs_err.max()),
        "ate_mae_on_support": _masked_mean(abs_err, support),
        "ate_mae_off_support": _masked_mean(abs_err, ~support),
        "ate_corr": float(np.corrcoef(tau_hat, ate)[0, 1]),
        "att_mae": float(np.abs(tau_hat - np.asarray(data["ATT"])).mean()),
        "atc_mae": float(np.abs(tau_hat - np.asarray(data["ATC"])).mean()),
    }


def marginal_qte_curve(
    y0: np.ndarray, y1: np.ndarray, n_bins: int = MARGINAL_QTE_BINS
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate Q1(u)-Q0(u) by sorting the two margins independently."""
    y0 = np.asarray(y0, dtype=np.float64)
    y1 = np.asarray(y1, dtype=np.float64)
    if y0.shape != y1.shape:
        raise ValueError(f"margin shapes differ: {y0.shape} versus {y1.shape}")
    if y0.ndim not in (1, 2):
        raise ValueError("margins must have shape (n,) or (n, K)")
    if not np.isfinite(y0).all() or not np.isfinite(y1).all():
        raise ValueError("margins contain non-finite values")
    if n_bins < 1 or len(y0) < n_bins:
        raise ValueError("n_bins must be positive and no larger than the sample size")

    scalar = y0.ndim == 1
    if scalar:
        y0, y1 = y0[:, None], y1[:, None]

    q0 = np.sort(y0, axis=0)
    q1 = np.sort(y1, axis=0)
    edges = np.linspace(0, len(y0), n_bins + 1).astype(int)
    curve = np.asarray(
        [q1[a:b].mean(axis=0) - q0[a:b].mean(axis=0)
         for a, b in zip(edges[:-1], edges[1:])]
    )
    u = (np.arange(n_bins) + 0.5) / n_bins
    return u, curve[:, 0] if scalar else curve


def score_marginal_qte(curve: np.ndarray, data: dict) -> dict:
    """Score a marginal-QTE curve against the generator's TAU_MARGINAL."""
    curve = np.asarray(curve, dtype=np.float64)
    truth = np.asarray(data["TAU_MARGINAL"], dtype=np.float64)
    if curve.shape != truth.shape:
        raise ValueError(f"marginal QTE has shape {curve.shape}; expected {truth.shape}")
    if not np.isfinite(curve).all():
        raise ValueError("marginal QTE contains non-finite values")

    support = np.asarray(data["ATE"]) != 0
    squared_error = (curve - truth) ** 2
    curve_sd = curve.std(axis=0)
    return {
        "marginal_qte_rmse": float(np.sqrt(squared_error.mean())),
        "marginal_qte_rmse_on_support": (
            float(np.sqrt(squared_error[:, support].mean()))
            if support.any() else float("nan")
        ),
        "marginal_qte_sd_on_support": _masked_mean(curve_sd, support),
        "marginal_qte_sd_off_support": _masked_mean(curve_sd, ~support),
        "true_marginal_qte_sd_on_support": _masked_mean(
            truth.std(axis=0), support
        ),
    }
