"""Halo ladder: per-pixel maps (names exactly as in HALO_PREREG.md), template regression,
class summaries and oracle floors. Pure numpy/scipy.

Per arm t in {0, 1} (S1 has arm 0 only; its oracle is the fitted Y):
    E_mu{t}       mean(gen_t) - mean(oracle_t)                         Sense-2 candidate
    E_sd{t}       sd(gen_t) - sd(oracle_t)
    R_sd{t}       log2(sd(gen_t) / sd(oracle_t))                       quiet over-dispersion
    KS{t}, W1_{t} two-sample KS statistic; exact 1-D Wasserstein-1     fidelity
    LEAK_MODEL{t} P_model(Y_k outside the pure-floor sliver [lo, hi]), ALL pixels (class-
                  masked in analysis); LEAK_DATA{t} the same on the oracle sample;
                  LEAK_X{t} = LEAK_MODEL - LEAK_DATA (excess leak, the S1 primary). NaN when
                  ``leak=False`` (C-smooth: no atoms, LEAK undefined).
    FLOORMASS{t}  P_model(Y_k < floor_thr) - P_oracle(Y_k < floor_thr)
    NBCORR{t}     |corr_gen - corr_ref| per pixel (mean over its incident edge-adjacent
                  pairs); NBCORR_pairs{t} per pair (Laura's ``neighbour_pairs(8)`` order)
    E_tau         tau_hat - ATE, tau_hat = mean(gen1 - gen0) under common random numbers
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy import stats

_MM = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _MM not in sys.path:
    sys.path.insert(0, _MM)
from sample_diagnostics import _pair_corr, neighbour_pairs  # noqa: E402

PAIRS = neighbour_pairs(8)
MAP_KEYS_ARM = ("E_mu", "E_sd", "R_sd", "KS", "W1_", "LEAK", "FLOORMASS", "NBCORR")


# Ported from ~/work/frugal-flows-frengression/validation/morphomnist/
# exp_frengression_recovery.py (w1_per_pixel, ~L536); copied, not imported, because that
# file lives in another worktree.
def w1_per_pixel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact Wasserstein-1 per coordinate for two EQUAL-size empirical measures."""
    a = np.sort(np.asarray(a, dtype=np.float64), axis=0)
    b = np.sort(np.asarray(b, dtype=np.float64), axis=0)
    return np.abs(a - b).mean(axis=0)


def w1_unequal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact 1-D W1 per coordinate for samples of different sizes (scipy)."""
    if len(a) == len(b):
        return w1_per_pixel(a, b)
    return np.array([stats.wasserstein_distance(a[:, k], b[:, k]) for k in range(a.shape[1])])


def _pixel_nbcorr(d_pairs: np.ndarray, K: int = 64) -> np.ndarray:
    tot, cnt = np.zeros(K), np.zeros(K)
    for (i, j), v in zip(PAIRS, d_pairs):
        tot[[i, j]] += v
        cnt[[i, j]] += 1
    return tot / cnt


def arm_maps(gen: np.ndarray, ref: np.ndarray, t: int, lo: float, hi: float,
             floor_thr: float, leak: bool = True) -> dict:
    """All per-arm maps of one generated sample against its oracle sample."""
    gen = np.asarray(gen, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    gs, rs = gen.std(0, ddof=1), ref.std(0, ddof=1)
    nan = np.full(gen.shape[1], np.nan)
    out_g = (gen < lo) | (gen > hi)
    out_r = (ref < lo) | (ref > hi)
    with np.errstate(invalid="ignore", divide="ignore"):
        nb = np.abs(_pair_corr(gen, PAIRS) - _pair_corr(ref, PAIRS))
        rsd = np.log2(gs / rs)
    return {f"E_mu{t}": gen.mean(0) - ref.mean(0), f"E_sd{t}": gs - rs, f"R_sd{t}": rsd,
            f"KS{t}": np.array([stats.ks_2samp(gen[:, k], ref[:, k]).statistic
                                for k in range(gen.shape[1])]),
            f"W1_{t}": w1_unequal(gen, ref),
            f"LEAK_MODEL{t}": out_g.mean(0) if leak else nan,
            f"LEAK_DATA{t}": out_r.mean(0) if leak else nan,
            f"LEAK_X{t}": out_g.mean(0) - out_r.mean(0) if leak else nan,
            f"FLOORMASS{t}": (gen < floor_thr).mean(0) - (ref < floor_thr).mean(0),
            f"NBCORR{t}": _pixel_nbcorr(nb), f"NBCORR_pairs{t}": nb,
            f"gen_mean{t}": gen.mean(0), f"gen_sd{t}": gs}


def maps(gen0, ref0, gen1=None, ref1=None, ate=None, tau_hat=None, *, lo: float, hi: float,
         floor_thr: float, leak: bool = True) -> dict:
    """Every declared map for one cell. ``tau_hat`` overrides the CRN mean difference
    (location-translation reads it off the model); otherwise mean(gen1 - gen0)."""
    out = arm_maps(gen0, ref0, 0, lo, hi, floor_thr, leak)
    if gen1 is not None:
        out.update(arm_maps(gen1, ref1, 1, lo, hi, floor_thr, leak))
        out["E_mu_diff"] = out["E_mu1"] - out["E_mu0"]
        crn = np.mean(np.asarray(gen1, np.float64) - np.asarray(gen0, np.float64), axis=0)
        out["tau_hat_crn"] = crn
        out["tau_hat"] = crn if tau_hat is None else np.asarray(tau_hat, np.float64)
        out["E_tau"] = out["tau_hat"] - np.asarray(ate, np.float64)
    return out


def template_regression(err_map: np.ndarray, templates: dict) -> dict:
    """OLS err_k = b0 + sum_j b_j t_jk over the 64 pixels (templates already standardised).
    A constant template is dropped (its coefficient NaN). Returns b0, b_<name>, r2."""
    names = [n for n, v in templates.items() if np.std(v) > 0]
    y = np.asarray(err_map, np.float64)
    ok = np.isfinite(y)
    A = np.column_stack([np.ones(len(y))] + [np.asarray(templates[n]) for n in names])[ok]
    coef, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    resid = y[ok] - A @ coef
    ss = ((y[ok] - y[ok].mean()) ** 2).sum()
    out = {"b0": float(coef[0]), "r2": float(1 - (resid ** 2).sum() / ss) if ss > 0 else float("nan")}
    out.update({f"b_{n[2:]}": float(c) for n, c in zip(names, coef[1:])})
    out.update({f"b_{n[2:]}": float("nan") for n in templates if n not in names})
    return out


def slope_on(err_map: np.ndarray, template_raw: np.ndarray, mask: np.ndarray) -> float:
    """Simple OLS slope of err on an UNSTANDARDISED template over ``mask`` pixels
    (S2/S4: E_tau on the imbalance / unadjusted-bias map, Dan's 0.14 vs 0.014 scale)."""
    x, y = np.asarray(template_raw)[mask], np.asarray(err_map)[mask]
    return float(np.polyfit(x, y, 1)[0])


FLOOR_CLASSES = ("exact_floor", "pure_floor", "mixture", "ink")
REGIONS = ("reg_disc", "reg_ring", "reg_far")


def summary_classes(classes: dict) -> dict:
    """The boolean pixel sets plus the floor-class x region cross-table
    ("<floor>&<region>", e.g. "mixture&reg_ring"); empty cells are dropped later."""
    out = {k: np.asarray(v, bool) for k, v in classes.items() if np.asarray(v).dtype == bool}
    for f in FLOOR_CLASSES:
        for r in REGIONS:
            out[f"{f}&{r}"] = out[f] & out[r]
    return out


def class_summaries(m: dict, classes: dict) -> dict:
    """{map: {class: {mean, median, rms}}} for every (64,) map and boolean class mask."""
    out = {}
    for k, v in m.items():
        v = np.asarray(v)
        if v.shape != (64,):
            continue
        out[k] = {}
        for c, mask in classes.items():
            mask = np.asarray(mask)
            if mask.dtype != bool or not mask.any():
                continue
            x = v[mask][np.isfinite(v[mask])]
            out[k][c] = ({"mean": float(x.mean()), "median": float(np.median(x)),
                          "rms": float(np.sqrt((x ** 2).mean()))} if len(x) else
                         {"mean": float("nan"), "median": float("nan"), "rms": float("nan")})
    return out


def oracle_floors(Y0: np.ndarray, Y1: np.ndarray | None, rng: np.random.Generator) -> dict:
    """Split-half floors of KS, W1, |E_mu|, |E_sd| from two disjoint halves of the oracle
    (``distribution_fidelity``-style): what a perfect model scores at this n."""
    out = {}
    for t, Yt in enumerate((Y0, Y1)):
        if Yt is None:
            continue
        p = rng.permutation(len(Yt))
        h = len(Yt) // 2
        a, b = Yt[p[:h]], Yt[p[h: 2 * h]]
        out[f"floor_KS{t}"] = np.array([stats.ks_2samp(a[:, k], b[:, k]).statistic
                                        for k in range(Yt.shape[1])])
        out[f"floor_W1_{t}"] = w1_per_pixel(a, b)
        out[f"floor_E_mu{t}"] = np.abs(a.mean(0) - b.mean(0))
        out[f"floor_E_sd{t}"] = np.abs(a.std(0) - b.std(0))
    return out


# ------------------------------------------------------------------ S10 (Amendment A5)
def retained_confounding(tau_hat, truth, naive) -> float:
    """rho = <tau_hat - truth, naive - truth> / ||naive - truth||^2 over the pixels
    (0 = the truth, 1 = the naive treated-minus-untreated difference)."""
    d = np.asarray(naive, np.float64) - np.asarray(truth, np.float64)
    return float(np.dot(np.asarray(tau_hat, np.float64) - np.asarray(truth, np.float64), d) / np.dot(d, d))


def s10_endpoints(tau_hat, truth, naive, disc, active_off, quiet, init=None) -> dict:
    """Per-cell S10 endpoints (all on the data / logit scale). ``disc`` is the effect support
    (the geometric disc when truth == 0); ``init`` the shift's starting vector (None: not a
    shift model)."""
    tau_hat, truth, naive = (np.asarray(v, np.float64) for v in (tau_hat, truth, naive))
    e = tau_hat - truth
    m = lambda v, k: float(np.mean(v[np.asarray(k, bool)]))
    out = {"tauhat_disc_mean": m(tau_hat, disc), "Etau_disc_mean": m(e, disc),
           "Etau_active_off_mean": m(e, active_off), "Etau_quiet_mean": m(e, quiet),
           "ate_mae": float(np.mean(np.abs(e))), "rho": retained_confounding(tau_hat, truth, naive),
           "dist_from_truth": float(np.linalg.norm(e)),
           "dist_naive_from_truth": float(np.linalg.norm(naive - truth))}
    if init is not None:
        out["dist_from_init"] = float(np.linalg.norm(tau_hat - np.asarray(init, np.float64)))
    return out
