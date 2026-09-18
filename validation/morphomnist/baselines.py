"""Classical per-pixel ATE estimators on the MorphoMNIST experiments.

Runs the standard estimators a careful applied statistician would reach for,
on exactly the datasets ``prepare_morphomnist_exps`` produces, and scores them
with exactly the metrics ``exp_ate_recovery`` uses. The point is a like-for-like
comparison: same data, same truth, same scores, so the only thing that differs
is the estimator.

Read the OLS row first. The effect in these experiments is ADDITIVE in logit
space and the outcome is modelled in logit space, so K independent regressions
of ``Y_k`` on ``(T, Z)`` are close to correctly specified and should be strong.
If they beat the flow on ATE recovery that is a fact worth knowing early: it
does not make the flow useless -- a regression cannot produce ``p(Y | do(T))``,
interventional samples, or the quantile-resolved ``tau(u)`` -- but it does
determine what the paper can claim about ATE recovery specifically.

Estimators
----------
    naive        difference in means. No adjustment; the size of its error is
                 the size of the confounding.
    ipw          inverse propensity weighting, propensity ESTIMATED by logistic
                 regression on the covariate basis. Hajek (self-normalised).
    ols          per-pixel OLS of Y_k on [T, basis(Z)]; tau_hat is the
                 coefficient on T. The natural competitor here.
    aipw         augmented IPW (doubly robust), 5-fold cross-fitted so the
                 nuisance models are not evaluated on their own training data.
    oracle_ipw   IPW using the TRUE propensity, which no real estimator has.
                 This is the sampling-noise floor, not a competitor.

Covariate basis (``--basis``)
-----------------------------
The frugal flow sees the covariates through a rank/quantile transform and a
flexible copula, so it can represent nonlinear covariate dependence. Giving the
baselines only a linear term would be an unfair comparison. Each continuous
covariate is therefore mapped to its rank and expanded:

    linear   [1, u]
    poly3    [1, u, u^2, u^3] per covariate, plus pairwise products (default)
    poly5    [1, ..., u^5] per covariate, plus pairwise products

Usage
-----
    python baselines.py --preset exp4_covariate_cate --size 8 --seed-data 101
    python baselines.py --preset exp1_rct_homogeneous --size 8 --seed-data 101 --base-shift 0
    python baselines.py --from-index        # every dataset the flow runs used, not yet done

Every invocation builds ONE dataset through exp_ate_recovery.build_data -- the same
class, the same function, the same defaults a flow run uses, so the bytes are the ones
the flow saw -- and writes one run folder under runs/baselines/:
    <UTC stamp>_baselines_<preset>_[<variant>_]k<K>_sd<seed>_d<digit>_<uid>/
        config.json    run_id, uid, dataset_id, data_hash, the generator config, basis
        arrays.npz     ATE, ATT, ATC, Y, X, ITE, PROPENSITY, tau_hat_<method> x5
        metrics.json   whole-image and regional scores per method
        plots/         ate_maps_<method>.png
        log.txt, wandb.json (baselines are not logged to wandb)
and one row per (dataset, method) into runs/baselines/index.csv. dataset_id and
data_hash are the join keys to runs/exp_ate_recovery/index.csv.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from prepare_morphomnist_exps import PRESETS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

METHODS = ("naive", "ipw", "ols", "aipw", "oracle_ipw")
BASES = ("linear", "poly3", "poly5")
RUNS_ROOT = os.path.join(SCRIPT_DIR, "runs", "baselines")


# --------------------------------------------------------------------------- #
# covariate basis
# --------------------------------------------------------------------------- #
def _ranks(a: np.ndarray) -> np.ndarray:
    """Column-wise rank scores in (0,1). Matches the flow's stage-one view of Z."""
    n = a.shape[0]
    order = np.argsort(a, axis=0, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(n)[:, None].repeat(a.shape[1], 1), axis=0)
    return (ranks + 0.5) / n


def design_matrix(Z: np.ndarray, basis: str = "poly3") -> np.ndarray:
    """``[1, expansion(Z)]``. Discrete (0/1) columns are passed through as-is.

    Rank-transforming first means the expansion is on a bounded, uniformly
    spread variable, so high powers stay conditioned -- a raw polynomial in a
    skewed covariate is numerically much worse behaved.
    """
    Z = np.asarray(Z, dtype=np.float64)
    is_binary = np.array([np.isin(np.unique(Z[:, j]), (0.0, 1.0)).all()
                          for j in range(Z.shape[1])])
    cont, disc = Z[:, ~is_binary], Z[:, is_binary]

    cols = [np.ones((len(Z), 1))]
    if cont.shape[1]:
        u = _ranks(cont)
        deg = {"linear": 1, "poly3": 3, "poly5": 5}[basis]
        for d in range(1, deg + 1):
            cols.append(u**d)
        # pairwise interactions between covariates (not with themselves)
        for j in range(u.shape[1]):
            for k in range(j + 1, u.shape[1]):
                cols.append((u[:, [j]] * u[:, [k]]))
    if disc.shape[1]:
        cols.append(disc[:, 1:] if disc.shape[1] > 1 else disc)  # drop one level
    return np.hstack(cols)


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares with a small ridge for conditioning. y may be (n,) or (n,K)."""
    XtX = X.T @ X
    XtX[np.diag_indices_from(XtX)] += 1e-8 * np.trace(XtX) / len(XtX)
    return np.linalg.solve(XtX, X.T @ y)


# --------------------------------------------------------------------------- #
# estimators -- each returns tau_hat of shape (K,)
# --------------------------------------------------------------------------- #
def est_naive(Y, T, X, _true_p, **kw):
    return Y[T].mean(0) - Y[~T].mean(0)


def _hajek(Y, T, p):
    """Self-normalised IPW. Stabilising the weights keeps a near-0/1 propensity
    from dominating the average, which matters here: propensities reach 0.999."""
    w1, w0 = T / p, (~T) / (1 - p)
    return (Y * w1[:, None]).sum(0) / w1.sum() - (Y * w0[:, None]).sum(0) / w0.sum()


def est_ipw(Y, T, X, _true_p, **kw):
    p = _fit_propensity(X, T)
    return _hajek(Y, T, p)


def est_oracle_ipw(Y, T, X, true_p, **kw):
    return _hajek(Y, T, np.asarray(true_p))


def est_ols(Y, T, X, _true_p, **kw):
    """Per-pixel OLS of Y on [T, basis(Z)]; tau_hat is the coefficient on T.

    Solved for all K pixels at once -- the design matrix does not depend on the
    pixel, so this is one factorisation and a matmul, not K regressions.
    """
    D = np.hstack([T[:, None].astype(np.float64), X])
    return _ols(D, Y)[0]


def _fit_propensity(X, T, folds=None):
    lr = LogisticRegression(max_iter=2000, C=1e3)
    lr.fit(X, T.astype(int))
    return np.clip(lr.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)


def est_aipw(Y, T, X, _true_p, n_folds=5, seed=0, **kw):
    """Cross-fitted AIPW. Nuisances are never evaluated on their own fold."""
    n, K = Y.shape
    mu1 = np.zeros((n, K))
    mu0 = np.zeros((n, K))
    p = np.zeros(n)
    for tr, te in KFold(n_folds, shuffle=True, random_state=seed).split(X):
        lr = LogisticRegression(max_iter=2000, C=1e3).fit(X[tr], T[tr].astype(int))
        p[te] = lr.predict_proba(X[te])[:, 1]
        for t, mu in ((True, mu1), (False, mu0)):
            m = tr[T[tr] == t]
            mu[te] = X[te] @ _ols(X[m], Y[m])
    p = np.clip(p, 1e-6, 1 - 1e-6)
    g1 = (T[:, None] / p[:, None]) * (Y - mu1) + mu1
    g0 = ((~T)[:, None] / (1 - p)[:, None]) * (Y - mu0) + mu0
    return (g1 - g0).mean(0)


ESTIMATORS = {"naive": est_naive, "ipw": est_ipw, "ols": est_ols,
              "aipw": est_aipw, "oracle_ipw": est_oracle_ipw}


# --------------------------------------------------------------------------- #
# scoring -- identical keys to exp_ate_recovery.evaluate
# --------------------------------------------------------------------------- #
def _regional(err: np.ndarray, masks) -> dict:
    disc, ring, far = masks
    e = np.where(np.isfinite(err), err, np.nan)
    out = {"mae_all": float(np.nanmean(np.abs(e))), "rmse_all": float(np.sqrt(np.nanmean(e ** 2)))}
    for name, m in (("disc", disc), ("ring", ring), ("far", far)):
        out[f"signed_{name}"] = float(np.nanmean(e[m]))
        out[f"mae_{name}"] = float(np.nanmean(np.abs(e[m])))
    return out


def score(tau_hat: np.ndarray, data: dict, masks) -> dict:
    """The whole-image scores exp_ate_recovery reports, plus the regional ones."""
    ATE = np.asarray(data["ATE"])
    support = ATE != 0
    err = tau_hat - ATE
    return {
        "ate_mae": float(np.abs(err).mean()),
        "ate_rmse": float(np.sqrt((err**2).mean())),
        "ate_max_abs_err": float(np.abs(err).max()),
        "ate_mae_on_support": float(np.abs(err[support]).mean()),
        "ate_mae_off_support": float(np.abs(err[~support]).mean()),
        "ate_corr": float(np.corrcoef(tau_hat, ATE)[0, 1]),
        "att_mae": float(np.abs(tau_hat - np.asarray(data["ATT"])).mean()),
        "atc_mae": float(np.abs(tau_hat - np.asarray(data["ATC"])).mean()),
        **_regional(err, masks),
    }


def dataset_config(preset: str, size: int, seed_data: int, radius=None, digit=0, n=None,
                   seed_assign=None, **generator_overrides):
    """The exp_ate_recovery.Config that builds this dataset -- the same class, the same
    build_data, so the bytes are the ones a flow run on these arguments saw."""
    import exp_ate_recovery as E
    return E.Config(preset=preset, size=size, radius=radius, digit=digit, n=n,
                    seed_data=seed_data, seed_assign=seed_assign, arm="flexible_continuous",
                    **{k: v for k, v in generator_overrides.items() if v is not None})


def run_name(cfg, uid: str) -> str:
    """``baselines_<preset>_[<variant>_]k<K>_sd<seed>_d<digit>_<uid>``. ``sd`` is the DATA seed:
    a baseline has no fit seed. ``bs<shift>`` whenever base_shift was set (``bs0`` for a
    zero effect), ``rct`` when confounding was switched off on a confounded preset,
    ``sa<k>`` when the assignment was re-drawn with its own seed."""
    import exp_ate_recovery as E
    var = []
    if cfg.base_shift is not None:
        var.append(f"bs{cfg.base_shift:g}")
    if cfg.ps_slope == 0 and E.PRESET_SHORT[cfg.preset][:2] != "e1":
        var.append("rct")
    if cfg.seed_assign is not None:
        var.append(f"sa{cfg.seed_assign}")
    return "_".join(["baselines", E.PRESET_SHORT[cfg.preset][:2]] + var
                    + [f"k{cfg.size ** 2}", f"sd{cfg.seed_data}", E.digit_tag(cfg), uid])


def run_one(cfg, basis: str, runs_root: str = RUNS_ROOT, plots: bool = True) -> dict:
    """Every estimator on one dataset, written as one run folder (one dataset, five
    tau_hat arrays, five score blocks). Returns the metrics record."""
    import secrets
    import time
    from dataclasses import asdict
    from datetime import datetime, timezone

    import exp_ate_recovery as E

    t0 = time.monotonic()
    data = E.build_data(cfg)
    uid = secrets.token_hex(3)
    name = run_name(cfg, uid)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_dir = os.path.join(runs_root, f"{stamp}_{name}")
    os.makedirs(run_dir, exist_ok=False)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"run_id": os.path.basename(run_dir), "wandb_name": name, "uid": uid,
                   "dataset_id": data["dataset_id"], "data_hash": data["data_hash"],
                   "config": {**asdict(cfg), "basis": basis, "methods": list(ESTIMATORS)},
                   "generator_config": data["generator_config"]}, f, indent=2)

    Y = np.asarray(data["Y"], dtype=np.float64)
    Tb = np.asarray(data["X"])[:, 0].astype(bool)
    X = design_matrix(np.asarray(data["Z"]), basis)
    true_p = np.asarray(data["PROPENSITY"])
    masks = E.region_masks(cfg.size, cfg.effective_radius)
    ITE = np.asarray(data["ITE"])
    Y0 = Y - Tb[:, None] * ITE
    Y1 = Y0 + ITE

    log_lines = [f"{os.path.basename(run_dir)}", f"dataset_id {data['dataset_id']}  data_hash {data['data_hash']}",
                 f"{Y.shape[0]} units x {Y.shape[1]} pixels, basis {basis}, seed_data {cfg.seed_data}"]
    metrics = {"run_id": os.path.basename(run_dir), "dataset_id": data["dataset_id"],
               "data_hash": data["data_hash"], "basis": basis, "n_units": int(Y.shape[0]),
               "n_pixels": int(Y.shape[1]), "methods": {}}
    arrays = {k: np.asarray(data[k]) for k in ("ATE", "ATT", "ATC", "Y", "X", "ITE", "PROPENSITY")}
    om = E.observed_maps(data)                 # observed difference and imbalance, all images
    arrays["obs_diff"] = om["obs_diff"]
    metrics["imbalance"] = E._regional_metrics("imb", om["imbalance"], masks)
    for mname, fn in ESTIMATORS.items():
        t1 = time.monotonic()
        tau = np.asarray(fn(Y, Tb, X, true_p, seed=cfg.seed_data))
        m = score(tau, data, masks)
        m.update(E._regional_metrics("vsobs", tau - om["obs_diff"], masks))   # estimate − observed
        m["seconds"] = time.monotonic() - t1
        e0 = e1 = None
        if mname == "naive":      # the only estimator with per-arm means to compare
            e0 = Y[~Tb].mean(axis=0) - Y0.mean(axis=0)
            e1 = Y[Tb].mean(axis=0) - Y1.mean(axis=0)
            arrays["e0_naive"], arrays["e1_naive"] = e0, e1
        metrics["methods"][mname] = m
        arrays[f"tau_hat_{mname}"] = tau
        if plots:
            os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
            E.plot_ate_maps(cfg.size, cfg.effective_radius, tau, arrays["ATE"],
                            os.path.join(run_dir, "plots", f"ate_maps_{mname}.png"),
                            title=f"{name}  |  {mname}", e0=e0, e1=e1, obs_diff=om["obs_diff"])
        log_lines.append(f"  {mname:10s} mae {m['mae_all']:.4f}  signed disc {m['signed_disc']:+.4f} "
                         f"ring {m['signed_ring']:+.4f} far {m['signed_far']:+.4f}  ({m['seconds']:.1f}s)")
    metrics["wall_s"] = time.monotonic() - t0
    np.savez(os.path.join(run_dir, "arrays.npz"), **arrays)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(run_dir, "wandb.json"), "w") as f:
        json.dump({"id": None, "name": name, "note": "baselines are not logged to wandb"}, f, indent=2)
    with open(os.path.join(run_dir, "log.txt"), "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print("\n".join(log_lines), flush=True)
    if os.path.realpath(runs_root) == os.path.realpath(RUNS_ROOT):
        import run_index
        run_index.upsert_baselines(run_dir)
        print(f"indexed in {run_index.BASELINES_INDEX}")
    else:
        print("not indexed: run folder is outside runs/baselines")
    return metrics


def existing_dataset_ids(runs_root: str = RUNS_ROOT) -> set:
    ids = set()
    for p in glob.glob(os.path.join(runs_root, "*", "config.json")):
        with open(p) as f:
            ids.add(json.load(f).get("dataset_id"))
    return ids


def datasets_from_flow_index(index_csv: str) -> list:
    """One Config per distinct dataset the flow runs used, read from their index."""
    import csv

    import exp_ate_recovery as E
    seen, out = set(), []
    with open(index_csv, newline="") as f:
        for r in csv.DictReader(f):
            if not r.get("dataset_id"):
                continue
            if r["dataset_id"] in seen:
                continue
            seen.add(r["dataset_id"])
            kw = {}
            for k in ("size", "radius", "digit", "n", "seed_data", "seed_assign", *E.GENERATOR_OVERRIDE_KEYS):
                v = r.get(f"cfg.{k}", "")
                if v in ("", None):
                    continue
                try:
                    kw[k] = int(float(v)) if k in ("size", "radius", "digit", "n", "seed_data", "seed_assign") else float(v)
                except ValueError:
                    kw[k] = v
            if "size" not in kw:            # the batch layout's config has size but not radius/digit
                kw["size"] = int(r["size"])
            kw.setdefault("seed_data", int(r["seed_data"]))
            kw.setdefault("digit", 0 if r.get("digit", "") == "" else int(float(r["digit"])))
            out.append((r["dataset_id"], dataset_config(r["preset_full"], **kw)))
    return out


def main(argv=None):
    import exp_ate_recovery as E
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="exp1_rct_homogeneous", choices=list(PRESETS))
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--radius", type=int, default=None, help="effect-map radius; default round(size/4), as exp_ate_recovery")
    ap.add_argument("--digit", type=int, default=0)
    ap.add_argument("--all-digits", action="store_true")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed-data", type=int, default=101)
    ap.add_argument("--seed-assign", type=int, default=None,
                    help="re-draw the treatment assignment with its own seed, images and noise fixed by --seed-data")
    for k in E.GENERATOR_OVERRIDE_KEYS:
        ap.add_argument(f"--{k.replace('_', '-')}", default=None,
                        type=(str if k in ("effect_mode", "h_shape", "g_shape", "effect") else float))
    ap.add_argument("--basis", default="poly3", choices=BASES)
    ap.add_argument("--runs-root", default=RUNS_ROOT)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--from-index", action="store_true",
                    help="run on every distinct dataset in runs/exp_ate_recovery/index.csv "
                         "that has no baseline folder yet")
    args = ap.parse_args(argv)

    if args.from_index:
        import run_index
        have = existing_dataset_ids(args.runs_root)
        todo = [(i, c) for i, c in datasets_from_flow_index(run_index.INDEX) if i not in have]
        print(f"{len(todo)} dataset(s) without a baseline folder")
        for i, cfg in todo:
            run_one(cfg, args.basis, args.runs_root, plots=not args.no_plots)
        return

    overrides = {k: getattr(args, k) for k in E.GENERATOR_OVERRIDE_KEYS}
    cfg = dataset_config(args.preset, args.size, args.seed_data, radius=args.radius,
                         digit=None if args.all_digits else args.digit, n=args.n,
                         seed_assign=args.seed_assign, **overrides)
    run_one(cfg, args.basis, args.runs_root, plots=not args.no_plots)


if __name__ == "__main__":
    main()
