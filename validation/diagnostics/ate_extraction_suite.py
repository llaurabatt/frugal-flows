"""Can the frugal flow extract the ATE across continuous OUTCOME families?

A no-notebook, fast, script-based generalisation of the notebook's identification
check. For each outcome family (Gaussian, Gamma, ...) and each causal-margin arm
(`gaussian` location-shift vs `flexible_continuous` treatment-conditioned spline)
it:

  1. generates a causl ground-truth dataset (known true ATE, per the family's link);
  2. fits the frugal flow;
  3. extracts the ATE MODEL-AGNOSTICALLY via paired common-random-numbers MC
     (sample the full fitted flow at fixed T, read dim 0 = Y) — works for every
     causal_model, unlike the gaussian-only `.ate` field;
  4. runs numerical pass/fail checks ("test sticks"): runs-clean, ATE recovery,
     interventional-mean match;
  5. renders two PNG figures (interventional-margin grid + ATE scorecard) sized
     for viewing on a phone.

Two tiers, like quick_sense_check.py:
  SMOKE (default, n=2000): runs-clean is HARD-gated; ATE/mean are informational.
  FULL  (--full, n=20000):  ATE/mean recovery additionally HARD-gated.

Usage (from validation/, in the frugal-flows-flowjax env):
  micromamba run -n frugal-flows-flowjax python -m diagnostics.ate_extraction_suite
  micromamba run -n frugal-flows-flowjax python -m diagnostics.ate_extraction_suite \
      --families gaussian,gamma --arms gaussian,flexible_continuous --full
"""

from __future__ import annotations

import argparse
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)  # must precede any jnp array creation

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless: write PNGs, no display
import matplotlib.pyplot as plt  # noqa: E402

_VALIDATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VALIDATION_DIR not in sys.path:
    sys.path.insert(0, _VALIDATION_DIR)

import data_processing_and_simulations.causl_sim_data_generation as causl_py  # noqa: E402
from diagnostics.outcome_families import FAMILIES  # noqa: E402
from diagnostics.quick_sense_check import (  # noqa: E402
    base_hyperparams,
    fit_model,
    model_args,  # noqa: F401  (kept for parity / external reuse)
)

ARM_LABEL = {
    "gaussian": "additive shift",
    "flexible_continuous": "spline",
    "location_translation": "loc-translation",
}


def intervene(key, ff, cond_dim, n_mc):
    """Paired-CRN interventional readout. Returns the do(0)/do(1) Y samples + summaries.

    Same base draw (same `key`) pushed through T=0 and T=1, so tau(u)=Q_1(u)-Q_0(u)
    is paired. ATE = mean(tau); tau_sd = std(tau) = effect heterogeneity across
    quantiles (~0 for a pure location shift, >0 for a genuine spline effect).
    """
    y0 = np.asarray(ff.sample(key, condition=jnp.zeros((n_mc, cond_dim)))[:, 0])
    y1 = np.asarray(ff.sample(key, condition=jnp.ones((n_mc, cond_dim)))[:, 0])
    tau = y1 - y0
    return {
        "y0": y0, "y1": y1,
        "mean0": float(np.mean(y0)), "mean1": float(np.mean(y1)),
        "var0": float(np.var(y0)), "var1": float(np.var(y1)),
        "ate": float(np.mean(tau)), "tau_sd": float(np.std(tau)),
        "frac_neg": float(np.mean(np.concatenate([y0, y1]) <= 0)),
        "anynan": bool(np.any(np.isnan(y0)) or np.any(np.isnan(y1))),
    }


TAU_CURVE_BINS = 40  # fixed => identical u-grid across seeds (stackable curves)


def tau_curve(y0, y1, n_bins=TAU_CURVE_BINS):
    """Quantile-resolved paired effect tau(u) = Q_1(u) - Q_0(u) on a fixed u-grid.

    Pairs are aligned by base draw (same key in `intervene`), so tau[i]=y1[i]-y0[i]
    is the effect at draw i's latent quantile. The causal margin is monotone, so
    ranking by the control outcome y0 recovers that quantile; binning tau by y0-rank
    then gives tau as a function of u in (0,1). A FIXED n_bins yields an identical
    u-grid across seeds, so per-seed curves stack directly for a bias (seed-mean)
    vs variance (seed-SD) decomposition.

    For a pure location shift the truth is flat at the ATE; any slope/curvature the
    spline shows on such a DGP is spurious. Returns (u_centers[n_bins], tau_of_u[n_bins]).
    """
    y0 = np.asarray(y0); y1 = np.asarray(y1)
    tau = (y1 - y0)[np.argsort(np.asarray(y0), kind="stable")]
    n = len(tau)
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    tau_of_u = np.array([tau[a:b].mean() if b > a else np.nan
                         for a, b in zip(edges[:-1], edges[1:])])
    u_centers = (np.arange(n_bins) + 0.5) / n_bins
    return u_centers, tau_of_u


def run_cell(key, fam, arm, Y, u_z, X, args):
    """Fit one (family, arm) and score it. Returns a result dict (+ samples for plots)."""
    cp = [args.const, args.ate]
    true_ate = fam.true_ate(cp)
    true_m0, true_m1 = fam.mean_do(cp, 0), fam.mean_do(cp, 1)

    try:
        ff, val_loss = fit_model(jr.fold_in(key, 1), Y, u_z, X, arm, args.epochs)
        m = intervene(jr.fold_in(key, 2), ff, X.shape[1], args.n_mc)
        crashed = None
    except Exception as e:  # keep the suite running; report the failure in the row
        return {
            "family": fam.name, "arm": arm, "crashed": repr(e),
            "true_ate": true_ate, "ate": float("nan"), "ate_relerr": float("nan"),
            "tau_sd": float("nan"), "mean0": float("nan"), "mean1": float("nan"),
            "true_m0": true_m0, "true_m1": true_m1, "frac_neg": float("nan"),
            "t_runs": False, "t_ate": False, "t_mean": False, "pass": False,
            "y0": None, "y1": None,
        }

    ate = m["ate"]
    ate_relerr = abs(ate - true_ate) / max(abs(true_ate), 1e-8)
    mean_err = max(abs(m["mean0"] - true_m0), abs(m["mean1"] - true_m1))

    t_runs = bool(np.isfinite(val_loss)) and (not m["anynan"]) and m["var0"] > 0 and m["var1"] > 0
    t_ate = ate_relerr < args.tol
    t_mean = mean_err < args.mean_tol
    passed = (t_runs and t_ate and t_mean) if args.full else t_runs

    return {
        "family": fam.name, "arm": arm, "crashed": crashed,
        "val_loss": val_loss, "true_ate": true_ate, "ate": ate, "ate_relerr": ate_relerr,
        "tau_sd": m["tau_sd"], "mean0": m["mean0"], "mean1": m["mean1"],
        "true_m0": true_m0, "true_m1": true_m1, "var0": m["var0"], "var1": m["var1"],
        "frac_neg": m["frac_neg"], "mean_err": mean_err,
        "t_runs": t_runs, "t_ate": t_ate, "t_mean": t_mean, "pass": passed,
        "y0": m["y0"], "y1": m["y1"],
    }


# ----------------------------------------------------------------------------- figures
def fig_margins(rows, families, arms, cp, outpath, seed):
    """Grid: rows = outcome families, cols = arms. Each cell overlays the model's
    interventional Y|do(0)/Y|do(1) on the analytic ground-truth shape (dashed)."""
    rng = np.random.default_rng(seed)
    nf, na = len(families), len(arms)
    fig, axes = plt.subplots(nf, na, figsize=(5.2 * na, 4.0 * nf), squeeze=False)
    by = {(r["family"], r["arm"]): r for r in rows}

    for i, fname in enumerate(families):
        fam = FAMILIES[fname]
        # shared bins per family row (truth do(0)+do(1)) so cells are comparable
        t0 = fam.sample_truth(cp, 0, 40000, rng)
        t1 = fam.sample_truth(cp, 1, 40000, rng)
        lo, hi = np.percentile(np.concatenate([t0, t1]), [0.5, 99.5])
        bins = np.linspace(lo, hi, 60)
        for j, arm in enumerate(arms):
            ax = axes[i][j]
            r = by.get((fname, arm))
            # truth outlines
            ax.hist(t0, bins=bins, density=True, histtype="step", color="C0", ls="--", lw=1.6, label="truth do(0)")
            ax.hist(t1, bins=bins, density=True, histtype="step", color="C1", ls="--", lw=1.6, label="truth do(1)")
            if r is not None and r["y0"] is not None:
                ax.hist(r["y0"], bins=bins, density=True, alpha=0.45, color="C0", label="model do(0)")
                ax.hist(r["y1"], bins=bins, density=True, alpha=0.45, color="C1", label="model do(1)")
                title = (f"{fname} / {ARM_LABEL.get(arm, arm)}\n"
                         f"ATE {r['ate']:+.2f} (true {r['true_ate']:+.2f}, "
                         f"err {r['ate_relerr']:.0%})")
            else:
                title = f"{fname} / {ARM_LABEL.get(arm, arm)}\n(crashed)"
            ax.set_title(title, fontsize=12)
            ax.set_xlim(lo, hi)
            ax.tick_params(labelsize=9)
            if i == 0 and j == 0:
                ax.legend(fontsize=8, framealpha=0.9)
    fig.suptitle("Interventional outcome margins: model vs causl truth", fontsize=15, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def fig_scorecard(rows, families, arms, outpath):
    """ATE recovery at a glance: true ATE (black bar) vs each arm's estimate (dots)."""
    fig, ax = plt.subplots(figsize=(max(6, 2.2 * len(families)), 5.2))
    by = {(r["family"], r["arm"]): r for r in rows}
    x = np.arange(len(families))
    width = 0.6
    # draw true ATE as a wide black tick per family
    for i, fname in enumerate(families):
        r_any = next((by[(fname, a)] for a in arms if (fname, a) in by), None)
        if r_any is None:
            continue
        ta = r_any["true_ate"]
        ax.hlines(ta, i - width / 2, i + width / 2, color="black", lw=3, zorder=3,
                  label="true ATE" if i == 0 else None)
        ax.annotate(f"{ta:+.2f}", (i, ta), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=10, fontweight="bold")
    # arm estimates
    colors = {"gaussian": "#d62728", "flexible_continuous": "#1f77b4", "location_translation": "#2ca02c"}
    offs = np.linspace(-0.18, 0.18, len(arms))
    for k, arm in enumerate(arms):
        xs, ys = [], []
        for i, fname in enumerate(families):
            r = by.get((fname, arm))
            if r is None or not np.isfinite(r["ate"]):
                continue
            xs.append(i + offs[k]); ys.append(r["ate"])
            ax.annotate(f"{r['ate_relerr']:.0%}", (i + offs[k], r["ate"]),
                        textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8,
                        color=colors.get(arm, "gray"))
        ax.scatter(xs, ys, s=130, color=colors.get(arm, "gray"), zorder=4,
                   label=f"{ARM_LABEL.get(arm, arm)} est", edgecolor="white", linewidth=1.2)
    ax.set_xticks(x); ax.set_xticklabels(families, fontsize=12)
    ax.set_ylabel("ATE = E[Y|do(1)] - E[Y|do(0)]", fontsize=12)
    ax.set_title("ATE recovery by outcome family and causal-margin arm\n"
                 "(% = relative error vs truth)", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=10, loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--families", default="gaussian,gamma", help="comma-separated outcome families")
    p.add_argument("--arms", default="gaussian,flexible_continuous", help="comma-separated causal_model arms")
    p.add_argument("--full", action="store_true", help="recovery tier: large n + HARD ATE/mean gating")
    p.add_argument("--n", type=int, default=None, help="train n (default 2000 smoke / 20000 full)")
    p.add_argument("--epochs", type=int, default=None, help="max epochs (default 400 smoke / 1500 full)")
    p.add_argument("--n-mc", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--const", type=float, default=0.0, help="causal_params[0] (log-mean for gamma)")
    p.add_argument("--ate", type=float, default=1.0, help="causal_params[1] (log-scale effect for gamma)")
    p.add_argument("--tol", type=float, default=0.15, help="relative ATE tol (full tier)")
    p.add_argument("--mean-tol", type=float, default=0.4, help="abs interventional-mean tol (full tier)")
    p.add_argument("--outdir", default=None)
    args = p.parse_args()

    if args.n is None:
        args.n = 20000 if args.full else 2000
    if args.epochs is None:
        args.epochs = 1500 if args.full else 400

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for f in families:
        if f not in FAMILIES:
            raise SystemExit(f"unknown family {f!r}; known: {list(FAMILIES)}")
    cp = [args.const, args.ate]
    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(outdir, exist_ok=True)

    tier = "FULL (hard ATE/mean gating)" if args.full else "SMOKE (runs-clean gated; ATE informational)"
    print(f"tier={tier}")
    print(f"families={families}  arms={arms}  n={args.n}  epochs={args.epochs}  seed={args.seed}")
    print(f"causal_params=[const={args.const}, ate={args.ate}]\n")

    key = jr.key(args.seed)
    rows = []
    for fi, fname in enumerate(families):
        fam = FAMILIES[fname]
        data = fam.generate(args.n, causal_params=cp, seed=args.seed)
        X, Y, Z_disc, Z_cont = data["X"], data["Y"], data["Z_disc"], data["Z_cont"]
        uz = causl_py.generate_uz_samples(Z_disc, Z_cont, False, args.seed, base_hyperparams(args.epochs))
        u_z = uz["uz_samples"]
        print(f"[{fname}] data X{tuple(X.shape)} Y{tuple(Y.shape)} u_z{tuple(u_z.shape)}  "
              f"treated_frac={float(np.asarray(X).mean()):.3f}  true_ate={fam.true_ate(cp):+.3f}")
        for ai, arm in enumerate(arms):
            r = run_cell(jr.fold_in(key, 100 * fi + ai), fam, arm, Y, u_z, X, args)
            if r.get("crashed"):
                print(f"    {arm:22s} CRASHED: {r['crashed'][:80]}")
            else:
                print(f"    {arm:22s} ATE={r['ate']:+.3f} (true {r['true_ate']:+.3f}, err {r['ate_relerr']:.1%})  "
                      f"tau_sd={r['tau_sd']:.3f}  E[Y|do]={r['mean0']:.2f}/{r['mean1']:.2f}  "
                      f"frac_neg={r['frac_neg']:.2f}")
            rows.append(r)
        print()

    # ---- numerical scorecard table ----
    def mk(b):
        return "OK" if b else "X"

    hdr = (f"{'family':<10}{'arm':<22}{'ATE':>8}{'true':>8}{'relerr':>8}{'tau_sd':>8}"
           f"{'runs':>6}{'ate':>5}{'mean':>6}  verdict")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        verdict = "PASS" if r["pass"] else "FAIL"
        if r.get("crashed"):
            print(f"{r['family']:<10}{r['arm']:<22}{'--':>8}{r['true_ate']:>8.2f}{'--':>8}{'--':>8}"
                  f"{mk(False):>6}{mk(False):>5}{mk(False):>6}  CRASH")
            continue
        print(f"{r['family']:<10}{r['arm']:<22}{r['ate']:>8.3f}{r['true_ate']:>8.2f}{r['ate_relerr']:>7.1%}"
              f"{r['tau_sd']:>8.3f}{mk(r['t_runs']):>6}{mk(r['t_ate']):>5}{mk(r['t_mean']):>6}  {verdict}")

    # ---- figures ----
    f1 = os.path.join(outdir, "ate_suite_margins.png")
    f2 = os.path.join(outdir, "ate_suite_scorecard.png")
    fig_margins(rows, families, arms, cp, f1, args.seed)
    fig_scorecard(rows, families, arms, f2)
    print(f"\nfigures: {f1}\n         {f2}")

    overall = all(r["pass"] for r in rows)
    gate = "smoke + ATE + mean" if args.full else "runs-clean only (ATE informational)"
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}  (gated: {gate})")
    if not args.full:
        print("note: ATE recovery is NOT gated in smoke tier (needs ~20k samples). Use --full for a hard check.")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
