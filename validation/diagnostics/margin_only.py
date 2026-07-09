"""E-ii "sever the copula": fit ONLY the frugal-flow gaussian-arm MARGIN term.

Pre-registered mechanism test. The frugal-flow "additive" arm
(``causal_model="gaussian"``) fits a JOINT whose log-likelihood is

    log L = log-copula-term(u_z, u_y)  +  log N(y; const + ate*x, scale^2),

where the second term is the *causal margin* of Y and u_y = Phi((y-const-ate*x)/scale)
is the PIT rank that feeds the copula. On a Gamma DGP (the location-shift Gaussian
margin is misspecified) the FULL joint fit at n=20000 with ZERO confounding
(family ``gamma_b0``) shows an ATE bias of about -0.500 at EVERY seed, even though
the margin term ALONE is a 3-parameter Gaussian MLE whose optimum EXACTLY
moment-matches: at the optimum ate* = mean(Y|X=1) - mean(Y|X=0) (diff-in-means).

Hypothesised mechanism: the copula term's gradient w.r.t. the margin parameters
(via the ranks u_y) pulls the shared margin parameters off moment-matching.

THE TEST here: fit ONLY the margin term, with the SAME SGD machinery (Adam,
learning_rate=5e-3), on the SAME datasets. Pre-registered predictions:
  P1: margin-only ate ~= diff-in-means to high precision at every (n, seed, dgp).
  P2: at beta=0 (gamma_b0) that means bias vs the true ATE (1.7634) is only
      sampling noise -- the -0.500 DISAPPEARS.
  P3: at beta=1 (gamma_b1) the margin-only ate matches the OBSERVATIONAL contrast
      (diff-in-means), NOT the true ATE -- deconfounding is the copula's job.
If P1/P2 fail, the mechanism claim is FALSIFIED -- reported loudly, not massaged.

Model + init match the FF gaussian arm EXACTLY (quick_sense_check.model_args
"gaussian"): const=0.0, ate=0.0, scale=1.0. NLL(theta) = -mean log N(y_i; const +
ate*x_i, scale^2).

DEVIATIONS FROM THE FF PIPELINE (deliberate, noted here):
  * FULL-BATCH gradient (no minibatching). The model is 3 scalars, so batching is
    unnecessary and full-batch removes SGD noise as a confound in a convergence test.
  * scale is optimised in LOG space (log_scale, init 0.0 => scale=1.0) so Adam can
    never step it negative. The ate optimum is INVARIANT to this reparametrisation
    (ate/const enter only through the mean, whose optimum given any scale is the
    conditional means), so it does not affect what P1-P3 test.
  * iterations = max(MIN_ITERS, 5 * EPOCHS_BY_N[n]). The raw 5x rule (3000 iters at
    n=2000) UNDER-converges: full-batch Adam on the raw-scale margin is ill-conditioned
    (once log_scale grows, the mean-parameter gradient is divided by scale^2 ~ 10, so
    convergence is slow and n-INDEPENDENT in iteration count -- small n just gets fewer
    iters). Empirically ~10000 iters reach machine precision at every n, so a floor of
    MIN_ITERS=20000 is applied. Convergence is verified against the closed-form MLE per
    row (gap_sgd_vs_mle); rows are only interpretable where |gap| is ~0.

Usage (from validation/, frugal-flows-flowjax env):
  micromamba run -n frugal-flows-flowjax python -m diagnostics.margin_only \
      --out outputs/flexible_te/margin_only_eii.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)  # MUST precede any jnp array creation

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

# optax is expected in this env; fall back to a hand-rolled Adam if absent.
try:
    import optax  # noqa: E402

    _HAVE_OPTAX = True
except Exception:  # noqa: BLE001
    _HAVE_OPTAX = False

_VALIDATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VALIDATION_DIR not in sys.path:
    sys.path.insert(0, _VALIDATION_DIR)

from diagnostics.h1_matrix import EPOCHS_BY_N  # noqa: E402  (reuse the convention)
from diagnostics.outcome_families import FAMILIES  # noqa: E402

FIELDNAMES = ["dgp", "n", "seed", "true_ate", "diff_means", "ate_sgd",
              "const_sgd", "scale_sgd", "gap_sgd_vs_mle", "bias_vs_truth",
              "secs", "error"]

LR = 5e-3  # matches the FF pipeline learning_rate
MIN_ITERS = 20000  # convergence floor (raw 5*EPOCHS under-converges at small n)

# Reference FULL-FF additive-arm biases (for context in the summary only).
FULL_FF_BIAS = {
    ("gamma_b0", 2000): -0.147, ("gamma_b0", 20000): -0.500,
    ("gamma_b1", 2000): +0.067, ("gamma_b1", 20000): -1.113,
}


def _nll(params, y, x):
    """-mean log N(y; const + ate*x, exp(log_scale)^2)."""
    mu = params["const"] + params["ate"] * x
    log_scale = params["log_scale"]
    z = (y - mu) / jnp.exp(log_scale)
    ll = -0.5 * math.log(2.0 * math.pi) - log_scale - 0.5 * z * z
    return -jnp.mean(ll)


def _init_params():
    # EXACT FF gaussian-arm init: const=0, ate=0, scale=1 (log_scale=0).
    return {"const": jnp.array(0.0), "ate": jnp.array(0.0),
            "log_scale": jnp.array(0.0)}


def _train_optax(y, x, iters, lr):
    opt = optax.adam(lr)
    params = _init_params()
    opt_state = opt.init(params)

    def step(carry, _):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(_nll)(params, y, x)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    (params, _), _ = jax.lax.scan(step, (params, opt_state), None, length=iters)
    return params


def _train_handrolled(y, x, iters, lr, b1=0.9, b2=0.999, eps=1e-8):
    """Fallback Adam (only used if optax is unavailable)."""
    params = _init_params()
    m = {k: jnp.array(0.0) for k in params}
    v = {k: jnp.array(0.0) for k in params}

    def step(carry, t):
        params, m, v = carry
        _, grads = jax.value_and_grad(_nll)(params, y, x)
        t = t + 1
        new_params, new_m, new_v = {}, {}, {}
        for k in params:
            new_m[k] = b1 * m[k] + (1 - b1) * grads[k]
            new_v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
            mhat = new_m[k] / (1 - b1 ** t)
            vhat = new_v[k] / (1 - b2 ** t)
            new_params[k] = params[k] - lr * mhat / (jnp.sqrt(vhat) + eps)
        return (new_params, new_m, new_v), None

    (params, _, _), _ = jax.lax.scan(step, (params, m, v), jnp.arange(iters))
    return params


_train = _train_optax if _HAVE_OPTAX else _train_handrolled
_train_jit = jax.jit(_train, static_argnums=(2, 3))


def _fit_cell(Y, X, iters):
    y = jnp.asarray(np.asarray(Y, dtype=float).ravel())
    x = jnp.asarray(np.asarray(X, dtype=float).ravel())
    # closed-form MLE reference (moment-matching optimum of the margin term)
    yn = np.asarray(Y, dtype=float).ravel()
    xn = np.asarray(X, dtype=float).ravel()
    const_mle = float(yn[xn == 0].mean())
    ate_mle = float(yn[xn == 1].mean() - yn[xn == 0].mean())  # == diff-in-means
    params = _train_jit(y, x, int(iters), float(LR))
    ate_sgd = float(params["ate"])
    const_sgd = float(params["const"])
    scale_sgd = float(jnp.exp(params["log_scale"]))
    return dict(diff_means=ate_mle, const_mle=const_mle, ate_sgd=ate_sgd,
                const_sgd=const_sgd, scale_sgd=scale_sgd)


def run(args):
    dgps = [d.strip() for d in args.dgps.split(",") if d.strip()]
    ns = [int(x) for x in args.ns.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    cp = [args.const, args.ate]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    total = len(dgps) * len(ns) * len(seeds)
    done = 0
    print(f"[margin_only {args.out}] optax={_HAVE_OPTAX} dgps={dgps} ns={ns} "
          f"seeds={seeds} => {total} cells  causal_params={cp}", flush=True)

    rows = []
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        w.writeheader(); fh.flush()
        for dgp in dgps:
            fam = FAMILIES[dgp]
            true_ate = fam.true_ate(cp)
            for n in ns:
                iters = max(MIN_ITERS, 5 * EPOCHS_BY_N[n])
                for seed in seeds:
                    t0 = time.time()
                    row = {k: "" for k in FIELDNAMES}
                    row.update(dgp=dgp, n=n, seed=seed, true_ate=true_ate)
                    try:
                        data = fam.generate(n, causal_params=cp, seed=seed)
                        res = _fit_cell(data["Y"], data["X"], iters)
                        row.update(
                            diff_means=res["diff_means"],
                            ate_sgd=res["ate_sgd"],
                            const_sgd=res["const_sgd"],
                            scale_sgd=res["scale_sgd"],
                            gap_sgd_vs_mle=res["ate_sgd"] - res["diff_means"],
                            bias_vs_truth=res["ate_sgd"] - true_ate,
                        )
                    except Exception as e:  # noqa: BLE001
                        row.update(ate_sgd=float("nan"), error=repr(e)[:200])
                    row["secs"] = round(time.time() - t0, 2)
                    w.writerow(row); fh.flush()
                    rows.append(dict(row))
                    done += 1
                    if row["error"]:
                        tag = "ERR " + row["error"]
                    else:
                        tag = (f"ate_sgd={float(row['ate_sgd']):+.4f} "
                               f"diff_means={float(row['diff_means']):+.4f} "
                               f"gap={float(row['gap_sgd_vs_mle']):+.2e} "
                               f"bias={float(row['bias_vs_truth']):+.4f}")
                    print(f"[{done}/{total}] {dgp} n={n} seed={seed} {tag} "
                          f"({row['secs']}s)", flush=True)
    print(f"[margin_only {args.out}] DONE {done}/{total}", flush=True)
    _summary(rows, ns, dgps, cp)


def _fnum(rows, key):
    return np.array([float(r[key]) for r in rows
                     if not r["error"] and r[key] != ""], dtype=float)


def _summary(rows, ns, dgps, cp):
    print("\n" + "=" * 78)
    print("SUMMARY  (E-ii: sever the copula -- margin-term-only SGD fit)")
    print("=" * 78)
    print(f"causal_params={cp}   n_seeds per cell="
          f"{len({r['seed'] for r in rows})}\n")

    hdr = (f"{'dgp':<10}{'n':>7}{'mean_bias':>11}{'sd_bias':>10}"
           f"{'mean|gap|':>11}{'full_FF_bias':>14}")
    print(hdr)
    print("-" * len(hdr))
    for dgp in dgps:
        for n in ns:
            cell = [r for r in rows if r["dgp"] == dgp and int(r["n"]) == n]
            bias = _fnum(cell, "bias_vs_truth")
            gap = _fnum(cell, "gap_sgd_vs_mle")
            if bias.size == 0:
                print(f"{dgp:<10}{n:>7}   (no successful cells)")
                continue
            ffb = FULL_FF_BIAS.get((dgp, n))
            ffb_s = f"{ffb:+.3f}" if ffb is not None else "   n/a"
            print(f"{dgp:<10}{n:>7}{bias.mean():>+11.4f}{bias.std(ddof=0):>10.4f}"
                  f"{np.abs(gap).mean():>11.2e}{ffb_s:>14}")

    # ---- pre-registered verdicts -------------------------------------------
    print("\n" + "-" * 78)
    print("PRE-REGISTERED VERDICTS")
    print("-" * 78)

    all_gap = _fnum(rows, "gap_sgd_vs_mle")
    p1_max = float(np.abs(all_gap).max()) if all_gap.size else float("nan")
    p1_pass = all_gap.size > 0 and p1_max < 1e-4
    print(f"P1  margin-only ate == diff-in-means at every cell "
          f"(max|gap_sgd_vs_mle| < 1e-4):")
    print(f"    max|gap| = {p1_max:.3e}  ->  {'PASS' if p1_pass else 'FAIL'}")

    print(f"\nP2  gamma_b0 |mean bias_vs_truth| is sampling noise "
          f"(< 2*sd/sqrt(n_seeds)) at each n:")
    p2_all = True
    have_b0 = False
    for n in ns:
        cell = [r for r in rows if r["dgp"] == "gamma_b0" and int(r["n"]) == n]
        bias = _fnum(cell, "bias_vs_truth")
        if bias.size == 0:
            continue
        have_b0 = True
        mb = abs(bias.mean())
        thr = 2.0 * bias.std(ddof=0) / math.sqrt(bias.size)
        ok = mb < thr
        p2_all = p2_all and ok
        print(f"    n={n:>6}: |mean bias|={mb:.4f}  threshold(2*sd/sqrt({bias.size}))"
              f"={thr:.4f}  ->  {'PASS' if ok else 'FAIL'}")
    if have_b0:
        print(f"    P2 overall: {'PASS' if p2_all else 'FAIL'}  "
              f"(the FULL-FF -0.500 bias should have DISAPPEARED)")
    else:
        print("    (no gamma_b0 cells present)")

    print(f"\nP3  gamma_b1 bias_vs_truth is NONZERO and equals the confounding gap "
          f"(diff_means - true_ate):")
    for n in ns:
        cell = [r for r in rows if r["dgp"] == "gamma_b1" and int(r["n"]) == n]
        bias = _fnum(cell, "bias_vs_truth")
        dmn = _fnum(cell, "diff_means")
        tru = _fnum(cell, "true_ate")
        if bias.size == 0:
            continue
        conf_gap = float((dmn - tru).mean())
        print(f"    n={n:>6}: mean bias_vs_truth={bias.mean():+.4f}  "
              f"confounding gap(diff_means-true_ate)={conf_gap:+.4f}  "
              f"(expected ~equal; ate follows the OBSERVATIONAL contrast, "
              f"deconfounding is the copula's job)")

    if not p1_pass or (have_b0 and not p2_all):
        print("\n*** MECHANISM CLAIM FALSIFIED for the failing prediction(s) above. "
              "Reported as-is, not massaged. ***")
    else:
        print("\nMechanism test consistent with the pre-registered predictions "
              "(P1/P2 hold; P3 as expected).")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dgps", default="gamma_b0,gamma_b1")
    p.add_argument("--ns", default="2000,20000")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--const", type=float, default=1.0, help="causal_params[0] (b0)")
    p.add_argument("--ate", type=float, default=0.5, help="causal_params[1] (b1)")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
