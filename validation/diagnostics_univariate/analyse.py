"""S11 analysis (halo prereg Amendment A6).

Reads every complete ``~/work/halo-runs/S11/fits/<block>/<dataset>/<arm>/result.json`` and writes
``~/work/halo-runs/_analysis/S11_tables.md``, ``S11_tables.json`` and ``S11_univariate.png``.

Per (block, DGP, arm): n fits, mean ATE estimate, bias = mean - truth, sd across data seeds
(ddof 1), RMSE = sqrt(mean((est - truth)^2)), naive mean, covers = |mean - truth| <= 2 sd
(the article's "mean +- 2 sigma contains the truth").

A6 pass clauses, article block, each new arm separately, as written:
 (i)   |mean bias| <= 0.10 at ATE 1 and <= 0.25 at ATE 5 on every model;
 (ii)  mean +- 2 sd covers the truth on every model (all six model x ATE cells);
 (iii) RMSE not worse than the better of `gaussian` and `location_translation` by more than
       0.05 (ATE 1) / 0.15 (ATE 5), per model x ATE cell. Decision on the point estimate
       RMSE_new - RMSE_better, where "better" = the existing arm with the lower RMSE on the
       full set of paired seeds. Reported beside it: a paired percentile bootstrap 95% CI over
       data seeds (B = 10,000; resample the seed indices common to the three arms with
       replacement; recompute RMSE_new and RMSE_better on the SAME resample; take the difference).
Misspecification block (no pass line): "biased" = |mean bias| > 2 sd / sqrt(n) (the July H1 rule).
"""
from __future__ import annotations

import glob
import json
import math
import os

import numpy as np

ROOT = os.path.expanduser("~/work/halo-runs/S11")
OUT = os.path.expanduser("~/work/halo-runs/_analysis")
ARMS = ("gaussian", "location_translation", "flexible_continuous",
        "location_translation_gaussian", "flexible_continuous_gaussian")
NEW = ("location_translation_gaussian", "flexible_continuous_gaussian")
SHORT = {"gaussian": "gauss", "location_translation": "LT", "flexible_continuous": "FC",
         "location_translation_gaussian": "LT-G", "flexible_continuous_gaussian": "FC-G"}
# NeurIPS 2024 Table 1, Frugal Flow column: mean +- 2 sigma (25 runs, N = 25,000)
PUBLISHED = {("M1", 1): (0.98, 0.12), ("M1", 5): (5.00, 0.24), ("M2", 1): (1.01, 0.10),
             ("M2", 5): (5.01, 0.18), ("M3", 1): (1.00, 0.09), ("M3", 5): (5.18, 0.30)}
BIAS_TOL = {1: 0.10, 5: 0.25}
RMSE_TOL = {1: 0.05, 5: 0.15}
B = 10_000


def load():
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "fits", "*", "*", "*", "result.json"))):
        with open(p) as fh:
            r = json.load(fh)
        if not r.get("complete") or r.get("smoke_max_epochs") is not None:
            continue
        r["block"] = p.split(os.sep)[-4]
        r["ate_arg"] = int(round(r["causal_params"][1])) if r["model"] != "gamma_margin" else 0
        rows.append(r)
    failed = sorted(glob.glob(os.path.join(ROOT, "fits", "*", "*", "*", "FAILED")))
    return rows, failed


def dgp_key(r):
    return (r["block"], r["model"], r["ate_arg"], r["n"])


def dgp_label(k):
    b, m, a, n = k
    return f"{m} n={n}" if m == "gamma_margin" else f"{m} ATE={a} n={n}"


def summarise(rows):
    groups = {}
    for r in rows:
        groups.setdefault(dgp_key(r), {}).setdefault(r["arm"], []).append(r)
    table = []
    for k in sorted(groups, key=lambda k: (["article", "small", "misspec"].index(k[0]), k[2], k[1], k[3])):
        for arm in ARMS:
            rs = sorted(groups[k].get(arm, []), key=lambda r: r["seed_data"])
            if not rs:
                continue
            est = np.array([r["ate_hat"] for r in rs], float)
            truth = rs[0]["true_ate"]
            finite = np.isfinite(est)
            e = est[finite]
            sd = float(np.std(e, ddof=1)) if len(e) > 1 else float("nan")
            mean = float(np.mean(e)) if len(e) else float("nan")
            shift = [r["shift_minus_ate_hat"] for r in rs if r.get("shift_minus_ate_hat") is not None]
            table.append(dict(
                block=k[0], model=k[1], ate=k[2], n=k[3], dgp=dgp_label(k), arm=arm, n_fits=len(rs),
                n_nonfinite=int((~finite).sum()), truth=truth, mean=mean, bias=mean - truth, sd=sd,
                rmse=float(np.sqrt(np.mean((e - truth) ** 2))) if len(e) else float("nan"),
                naive=float(np.mean([r["naive"] for r in rs])), ols=float(np.mean([r["ols"] for r in rs])),
                covers=bool(abs(mean - truth) <= 2 * sd), real_bias=bool(abs(mean - truth) > 2 * sd / math.sqrt(len(e))),
                seeds=[r["seed_data"] for r in rs], est=est.tolist(),
                max_abs_shift_minus_ate=(float(np.max(np.abs(shift))) if shift else None),
                median_epochs=float(np.median([r["epochs_run"] for r in rs])),
                median_best_epoch=float(np.median([r["best_epoch"] for r in rs])),
                frac_hit_cap=float(np.mean([r["epochs_run"] >= r["hp"]["FHP"]["max_epochs"] for r in rs])),
                median_fit_s=float(np.median([r["wall_fit_s"] for r in rs])),
                published=PUBLISHED.get((k[1], k[2])) if k[0] == "article" else None))
    return table, groups


def paired_rmse_diff(groups, k, new, rng):
    """RMSE_new - RMSE_better over data seeds common to new, gaussian, location_translation."""
    by = {a: {r["seed_data"]: r["ate_hat"] for r in groups[k].get(a, [])} for a in (new, "gaussian", "location_translation")}
    common = sorted(set.intersection(*(set(v) for v in by.values())))
    if len(common) < 2:
        return None
    truth = groups[k][new][0]["true_ate"]
    se = {a: (np.array([by[a][s] for s in common]) - truth) ** 2 for a in by}
    rm = {a: float(np.sqrt(se[a].mean())) for a in se}
    better = min(("gaussian", "location_translation"), key=lambda a: rm[a])
    diff = rm[new] - rm[better]
    idx = rng.integers(0, len(common), size=(B, len(common)))
    boot = np.sqrt(se[new][idx].mean(1)) - np.sqrt(se[better][idx].mean(1))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(n_seeds=len(common), better=better, rmse_new=rm[new], rmse_better=rm[better],
                rmse_gaussian=rm["gaussian"], rmse_lt=rm["location_translation"],
                diff=diff, ci=[float(lo), float(hi)])


def clauses(table, groups):
    rng = np.random.default_rng(20261001)
    res = {}
    art = {(t["model"], t["ate"], t["arm"]): t for t in table if t["block"] == "article"}
    for new in NEW:
        out = {"i": [], "ii": [], "iii": []}
        for m in ("M1", "M2", "M3"):
            for a in (1, 5):
                t = art.get((m, a, new))
                if t is None:
                    out["i"].append(dict(model=m, ate=a, missing=True, ok=False))
                    out["ii"].append(dict(model=m, ate=a, missing=True, ok=False))
                    out["iii"].append(dict(model=m, ate=a, missing=True, ok=False))
                    continue
                out["i"].append(dict(model=m, ate=a, bias=t["bias"], tol=BIAS_TOL[a], n=t["n_fits"],
                                     ok=bool(abs(t["bias"]) <= BIAS_TOL[a])))
                out["ii"].append(dict(model=m, ate=a, mean=t["mean"], sd=t["sd"], lo=t["mean"] - 2 * t["sd"],
                                      hi=t["mean"] + 2 * t["sd"], ok=t["covers"]))
                d = paired_rmse_diff(groups, ("article", m, a, 25000), new, rng)
                if d is None:
                    out["iii"].append(dict(model=m, ate=a, missing=True, ok=False))
                else:
                    out["iii"].append(dict(model=m, ate=a, tol=RMSE_TOL[a], ok=bool(d["diff"] <= RMSE_TOL[a]), **d))
        complete = all(not c.get("missing") for v in out.values() for c in v)
        res[new] = dict(clauses=out, complete=complete,
                        **{f"pass_{c}": all(x["ok"] for x in out[c]) for c in out},
                        overall=complete and all(all(x["ok"] for x in out[c]) for c in out))
    return res


def fmt(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def write_md(table, cl, failed, path):
    L = ["# S11 — univariate stress test of the Gaussian-scale arms (halo prereg Amendment A6)", ""]
    L.append("## Pass clauses (article block), per new arm")
    for new, c in cl.items():
        verdict = "PASS" if c["overall"] else ("FAIL" if c["complete"] else "INCOMPLETE")
        L.append(f"\n### {new}: **{verdict}**  (i {'PASS' if c['pass_i'] else 'FAIL'}, "
                 f"ii {'PASS' if c['pass_ii'] else 'FAIL'}, iii {'PASS' if c['pass_iii'] else 'FAIL'})")
        L.append("\n| model | ATE | (i) bias / tol | (ii) mean ± 2 sd | (iii) RMSE new − better [95% CI] / tol | better arm |")
        L.append("|---|---|---|---|---|---|")
        for x1, x2, x3 in zip(c["clauses"]["i"], c["clauses"]["ii"], c["clauses"]["iii"]):
            if x1.get("missing"):
                L.append(f"| {x1['model']} | {x1['ate']} | missing | missing | missing | |")
                continue
            ok = lambda b: "ok" if b else "**FAIL**"
            iii = ("missing" if x3.get("missing") else
                   f"{x3['diff']:+.3f} [{x3['ci'][0]:+.3f}, {x3['ci'][1]:+.3f}] / {x3['tol']} {ok(x3['ok'])}")
            L.append(f"| {x1['model']} | {x1['ate']} | {x1['bias']:+.3f} / {x1['tol']} {ok(x1['ok'])} | "
                     f"{x2['mean']:.3f} ± {2 * x2['sd']:.3f} = [{x2['lo']:.3f}, {x2['hi']:.3f}] {ok(x2['ok'])} | "
                     f"{iii} | {'' if x3.get('missing') else SHORT[x3['better']]} |")
    L.append("\n## Full table")
    L.append("\n| block | DGP | arm | n | mean | bias | sd | RMSE | naive | OLS | covers | real bias (2sd/√n) | published FF | epochs (med) / hit cap | fit s (med) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in table:
        pub = f"{t['published'][0]:.2f} ± {t['published'][1]:.2f}" if t["published"] else ""
        L.append(f"| {t['block']} | {t['dgp']} | {t['arm']} | {t['n_fits']}{'' if not t['n_nonfinite'] else f' ({t['n_nonfinite']} non-finite)'} | "
                 f"{fmt(t['mean'])} | {t['bias']:+.3f} | {fmt(t['sd'])} | {fmt(t['rmse'])} | {fmt(t['naive'])} | {fmt(t['ols'])} | "
                 f"{'yes' if t['covers'] else 'no'} | {'yes' if t['real_bias'] else 'no'} | {pub} | "
                 f"{t['median_epochs']:.0f} / {t['frac_hit_cap']:.0%} | {t['median_fit_s']:.0f} |")
    L.append("\n## Shift parameter vs sampled ATE (shift arms)")
    L.append("\n| block | DGP | arm | max over seeds of abs(shift on Y scale − sampled ATE) |")
    L.append("|---|---|---|---|")
    for t in table:
        if t["max_abs_shift_minus_ate"] is not None:
            L.append(f"| {t['block']} | {t['dgp']} | {t['arm']} | {t['max_abs_shift_minus_ate']:.2e} |")
    mis = [t for t in table if t["block"] == "misspec"]
    if mis:
        L.append("\n## Misspecification block vs predictions")
        L.append("Prediction (A6): `location_translation_gaussian` biased; `flexible_continuous_gaussian` not.")
        for t in mis:
            L.append(f"- {t['arm']}: mean {t['mean']:.3f} (truth {t['truth']:.4f}), bias {t['bias']:+.3f}, sd {t['sd']:.3f}, "
                     f"real bias by 2sd/√n: {'yes' if t['real_bias'] else 'no'}; covers by ±2sd: {'yes' if t['covers'] else 'no'}")
    L.append(f"\n## Failed fits: {len(failed)}")
    for f in failed:
        L.append(f"- {f}")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def plot(table, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dgps = []
    for t in table:
        if (t["block"], t["dgp"]) not in dgps:
            dgps.append((t["block"], t["dgp"]))
    nc = 3
    nr = math.ceil(len(dgps) / nc)
    fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.2 * nr), squeeze=False)
    colors = {"gaussian": "#7f7f7f", "location_translation": "#1f77b4", "flexible_continuous": "#2ca02c",
              "location_translation_gaussian": "#d62728", "flexible_continuous_gaussian": "#9467bd"}
    for ax, (b, d) in zip(axes.ravel(), dgps):
        ts = [t for t in table if t["block"] == b and t["dgp"] == d]
        for i, t in enumerate(ts):
            ax.errorbar(i, t["mean"], yerr=2 * t["sd"], fmt="o", color=colors[t["arm"]], capsize=3)
            ax.scatter(np.full(len(t["est"]), i + 0.18), t["est"], s=6, color=colors[t["arm"]], alpha=0.5)
        ax.axhline(ts[0]["truth"], color="k", lw=1, label="truth")
        ax.axhline(ts[0]["naive"], color="k", lw=1, ls="--", label="naive")
        if ts[0]["published"]:
            m, w = ts[0]["published"]
            ax.axhspan(m - w, m + w, color="orange", alpha=0.15, label="published FF ±2σ")
        ax.set_xticks(range(len(ts)), [SHORT[t["arm"]] for t in ts])
        ax.set_title(f"{b}: {d}", fontsize=9)
    for ax in axes.ravel()[len(dgps):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle("S11: ATE estimates by arm (mean ± 2 sd over data seeds; dots = seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, failed = load()
    table, groups = summarise(rows)
    cl = clauses(table, groups)
    with open(os.path.join(OUT, "S11_tables.json"), "w") as fh:
        json.dump(dict(table=table, clauses=cl, failed=failed, n_results=len(rows)), fh, indent=1)
    write_md(table, cl, failed, os.path.join(OUT, "S11_tables.md"))
    if table:
        plot(table, os.path.join(OUT, "S11_univariate.png"))
    print(open(os.path.join(OUT, "S11_tables.md")).read())


if __name__ == "__main__":
    main()
