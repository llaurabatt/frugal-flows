"""Score the sub5k grid (grid_e2e4.py --suite sub5k): 4 frugal flows + frengression, E2 and E4, n = 5000.

Per run: ATE MAE over the 64 pixels, signed disc bias (E2), retained confounding rho (projection of the
error map on the naive error map: 0 = fully adjusted, 1 = naive), best epoch / iterations, wall time.
Writes ~/work/halo-runs/sub5k/results.md and results.png; --wandb uploads both as one summary run.
Runs on partial results. The uniform arms are on raw Y and the Gaussian arms on standardised Y, so a
uniform-vs-Gaussian contrast here includes the standardisation gain (about 15-18% on 60k E2).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MM = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, MM)
import exp_ate_recovery as E  # noqa: E402

FLOW_ROOT, FR_ROOT = os.path.join(MM, "runs", "sub5k"), os.path.join(MM, "runs", "sub5k_frengression")
OUT = os.path.expanduser("~/work/halo-runs/sub5k")
EXPS = {"exp2_confounded_homogeneous": "E2", "exp4_covariate_cate": "E4"}
ARM_CODE = {("flexible_continuous", "none"): "U-flex-raw", ("location_translation", "none"): "U-LT-raw",
            ("flexible_continuous_gaussian", "standardize"): "G-flex-std",
            ("location_translation_gaussian", "standardize"): "G-LT-std"}
ROWS = ["U-flex-raw", "U-LT-raw", "G-flex-std", "G-LT-std", "freng"]
_naive = {}


def naive(preset, k):
    """Naive difference-in-means map and true ATE on the n=5000 subsample, rebuilt from the generator."""
    if (preset, k) not in _naive:
        cfg = E.Config(preset=preset, size=8, seed_data=101, n=5000, seed_assign=k, digit=None)
        d = E.build_data(cfg)
        Y, T = np.asarray(d["Y"], np.float64), np.asarray(d["X"]).ravel()
        _naive[(preset, k)] = (Y[T == 1].mean(0) - Y[T == 0].mean(0), np.asarray(d["ATE"], np.float64))
    return _naive[(preset, k)]


def collect():
    recs = []
    for root, kind in ((FLOW_ROOT, "flow"), (FR_ROOT, "freng")):
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            mp, cp, ap = (os.path.join(d, f) for f in ("metrics.json", "config.json", "arrays.npz"))
            if not (os.path.exists(mp) and os.path.exists(ap)):
                continue
            m, c = json.load(open(mp)), json.load(open(cp))
            c = c.get("config", c)
            if c.get("n") != 5000 or c.get("preset") not in EXPS:
                continue
            code = "freng" if kind == "freng" else ARM_CODE.get((c.get("arm"), c.get("y_scaling", "none")))
            if code is None:
                continue
            k = int(c["seed_assign"])
            tau = np.load(ap)["tau_hat"].astype(np.float64)
            nv, ate = naive(c["preset"], k)
            err, nerr = tau - ate, nv - ate
            recs.append({"exp": EXPS[c["preset"]], "code": code, "k": k, "run": os.path.basename(d),
                         "mae": float(np.abs(err).mean()), "naive_mae": float(np.abs(nerr).mean()),
                         "disc": m.get("signed_disc"), "rho": float(err @ nerr / (nerr @ nerr)),
                         "best": m.get("best_epoch", m.get("loss_min_iter")),
                         "wall_min": (m.get("wall_time_s") or float("nan")) / 60})
    return recs


def fmt(x, p=4):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{p}f}"


def report(recs):
    lines = ["# sub5k grid: 4 frugal flows + frengression, n = 5000 all digits, E2 and E4", "",
             "Uniform arms on raw Y, Gaussian arms on standardised Y (so U vs G includes the standardisation "
             "gain). One fit per dataset, fit seed = dataset. rho: 0 = adjusted, 1 = naive.", ""]
    means = {}
    for exp in ("E2", "E4"):
        R = [r for r in recs if r["exp"] == exp]
        if not R:
            continue
        nv = {r["k"]: r["naive_mae"] for r in R}
        lines += [f"## {exp} (naive MAE: " + ", ".join(f"k{k} {v:.3f}" for k, v in sorted(nv.items())) + ")", "",
                  "| arm | k1 | k2 | k3 | mean | mean rho | mean disc bias | best epoch/iter | wall min |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for code in ROWS:
            rr = {r["k"]: r for r in R if r["code"] == code}
            if not rr:
                continue
            v = [rr[k]["mae"] for k in rr]
            means[(exp, code)] = (np.mean(v), len(v))
            cell = lambda k: fmt(rr[k]["mae"]) if k in rr else "…"
            wall = ", ".join(f"{rr[k]['wall_min']:.0f}" for k in sorted(rr))
            disc = [rr[k]["disc"] for k in rr if rr[k]["disc"] is not None]
            lines.append(f"| {code} | {cell(1)} | {cell(2)} | {cell(3)} | {fmt(np.mean(v))} (n={len(v)}) | "
                         f"{fmt(np.mean([rr[k]['rho'] for k in rr]), 3)} | {fmt(np.mean(disc), 3) if disc else '—'} | "
                         f"{', '.join(str(rr[k]['best']) for k in sorted(rr))} | "
                         f"{wall} |")
        lines += ["", "Paired ratios (mean over datasets where both arms finished; < 1 favours the first arm):", ""]
        for a, b, note in (("G-flex-std", "U-flex-raw", "Gaussian vs uniform, flexible (std vs raw)"),
                           ("G-LT-std", "U-LT-raw", "Gaussian vs uniform, LT (std vs raw)"),
                           ("U-LT-raw", "U-flex-raw", "LT vs flexible, uniform"),
                           ("G-LT-std", "G-flex-std", "LT vs flexible, Gaussian"),
                           ("G-flex-std", "freng", "Gaussian flexible vs frengression"),
                           ("U-flex-raw", "freng", "uniform flexible vs frengression")):
            ka = {r["k"]: r["mae"] for r in R if r["code"] == a}
            kb = {r["k"]: r["mae"] for r in R if r["code"] == b}
            ks = sorted(set(ka) & set(kb))
            if ks:
                ratios = [ka[k] / kb[k] for k in ks]
                lines.append(f"- {note}: {np.mean([ka[k] for k in ks]):.4f} / {np.mean([kb[k] for k in ks]):.4f} "
                             f"= {np.mean([ka[k] for k in ks]) / np.mean([kb[k] for k in ks]):.2f}x "
                             f"(per dataset {', '.join(f'{x:.2f}' for x in ratios)}; first arm wins {sum(x < 1 for x in ratios)}/{len(ks)})")
        lines.append("")
    return "\n".join(lines), means


def figure(recs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, exp in zip(axes, ("E2", "E4")):
        R = [r for r in recs if r["exp"] == exp]
        for i, code in enumerate(ROWS):
            v = [r["mae"] for r in R if r["code"] == code]
            if v:
                ax.bar(i, np.mean(v), color="C0" if code.startswith("U") else "C1" if code.startswith("G") else "0.5",
                       alpha=0.5)
                ax.scatter([i] * len(v), v, color="k", s=14, zorder=3)
        ax.set_xticks(range(len(ROWS)), ROWS, rotation=25)
        ax.set_title(f"{exp}, n = 5000: ATE MAE (dots = datasets 1-3)")
        ax.set_ylabel("ATE MAE")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    recs = collect()
    md, _ = report(recs)
    open(os.path.join(OUT, "results.md"), "w").write(md)
    json.dump(recs, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    if recs:
        figure(recs, os.path.join(OUT, "results.png"))
    print(md)
    print(f"\n{len(recs)} runs scored (30 expected)")
    if a.wandb and recs:
        import pandas as pd
        import wandb
        run = wandb.init(project="Frugal Images", group="sub5k_e2e4", name="sub5k_e2e4_summary",
                         tags=["sub5k", "summary"], job_type="analysis")
        df = pd.DataFrame(recs)
        for col in df.columns:
            if df[col].dtype == object and col not in ("exp", "code", "run"):
                df[col] = df[col].astype(str)
        run.log({"table/per_run": wandb.Table(dataframe=df), "plots/results": wandb.Image(os.path.join(OUT, "results.png")),
                 "report": wandb.Html("<pre>" + md + "</pre>")})
        run.finish()


if __name__ == "__main__":
    main()
