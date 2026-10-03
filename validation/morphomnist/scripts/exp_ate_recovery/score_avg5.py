"""5-fit averaging of the Gaussian spline (G-flex-std) at n = 5000 — Laura's criterion.

Per (experiment, dataset k): the flow's fits are the sub5k fit (fit seed k, runs/sub5k) plus the avg5 fits
(fit seeds 1001-1004, runs/gausshp, read from the avg5 launcher log). The averaged map is the mean of the
fits' tau_hat maps. Frengression: the sub5k fit (seed k) plus any extra seeds found in runs/sub5k_frengression.
Reports flow single (mean over fits), flow averaged, frengression single (seed k), frengression averaged
(when extra seeds exist), with paired ratios and wins over datasets. Writes results.md in the avg5 log dir.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MM = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import score_sub5k as S  # noqa: E402

LOG = os.path.expanduser(os.environ.get("FF_RUNS_LOG", "~/work/halo-runs") + "/avg5_gauss")
SHORT2PRESET = {v: k for k, v in S.EXPS.items()}


def tau(d):
    return np.load(os.path.join(d, "arrays.npz"))["tau_hat"].astype(np.float64)


def main():
    flows, frs = {}, {}
    for r in json.load(open(os.path.expanduser(os.environ.get("FF_RUNS_LOG", "~/work/halo-runs") + "/sub5k/results.json"))):
        if r["code"] == "G-flex-std":
            flows.setdefault((r["exp"], r["k"]), []).append(tau(os.path.join(S.FLOW_ROOT, r["run"])))
        if r["code"] == "freng":
            frs.setdefault((r["exp"], r["k"]), []).append(tau(os.path.join(S.FR_ROOT, r["run"])))
    pat = re.compile(r"END\s+hp-base:(E\d):k(\d+):s(\d+) rc=0 done (\S+)")
    seen = set()
    for lf in sorted(glob.glob(os.path.expanduser(os.environ.get("FF_RUNS_LOG", "~/work/halo-runs") + "/avg5_gauss*/launcher.log"))):
      for line in open(lf):
        m = pat.search(line)
        if m and m[4] not in seen:
            seen.add(m[4])
            flows.setdefault((m[1], int(m[2])), []).append(tau(os.path.join(MM, "runs", "gausshp", m[4])))
    for d in glob.glob(os.path.join(S.FR_ROOT, "*_frengression_*_s100[1-4]_*")):  # extra frengression seeds
        if os.path.exists(os.path.join(d, "arrays.npz")):
            mm = re.search(r"frengression_(e\d)_sa(\d+)_", os.path.basename(d))
            frs.setdefault((mm[1].upper(), int(mm[2])), []).append(tau(d))
    lines = ["# Gaussian spline 5-fit averaging, n = 5000 (Laura's criterion: averaged flow map vs frengression)", ""]
    for exp in ("E1", "E2", "E3", "E4", "E5", "E6"):
        ks = sorted(k for (e, k) in flows if e == exp and len(flows[(e, k)]) == 5)
        if not ks:
            continue
        rows = []
        for k in ks:
            ate = S.naive(SHORT2PRESET[exp], k)[1]
            mae = lambda t: float(np.abs(t - ate).mean())
            f = flows[(exp, k)]
            fr = frs.get((exp, k), [])
            rows.append({"k": k, "f_single": np.mean([mae(t) for t in f]), "f_avg": mae(np.mean(f, 0)),
                         "fr_single": mae(fr[0]) if fr else np.nan,
                         "fr_avg": mae(np.mean(fr, 0)) if len(fr) == 5 else np.nan, "n_fr": len(fr)})
        g = lambda key: np.array([r[key] for r in rows])
        lines += [f"## {exp} ({len(ks)} datasets with 5 flow fits)", "",
                  "| k | flow single (mean of 5) | flow avg5 | frengression single | frengression avgN | N fr |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| {r['k']} | {r['f_single']:.4f} | {r['f_avg']:.4f} | {r['fr_single']:.4f} | "
                  f"{r['fr_avg']:.4f} | {r['n_fr']} |" for r in rows]
        lines += [f"| mean | {g('f_single').mean():.4f} | {g('f_avg').mean():.4f} | {np.nanmean(g('fr_single')):.4f} | "
                  f"{np.nanmean(g('fr_avg')) if np.isfinite(g('fr_avg')).any() else float('nan'):.4f} | |", ""]
        fa, fs, frs1 = g("f_avg"), g("f_single"), g("fr_single")
        lines.append(f"- Averaging gain (flow single / flow avg5): {fs.mean() / fa.mean():.2f}x")
        lines.append(f"- Flow avg5 vs frengression single: {fa.mean() / np.nanmean(frs1):.2f}x, "
                     f"flow wins {int(np.sum(fa < frs1))}/{len(ks)}")
        fra = g("fr_avg")
        ok = np.isfinite(fra)
        if ok.any():
            lines.append(f"- Flow avg5 vs frengression averaged (datasets with extra frengression seeds, n={ok.sum()}): "
                         f"{fa[ok].mean() / fra[ok].mean():.2f}x, flow wins {int(np.sum(fa[ok] < fra[ok]))}/{ok.sum()}")
        lines.append("")
    md = "\n".join(lines)
    open(os.path.join(LOG, "results.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
