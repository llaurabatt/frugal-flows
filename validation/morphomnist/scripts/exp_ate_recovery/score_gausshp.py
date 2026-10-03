"""Score the Gaussian-spline hyperparameter sweep (grid_e2e4.py --suite gausshp).

Each finished run is mapped to its sweep setting through the launcher log ("END hp-<name>:<E>:k<k> ... done <dir>").
Per setting: ATE MAE per (experiment, tuning dataset), the mean, and the paired ratio to "base" (Laura's settings),
as a geometric mean over the (experiment, dataset) pairs, with wins out of pairs. Writes results.md in the log dir.
Selection on ATE MAE uses the simulation truth on TUNING datasets (k = 11-13) only; the reported datasets (k = 1-10)
are never used to choose.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MM = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.join(MM, "runs", "gausshp")
LOG = os.path.expanduser(os.environ.get("FF_RUNS_LOG", "~/work/halo-runs") + "/gausshp")


def collect(logdir):
    pat = re.compile(r"END\s+hp-(\S+):(E\d):k(\d+) rc=0 done (\S+)")
    recs = []
    for line in open(os.path.join(logdir, "launcher.log")):
        m = pat.search(line)
        if not m:
            continue
        name, exp, k, run = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        mp = os.path.join(ROOT, run, "metrics.json")
        if not os.path.exists(mp):
            continue
        mt = json.load(open(mp))
        recs.append({"name": name, "exp": exp, "k": k, "mae": mt["ate_mae"], "best": mt.get("best_epoch"),
                     "wall_min": mt.get("wall_time_s", float("nan")) / 60, "run": run})
    return recs


def report(recs):
    by = defaultdict(dict)
    for r in recs:
        by[r["name"]][(r["exp"], r["k"])] = r
    base = by.get("base", {})
    exps = sorted({r["exp"] for r in recs})
    lines = ["# Gaussian spline hyperparameter sweep (n = 5000, tuning datasets)", "",
             "Per experiment: mean ATE MAE, then geometric-mean ratio vs base over paired datasets and wins "
             "(< 1 is better). Overall = geometric mean over all pairs.", "",
             "| setting | " + " | ".join(f"{e} mean | {e} ratio (wins)" for e in exps)
             + " | overall ratio | t | best epoch | wall min |",
             "|---|" + "---|---|" * len(exps) + "---|---|---|---|"]
    rows = []
    for name, d in by.items():
        both = [p for p in d if p in base]
        lr = np.array([np.log(d[p]["mae"] / base[p]["mae"]) for p in both])
        ratio = float(np.exp(lr.mean())) if len(lr) else float("nan")
        t = float(lr.mean() / (lr.std(ddof=1) / np.sqrt(len(lr)))) if len(lr) > 2 and lr.std() > 0 else float("nan")
        rows.append((ratio, name, d, t))
    for ratio, name, d, t in sorted(rows, key=lambda r: (r[1] != "base", r[0])):
        cells = []
        for e in exps:
            pe = [p for p in d if p[0] == e]
            pb = [p for p in pe if p in base]
            m = np.mean([d[p]["mae"] for p in pe]) if pe else float("nan")
            if name == "base" or not pb:
                cells.append(f"{m:.4f} | —")
            else:
                r = np.exp(np.mean([np.log(d[p]["mae"] / base[p]["mae"]) for p in pb]))
                cells.append(f"{m:.4f} | {r:.2f} ({sum(d[p]['mae'] < base[p]['mae'] for p in pb)}/{len(pb)})")
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {ratio:.3f} | {t:.2f} | "
                     f"{np.median([d[p]['best'] for p in d]):.0f} | {np.median([d[p]['wall_min'] for p in d]):.1f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default=LOG)
    a = ap.parse_args()
    recs = collect(a.logdir)
    md = report(recs)
    open(os.path.join(a.logdir, "results.md"), "w").write(md)
    print(md)
    print(f"\n{len(recs)} runs scored")


if __name__ == "__main__":
    main()
