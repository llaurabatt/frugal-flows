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
LOG = os.path.expanduser("~/work/halo-runs/gausshp")


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
    pairs = sorted({(r["exp"], r["k"]) for r in recs})
    lines = ["# Gaussian spline hyperparameter sweep (n = 5000, tuning datasets)", "",
             "Ratio = geometric mean of (setting / base) ATE MAE over paired (experiment, dataset); < 1 is better.", "",
             "| setting | " + " | ".join(f"{e} k{k}" for e, k in pairs) + " | mean E2 | mean E4 | ratio vs base | wins | best epoch | wall min |",
             "|---|" + "---|" * (len(pairs) + 6)]
    rows = []
    for name, d in by.items():
        both = [p for p in d if p in base]
        ratio = float(np.exp(np.mean([np.log(d[p]["mae"] / base[p]["mae"]) for p in both]))) if both else float("nan")
        wins = sum(d[p]["mae"] < base[p]["mae"] for p in both)
        rows.append((ratio if name != "base" else 1.0, name, d, both, wins))
    for ratio, name, d, both, wins in sorted(rows, key=lambda t: t[0]):
        cells = " | ".join(f"{d[p]['mae']:.4f}" if p in d else "…" for p in pairs)
        m = lambda e: np.mean([d[p]["mae"] for p in d if p[0] == e]) if any(p[0] == e for p in d) else float("nan")
        lines.append(f"| {name} | {cells} | {m('E2'):.4f} | {m('E4'):.4f} | {ratio:.2f} | "
                     f"{'—' if name == 'base' else f'{wins}/{len(both)}'} | "
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
