"""S11 driver (halo prereg Amendment A6): enumerate, generate, fit, check completeness.

Blocks (A6):
  article   M1, M2, M3 x ATE {1, 5} x N 25,000 x data seeds 1..10 x 5 arms   = 300
  small     M2 x ATE 1 x N {2,000, 5,000} x 10 x 5                            = 100
  misspec   gamma_margin x N 20,000 x 10 x 5                                  =  50
Fit order: article ATE 1, article ATE 5, small, misspec (``--order`` can put misspec second).
seed_fit = seed_data + 100.

Datasets are generated first, sequentially, in env ``frugal-flows`` (R causl). Fits run as one
subprocess each in env ``frugal-flows-halo`` with PYTHONPATH = this worktree, 1 thread each, XLA
flags pinned. Concurrency: ``--conc`` normally, but capped at ``--conc-shared`` (2) whenever the
S10 run (``halo_driver.py --stage S10``) is alive, re-checked before every launch.

  python driver.py --blocks article --resume --conc 8
  python driver.py --check                      # completeness only
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.expanduser(os.environ.get("FF_RUNS_LOG", "~/work/halo-runs") + "/S11")
DATA, FITS, LOGS = (os.path.join(ROOT, d) for d in ("data", "fits", "_logs"))
MAMBA = os.path.expanduser("~/.local/bin/micromamba")
ARMS = ("gaussian", "location_translation", "flexible_continuous",
        "location_translation_gaussian", "flexible_continuous_gaussian")
SEEDS = tuple(range(1, 11))
XLA = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"


def datasets(block):
    """(block, model, ate, n, seed) for a block, in run order."""
    if block == "article1":
        return [("article", m, 1, 25000, s) for m in ("M1", "M2", "M3") for s in SEEDS]
    if block == "article5":
        return [("article", m, 5, 25000, s) for m in ("M1", "M2", "M3") for s in SEEDS]
    if block == "small":
        return [("small", "M2", 1, n, s) for n in (2000, 5000) for s in SEEDS]
    if block == "misspec":
        return [("misspec", "gamma_margin", 0, 20000, s) for s in SEEDS]
    raise ValueError(block)


def data_path(model, ate, n, seed):
    tag = f"{model}_n{n}_s{seed}" if model == "gamma_margin" else f"{model}_ate{ate}_n{n}_s{seed}"
    return os.path.join(DATA, tag + ".npz")


def cell_id(block, model, ate, n, seed, arm):
    tag = os.path.basename(data_path(model, ate, n, seed))[:-4]
    return f"{block}__{tag}__{arm}"


def fit_dir(block, model, ate, n, seed, arm):
    return os.path.join(FITS, block, os.path.basename(data_path(model, ate, n, seed))[:-4], arm)


def is_complete(d):
    p = os.path.join(d, "result.json")
    if not os.path.exists(p):
        return False
    try:
        with open(p) as fh:
            r = json.load(fh)
        return bool(r.get("complete")) and r.get("smoke_max_epochs") is None
    except Exception:  # noqa: BLE001
        return False


def s10_alive():
    r = subprocess.run(["pgrep", "-f", "[h]alo_driver.py --stage S10"], capture_output=True, text=True)
    return bool(r.stdout.strip())


def pinned_env():
    e = dict(os.environ)
    e["PYTHONPATH"] = REPO
    e["XLA_FLAGS"] = XLA
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS"):
        e[k] = "1"
    e.pop("JAX_ENABLE_X64", None)  # package default precision (float64), recorded per fit
    e["PYTHONHASHSEED"] = "0"
    return e


def generate_missing(dsets):
    for _, model, ate, n, seed in dsets:
        p = data_path(model, ate, n, seed)
        if os.path.exists(p):
            continue
        cmd = [MAMBA, "run", "-n", "frugal-flows", "python", os.path.join(HERE, "gen_causl.py"),
               "--model", model, "--ate", str(ate), "--n", str(n), "--seed", str(seed), "--out", p]
        log = os.path.join(LOGS, "gen__" + os.path.basename(p)[:-4] + ".log")
        with open(log, "w") as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
        print(f"[gen] {os.path.basename(p)} rc={rc}", flush=True)
        if rc != 0 or not os.path.exists(p):
            raise SystemExit(f"generation failed: {p} (see {log})")


def fit_cells(dsets, arms):
    return [(b, m, a, n, s, arm) for (b, m, a, n, s) in dsets for arm in arms]


def run_fits(cells, conc, conc_shared, resume):
    todo = [c for c in cells if not (resume and is_complete(fit_dir(*c)))]
    print(f"[fit] {len(cells)} cells, {len(cells) - len(todo)} already complete, {len(todo)} to run", flush=True)
    running = {}
    env = pinned_env()
    t_start = time.time()
    done = 0
    while todo or running:
        for pid, (proc, c, t0, fh) in list(running.items()):
            if proc.poll() is not None:
                fh.close()
                ok = is_complete(fit_dir(*c))
                done += 1
                print(f"[fit] {'OK ' if ok else 'FAIL'} {cell_id(*c)} {time.time() - t0:.0f}s "
                      f"({done} finished, {len(todo)} queued, {time.time() - t_start:.0f}s elapsed)", flush=True)
                del running[pid]
        cap = conc_shared if s10_alive() else conc
        while todo and len(running) < cap:
            c = todo.pop(0)
            b, m, a, n, s, arm = c
            out = fit_dir(*c)
            os.makedirs(out, exist_ok=True)
            cmd = [MAMBA, "run", "-n", "frugal-flows-halo", "python", os.path.join(HERE, "fit_one.py"),
                   "--data", data_path(m, a, n, s), "--arm", arm, "--seed-fit", str(s + 100), "--out", out]
            fh = open(os.path.join(LOGS, cell_id(*c) + ".log"), "w")
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=HERE)
            running[proc.pid] = (proc, c, time.time(), fh)
            print(f"[fit] start {cell_id(*c)} (cap {cap}, running {len(running)})", flush=True)
        time.sleep(5)


def check(cells):
    n_ok = n_fail = n_missing = 0
    failed = []
    for c in cells:
        d = fit_dir(*c)
        if is_complete(d):
            n_ok += 1
        elif os.path.exists(os.path.join(d, "FAILED")):
            n_fail += 1
            failed.append(cell_id(*c))
        else:
            n_missing += 1
    print(f"[check] expected {len(cells)}: complete {n_ok}, FAILED {n_fail}, missing {n_missing}")
    for f in failed:
        print(f"  FAILED {f}")
    return n_ok, n_fail, n_missing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--blocks", default="article1,article5,small,misspec",
                   help="comma list from article1, article5, small, misspec, in run order")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--conc", type=int, default=8)
    p.add_argument("--conc-shared", type=int, default=2, help="cap while S10 is running")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--gen-only", action="store_true", help="generate missing datasets, no fits")
    a = p.parse_args()
    for d in (DATA, FITS, LOGS):
        os.makedirs(d, exist_ok=True)
    blocks = [b.strip() for b in a.blocks.split(",") if b.strip()]
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    dsets = list(itertools.chain.from_iterable(datasets(b) for b in blocks))
    cells = fit_cells(dsets, arms)
    if a.check:
        check(cells)
        return
    if a.dry_run:
        for c in cells:
            print(cell_id(*c), "complete" if is_complete(fit_dir(*c)) else "todo")
        print(len(cells), "cells")
        return
    print(f"[driver] blocks={blocks} arms={arms} conc={a.conc} conc_shared={a.conc_shared} xla='{XLA}'", flush=True)
    generate_missing(dsets)
    if a.gen_only:
        return
    run_fits(cells, a.conc, a.conc_shared, a.resume)
    check(cells)


if __name__ == "__main__":
    main()
