"""E2 + E4 three-model grid on the collaborator's datasets (2026-10-02 evening).

Datasets k = 1-3 (seed_data 101, all ten digits, 8x8), fit seeds {k, 1001-1004}, paper hyperparameters:
  U-raw   the collaborator's flow (flexible_continuous, raw Y, copula width 16)      E4: 15 fits
  G-std   Gaussian spline (flexible_continuous_gaussian, standardised Y, width 16)    E4: 15 fits, E2: seeds 1001-1004 (12)
  G-LT    Gaussian location translation (width 16, naive start, shift lr x10)          E4: seed k only (3)
  freng   frengression (her defaults, 5000 iterations, 2 threads)                        E4: 15 fits
E2 U-raw reuses her 15 saved fits (runs/exp_ate_recovery), E2 G-std seed k reuses runs/gaussian_scale, E2
frengression is runs/frengression (seed k) + ~/work/halo-runs/gsw/fr_avg5 (seeds 1001-1004).

Scheduler: a thread budget (flow fit = 1 thread, frengression fit = 2), queue order = priority; skip-done.
No W&B logging. --wait-file: do not start until that file contains "ALL DONE" (the running E2 frengression queue).
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MM = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, MM)
sys.path.insert(0, HERE)
import grid_gaussian_scale as G  # noqa: E402  (PAPER flags, arm table, run-name prefix)

FLOW_ROOT = os.path.join(MM, "runs", "e2e4")
FR_ROOT = os.path.join(MM, "runs", "e2e4_frengression")
E2, E4 = "exp2_confounded_homogeneous", "exp4_covariate_cate"
SHORT = {E2: "E2", E4: "E4"}
PAPER = [a for a in G.PAPER if a not in ("--wandb", "--wandb-group", G.GROUP, "--wandb-tags", "gaussian-scale")]


def flow_cell(code, preset, k, s):
    arm, ys, _, ex = G.ARMS[code]
    args = PAPER + ["--preset", preset, "--seed-assign", str(k), "--seed-fit", str(s), "--arm", arm,
                    "--y-scaling", ys, "--copula-nn-width", "16"] + list(ex)
    if code == "G-LT":
        args += ["--shift-lr-mult", "10"]
    return {"kind": "flow", "label": f"{code}:{SHORT[preset]}:k{k}:s{s}", "args": args, "threads": 1}


def fr_cell(preset, k, s):
    args = ["--all-digits", "--preset", preset, "--seed-assign", str(k), "--seed-fit", str(s), "--size", "8",
            "--seed-data", "101", "--num-iters", "5000", "--threads", "2"]
    return {"kind": "freng", "label": f"freng:{SHORT[preset]}:k{k}:s{s}", "args": args, "threads": 2,
            "glob": f"*_frengression_{SHORT[preset].lower()}_sa{k}_k64_s{s}_d0-9_*"}


def queue(n_fits: int = 5):
    """Phase 1 = the first 3 fit seeds {k, 1001, 1002} of everything (E4 first, then the E2 Gaussian gaps);
    phase 2 = seeds 1003, 1004 (only with n_fits = 5)."""
    ks = (1, 2, 3)
    q = []
    for k in ks:                                   # E4 seed-k cells first: one of everything per dataset
        q += [flow_cell("U-raw", E4, k, k), flow_cell("G-std", E4, k, k), fr_cell(E4, k, k), flow_cell("G-LT", E4, k, k)]
    for s in (1001, 1002):
        for k in ks:
            q += [flow_cell("U-raw", E4, k, s), flow_cell("G-std", E4, k, s), fr_cell(E4, k, s)]
    for s in (1001, 1002):
        for k in ks:
            q.append(flow_cell("G-std", E2, k, s))
    if n_fits == 5:
        for s in (1003, 1004):
            for k in ks:
                q += [flow_cell("U-raw", E4, k, s), flow_cell("G-std", E4, k, s), fr_cell(E4, k, s),
                      flow_cell("G-std", E2, k, s)]
    return q


def done_dir(c):
    if c["kind"] == "flow":
        pre = c["prefix"]
        for d in glob.glob(os.path.join(FLOW_ROOT, "*")):
            n = os.path.basename(d)
            if n.split("_", 1)[-1].startswith(pre) and os.path.exists(os.path.join(d, "metrics.json")) \
                    and os.path.exists(os.path.join(d, "model.eqx")):
                return d
        return None
    for d in glob.glob(os.path.join(FR_ROOT, c["glob"])):
        if os.path.exists(os.path.join(d, "metrics.json")):
            return d
    return None


def log(msg, fh):
    line = f"[{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--logdir", default=os.path.expanduser("~/work/halo-runs/e2e4"))
    ap.add_argument("--wait-file", default=None)
    ap.add_argument("--fits", type=int, default=3, choices=(3, 5))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stagger-s", type=float, default=20.0)
    a = ap.parse_args()
    for d in (FLOW_ROOT, FR_ROOT, a.logdir, os.path.join(a.logdir, "fits")):
        os.makedirs(d, exist_ok=True)
    fh = open(os.path.join(a.logdir, "launcher.log"), "a")
    cells = queue(a.fits)
    for c in cells:
        if c["kind"] == "flow":
            c["prefix"] = G.name_prefix(c["args"])
    log(f"queue: {len(cells)} cells ({a.fits} fits per arm), thread budget {a.threads}", fh)
    for i, c in enumerate(cells):
        log(f"  {i + 1:2d} {c['label']:<22} {c.get('prefix', c.get('glob'))}", fh)
    if a.dry_run:
        return
    if a.wait_file:
        log(f"waiting for 'ALL DONE' in {a.wait_file}", fh)
        while not (os.path.exists(a.wait_file) and "ALL DONE" in open(a.wait_file).read()):
            time.sleep(30)
        log("wait-file satisfied, starting", fh)

    base = dict(os.environ)
    base.update({"JAX_PLATFORMS": "cpu", "TF_CPP_MIN_LOG_LEVEL": "2", "WANDB_MODE": "disabled"})
    py = [os.environ.get("MAMBA_EXE", "micromamba"), "run", "-n", "frugal-flows-e2w", "python"]
    running, status, pending = {}, {}, list(cells)
    while pending or running:
        for lab, (p, c) in list(running.items()):
            rc = p.poll()
            if rc is not None:
                d = done_dir(c)
                status[lab] = "done" if (rc == 0 and d) else "failed"
                log(f"END   {lab} rc={rc} {status[lab]} {os.path.basename(d) if d else ''}", fh)
                del running[lab]
        used = sum(c["threads"] for _, c in running.values())
        while pending:
            c = pending[0]
            d = done_dir(c)
            if d:
                status[c["label"]] = "skipped"
                log(f"SKIP  {c['label']} ({os.path.basename(d)})", fh)
                pending.pop(0)
                continue
            if used + c["threads"] > a.threads:
                break
            pending.pop(0)
            env = dict(base)
            n = str(c["threads"])
            env.update({"OMP_NUM_THREADS": n, "MKL_NUM_THREADS": n, "OPENBLAS_NUM_THREADS": n,
                        "VECLIB_MAXIMUM_THREADS": n, "NUMEXPR_NUM_THREADS": n})
            if c["kind"] == "flow":
                env["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
                cmd = py + [os.path.join(MM, "exp_ate_recovery.py")] + c["args"] + ["--runs-root", FLOW_ROOT]
            else:
                cmd = py + [os.path.join(MM, "exp_frengression_recovery.py")] + c["args"] + ["--runs-root", FR_ROOT]
            lf = open(os.path.join(a.logdir, "fits", c["label"].replace(":", "_") + ".log"), "a")
            p = subprocess.Popen(cmd, cwd=MM, env=env, stdout=lf, stderr=subprocess.STDOUT)
            running[c["label"]] = (p, c)
            used += c["threads"]
            status[c["label"]] = "running"
            log(f"START {c['label']} pid={p.pid} (threads in use {used})", fh)
            time.sleep(a.stagger_s)
        time.sleep(15)
    log("grid finished: " + str({s: sum(v == s for v in status.values()) for s in ("done", "skipped", "failed")}), fh)
    open(os.path.join(a.logdir, "DONE"), "w").write(dt.datetime.now().isoformat())


if __name__ == "__main__":
    main()
