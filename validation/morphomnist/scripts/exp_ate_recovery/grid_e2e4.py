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
ALL_E = {"E1": "exp1_rct_homogeneous", "E2": E2, "E3": "exp3_confounded_heterogeneous", "E4": E4,
         "E5": "exp5_quantile_effect", "E6": "exp6_spatial_cate"}
SHORT = {v: k for k, v in ALL_E.items()}
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


# --- sub5k suite (2026-10-02 16:17 agreement): Laura's 4 flows on a 5,000-image all-digit subsample -----
# datasets k = 1-5, one fit per dataset (fit seed = k), W&B group sub5k_e2e4; uniform arms raw, Gaussian arms standardised
SUB5K_GROUP = "sub5k_e2e4"
SUB5K_ARMS = {  # code -> (arm, y_scaling, extra flags)
    "U-flex-raw": ("flexible_continuous", "none", ["--conditioner", "mlp"]),
    "U-LT-raw": ("location_translation", "none", []),                       # ate_init 0.5 (default), lr x1
    "G-flex-std": ("flexible_continuous_gaussian", "standardize", []),
    "G-LT-std": ("location_translation_gaussian", "standardize", ["--shift-init", "scalar"]),  # 0.5, lr x1
    # added 18:45 (Dan): separate scale from scaling; G-LT with the head start (uniform LT has no per-pixel start)
    "U-flex-std": ("flexible_continuous", "standardize", ["--conditioner", "mlp"]),
    "G-LT-head": ("location_translation_gaussian", "standardize", ["--shift-init", "naive", "--shift-lr-mult", "10"]),
}
SUB5K_FIRST4 = ["U-flex-raw", "U-LT-raw", "G-flex-std", "G-LT-std"]
SUB5K_NEW = ["U-flex-std", "G-LT-head"]


def sub5k_flow_cell(code, preset, k, s=None):
    s = k if s is None else s
    arm, ys, ex = SUB5K_ARMS[code]
    args = PAPER + ["--n", "5000", "--preset", preset, "--seed-assign", str(k), "--seed-fit", str(s),
                    "--arm", arm, "--y-scaling", ys, "--copula-nn-width", "16"] + list(ex) \
        + ["--wandb", "--wandb-group", SUB5K_GROUP, "--wandb-tags", f"sub5k,{code}"]
    lab = f"{code}:{SHORT[preset]}:k{k}" + ("" if s == k else f":s{s}")
    return {"kind": "flow", "label": lab, "args": args, "threads": 1}


def sub5k_fr_cell(preset, k, s=None):
    s = k if s is None else s
    args = ["--all-digits", "--n", "5000", "--preset", preset, "--seed-assign", str(k), "--seed-fit", str(s),
            "--size", "8", "--seed-data", "101", "--num-iters", "5000", "--threads", "2",
            "--wandb", "--wandb-group", SUB5K_GROUP, "--wandb-tags", "sub5k,freng"]
    lab = f"freng:{SHORT[preset]}:k{k}" + ("" if s == k else f":s{s}")
    return {"kind": "freng", "label": lab, "args": args, "threads": 2,
            "glob": f"*_frengression_{SHORT[preset].lower()}_sa{k}_k64_s{s}_d0-9_*"}


def queue_sub5k(n_datasets: int = 5, exps=("E2", "E4"), new_arms: bool = False):
    """Dataset-major: all arms on each experiment for k = 1, then k = 2, ... (one fit per dataset, fit seed k).
    new_arms adds U-flex-std and G-LT-head: queued first on E2/E4 (whose other arms already exist), then every
    experiment gets all 6 flows + frengression."""
    q = []
    codes = SUB5K_FIRST4 + (SUB5K_NEW if new_arms else [])
    if new_arms:
        for k in range(1, n_datasets + 1):
            for e in ("E2", "E4"):
                if e in exps:
                    q += [sub5k_flow_cell(c, ALL_E[e], k) for c in SUB5K_NEW]
    for k in range(1, n_datasets + 1):
        for e in exps:
            q += [sub5k_flow_cell(c, ALL_E[e], k) for c in codes] + [sub5k_fr_cell(ALL_E[e], k)]
    return q


# --- gausshp suite: one-at-a-time hyperparameter sweep of the Gaussian spline (G-flex-std) ------------------
# n = 5000, E2 + E4, TUNING datasets k = 11-13 (disjoint from the k = 1-10 reported in sub5k), W&B group gausshp
GAUSSHP_GROUP = "gausshp_sub5k"
GAUSSHP = {  # name -> flag overrides on top of the paper flags (lr 1e-3, patience 30, width 48, knots 8, copw 16)
    "base": [],
    "lr3e-4": ["--learning-rate", "0.0003"],
    "lr3e-3": ["--learning-rate", "0.003"],
    "pat100": ["--max-patience", "100"],
    "mw24": ["--nn-width", "24"],
    "mw128": ["--nn-width", "128"],
    "kn4": ["--rqs-knots", "4"],
    "kn16": ["--rqs-knots", "16"],
    "copw50": ["--copula-nn-width", "50"],
    "batch256": ["--batch-size", "256"],
}


def _override(args, extra):
    args = list(args)
    for i in range(0, len(extra), 2):
        if extra[i] in args:
            args[args.index(extra[i]) + 1] = extra[i + 1]
        else:
            args += extra[i:i + 2]
    return args


def queue_gausshp(ks=(11, 12, 13), exps=("E2", "E4"), names=None):
    q = []
    for k in ks:
        for e in exps:
            for name in (names or GAUSSHP):
                c = sub5k_flow_cell("G-flex-std", ALL_E[e], k)
                c["args"] = _override(c["args"], GAUSSHP[name])
                c["args"] = _override(c["args"], ["--wandb-group", GAUSSHP_GROUP, "--wandb-tags", f"gausshp,{name}"])
                c["label"] = f"hp-{name}:{e}:k{k}"
                q.append(c)
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
    ap.add_argument("--fits", type=int, default=3, choices=(3, 5), help="e2e4 suite: fits per cell")
    ap.add_argument("--datasets", type=int, default=5, help="sub5k suite: datasets k = 1..N, one fit each")
    ap.add_argument("--exps", default="E2,E4", help="sub5k suite: comma-separated experiments (E1-E6)")
    ap.add_argument("--new-arms", action="store_true", help="sub5k suite: add U-flex-std and G-LT-head")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stagger-s", type=float, default=20.0)
    ap.add_argument("--suite", default="e2e4", choices=("e2e4", "sub5k", "gausshp"))
    a = ap.parse_args()
    global FLOW_ROOT, FR_ROOT
    if a.suite == "sub5k":
        FLOW_ROOT, FR_ROOT = os.path.join(MM, "runs", "sub5k"), os.path.join(MM, "runs", "sub5k_frengression")
        if a.logdir == ap.get_default("logdir"):
            a.logdir = os.path.expanduser("~/work/halo-runs/sub5k")
    if a.suite == "gausshp":
        FLOW_ROOT = os.path.join(MM, "runs", "gausshp")
        if a.logdir == ap.get_default("logdir"):
            a.logdir = os.path.expanduser("~/work/halo-runs/gausshp")
    for d in (FLOW_ROOT, FR_ROOT, a.logdir, os.path.join(a.logdir, "fits")):
        os.makedirs(d, exist_ok=True)
    fh = open(os.path.join(a.logdir, "launcher.log"), "a")
    cells = (queue_sub5k(a.datasets, tuple(a.exps.split(",")), a.new_arms) if a.suite == "sub5k"
             else queue_gausshp() if a.suite == "gausshp" else queue(a.fits))
    for c in cells:
        if c["kind"] == "flow":
            c["prefix"] = G.name_prefix(c["args"])
    log(f"queue [{a.suite}]: {len(cells)} cells, thread budget {a.threads}", fh)
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
    base.update({"JAX_PLATFORMS": "cpu", "TF_CPP_MIN_LOG_LEVEL": "2"})
    if a.suite == "e2e4":
        base["WANDB_MODE"] = "disabled"
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
