"""Overnight Gaussian-scale paper grid (docs/gaussian_scale/OVERNIGHT_PREREG.md).

Queue of exp_ate_recovery.py fits, N_SLOTS concurrent, 1 thread each (pinned XLA flags), in round
priority order: no fit of a round starts before every fit of the earlier rounds has started; within a
round, dataset-major. A cell is DONE (skipped) when a run folder with its exact name prefix holds
metrics.json and model.eqx. After each round has finished (every cell done or failed) the analysis
script runs; at the end it runs once more and the optional --after command (S11 resume) is started.

No new fit starts after --start-cutoff (local HH:MM, the next occurrence after launch); fits running
then are left to finish. Cells not started are listed in the log; rerun this script (skip-done) to
continue them.

Usage: see validation/morphomnist/scripts/exp_ate_recovery/grid_gaussian_scale.sh
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MM = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.abspath(os.path.join(MM, "..", ".."))
sys.path.insert(0, MM)

RUNS_ROOT = os.path.join(MM, "runs", "gaussian_scale")
GROUP = "grid_8x8_alldigits_gaussian"
E2, E1 = "exp2_confounded_homogeneous", "exp1_rct_homogeneous"

PAPER = ["--all-digits", "--size", "8", "--seed-data", "101", "--model", "ff", "--nn-width", "48",
         "--nn-depth", "1", "--flow-layers", "4", "--rqs-knots", "8", "--learning-rate", "0.001",
         "--batch-size", "100", "--max-epochs", "1000", "--max-patience", "30", "--n-mc", "5000",
         "--save-model", "--wandb", "--wandb-group", GROUP, "--wandb-tags", "gaussian-scale"]

# arm code -> (cli arm, y_scaling, copula width, extra flags)
ARMS = {
    "U-raw": ("flexible_continuous", "none", 16, ["--conditioner", "mlp"]),
    "U-std": ("flexible_continuous", "standardize", 16, ["--conditioner", "mlp"]),
    "G-raw": ("flexible_continuous_gaussian", "none", 50, []),
    "G-std": ("flexible_continuous_gaussian", "standardize", 50, []),
    "G-LT": ("location_translation_gaussian", "standardize", 50, ["--shift-init", "naive"]),
}


def cell(rnd, code, preset, k, seed_fit=None, copw=None, extra=()):
    arm, ys, cw, ex = ARMS[code]
    cw = cw if copw is None else copw
    seed_fit = k if seed_fit is None else seed_fit
    args = PAPER + ["--preset", preset, "--seed-assign", str(k), "--seed-fit", str(seed_fit), "--arm", arm,
                    "--y-scaling", ys, "--copula-nn-width", str(cw)] + list(ex) + list(extra)
    label = (f"{rnd}:{code}{'' if copw is None else f'-copw{copw}'}"
             f"{'-zshuf' if '--z-shuffle-seed' in extra else ''}:{'E2' if preset == E2 else 'E1'}:k{k}:s{seed_fit}")
    return {"round": rnd, "code": code, "label": label, "args": args, "shift_lr_mult": None}


def queue(shift_lr_mult: float, b_seeds, with_e: bool, order: str):
    rounds = {}
    a = []
    for k in range(1, 7):
        a.append(cell("A", "G-std", E2, k))
        if k <= 3:
            a += [cell("A", "G-LT", E2, k), cell("A", "U-std", E2, k), cell("A", "G-raw", E2, k)]
    rounds["A"] = a
    rounds["B"] = [cell("B", "G-std", E2, k, seed_fit=s) for k in (1, 2, 3) for s in b_seeds]
    c = []
    for k in (1, 2, 3):
        if k == 1:
            c.append(cell("C", "G-std", E2, 1, extra=["--z-shuffle-seed", "7"]))
        c.append(cell("C", "G-LT", E2, k, extra=["--z-shuffle-seed", "7"]))
        if k == 1:
            c.append(cell("C", "U-raw", E2, 1))
        c.append(cell("C", "G-std", E2, k, copw=16))
    rounds["C"] = c
    d = []
    for k in range(1, 7):
        d.append(cell("D", "G-std", E1, k))
        if k <= 3:
            d.append(cell("D", "G-LT", E1, k))
    rounds["D"] = d
    e = []
    if with_e:
        for k in (1, 2, 3):
            e += [cell("E", "U-std", E2, k, seed_fit=1001), cell("E", "G-raw", E2, k, seed_fit=1001)]
        for k in (4, 5, 6):
            e.append(cell("E", "G-LT", E2, k))
            e += [cell("E", "G-std", E2, k, seed_fit=s) for s in (1001, 1002, 1003, 1004)]
    rounds["E"] = e
    out = []
    for r in order:
        for c_ in rounds[r]:
            if c_["code"] == "G-LT" and shift_lr_mult != 1.0:
                c_["args"] = c_["args"] + ["--shift-lr-mult", f"{shift_lr_mult:g}"]
            out.append(c_)
    return out


def name_prefix(args) -> str:
    """The run-folder name of this cell without timestamp and uid (exp_ate_recovery.wandb_name_for)."""
    import exp_ate_recovery as E
    from dataclasses import asdict, fields
    ns = {}
    i = 0
    flags = {"--all-digits"}
    bools = {"--save-model": ("save_model", True), "--wandb": ("wandb", True)}
    while i < len(args):
        a = args[i]
        if a in flags:
            i += 1
            continue
        if a in bools:
            ns[bools[a][0]] = bools[a][1]
            i += 1
            continue
        key = a[2:].replace("-", "_")
        ns[key] = args[i + 1]
        i += 2
    typed = {}
    ftypes = {f.name: f for f in fields(E.Config)}
    for k, v in ns.items():
        f = ftypes[k]
        if isinstance(v, bool):
            typed[k] = v
        elif f.type.startswith("int"):
            typed[k] = int(v)
        elif f.type.startswith("float"):
            typed[k] = float(v)
        else:
            typed[k] = v
    cfg = E.Config(**typed)
    cfg = E.Config(**{**asdict(cfg), "digit": None})
    return E.wandb_name_for(cfg, "XXXXXX")[:-6]


def is_done(prefix: str) -> str | None:
    if not os.path.isdir(RUNS_ROOT):
        return None
    for name in sorted(os.listdir(RUNS_ROOT)):
        if name.split("_", 1)[-1].startswith(prefix):
            d = os.path.join(RUNS_ROOT, name)
            if os.path.exists(os.path.join(d, "metrics.json")) and os.path.exists(os.path.join(d, "model.eqx")):
                return d
    return None


def log(msg, fh):
    line = f"[{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--logdir", default=os.path.expanduser("~/work/halo-runs/gsw"))
    ap.add_argument("--order", default="ACBD", help="round priority order, e.g. ACBD or ACBDE")
    ap.add_argument("--b-seeds", default="1001,1002")
    ap.add_argument("--with-e", action="store_true")
    ap.add_argument("--shift-lr-mult", type=float, default=1.0)
    ap.add_argument("--start-cutoff", default=None, help="HH:MM local: no new fit starts after this")
    ap.add_argument("--after", default=None, help="shell command run (detached) when the grid ends")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stagger-s", type=float, default=20.0)
    a = ap.parse_args()

    os.makedirs(RUNS_ROOT, exist_ok=True)
    os.makedirs(a.logdir, exist_ok=True)
    fits_log = os.path.join(a.logdir, "fits")
    os.makedirs(fits_log, exist_ok=True)
    fh = open(os.path.join(a.logdir, "launcher.log"), "a")
    order = [r for r in a.order if r in "ABCDE"]
    cells = queue(a.shift_lr_mult, [int(s) for s in a.b_seeds.split(",") if s], a.with_e or "E" in order, order)
    for c in cells:
        c["prefix"] = name_prefix(c["args"])
    cutoff = None
    if a.start_cutoff:
        hh, mm = map(int, a.start_cutoff.split(":"))
        now = dt.datetime.now()
        cutoff = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if cutoff <= now:
            cutoff += dt.timedelta(days=1)
    log(f"queue: {len(cells)} cells, order {''.join(order)}, slots {a.slots}, cutoff {cutoff}, "
        f"shift_lr_mult {a.shift_lr_mult:g}", fh)
    json.dump([{k: c[k] for k in ("round", "label", "prefix", "args")} for c in cells],
              open(os.path.join(a.logdir, "queue.json"), "w"), indent=1)
    for i, c in enumerate(cells):
        log(f"  {i + 1:2d} {c['label']:<28} {c['prefix']}", fh)
    if a.dry_run:
        return

    env = dict(os.environ)
    env.update({"XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "JAX_PLATFORMS": "cpu",
                "TF_CPP_MIN_LOG_LEVEL": "2", "WANDB_SILENT": "true"})
    py = [os.environ.get("MAMBA_EXE", "micromamba"), "run", "-n", "frugal-flows-e2w", "python"]
    status = {}                       # label -> "done" | "skipped" | "failed" | "running" | "not_started"
    running = {}                      # label -> Popen
    analysed = set()
    pending = list(cells)

    def analyse(tag):
        cmd = py + [os.path.join(HERE, "analyse_gaussian_scale.py")]
        with open(os.path.join(a.logdir, f"analysis_{tag}.log"), "w") as out:
            rc = subprocess.call(cmd, cwd=REPO, env=env, stdout=out, stderr=subprocess.STDOUT)
        log(f"analysis after {tag}: rc={rc}", fh)

    def round_finished(r):
        return all(status.get(c["label"]) in ("done", "skipped", "failed", "not_started")
                   for c in cells if c["round"] == r)

    while pending or running:
        # reap
        for lab, p in list(running.items()):
            rc = p.poll()
            if rc is not None:
                c = next(c for c in cells if c["label"] == lab)
                d = is_done(c["prefix"])
                status[lab] = "done" if (rc == 0 and d) else "failed"
                log(f"END   {lab} rc={rc} {status[lab]} {os.path.basename(d) if d else ''}", fh)
                del running[lab]
        # rounds that just finished -> analysis (background thread; it reads finished folders only)
        for r in order:
            if r not in analysed and any(c["round"] == r for c in cells) and round_finished(r):
                analysed.add(r)
                threading.Thread(target=analyse, args=(f"round{r}",), daemon=False).start()
        # launch
        while pending and len(running) < a.slots:
            c = pending[0]
            d = is_done(c["prefix"])
            if d:
                status[c["label"]] = "skipped"
                log(f"SKIP  {c['label']} (done: {os.path.basename(d)})", fh)
                pending.pop(0)
                continue
            if cutoff is not None and dt.datetime.now() >= cutoff:
                for c2 in pending:
                    status[c2["label"]] = "not_started"
                    log(f"CUTOFF not started: {c2['label']}", fh)
                pending = []
                break
            pending.pop(0)
            lf = open(os.path.join(fits_log, c["label"].replace(":", "_") + ".log"), "a")
            p = subprocess.Popen(py + [os.path.join(MM, "exp_ate_recovery.py")] + c["args"] + ["--runs-root", RUNS_ROOT],
                                 cwd=MM, env=env, stdout=lf, stderr=subprocess.STDOUT)
            running[c["label"]] = p
            status[c["label"]] = "running"
            log(f"START {c['label']} pid={p.pid}", fh)
            time.sleep(a.stagger_s)
        time.sleep(15)

    for r in order:
        if r not in analysed and any(c["round"] == r for c in cells):
            analysed.add(r)
    log("grid finished: " + json.dumps({s: sum(v == s for v in status.values())
                                        for s in ("done", "skipped", "failed", "not_started")}), fh)
    analyse("final")
    if a.after:
        log(f"starting after-command: {a.after}", fh)
        subprocess.Popen(a.after, shell=True, start_new_session=True)
    log("launcher exit", fh)


if __name__ == "__main__":
    main()
