"""Halo ladder driver: enumerate a stage's cells (HALO_PREREG.md), launch one pinned
subprocess per cell, fail closed on missing / duplicate identities.

    python halo_driver.py --stage {S0,S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,all} [--conc 8] [--threads 1] [--resume]
                          [--dry-run] [--smoke] [--runs-root ~/work/halo-runs]

Outputs: <runs-root>/<stage>/<run_id>/ (halo_fit.py), <runs-root>/<stage>/_stage.json (cell
list + prereg sha256), <runs-root>/<stage>/_logs/<run_id>.log, a launcher lock per stage.
"""
from __future__ import annotations

import argparse
import atexit
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HALO_DIR = os.path.dirname(os.path.abspath(__file__))
PREREG = os.path.join(HALO_DIR, "HALO_PREREG.md")
PY = sys.executable
SEEDS_DATA = tuple(range(31, 41))
SEEDS_FIT = (41, 42)
SEED_MC, SEED_MC2 = 7, 8
PRESET_SHORT = {"E1": "e1", "E2": "e2"}
COMMON = dict(corpus="A", width=48, depth=1, layers=4, knots=8, lr=1e-2, max_epochs=300, patience=30,
              batch=100, n_mc=5000, x64=False, synthetic=None, seed_mc=SEED_MC, seed_mc2=None)
PS_SLOPE = {"E1": 0.0, "E2": 1.2}


# ------------------------------------------------------------------ cell enumeration
def _configs(stage: str) -> list[dict]:
    """The distinct (non-seed) configurations of a stage (HALO_PREREG.md v1)."""
    U = dict(task="uncond", preset="E1", base_shift=0.0)
    S = dict(rank_mode="spread")
    if stage == "S1":
        c = [dict(arm="A1s", preproc=p, **S, **U) for p in ("P0", "P1", "P2", "P3")]
        c += [dict(arm="A1", preproc="P0", rank_mode="modulo", **U),
              dict(arm="A1w", preproc="P0", width=72, **S, **U)]
        c += [dict(arm="A2", preproc=p, rank_mode="none", **U) for p in ("P0", "P1")]
        c += [dict(arm="A3", preproc="P1", rank_mode="none", **U),
              dict(arm="A4", preproc="P1", rank_mode="none", **U),
              dict(arm="Csmooth", preproc="P0", synthetic="smooth", **S, **U),
              dict(arm="Czinf", preproc="P0", synthetic="zinf", **S, **U)]
        c += [dict(arm=a, preproc=p, corpus="B", rank_mode=r, **U)
              for a, p, r in (("A1s", "P0", "spread"), ("A1s", "P1", "spread"), ("A2", "P1", "none"))]
        return c
    if stage == "S2":
        arms = [("ff_cond", "P0"), ("ff_cond", "P1"), ("a2_cond", "P1"), ("lt", "P0"), ("sep", "P0")]
        c = [dict(arm=a, preproc=p, task="cond", preset="E1", base_shift=b,
                  rank_mode="none" if a == "a2_cond" else "spread")
             for b in (0.0, 1.0) for a, p in arms]
        c += [dict(arm="ff_cond", preproc=p, task="cond", preset="E1", base_shift=1.0, corpus="B",
                   rank_mode="spread") for p in ("P0", "P1")]
        return c
    if stage == "S3":
        tasks = [("uncond", 0.0), ("cond", 0.0), ("cond", 1.0)]
        return [dict(arm=a, preproc=p, task=t, preset="E1", base_shift=b, rank_mode="none")
                for a in ("zuko_nsf", "zuko_maf", "zuko_nsf_coupling")
                for t, b in tasks for p in ("P0", "P1")]
    if stage == "S4":
        c = [dict(arm="ff_full", preproc=p, task="cond", preset=e, base_shift=1.0, rank_mode="spread")
             for e in ("E1", "E2") for p in ("P0", "P1")]
        c += [dict(arm="ff_cond", preproc=p, task="cond", preset="E2", base_shift=1.0,
                   rank_mode="spread") for p in ("P0", "P1")]
        return c
    if stage == "S5":                                   # Amendment A1: P5 (frengression scaling)
        c = [dict(arm="A1s", preproc="P5", corpus=k, rank_mode="spread", **U) for k in ("A", "B")]
        c += [dict(arm="ff_cond", preproc="P5", task="cond", preset="E1", base_shift=1.0, rank_mode="spread")]
        c += [dict(arm="ff_full", preproc="P5", task="cond", preset=e, base_shift=1.0, rank_mode="spread")
              for e in ("E1", "E2")]
        return c
    if stage == "S6":                                   # Amendment A2: N, LT-N and LT at P1, E1
        return [dict(arm=a, preproc="P1", task="cond", preset="E1", base_shift=b, rank_mode="spread")
                for b in (1.0, 0.0) for a in ("n_cond", "lt_n", "lt")]
    if stage == "S7":                                   # Amendment A3: copula hidden ranks old vs new
        F = dict(arm="ff_full", preproc="P0", task="cond", base_shift=1.0, rank_mode="spread")
        c = [dict(**F, preset=e, copula_width=w, copula_rank_rule=r)
             for w, r in ((50, "old"), (16, "old"), (16, "new")) for e in ("E2", "E1")]
        c += [dict(**F, preset="E2", copula_width=16, copula_rank_rule=r, paper_setting=True,
                   **S7_PAPER) for r in ("old", "new")]
        return c
    if stage == "S8":                                   # Amendment A4: U, N, LT-N latent calibration
        return [dict(arm=a, preproc="P1", task="cond", preset="E1", base_shift=1.0, rank_mode="spread")
                for a in ("ff_cond", "n_cond", "lt_n")]
    if stage == "S9":                                   # Amendment A4: Gaussian-scale full FF (package)
        G = dict(preproc="P1", task="cond", base_shift=1.0, rank_mode="spread")
        c = [dict(arm=a, preset=e, **G) for a in ("gff_flex", "gff_shift") for e in ("E2", "E1")]
        c += [dict(arm=a, preset="E2", paper_setting=True, **S9_PAPER, **G) for a in ("gff_flex", "gff_shift")]
        return c
    if stage == "S10":                                  # Amendment A5: is the confounded recovery genuine?
        G = dict(preproc="P1", task="cond", preset="E2", rank_mode="spread", placebo_covariate=False)
        # truth x init at ps_slope 1.2; (base_shift 1, init zero) = the S9 gff_shift E2 sf41 cells (reused)
        c = [dict(arm="gff_shift", base_shift=b, shift_init=i, ps_slope=1.2, **G)
             for b in (1.0, 0.0, -1.0) for i in S10_INITS if (b, i) != (1.0, "zero")]
        c += [dict(arm="gff_shift", base_shift=1.0, shift_init=i, ps_slope=2.4, **G) for i in ("zero", "naive")]
        c += [dict(arm="lt_n", base_shift=1.0, shift_init="zero", ps_slope=1.2, **G)]          # Anchor A
        c += [dict(arm="gff_shift", base_shift=1.0, shift_init="zero", ps_slope=1.2,             # Anchor B
                   **{**G, "placebo_covariate": True})]
        return c
    raise ValueError(stage)


S10_INITS = ("zero", "naive", "plus2")
S10_SEEDS = tuple(range(31, 36))

# A3 paper-setting check (exploratory): Laura's agreed runner setting (6089a1a) at width 16
S7_PAPER = dict(lr=1e-3, max_epochs=1000, patience=30)
S7_PAPER_SEEDS = tuple(range(31, 36))
# A4 paper-setting check (exploratory) for S9: same settings and seeds as S7's
S9_PAPER = dict(S7_PAPER)


def identity_sha(ident: dict) -> str:
    return hashlib.sha256(json.dumps(ident, sort_keys=True, default=str).encode()).hexdigest()[:8]


def run_id_of(ident: dict) -> str:
    s7 = ""                                             # A3 keys exist only on S7 cells
    if "copula_rank_rule" in ident:
        s7 = f"R{ident['copula_rank_rule']}_W{ident['copula_width']}_" + ("paper_" if ident.get("paper_setting") else "")
    elif ident.get("paper_setting"):                    # S9 (A4) paper-setting cells
        s7 = "paper_"
    elif "shift_init" in ident:                         # S10 (A5) cells
        s7 = f"init{ident['shift_init']}_ps{ident['ps_slope']:g}_" + ("placebo_" if ident["placebo_covariate"] else "")
    return (f"{ident['stage']}_{ident['corpus']}_{ident['arm']}_{ident['preproc']}_{ident['task']}_bs{ident['base_shift']}_"
            f"{PRESET_SHORT[ident['preset']]}_{s7}sd{ident['seed_data']}_sf{ident['seed_fit']}_"
            f"{identity_sha(ident)}")


def is_primary(stage: str, c: dict) -> bool:
    """Prereg v1.1: configs in the primary families (+ all Corpus B) keep seed_fit {41, 42};
    every other config is exploratory and runs at seed_fit 41 only."""
    if c.get("corpus", "A") == "B" or stage == "S5":   # S5: every config primary (Amendment A1)
        return True
    ap = (c["arm"], c["preproc"])
    if stage == "S1":
        return ap in {("A1s", "P0"), ("A1s", "P1"), ("A2", "P1"), ("A1", "P0")}
    if stage == "S2":
        return c["base_shift"] == 1.0 and ap in {("ff_cond", "P0"), ("ff_cond", "P1"),
                                                  ("a2_cond", "P1"), ("lt", "P0")}
    if stage == "S4":
        return c["arm"] == "ff_full" and c["preset"] == "E2"
    if stage == "S6":                                   # A2: tau=1 primary; tau=0 exploratory
        return c["base_shift"] == 1.0
    if stage == "S7":                                   # A3: E2 primary; E1 and paper setting exploratory
        return c["preset"] == "E2" and not c.get("paper_setting", False)
    if stage == "S9":                                   # A4: E2 primary; E1 and paper setting exploratory
        return c["preset"] == "E2" and not c.get("paper_setting", False)
    # S8 (A4): descriptive, seed_fit 41 only; S10 (A5): seed_fit 41 only, seeds 31-35
    return False


def enumerate_cells(stage: str, smoke: bool = False, xla_flags: str | None = None) -> list[dict]:
    """Every cell of a stage, primary configs first. smoke: one cell per distinct config
    (seed_data 31, seed_fit 41, max_epochs 5, n_mc 2000). seed_mc2 on seed_data-31 cells only."""
    from halo_fit import identity_of
    out = []
    # stable; S7's / S9's paper-setting cells (<= 1000 epochs) are queued LAST (Amendments A3, A4)
    configs = sorted(_configs(stage), key=lambda c: (bool(c.get("paper_setting")), not is_primary(stage, c)))
    for c in configs:
        fits = SEEDS_FIT if is_primary(stage, c) else SEEDS_FIT[:1]
        seeds = S7_PAPER_SEEDS if c.get("paper_setting") else S10_SEEDS if stage == "S10" else SEEDS_DATA
        for sd in ((31,) if smoke else seeds):
            for sf in ((41,) if smoke else fits):
                cell = {**COMMON, **c, "stage": stage, "seed_data": sd, "seed_fit": sf,
                        "ps_slope": c.get("ps_slope", PS_SLOPE[c["preset"]]), "xla_flags": xla_flags,
                        "primary": is_primary(stage, c)}
                if sd == 31:
                    cell["seed_mc2"] = SEED_MC2
                if smoke:
                    cell.update(max_epochs=5, n_mc=2000)
                out.append(cell)
    if stage == "S1" and not smoke:                                # P4 demonstration cells
        out += [{**COMMON, "stage": "S1", "arm": "A1s", "preproc": "P4", "task": "uncond",
                 "preset": "E1", "base_shift": 0.0, "rank_mode": "spread", "seed_data": 31,
                 "seed_fit": 41, "ps_slope": 0.0, "xla_flags": xla_flags, "seed_mc2": SEED_MC2,
                 "primary": False}]                                  # exploratory: one fit
    if stage == "S1" and smoke:
        out.append({**out[0], "preproc": "P4"})
    for cell in out:
        assert cell["seed_fit"] != cell["seed_data"], "seed alias"
        ident = identity_of(cell)
        cell["identity_sha"] = identity_sha(ident)
        cell["run_id"] = run_id_of(ident)
    ids = [c["run_id"] for c in out]
    assert len(set(ids)) == len(ids), "duplicate identities"
    return out


# ------------------------------------------------------------------ process control
# pinned_env / _done_identities / filter_done / acquire_stage_lock / assert_stage_complete
# are adapted (with attribution) from ~/work/ff-capacity-runs/_resume/capacity_driver.py
# (~L505-696). There: identities parsed from argv and "done" = a finite ate_mae. Here:
# identities are the cell's config.json "config" block and "done" = metrics.json with
# complete == true.
def pinned_env(threads: int) -> dict:
    """One worker, ``threads`` cores; the XLA flag string is load-bearing (30% shifts in
    ate_mae across flag sets are documented), so it is recorded in every config.json."""
    e = dict(os.environ)
    e["XLA_FLAGS"] = (e.get("XLA_FLAGS", "") + f" --xla_cpu_multi_thread_eigen=false"
                      f" intra_op_parallelism_threads={threads}").strip()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS"):
        e[k] = str(threads)
    e["JAX_ENABLE_X64"] = "0"
    e["PYTHONHASHSEED"] = "0"
    return e


def _done_identities(stage_root: str):
    """(done run_ids, in-flight run_ids): a dir with config.json but no complete metrics is
    in flight or crashed (FAILED marker distinguishes the latter)."""
    done, inflight = [], []
    if not os.path.isdir(stage_root):
        return done, inflight
    for name in sorted(os.listdir(stage_root)):
        d = os.path.join(stage_root, name)
        if not os.path.exists(os.path.join(d, "config.json")):
            continue
        try:
            ok = json.load(open(os.path.join(d, "metrics.json"))).get("complete") is True
        except Exception:
            ok = False
        (done if ok else inflight).append(name)
    return done, inflight


def filter_done(cells, stage_root, relaunch_incomplete=False):
    done, inflight = _done_identities(stage_root)
    failed = {n for n in inflight if os.path.exists(os.path.join(stage_root, n, "FAILED"))}
    todo, skipped, held = [], 0, []
    for c in cells:
        if c["run_id"] in done:
            skipped += 1
        elif c["run_id"] in inflight and c["run_id"] not in failed and not relaunch_incomplete:
            held.append(c["run_id"])
        else:
            todo.append(c)
    return todo, skipped, held


def acquire_stage_lock(stage_root: str) -> str:
    """Refuse a second launcher on a stage (two launchers made the R15-F1 duplicate)."""
    os.makedirs(stage_root, exist_ok=True)
    path = os.path.join(stage_root, ".launcher.lock")
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            prev = int(open(path).read().split()[0])
            os.kill(prev, 0)
            sys.exit(f"refusing to start: live launcher pid {prev} holds {path}")
        except (OSError, ValueError, IndexError):
            print(f"stale lock {path} removed", file=sys.stderr)
            os.unlink(path)
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, f"{os.getpid()} {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n".encode())
    os.close(fd)
    atexit.register(lambda: os.path.exists(path) and os.unlink(path))
    return path


def assert_stage_complete(expected, stage_root) -> list[str]:
    """Every expected identity has exactly one complete record; prints missing/duplicate."""
    done, _ = _done_identities(stage_root)
    shas = {}
    for name in done:
        try:
            shas.setdefault(json.load(open(os.path.join(stage_root, name, "config.json")))["identity_sha"], []).append(name)
        except Exception:
            pass
    missing = [c["run_id"] for c in expected if c["run_id"] not in done]
    dup = [f"{k} x{len(v)}" for k, v in shas.items() if len(v) > 1]
    for m in missing[:30]:
        print(f"   missing: {m}", file=sys.stderr)
    for m in dup:
        print(f"   duplicate: {m}", file=sys.stderr)
    problems = ([f"{len(missing)} missing"] if missing else []) + ([f"{len(dup)} duplicated"] if dup else [])
    print(f"stage {os.path.basename(stage_root)}: {len(expected) - len(missing)}/{len(expected)} complete"
          + (f"; INCOMPLETE: {', '.join(problems)}" if problems else ""), file=sys.stderr)
    return problems


def run_cell(cell, stage_root, threads, smoke):
    cfg_path = os.path.join(stage_root, "_cells", cell["run_id"] + ".json")
    log = os.path.join(stage_root, "_logs", cell["run_id"] + ".log")
    cmd = [PY, os.path.join(HALO_DIR, "halo_fit.py"), "--config", cfg_path] + (["--smoke"] if smoke else [])
    t0 = time.monotonic()
    with open(log, "w") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=pinned_env(threads), cwd=HALO_DIR)
    secs = time.monotonic() - t0
    print(f"  [{'ok' if rc == 0 else 'FAIL rc=%d' % rc}] {cell['run_id']}  {secs:.0f}s", flush=True)
    return {"run_id": cell["run_id"], "rc": rc, "secs": secs}


def run_stage(stage, a) -> int:
    root = os.path.expanduser(a.runs_root)
    stage_root = os.path.join(root, stage)
    if stage == "S0":
        cmd = [PY, os.path.join(HALO_DIR, "halo_data.py"), "--stage", "S0", "--out", stage_root]
        print(" ".join(cmd))
        return 0 if a.dry_run else subprocess.call(cmd, env=pinned_env(a.threads), cwd=HALO_DIR)
    xla = pinned_env(a.threads)["XLA_FLAGS"]
    cells = enumerate_cells(stage, smoke=a.smoke, xla_flags=xla)
    if stage == "S7" and not a.dry_run:               # A3 deterministic connectivity table
        os.makedirs(stage_root, exist_ok=True)
        subprocess.call([PY, os.path.join(HALO_DIR, "halo_ranks.py"), "--out",
                         os.path.join(stage_root, "connectivity.json")], env=pinned_env(a.threads), cwd=HALO_DIR)
    for c in cells:
        c["out_dir"] = os.path.join(stage_root, c["run_id"])
    todo, skipped, held = filter_done(cells, stage_root, a.relaunch_incomplete) if a.resume else (cells, 0, [])
    print(f"{stage}: {len(cells)} cells, {skipped} done, {len(held)} held, {len(todo)} to run")
    if a.dry_run:
        for c in todo[:20]:
            print("  ", c["run_id"])
        return 0
    acquire_stage_lock(stage_root)
    for sub in ("_cells", "_logs"):
        os.makedirs(os.path.join(stage_root, sub), exist_ok=True)
    for c in cells:
        json.dump(c, open(os.path.join(stage_root, "_cells", c["run_id"] + ".json"), "w"), indent=1)
    stage_rec = {"stage": stage, "smoke": a.smoke, "n_cells": len(cells), "xla_flags": xla,
                 "prereg_sha256": hashlib.sha256(open(PREREG, "rb").read()).hexdigest(),
                 "started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "cells": [{"run_id": c["run_id"], "identity_sha": c["identity_sha"]} for c in cells]}
    json.dump(stage_rec, open(os.path.join(stage_root, "_stage.json"), "w"), indent=1)
    with ThreadPoolExecutor(a.conc) as ex:
        res = list(ex.map(lambda c: run_cell(c, stage_root, a.threads, a.smoke), todo))
    stage_rec["results"] = res
    json.dump(stage_rec, open(os.path.join(stage_root, "_stage.json"), "w"), indent=1)
    problems = assert_stage_complete(cells, stage_root) + (["held"] if held else [])
    return 1 if problems else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "all"])
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--relaunch-incomplete", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--runs-root", default=os.path.expanduser(os.environ.get("FF_RUNS_LOG", "~/work/halo-runs")))
    a = ap.parse_args(argv)
    stages = ["S0", "S1", "S2", "S4", "S3"] if a.stage == "all" else [a.stage]
    rc = 0
    for s in stages:
        rc |= run_stage(s, a)
    return rc


if __name__ == "__main__":
    sys.path.insert(0, HALO_DIR)
    sys.exit(main())
