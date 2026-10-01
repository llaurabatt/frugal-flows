"""S10 (Amendment A5) paired frengression: run the frengression runner on the halo datasets and
score its tau_hat with the S10 endpoints.

    python halo_frengression.py run  [--conc 4] [--threads 2] [--smoke] [--root ~/work/halo-runs/S10/frengression]
    python halo_frengression.py post [--root ...]

``run`` launches ``validation/morphomnist/exp_frengression_recovery.py`` once per (preset, seed_data)
(E2 and E1, seed_data 31-40) as a subprocess at the runner's frozen settings, passed explicitly
(lr 1e-3, hidden_dim 100, num_layer 3, noise_dim 64, y_scaling per_pixel, y_sd_floor 0.25,
num_iters 5000, n_mc 50000), size 8, digit 0, n None (all 5,923 digit-0 images, as the halo
cells), seed_fit 41 (the halo seed_fit), into ``<root>/fr_<e1|e2>_sd<S>``. The runner owns its
own import order (torch before jax); this launcher imports neither.

``post`` (a separate process: no torch) rebuilds each cell's dataset with
``halo_data.build_dataset`` and checks it is the SAME dataset (data_hash and dataset_id equal to
the runner's, X and the ATE vector equal, and the data_hash equal to the halo S9/S4 cells' of the
same preset and seed when present), then writes ``halo_s10.json`` (S10 endpoints on the runner's
saved tau_hat) per cell and ``_dataset_check.json`` for the whole set.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HALO_DIR = os.path.dirname(os.path.abspath(__file__))
MM_DIR = os.path.abspath(os.path.join(HALO_DIR, "..", ".."))
WORKTREE = os.path.abspath(os.path.join(MM_DIR, "..", ".."))
RUNNER = os.path.join(MM_DIR, "exp_frengression_recovery.py")
PRESETS = {"E2": "exp2_confounded_homogeneous", "E1": "exp1_rct_homogeneous"}
SEEDS = tuple(range(31, 41))
SEED_FIT = 41
FROZEN = ["--lr", "1e-3", "--hidden-dim", "100", "--num-layer", "3", "--noise-dim", "64",
          "--y-scaling", "per_pixel", "--y-sd-floor", "0.25", "--num-iters", "5000", "--n-mc", "50000",
          "--size", "8", "--digit", "0"]
DEFAULT_ROOT = os.path.expanduser("~/work/halo-runs/S10/frengression")


def cells(smoke: bool) -> list[tuple[str, int]]:
    return [("E2", 31)] if smoke else [(e, s) for e in ("E2", "E1") for s in SEEDS]


def cell_dir(root: str, e: str, s: int) -> str:
    return os.path.join(root, f"fr_{e.lower()}_sd{s}")


def _done(d: str) -> bool:
    try:
        return json.load(open(os.path.join(d, "metrics.json"))).get("status") == "ok"
    except Exception:
        return False


def run(a) -> int:
    root = os.path.expanduser(a.root)
    os.makedirs(os.path.join(root, "_logs"), exist_ok=True)
    env = dict(os.environ, PYTHONPATH=WORKTREE, JAX_ENABLE_X64="0", PYTHONHASHSEED="0")
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[k] = str(a.threads)
    todo = [(e, s) for e, s in cells(a.smoke) if not _done(cell_dir(root, e, s))]
    print(f"frengression: {len(cells(a.smoke))} cells, {len(todo)} to run", flush=True)

    def one(es):
        e, s = es
        d = cell_dir(root, e, s)
        cmd = [sys.executable, RUNNER, "--preset", PRESETS[e], "--seed-data", str(s), "--seed-fit", str(SEED_FIT),
               *FROZEN, "--threads", str(a.threads), "--no-plots", "--run-dir", d, "--overwrite"]
        if a.smoke:
            cmd[cmd.index("--num-iters") + 1] = "50"
            cmd[cmd.index("--n-mc") + 1] = "5000"
        t0 = time.monotonic()
        with open(os.path.join(root, "_logs", os.path.basename(d) + ".log"), "w") as fh:
            fh.write(" ".join(cmd) + "\n")
            fh.flush()
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=MM_DIR)
        ok = rc == 0 and _done(d)
        print(f"  [{'ok' if ok else 'FAIL rc=%d' % rc}] {os.path.basename(d)} {time.monotonic() - t0:.0f}s", flush=True)
        return ok

    with ThreadPoolExecutor(a.conc) as ex:
        res = list(ex.map(one, todo))
    missing = [os.path.basename(cell_dir(root, e, s)) for e, s in cells(a.smoke) if not _done(cell_dir(root, e, s))]
    print(f"frengression complete: {len(cells(a.smoke)) - len(missing)}/{len(cells(a.smoke))}"
          + (f"; MISSING {missing}" if missing else ""), flush=True)
    return 1 if missing or not all(res) else 0


def _halo_hashes(runs_root: str) -> dict:
    """(preset short, seed_data) -> set of data_hash recorded by halo P1 cells (S9 gff, S4 ff_full)."""
    out: dict = {}
    for st, pat in (("S9", "S9_A_gff_*_P1_cond_bs1.0_*"), ("S4", "S4_A_ff_full_P1_cond_bs1.0_*")):
        for d in glob.glob(os.path.join(runs_root, st, pat)):
            try:
                c = json.load(open(os.path.join(d, "config.json")))["config"]
                h = json.load(open(os.path.join(d, "metrics.json")))["data_hash"]
            except Exception:
                continue
            if c.get("paper_setting"):
                continue
            out.setdefault((c["preset"], c["seed_data"]), set()).add(h)
    return out


def post(a) -> int:
    import numpy as np
    sys.path.insert(0, HALO_DIR)
    import halo_data as hd
    import halo_metrics as hmx
    root = os.path.expanduser(a.root)
    halo_h = _halo_hashes(os.path.dirname(os.path.dirname(root.rstrip("/"))))
    checks, bad = [], 0
    for e, s in cells(a.smoke):
        d = cell_dir(root, e, s)
        if not _done(d):
            checks.append({"cell": os.path.basename(d), "status": "missing"})
            bad += 1
            continue
        conf = json.load(open(os.path.join(d, "config.json")))
        z = np.load(os.path.join(d, "arrays.npz"))
        data = hd.build_dataset({"preset": e, "base_shift": 1.0, "seed_data": s, "preproc": "P1", "corpus": "A"})
        ref = hd.class_reference(s, "A")
        cls = hd.pixel_classes(ref["Y"], data["disc"], ref["RAW"])
        hh = halo_h.get((e, s), set())
        chk = {"cell": os.path.basename(d), "preset": e, "seed_data": s,
               "runner_data_hash": conf.get("data_hash"), "halo_data_hash": data["data_hash"],
               "runner_dataset_id": conf.get("dataset_id"), "halo_dataset_id": data["dataset_id"],
               "data_hash_equal": conf.get("data_hash") == data["data_hash"],
               "dataset_id_equal": conf.get("dataset_id") == data["dataset_id"],
               "X_equal": bool(np.array_equal(np.asarray(z["X"], np.float64), data["X"])),
               "ATE_equal": bool(np.array_equal(np.asarray(z["ATE"], np.float64), data["ATE"])),
               "halo_cells_hashes": sorted(hh),
               "halo_cells_hash_equal": (data["data_hash"] in hh and len(hh) == 1) if hh else None,
               "n_units": int(json.load(open(os.path.join(d, "metrics.json"))).get("n_units", -1)),
               "runner_config": {k: conf["config"].get(k) for k in ("lr", "hidden_dim", "num_layer", "noise_dim",
                                                                     "y_scaling", "y_sd_floor", "num_iters", "n_mc",
                                                                     "size", "digit", "n", "seed_fit")}}
        chk["identical"] = all(chk[k] for k in ("data_hash_equal", "dataset_id_equal", "X_equal", "ATE_equal")) \
            and chk["halo_cells_hash_equal"] is not False and chk["n_units"] == len(data["Y"])
        bad += not chk["identical"]
        naive = hd.naive_diff(data["Y"], data["X"])
        ep = hmx.s10_endpoints(z["tau_hat"], data["ATE"], naive, cls["disc"], cls["active_off"], cls["quiet"])
        met = json.load(open(os.path.join(d, "metrics.json")))
        json.dump({"preset": e, "seed_data": s, "endpoints": ep, "tau_hat": np.asarray(z["tau_hat"]).tolist(),
                   "truth_map": data["ATE"].tolist(), "naive_map": naive.tolist(), "dataset_check": chk,
                   "runner_ate_mae": met.get("ate_mae"), "n_iters_run": met.get("n_iters_run"),
                   "diverged": met.get("diverged")},
                  open(os.path.join(d, "halo_s10.json"), "w"), indent=1)
        checks.append(chk)
        print(f"{chk['cell']}: identical={chk['identical']} ate_mae={ep['ate_mae']:.4f} "
              f"(runner {met.get('ate_mae')}) rho={ep['rho']:+.3f}", flush=True)
    json.dump({"n": len(checks), "n_not_identical_or_missing": bad, "checks": checks},
              open(os.path.join(root, "_dataset_check.json"), "w"), indent=1)
    print(f"dataset check: {len(checks) - bad}/{len(checks)} identical", flush=True)
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run", "post"])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--smoke", action="store_true", help="one E2 cell, 50 iterations, n_mc 5000")
    a = ap.parse_args(argv)
    return run(a) if a.action == "run" else post(a)


if __name__ == "__main__":
    sys.exit(main())
