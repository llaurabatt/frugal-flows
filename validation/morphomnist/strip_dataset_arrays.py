"""Remove the dataset copies (Y, ITE) from old runs' arrays.npz (2026-10-01).

From 2026-10-01 runs do not save Y and ITE; dataset_store rebuilds them from config.json.
This removes them from runs made before then, ONLY after checking, run by run, that:
  1. the run is finished (metrics.json and arrays.npz exist);
  2. rebuilding its dataset reproduces the recorded dataset_id and data_hash, AND the rebuilt Y and
     ITE are bit-for-bit identical to the ones stored in this run's arrays.npz;
  3. the rewritten file, read back from a temporary path, holds every other array unchanged.
Only then is arrays.npz replaced (atomic rename). Anything else is skipped and logged.

  python strip_dataset_arrays.py            # dry run: report only
  python strip_dataset_arrays.py --apply    # rewrite; manifest in runs/datasets/strip_manifest_<stamp>.csv
"""
import argparse
import collections
import csv
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dataset_store as DS  # noqa: E402

ROOTS = [os.path.join(HERE, "runs", r) for r in ("exp_ate_recovery", "frengression", "baselines")]

p = argparse.ArgumentParser()
p.add_argument("--apply", action="store_true")
args = p.parse_args()

candidates = []
for root in ROOTS:
    for d in sorted(glob.glob(os.path.join(root, "2*/"))):
        a, m, c = (os.path.join(d, f) for f in ("arrays.npz", "metrics.json", "config.json"))
        if not (os.path.exists(a) and os.path.exists(m) and os.path.exists(c)):
            continue
        with np.load(a) as z:
            if any(k in z.files for k in DS.DROPPED_KEYS):
                candidates.append(d)
by_ds = collections.defaultdict(list)
for d in candidates:
    by_ds[json.load(open(os.path.join(d, "config.json"))).get("dataset_id")].append(d)
print(f"{len(candidates)} runs hold Y/ITE, {len(by_ds)} distinct datasets", flush=True)

stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
rows, freed, n_ok, n_skip = [], 0, 0, 0
for did, runs in by_ds.items():
    try:
        data = DS.build_for_run(runs[0])      # checks dataset_id / data_hash against runs[0]'s record
        ref = {k: np.asarray(data[k]) for k in DS.DROPPED_KEYS}
    except Exception as e:                    # noqa: BLE001
        for d in runs:
            rows.append((d, "skip", f"rebuild failed: {str(e)[:150]}", 0)); n_skip += 1
        continue
    for d in runs:
        rec = json.load(open(os.path.join(d, "config.json")))
        if rec.get("data_hash") is not None and rec["data_hash"] != data["data_hash"]:
            rows.append((d, "skip", "data_hash differs from the rebuilt dataset", 0)); n_skip += 1; continue
        path = os.path.join(d, "arrays.npz")
        with np.load(path) as z:
            stored = {k: z[k] for k in z.files}
        same = all(k not in stored or np.array_equal(stored[k], ref[k].astype(stored[k].dtype)) for k in DS.DROPPED_KEYS)
        if not same:
            rows.append((d, "skip", "stored Y/ITE differ from the rebuilt dataset", 0)); n_skip += 1; continue
        keep = {k: v for k, v in stored.items() if k not in DS.DROPPED_KEYS}
        before = os.path.getsize(path)
        if not args.apply:
            gain = sum(stored[k].nbytes for k in DS.DROPPED_KEYS if k in stored)
            rows.append((d, "would strip", "match", gain)); freed += gain; n_ok += 1; continue
        tmp = path + ".strip_tmp.npz"
        np.savez(tmp, **keep)
        with np.load(tmp) as z:
            ok = set(z.files) == set(keep) and all(np.array_equal(z[k], keep[k]) for k in keep)
        if not ok:
            os.remove(tmp)
            rows.append((d, "skip", "rewritten file did not read back identically", 0)); n_skip += 1; continue
        os.replace(tmp, path)
        gain = before - os.path.getsize(path)
        rows.append((d, "stripped", "match", gain)); freed += gain; n_ok += 1
    print(f"dataset {did}: {len(runs)} runs done; total {'freed' if args.apply else 'would free'} {freed/1e9:.2f} GB", flush=True)

out = os.path.join(DS.CACHE_DIR, f"strip_manifest_{stamp}{'' if args.apply else '_dryrun'}.csv")
os.makedirs(DS.CACHE_DIR, exist_ok=True)
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["run_dir", "action", "reason", "bytes"]); w.writerows(rows)
print(f"=== {'APPLIED' if args.apply else 'DRY RUN'}: {n_ok} runs {'stripped' if args.apply else 'would be stripped'}, "
      f"{n_skip} skipped, {freed/1e9:.2f} GB; manifest {out}")
for reason, cnt in collections.Counter(r[2] for r in rows if r[1] == "skip").items():
    print(f"  skipped: {cnt} x {reason}")
