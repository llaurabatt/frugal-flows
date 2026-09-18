"""One row per run: runs/exp_ate_recovery/index.csv, built from the run folders.

    python run_index.py                    # rebuild the whole index from every folder
    python run_index.py --upsert <run dir> # add or replace one folder's row (what
                                           # exp_ate_recovery.py calls when a run finishes)
    python run_index.py --query "<pandas expression>"   # e.g. "preset == 'E1' and K == 64"

The folders are the source of truth; the index is derived from them through
run_tables.read_run (which reads every folder layout) and can always be regenerated.
Columns come in blocks: identity/setup, architecture/optimisation, training,
performance, then every raw config key as cfg.<key>. Missing values are empty cells,
never the "not recorded" strings the tables print, so numeric columns stay numeric.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_tables import PRESET_TAG, read_run, regions  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "exp_ate_recovery")
INDEX = os.path.join(ROOT, "index.csv")


def _f(x):
    """A float or empty; never a string, never NaN (which the tables print as 'non-finite')."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return ""
    return v if np.isfinite(v) else ""


def row_for(d: str) -> dict:
    R = read_run(d)
    base = os.path.basename(d.rstrip("/"))
    wj = json.load(open(f"{d}/wandb.json")) if os.path.exists(f"{d}/wandb.json") else {}
    disc, ring, far = regions(R["size"], R["radius"])
    err = R["tau"] - R["ate"]
    e = np.where(np.isfinite(err), err, np.nan)
    fin = lambda z: None if z is None else np.where(np.isfinite(z), z, np.nan)  # noqa: E731
    e0, e1 = fin(R["e0"]), fin(R["e1"])
    v, t = np.asarray(R["val"]), np.asarray(R["train"])
    b = int(np.argmin(v))
    w = min(30, len(v))
    row = {
        # ---- identity and setup (Table 1)
        "run_id": base, "uid": R["record"].get("uid", base[-6:]), "stamp": base[:20],
        "wandb_name": R["record"].get("wandb_name", wj.get("name")), "wandb_id": wj.get("id") or "",
        "wandb_url": wj.get("url", ""), "layout": R["layout"],
        # dataset fingerprints (prepare_morphomnist_exps.dataset_identity): the join key to
        # baselines and to other fits on the same data, and the proof the bytes matched
        "dataset_id": R["record"].get("dataset_id") or R["metrics"].get("dataset_id", ""),
        "data_hash": R["record"].get("data_hash") or R["metrics"].get("data_hash", ""),
        # the model tag as the name carries it (check_runs.py verifies it against the config);
        # the raw config uses different words per layout (fullff / standalone / ...)
        "model": re.match(r"^(ff|margin_sep|margin_zero|margin)_", base[21:]).group(1),
        "model_desc": R["model"],
        # what differs from the plain fit, as the name carries it (empty for a plain fit):
        # everything between the arm and the k<K> field, e.g. "coplam4", "copw200", "bs0.5"
        "variant": (re.search(r"_(?:flexcont|loctrans)(?:-trf)?_(.*?)_k\d+_s\d+_", base[21:]) or [None, ""])[1],
        "preset": PRESET_TAG.get(R["preset"], R["preset"]), "preset_full": R["preset"],
        "arm": R["arm"], "conditioner": R["conditioner"] or "",
        "digit": "" if R["digit"] is None and R["n"] != 5923 else (0 if R["digit"] is None else R["digit"]),
        "n": R["n"], "n_train": R["n_train"], "n_val": R["n_val"],
        "seed_data": R["seed_data"], "seed_fit": R["seed_fit"],
        "seed_assign": R["cfg"].get("seed_assign") if R["cfg"].get("seed_assign") is not None else "",
        "size": R["size"], "K": R["K"], "radius": R["radius"],
        "ps_slope": R["ps_slope"], "base_shift": R["base_shift"],
        "true_effect_disc": _f(R["ate"][disc].mean()),
        "n_disc": int(disc.sum()), "n_ring": int(ring.sum()), "n_far": int(far.sum()),
        "n_mc": R["n_mc"],
        # ---- architecture and optimisation (Table 4)
        "nn_width": R["nn_width"], "nn_depth": R["nn_depth"], "flow_layers": R["flow_layers"],
        "rqs_knots": R["knots"], "nn_heads": R["heads"] if R["conditioner"] == "transformer" else "",
        "copula": R["has_copula"],
        "copula_nn_width": R["cop_width"] if R["has_copula"] else "",
        "copula_nn_depth": R["cop_depth"] if R["has_copula"] else "",
        "copula_flow_layers": R["cop_layers"] if R["has_copula"] else "",
        "copula_rqs_knots": R["cop_knots"] if R["has_copula"] else "",
        "n_params": R["n_params"] or "", "lr": R["lr"], "batch_size": R["batch_size"],
        "precision": R["precision"] if "not recorded" not in str(R["precision"]) else "",
        # ---- training (Table 3)
        "termination": R["termination"] or "", "converged": R["converged"],
        "epochs_run": R["epochs"], "best_epoch": R["best_epoch"],
        "val_loss_best": _f(v[b]), "train_loss_best": _f(t[b]), "train_val_gap_best": _f(t[b] - v[b]),
        "val_loss_first": _f(v[0]), "val_loss_last": _f(v[-1]),
        "val_gain_last30_before_best": _f(v[b] - v[b - 30:b].min()) if b >= 30 else "",
        "val_sd_last30": _f(v[-w:].std()), "val_min_last30": _f(v[-w:].min()), "val_max_last30": _f(v[-w:].max()),
        "wall_s": _f(R["wall_s"]),
        # ---- performance (Table 2)
        "mae_all": _f(np.nanmean(np.abs(e))), "rmse_all": _f(np.sqrt(np.nanmean(e ** 2))),
        "signed_disc": _f(np.nanmean(e[disc])), "signed_ring": _f(np.nanmean(e[ring])), "signed_far": _f(np.nanmean(e[far])),
        "mae_disc": _f(np.nanmean(np.abs(e[disc]))), "mae_ring": _f(np.nanmean(np.abs(e[ring]))), "mae_far": _f(np.nanmean(np.abs(e[far]))),
        "e0_disc": _f(np.nanmean(e0[disc])) if e0 is not None else "",
        "e0_ring": _f(np.nanmean(e0[ring])) if e0 is not None else "",
        "e0_far": _f(np.nanmean(e0[far])) if e0 is not None else "",
        "e1_disc": _f(np.nanmean(e1[disc])) if e1 is not None else "",
        "e1_ring": _f(np.nanmean(e1[ring])) if e1 is not None else "",
        "e1_far": _f(np.nanmean(e1[far])) if e1 is not None else "",
        "se_disc": _f(R["se"]["reference_disc"]), "se_ring": _f(R["se"]["reference_ring"]), "se_far": _f(R["se"]["far_region"]),
        # against the images: imbalance (observed − true), estimate − observed, sampled arm
        # means − that arm's images (all over the full dataset)
        **{f"imb_{r}": _f(np.nanmean(np.where(np.isfinite(R["imbalance"]), R["imbalance"], np.nan)[m]))
           if R.get("imbalance") is not None else "" for r, m in (("disc", disc), ("ring", ring), ("far", far))},
        **({"vsobs_mae_all": _f(np.nanmean(np.abs(np.where(np.isfinite(R["tau"] - R["obs_diff"]), R["tau"] - R["obs_diff"], np.nan))))}
           if R.get("obs_diff") is not None else {"vsobs_mae_all": ""}),
        **{f"vsobs_signed_{r}": _f(np.nanmean(np.where(np.isfinite(R["tau"] - R["obs_diff"]), R["tau"] - R["obs_diff"], np.nan)[m]))
           if R.get("obs_diff") is not None else "" for r, m in (("disc", disc), ("ring", ring), ("far", far))},
        **{f"d0_{r}": _f(np.nanmean(np.where(np.isfinite(R["d0"]), R["d0"], np.nan)[m])) if R.get("d0") is not None else ""
           for r, m in (("disc", disc), ("ring", ring), ("far", far))},
        **{f"d1_{r}": _f(np.nanmean(np.where(np.isfinite(R["d1"]), R["d1"], np.nan)[m])) if R.get("d1") is not None else ""
           for r, m in (("disc", disc), ("ring", ring), ("far", far))},
        "nonfinite_values": R["nonfinite"] if R["nonfinite"] is not None else "",
        "nonfinite_pixels": int((~np.isfinite(err)).sum()),
    }
    if R["layout"] == "margin_only" and R.get("arm_regional"):
        for reg, key in (("disc", "disc"), ("ring", "ring")):
            if R["arm_regional"].get(key):
                row[f"e0_{reg}"], row[f"e1_{reg}"] = R["arm_regional"][key]
    if R["sep"]:
        row["epochs_run_arm1"] = R.get("arm_info", {}).get("arm1_epochs_run", "")
        row["best_epoch_arm1"] = R.get("arm_info", {}).get("arm1_best_epoch", "")
    # ---- every raw config key, so nothing identifying is lost
    for k, val in sorted(R["cfg"].items()):
        if isinstance(val, (dict, list)):
            val = json.dumps(val)
        row[f"cfg.{k}"] = "" if val is None else val
    return row


def _read_index(path=INDEX):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_index(rows, path=INDEX):
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    # config columns last, everything else in first-seen order
    cols = [c for c in cols if not c.startswith("cfg.")] + sorted(c for c in cols if c.startswith("cfg."))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def rebuild() -> int:
    rows = [row_for(d.rstrip("/")) for d in sorted(glob.glob(f"{ROOT}/2*/"))]
    _write_index(rows)
    return len(rows)


def upsert(run_dir: str) -> str:
    row = row_for(run_dir)
    rows = [r for r in _read_index() if r.get("uid") != row["uid"]]
    rows.append(row)
    rows.sort(key=lambda r: r["run_id"])
    _write_index(rows)
    return row["run_id"]


# ----------------------------------------------------------------------------- baselines
BASELINES_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "baselines")
BASELINES_INDEX = os.path.join(BASELINES_ROOT, "index.csv")


def baseline_rows_for(d: str) -> list:
    """One row per estimator in a baselines run folder, with the same column names as the
    flow index wherever the meaning is the same, so the two join on dataset_id."""
    cj = json.load(open(f"{d}/config.json"))
    mj = json.load(open(f"{d}/metrics.json"))
    c, g = cj["config"], cj.get("generator_config", {})
    base = os.path.basename(d.rstrip("/"))
    a = np.load(f"{d}/arrays.npz")
    ate = a["ATE"]
    size = int(c["size"])
    radius = c.get("radius") if c.get("radius") is not None else max(1, round(size / 4))
    disc, ring, far = regions(size, radius)
    rows = []
    for method, m in mj["methods"].items():
        row = {
            "run_id": base, "uid": cj["uid"], "stamp": base[:20], "method": method, "basis": c["basis"],
            "dataset_id": cj["dataset_id"], "data_hash": cj["data_hash"],
            "model": f"baseline_{method}",
            "preset": PRESET_TAG.get(c["preset"], c["preset"]), "preset_full": c["preset"],
            "variant": (re.search(r"^baselines_e\d_(.*?)_k\d+_sd\d+_", base[21:]) or [None, ""])[1],
            "digit": "" if c.get("digit") is None else c["digit"],
            "n": mj["n_units"], "seed_data": c["seed_data"],
            "seed_assign": c.get("seed_assign") if c.get("seed_assign") is not None else "",
            "size": size, "K": size * size,
            "radius": radius, "ps_slope": g.get("ps_slope", ""), "base_shift": c.get("base_shift"),
            "true_effect_disc": _f(ate[disc].mean()),
            "n_disc": int(disc.sum()), "n_ring": int(ring.sum()), "n_far": int(far.sum()),
            "mae_all": _f(m["mae_all"]), "rmse_all": _f(m["rmse_all"]),
            "signed_disc": _f(m["signed_disc"]), "signed_ring": _f(m["signed_ring"]), "signed_far": _f(m["signed_far"]),
            "mae_disc": _f(m["mae_disc"]), "mae_ring": _f(m["mae_ring"]), "mae_far": _f(m["mae_far"]),
            "ate_mae_on_support": _f(m["ate_mae_on_support"]), "ate_mae_off_support": _f(m["ate_mae_off_support"]),
            "ate_corr": _f(m["ate_corr"]), "att_mae": _f(m["att_mae"]), "atc_mae": _f(m["atc_mae"]),
            "nonfinite_pixels": int((~np.isfinite(a[f"tau_hat_{method}"])).sum()),
            "seconds": _f(m.get("seconds")),
            # against the images (same names as the flow index)
            **{f"imb_{r}": _f(mj.get("imbalance", {}).get(f"imb_signed_{r}")) for r in ("disc", "ring", "far")},
            "vsobs_mae_all": _f(m.get("vsobs_mae_all")),
            **{f"vsobs_signed_{r}": _f(m.get(f"vsobs_signed_{r}")) for r in ("disc", "ring", "far")},
        }
        if method == "naive" and "e0_naive" in a.files:
            e0, e1 = a["e0_naive"], a["e1_naive"]
            for reg, mk in (("disc", disc), ("ring", ring), ("far", far)):
                row[f"e0_{reg}"] = _f(e0[mk].mean())
                row[f"e1_{reg}"] = _f(e1[mk].mean())
        for k, val in sorted(g.items()):
            row[f"cfg.{k}"] = "" if val is None else (json.dumps(val) if isinstance(val, (dict, list)) else val)
        rows.append(row)
    return rows


def rebuild_baselines() -> int:
    rows = []
    for d in sorted(glob.glob(f"{BASELINES_ROOT}/2*/")):
        rows += baseline_rows_for(d.rstrip("/"))
    _write_index(rows, BASELINES_INDEX)
    return len(rows)


def upsert_baselines(run_dir: str) -> str:
    new = baseline_rows_for(run_dir)
    uid = new[0]["uid"]
    rows = [r for r in _read_index(BASELINES_INDEX) if r.get("uid") != uid] + new
    rows.sort(key=lambda r: (r["run_id"], r["method"]))
    _write_index(rows, BASELINES_INDEX)
    return new[0]["run_id"]


def compare(where: str | None = None, columns=("mae_all", "signed_disc", "signed_ring", "signed_far")):
    """Join the two indexes on dataset_id and print every method on every dataset."""
    import warnings

    import pandas as pd
    warnings.simplefilter("ignore")   # pandas/numpy notices about string concatenation and concat
    ff = pd.read_csv(INDEX)
    bl = pd.read_csv(BASELINES_INDEX) if os.path.exists(BASELINES_INDEX) else pd.DataFrame()
    # a flow row's label: model, arm (-trf), variant, fit seed -- everything that tells two
    # fits on the same dataset apart, e.g. ff_loctrans_s101, ff_flexcont_coplam4_s101
    var = ff["variant"].fillna("").astype(str)
    trf = np.where(ff["conditioner"].fillna("") == "transformer", "-trf", "")
    ff = ff.assign(method=ff["model"] + "_" + ff["arm"] + trf
                   + np.where(var != "", "_" + var, "") + "_s" + ff["seed_fit"].astype(str))
    both = pd.concat([ff, bl], ignore_index=True, sort=False)
    both = both[both["dataset_id"].notna()]
    if where:
        both = both.query(where)
    # the join is only valid where the bytes agree
    for did, grp in both.groupby("dataset_id"):
        if grp["data_hash"].nunique() > 1:
            print(f"WARNING dataset {did}: differing data_hash across runs -> {sorted(grp['data_hash'].unique())}")
    keys = ["dataset_id", "preset", "K", "seed_data", "base_shift", "ps_slope"]
    keys = [k for k in keys if k in both.columns]
    out = both[keys + ["method", "termination"] + list(columns)] if "termination" in both.columns else both[keys + ["method"] + list(columns)]
    with pd.option_context("display.width", 250, "display.max_rows", 1000, "display.max_columns", 40):
        print(out.sort_values(keys + ["method"]).to_string(index=False))
    print(f"{both['dataset_id'].nunique()} datasets, {len(both)} rows", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upsert", metavar="RUN_DIR", default=None)
    ap.add_argument("--query", metavar="EXPR", default=None,
                    help="pandas .query() expression over the index; prints the matching rows' key columns")
    ap.add_argument("--columns", default="run_id,model,preset,K,seed_fit,termination,mae_all,signed_disc,signed_ring,signed_far",
                    help="columns to print with --query (comma-separated)")
    ap.add_argument("--baselines", action="store_true", help="rebuild runs/baselines/index.csv instead")
    ap.add_argument("--compare", nargs="?", const="", default=None, metavar="EXPR",
                    help="join both indexes on dataset_id and print every method per dataset; "
                         "optional pandas expression to filter, e.g. \"preset == 'E1'\"")
    args = ap.parse_args()
    if args.baselines:
        print(f"rebuilt {BASELINES_INDEX}: {rebuild_baselines()} rows", file=sys.stderr)
    elif args.compare is not None:
        compare(args.compare or None)
    elif args.upsert:
        print(f"indexed {upsert(args.upsert)}", file=sys.stderr)
    elif args.query is not None:
        import pandas as pd
        df = pd.read_csv(INDEX)
        sub = df.query(args.query) if args.query else df
        with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 500):
            print(sub[[c for c in args.columns.split(",") if c in sub.columns]].to_string(index=False))
        print(f"{len(sub)} of {len(df)} runs", file=sys.stderr)
    else:
        print(f"rebuilt {INDEX}: {rebuild()} runs", file=sys.stderr)


if __name__ == "__main__":
    main()
