"""Check every run folder for naming and config consistency, on disk and against wandb.

    python check_runs.py [--no-wandb]

For each folder under runs/exp_ate_recovery/ it checks:
  1. the folder name has the form <UTC stamp>_<wandb name>, and the wandb name has the form
     <model>_<preset>_<arm>[-trf]_[<variant>_]k<K>_s<seed>_d<digit>_<uid>
  2. config.json exists and its run_id / wandb_name / uid agree with the folder name
  3. every field the name encodes (model, preset, arm, conditioner, K, seed, digit, and the
     variants that config records: copw, bs, rct) agrees with the config
  4. metrics.json (when present) carries the folder's run_id
  5. the files its layout requires are present
  6. wandb.json exists, names the same wandb name, and (unless --no-wandb) the run fetched by
     id has that name and the same run_id / wandb_name / uid / preset / seed / size in its config
  7. uids, wandb names and wandb ids are unique across folders
  8. every wandb run owned by the user has a folder (unless --no-wandb)
Prints every problem and a summary; exit code 1 if anything failed.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "exp_ate_recovery")
STAMP = r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z"
NAME = re.compile(
    r"^(?P<model>ff|margin|margin_sep)_(?P<preset>e[1-6])_(?P<arm>flexcont|loctrans)(?P<trf>-trf)?"
    r"(?P<variants>(?:_[a-z]+[0-9.]*(?:ep)?)*)_k(?P<k>\d+)_s(?P<seed>\d+)_d(?P<digit>\d|0-9)_(?P<uid>[0-9a-f]{6})$")
PRESET_OF = {"exp1_rct_homogeneous": "e1", "exp2_confounded_homogeneous": "e2", "exp3_confounded_heterogeneous": "e3",
             "exp4_covariate_cate": "e4", "exp5_quantile_effect": "e5", "exp6_spatial_cate": "e6"}
ARM_OF = {"flexible_continuous": "flexcont", "location_translation": "loctrans"}
LAYOUTS = {  # required files per layout, keyed by the file that identifies the layout
    "metrics.json": ["config.json", "metrics.json", "arrays.npz", "log.txt", "wandb.json"],
    "result.json": ["config.json", "result.json", "result.npz", "wandb.json"],
    "history.npz": ["config.json", "history.npz", "log.txt", "wandb.json"],
}


def main():
    use_wandb = "--no-wandb" not in sys.argv
    root = ROOT
    if "--root" in sys.argv:                       # check a trial folder instead of the real runs
        root = sys.argv[sys.argv.index("--root") + 1]
    problems, folders = [], sorted(glob.glob(f"{root}/*/"))
    folders = [f.rstrip("/") for f in folders if not os.path.basename(f.rstrip("/")).startswith("_")]
    seen_uid, seen_wn, seen_wid = Counter(), Counter(), Counter()
    linked = {}

    def bad(folder, msg):
        problems.append(f"{os.path.basename(folder)}: {msg}")

    for d in folders:
        base = os.path.basename(d)
        # 1. name shape
        m = re.match(rf"^({STAMP})_(.+)$", base)
        if not m:
            bad(d, "folder name does not start with a UTC stamp")
            continue
        stamp, wn = m.groups()
        nm = NAME.match(wn)
        if not nm:
            bad(d, f"wandb name part does not match the convention: {wn}")
            continue
        seen_uid[nm["uid"]] += 1
        seen_wn[wn] += 1
        # 2. config.json
        if not os.path.exists(f"{d}/config.json"):
            bad(d, "no config.json")
            continue
        cj = json.load(open(f"{d}/config.json"))
        c = cj.get("config", cj)
        for key, want in (("run_id", base), ("wandb_name", wn), ("uid", nm["uid"])):
            if cj.get(key) != want:
                bad(d, f"config.json {key} = {cj.get(key)!r}, folder says {want!r}")
            if isinstance(c, dict) and c is not cj and c.get(key) not in (None, want):
                bad(d, f"config.json config.{key} = {c.get(key)!r}, folder says {want!r}")
        # 3. name fields vs config
        if c.get("preset") and PRESET_OF.get(c["preset"]) != nm["preset"]:
            bad(d, f"name preset {nm['preset']} vs config preset {c.get('preset')}")
        arm_cfg = ARM_OF.get(c.get("arm", "flexible_continuous"))
        if arm_cfg != nm["arm"]:
            bad(d, f"name arm {nm['arm']} vs config arm {c.get('arm')}")
        cond = c.get("conditioner") or (c.get("margin") or {}).get("conditioner")
        if (cond == "transformer") != bool(nm["trf"]):
            bad(d, f"name {'has' if nm['trf'] else 'lacks'} -trf but config conditioner is {cond}")
        k_cfg = c.get("K") or (int(c["size"]) ** 2 if c.get("size") else None)
        if k_cfg and int(nm["k"]) != k_cfg:
            bad(d, f"name K={nm['k']} vs config {k_cfg}")
        if c.get("seed_fit") is not None and int(nm["seed"]) != c["seed_fit"]:
            bad(d, f"name seed {nm['seed']} vs config seed_fit {c['seed_fit']}")
        if "digit" in c:
            want = "0-9" if c["digit"] is None else str(c["digit"])
            if nm["digit"] != want:
                bad(d, f"name digit {nm['digit']} vs config digit {c['digit']}")
        model = c.get("model")
        want_model = {"fullff": "ff", None: "ff", "ff": "ff", "separate": "margin_sep", "margin_sep": "margin_sep",
                      "standalone": "margin", "margin": "margin"}.get(model)
        if model and "no copula" in str(model):
            want_model = "margin"
        if want_model and nm["model"] != want_model:
            bad(d, f"name model {nm['model']} vs config model {model!r} (expected {want_model})")
        variants = [v for v in nm["variants"].split("_") if v]
        copw = [v for v in variants if v.startswith("copw") and not v.startswith("copwd")]
        w = c.get("copula_nn_width")
        if w is not None and w != 50 and f"copw{w}" not in variants:
            bad(d, f"config copula_nn_width={w} but name has no copw{w}")
        if copw and (w is None or f"copw{w}" != copw[0]):
            bad(d, f"name has {copw[0]} but config copula_nn_width={w}")
        # effect<size>: present iff base_shift is set to something other than the default 1.0
        eff = [v for v in variants if v.startswith("effect")]
        bshift = c.get("base_shift")
        if bshift is not None and float(bshift) != 1.0 and f"effect{float(bshift):g}" not in variants:
            bad(d, f"config base_shift={bshift} but name has no effect{float(bshift):g} tag")
        if eff and (bshift is None or f"effect{float(bshift):g}" != eff[0]):
            bad(d, f"name has {eff[0]} but config base_shift={bshift}")
        sa = [v for v in variants if v.startswith("sa")]
        if c.get("seed_assign") is not None and f"sa{c['seed_assign']}" not in variants:
            bad(d, f"config seed_assign={c['seed_assign']} but name has no sa tag")
        if sa and (c.get("seed_assign") is None or f"sa{c['seed_assign']}" != sa[0]):
            bad(d, f"name has {sa[0]} but config seed_assign={c.get('seed_assign')}")
        if ("rct" in variants) != (c.get("ps_slope") == 0 and nm["preset"] != "e1"):
            bad(d, f"rct tag {'present' if 'rct' in variants else 'absent'} but config ps_slope={c.get('ps_slope')}")
        # 4. metrics.json
        if os.path.exists(f"{d}/metrics.json"):
            mj = json.load(open(f"{d}/metrics.json"))
            if mj.get("run_id") != base:
                bad(d, f"metrics.json run_id = {mj.get('run_id')!r}")
        # 5. required files
        layout = next((k for k in LAYOUTS if os.path.exists(f"{d}/{k}")), None)
        if layout is None:
            bad(d, "no metrics.json, result.json or history.npz: unknown layout")
        else:
            for f in LAYOUTS[layout]:
                if not os.path.exists(f"{d}/{f}"):
                    bad(d, f"missing {f} (layout {layout})")
        # 6. wandb.json
        if os.path.exists(f"{d}/wandb.json"):
            wj = json.load(open(f"{d}/wandb.json"))
            if wj.get("name") != wn:
                bad(d, f"wandb.json name = {wj.get('name')!r}")
            if wj.get("id"):
                seen_wid[wj["id"]] += 1
                linked[wj["id"]] = (d, wn, base, nm["uid"], c)
    # 7. uniqueness
    for label, cnt in (("uid", seen_uid), ("wandb name", seen_wn), ("wandb id", seen_wid)):
        for k, v in cnt.items():
            if v > 1:
                problems.append(f"{label} {k} used by {v} folders")
    # 6b/8. wandb
    n_wandb = None
    if use_wandb:
        import wandb
        api = wandb.Api()
        mine = {r.id: r for r in api.runs("proj-lb/Frugal Images") if r.user.username == "laura-battaglia"}
        n_wandb = len(mine)
        for wid, (d, wn, base, uid, c) in linked.items():
            r = mine.get(wid)
            if r is None:
                bad(d, f"wandb id {wid} not found among the user's runs")
                continue
            if r.name != wn:
                bad(d, f"wandb run {wid} is named {r.name!r}")
            for key, want in (("run_id", base), ("wandb_name", wn), ("uid", uid)):
                if r.config.get(key) != want:
                    bad(d, f"wandb config.{key} = {r.config.get(key)!r}")
            for key in ("preset", "seed_fit", "size"):
                if key in c and r.config.get(key) != c[key]:
                    bad(d, f"wandb config.{key} = {r.config.get(key)!r}, local {c[key]!r}")
        for wid, r in mine.items():
            if wid not in linked:
                problems.append(f"wandb run {r.name} ({wid}) has no folder")
    # report
    print(f"{len(folders)} run folders checked" + (f", {len(linked)} linked to wandb, {n_wandb} wandb runs owned" if use_wandb else ""))
    for p in problems:
        print("  PROBLEM", p)
    print(f"{len(problems)} problems")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
