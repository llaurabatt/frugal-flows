"""Build the four summary tables for one run folder.

    python run_tables.py runs/<family>/<run folder> [--out tables.md]

Reads only what the folder holds (config.json, metrics.json / result.json, arrays.npz /
result.npz / history.npz, log.txt) and writes tables.md next to them. Three folder layouts
are handled:

  exp_ate_recovery.py runs   config.json + metrics.json + arrays.npz   (margin + copula)
  overnight.py runs          config.json + result.json  + result.npz   (margin-only variants
                             and the transformer frugal-flow runs)
  margin_only.py runs        config.json + history.npz  + log.txt      (one standalone fit)

A value the folder does not record is printed as "not recorded" -- never guessed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_morphomnist_exps import PRESETS  # ps_slope per preset

PRESET_TAG = {"exp1_rct_homogeneous": "E1", "exp2_confounded_homogeneous": "E2",
              "exp3_confounded_heterogeneous": "E3", "exp4_covariate_cate": "E4",
              "exp5_quantile_effect": "E5", "exp6_spatial_cate": "E6"}
NR = "not recorded"


# ----------------------------------------------------------------------------- helpers
def load_json(p):
    with open(p) as f:
        return json.load(f)


def regions(s: int, radius: int):
    """Disc / ring / far masks as flat boolean arrays, for an s x s image."""
    xx, yy = np.meshgrid(np.arange(s), np.arange(s), indexing="ij")
    c = (s - 1) / 2
    disc = ((xx - c) ** 2 + (yy - c) ** 2) <= radius ** 2
    ring = np.zeros_like(disc)
    for i in range(s):
        for j in range(s):
            if disc[i, j]:
                continue
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < s and 0 <= jj < s and disc[ii, jj]:
                    ring[i, j] = True
    far = ~disc & ~ring
    return disc.ravel(), ring.ravel(), far.ravel()


def fmt(x, signed=False, nd=4):
    if x is None:
        return NR
    if isinstance(x, str):
        return x
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    if not np.isfinite(x):
        return "non-finite"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def md_table(rows, header=None):
    out = []
    if header:
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "---|" * len(header))
    else:
        out.append("| " + " | ".join([""] * len(rows[0])) + " |")
        out.append("|" + "---|" * len(rows[0]))
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |")
    return "\n".join(out)


def param_count_from_log(log_path):
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        m = re.search(r"([\d,]+) trainable param", f.read())
    return int(m.group(1).replace(",", "")) if m else None


# ----------------------------------------------------------------------------- readers
def read_run(d: str) -> dict:
    """Return one flat dict of everything the four tables need, from either layout."""
    cj = load_json(f"{d}/config.json")
    c = cj.get("config", cj)
    R = {"run": os.path.basename(d.rstrip("/")), "cfg": c, "record": cj}

    if os.path.exists(f"{d}/result.json"):                   # ---- overnight.py layout
        rj = load_json(f"{d}/result.json")
        a = np.load(f"{d}/result.npz")
        met, info, fitcfg = rj["metrics"], rj["info"], rj["config"]
        R.update(layout="overnight", metrics=met, info=info)
        model = c.get("model")
        bs0 = c.get("base_shift")
        R["model"] = {"standalone": "margin only, effect set to zero" if bs0 == 0.0 else "margin only, no copula",
                      "separate": "one margin per treatment arm, no copula",
                      "fullff": "margin + copula"}.get(model, NR)
        R["has_copula"] = model == "fullff"
        R["sep"] = model == "separate"
        R.update(preset=c.get("preset"), size=c.get("size"), K=c.get("K"),
                 n=c.get("n_units"), n_train=c.get("n_train"), n_val=c.get("n_val"),
                 seed_data=c.get("seed_data"), seed_fit=c.get("seed_fit"),
                 n_mc=c.get("nmc_separate") if R["sep"] else c.get("nmc_final"),
                 digit=c.get("digit"), radius=None,
                 base_shift=bs0)
        R.update(tau=a["tau_hat"], ate=a["ATE"], e0=a["e0_map"], e1=a["e1_map"],
                 val=a["val_nll"], train=a["train_obj"])
        R.update(termination=info.get("termination"), epochs=info.get("epochs_run"),
                 best_epoch=info.get("best_epoch"), converged=info.get("converged"),
                 wall_s=info.get("wall_s"),
                 patience=c.get("patience"), epoch_cap=c.get("epoch_cap"), wall_cap_s=c.get("wall_cap_s"))
        if R["sep"]:
            R.update(arm_info={k: v for k, v in info.items() if k.startswith("arm")},
                     val1=a.get("arm1_val_nll"), train1=a.get("arm1_train_obj"),
                     wall_s=(info.get("arm0_wall_s") or 0) + (info.get("arm1_wall_s") or 0) or None)
        R.update(se={r: met.get(f"final_{r}_se_paired") for r in ("reference_disc", "reference_ring", "far_region")},
                 nonfinite=met.get("final_nonfinite_total"))
        m = c.get("margin") or fitcfg.get("margin") or {}
        R.update(arm="flexcont", conditioner=m.get("conditioner") or c.get("conditioner"),
                 nn_width=m.get("nn_width"), nn_depth=m.get("nn_depth"), flow_layers=m.get("flow_layers"),
                 knots=m.get("RQS_knots"), heads=m.get("nn_heads"), expansion=m.get("expansion", 2),
                 cop_width=None, cop_depth=None, cop_layers=None, cop_knots=None,
                 lr=fitcfg.get("lr"), batch_size=c.get("batch_size"),
                 precision="float32" if c.get("x64") is False else (NR if c.get("x64") is None else "float64"))
        R["precision_note"] = "" if c.get("x64") is not None else " (overnight.py sets jax_enable_x64=False; not in config)"
        R["batch_note"] = "" if c.get("batch_size") is not None else " (exp_ate_recovery.Config default; not in config)"
        R["batch_size"] = c.get("batch_size", 100)
        R["cop_note"] = " (library default; not in config)" if R["has_copula"] else ""
        if R["has_copula"]:
            R.update(cop_width=c.get("copula_nn_width", 50), cop_depth=c.get("copula_nn_depth", 1),
                     cop_layers=c.get("copula_flow_layers", 4), cop_knots=c.get("copula_rqs_knots", 8))
    elif os.path.exists(f"{d}/history.npz"):                 # ---- margin_only.py layout
        a = np.load(f"{d}/history.npz")
        log = open(f"{d}/log.txt").read() if os.path.exists(f"{d}/log.txt") else ""
        R.update(layout="margin_only", metrics={}, info={})
        R["model"] = "margin only, no copula"
        R["has_copula"] = False
        R["sep"] = False
        m = re.search(r"train (\d+) rows, validation (\d+) rows", log)
        n_tr, n_va = (int(m.group(1)), int(m.group(2))) if m else (None, None)
        R.update(preset=c["preset"], size=c["size"], K=int(c["size"]) ** 2,
                 n=(n_tr + n_va) if m else None, n_train=n_tr, n_val=n_va,
                 seed_data=c.get("seed_data"), seed_fit=c.get("seed_fit"), n_mc=c.get("n_mc"),
                 digit=c.get("digit"), radius=None, base_shift=c.get("base_shift"))
        R.update(tau=a["tau_hat"], ate=a["ATE"], e0=None, e1=None, val=a["val_loss"], train=a["train_loss"])
        # the final arm errors exist only in the log, for disc ("inside") and ring
        arm = {}
        for reg, key in (("disc", "inside"), ("ring", "ring")):
            mm = re.search(rf"{key}\s+e0 ([+-][\d.]+)\s+e1 ([+-][\d.]+)", log)
            arm[reg] = (float(mm.group(1)), float(mm.group(2))) if mm else None
        R["arm_regional"] = arm
        mm = re.search(r"epochs run (\d+)\s+best validation loss [-\d.]+ at epoch (\d+)\s+ended on the (\w+) rule\s+wall (\d+)s", log)
        ep, best, rule, wall = (int(mm.group(1)), int(mm.group(2)), mm.group(3), int(mm.group(4))) if mm else (len(a["epoch"]), int(np.argmin(a["val_loss"])) + 1, None, None)
        R.update(termination=rule, epochs=ep, best_epoch=best, converged=(rule == "patience"), wall_s=wall,
                 patience=c.get("max_patience"), epoch_cap=c.get("max_epochs"), wall_cap_s=None)
        mm = re.search(r"non-finite draws dropped: ([\d.]+)%", log)
        R.update(se={r: None for r in ("reference_disc", "reference_ring", "far_region")}, nonfinite=None,
                 dropped_frac=float(mm.group(1)) if mm else None)
        R.update(arm="flexcont", conditioner=c.get("conditioner"), nn_width=c.get("nn_width"), nn_depth=c.get("nn_depth"),
                 flow_layers=c.get("flow_layers"), knots=c.get("rqs_knots"), heads=c.get("nn_heads"), expansion=c.get("expansion", 2),
                 cop_width=None, cop_depth=None, cop_layers=None, cop_knots=None,
                 lr=c.get("learning_rate"), batch_size=c.get("batch_size"),
                 precision="float64" if c.get("x64") else "float32", precision_note="", batch_note="", cop_note="")
    else:                                                    # ---- exp_ate_recovery.py layout
        met = load_json(f"{d}/metrics.json")
        a = np.load(f"{d}/arrays.npz")
        R.update(layout="exp_ate_recovery", metrics=met, info={})
        # runs made before 2026-09-18 have no model field and are all margin + copula
        mdl = c.get("model", "ff")
        R["model"] = {"ff": "margin + copula",
                      "margin": "margin only, effect set to zero" if c.get("base_shift") == 0.0 else "margin only, no copula",
                      "margin_sep": "one margin per treatment arm, no copula"}[mdl]
        R["has_copula"] = mdl == "ff"
        R["sep"] = mdl == "margin_sep"
        if R["sep"]:
            cap = c.get("max_epochs")
            rule = lambda ep: None if ep is None else ("patience" if cap and ep < cap else "epoch cap")  # noqa: E731
            R.update(arm_info={"arm0_termination": rule(met.get("n_epochs_run")),
                               "arm1_termination": rule(met.get("n_epochs_run_arm1")),
                               "arm0_epochs_run": int(met["n_epochs_run"]), "arm1_epochs_run": met.get("n_epochs_run_arm1"),
                               "arm0_best_epoch": int(np.argmin(a["loss_val"])) + 1,
                               "arm1_best_epoch": int(np.argmin(a["loss_val_arm1"])) + 1 if "loss_val_arm1" in a.files else None},
                     val1=a["loss_val_arm1"] if "loss_val_arm1" in a.files else None,
                     train1=a["loss_train_arm1"] if "loss_train_arm1" in a.files else None)
        n = int(a["Y"].shape[0])
        nv = round(0.1 * n)
        R.update(preset=c["preset"], size=c["size"], K=int(c["size"]) ** 2, n=n, n_train=n - nv, n_val=nv,
                 seed_data=c.get("seed_data"), seed_fit=c.get("seed_fit"), n_mc=c.get("n_mc"),
                 digit=c.get("digit"), radius=cj.get("effective_radius"), base_shift=c.get("base_shift"))
        Y, X, ITE = a["Y"], a["X"][:, 0], a["ITE"]
        Y0 = Y - X[:, None] * ITE
        Y1 = Y0 + ITE
        if "mc_mean0" in a.files:          # flexcont: effect read out by sampling both arms
            e0, e1 = a["mc_mean0"] - Y0.mean(0), a["mc_mean1"] - Y1.mean(0)
        else:                              # loctrans: effect read from the ate parameter, no sampled arms
            e0 = e1 = None
        R.update(tau=a["tau_hat"], ate=a["ATE"], e0=e0, e1=e1, val=a["loss_val"], train=a["loss_train"])
        ep = int(met["n_epochs_run"])
        cap = c.get("max_epochs")
        R.update(termination="patience" if cap and ep < cap else "epoch cap", epochs=ep,
                 best_epoch=int(np.argmin(a["loss_val"])) + 1,
                 converged=bool(cap and ep < cap), wall_s=met.get("flow_fit_s"),
                 patience=c.get("max_patience"), epoch_cap=cap, wall_cap_s=None)
        R.update(se={r: None for r in ("reference_disc", "reference_ring", "far_region")},
                 nonfinite=None,
                 dropped=(met.get("mc_n") or 0) - (met.get("mc_n_used") or 0), anynan=met.get("mc_anynan"))
        cond = c.get("conditioner") if c["arm"] == "flexible_continuous" else None
        R.update(arm={"flexible_continuous": "flexcont", "location_translation": "loctrans"}[c["arm"]],
                 conditioner=cond, nn_width=c.get("nn_width"), nn_depth=c.get("nn_depth"),
                 flow_layers=c.get("flow_layers"), knots=c.get("rqs_knots"), heads=c.get("nn_heads"),
                 expansion=c.get("expansion", 2),
                 cop_width=c.get("copula_nn_width"), cop_depth=c.get("copula_nn_depth"),
                 cop_layers=c.get("copula_flow_layers"), cop_knots=c.get("copula_rqs_knots"),
                 lr=c.get("learning_rate"), batch_size=c.get("batch_size"),
                 precision="float64" if c.get("x64") else "float32",
                 precision_note="", batch_note="",
                 cop_note="" if c.get("copula_nn_width") is not None else " (not in config; the copula ran at the library default)")
        if c.get("copula_nn_width") is None:
            R.update(cop_width=50, cop_depth=1, cop_layers=4, cop_knots=8)
    R["n_params"] = param_count_from_log(f"{d}/log.txt")
    R["ps_slope"] = PRESETS[R["preset"]].ps_slope if R["preset"] in PRESETS else None
    if R["radius"] is None:
        R["radius"] = round(R["size"] / 4)
    return R


# ----------------------------------------------------------------------------- tables
def table1(R):
    disc, ring, far = regions(R["size"], R["radius"])
    ate = R["ate"]
    if np.all(ate == 0):
        effect = "0 on every pixel"
    elif np.allclose(ate[disc], ate[disc][0]) and np.all(ate[~disc] == 0):
        effect = f"{ate[disc][0]:g} on every disc pixel, 0 elsewhere"
    else:
        effect = f"heterogeneous: mean {ate[disc].mean():+.3f} on the disc, {ate[~disc].mean():+.3f} elsewhere"
    if R["ps_slope"] is None:
        treat = NR
    elif R["ps_slope"] == 0:
        treat = "binary, randomised"
    else:
        treat = f"binary, confounded by thickness (propensity slope {R['ps_slope']:g})"
    if R["digit"] is None and R["n"] == 5923:
        digit = "digit 0 (not in config; inferred from n = 5,923, the digit-0 count)"
    elif R["digit"] is None:
        digit = "all ten digits" if R["cfg"].get("digit", 0) is None and R["layout"] == "exp_ate_recovery" else NR
    else:
        digit = f"digit {R['digit']}"
    rows = [
        ("Run", R["run"]),
        ("Model", R["model"]),
        ("Preset", PRESET_TAG.get(R["preset"], R["preset"])),
        ("Dataset", f"MorphoMNIST, {digit}"),
        ("Number of images", fmt(R["n"])),
        ("Training sample size", fmt(R["n_train"])),
        ("Validation sample size", fmt(R["n_val"])),
        ("Data seed", fmt(R["seed_data"])),
        ("Fit seed", fmt(R["seed_fit"])),
        ("Image size", f"{R['size']} × {R['size']}"),
        ("Number of pixels", fmt(R["K"])),
        ("Treatment", treat),
        ("True effect", effect),
        ("Disc", f"{int(disc.sum())} pixels, radius {R['radius']} around the centre"),
        ("Ring", f"{int(ring.sum())} pixels adjacent to the disc"),
        ("Far region", f"{int(far.sum())} pixels"),
        ("Monte Carlo draws", fmt(R["n_mc"])),
    ]
    return "Table 1. Setup", None, rows


def table2(R):
    disc, ring, far = regions(R["size"], R["radius"])
    K = R["K"]
    err = R["tau"] - R["ate"]
    fin = np.isfinite(err)
    e = np.where(fin, err, np.nan)
    reg = (("disc", disc, "reference_disc"), ("ring", ring, "reference_ring"), ("far", far, "far_region"))
    n_ = {r: int(m.sum()) for r, m, _ in reg}
    rows = [
        ("MAE, all pixels", f"mean absolute difference between estimated and true effect over the {K} pixels",
         fmt(np.nanmean(np.abs(e)))),
        ("RMSE, all pixels", f"root mean squared difference between estimated and true effect over the {K} pixels",
         fmt(np.sqrt(np.nanmean(e ** 2)))),
    ]
    for r, m, _ in reg:
        rows.append((f"Signed error, {r}", f"mean of (estimated − true effect) over the {n_[r]} {r} pixels",
                     fmt(np.nanmean(e[m]), signed=True)))
    for r, m, _ in reg:
        rows.append((f"MAE, {r}", f"mean absolute difference between estimated and true effect over the {n_[r]} {r} pixels",
                     fmt(np.nanmean(np.abs(e[m])))))
    for r, m, _ in reg:
        if R["e0"] is not None:
            u = fmt(np.nanmean(np.where(np.isfinite(R["e0"]), R["e0"], np.nan)[m]), signed=True)
            t = fmt(np.nanmean(np.where(np.isfinite(R["e1"]), R["e1"], np.nan)[m]), signed=True)
        elif R.get("arm_regional", {}).get(r):
            u, t = (fmt(x, signed=True) for x in R["arm_regional"][r])
        else:
            u = t = "not recorded"
        rows.append((f"Untreated-arm error, {r}",
                     f"sampled mean under T = 0 minus the true untreated mean, averaged over the {n_[r]} {r} pixels", u))
        rows.append((f"Treated-arm error, {r}",
                     f"sampled mean under T = 1 minus the true treated mean, averaged over the {n_[r]} {r} pixels", t))
    for r, m, key in reg:
        se = R["se"][key]
        if se is None:
            val = "not recorded (per-draw values not saved)"
        elif not np.isfinite(se):
            val = "non-finite (region contains the non-finite pixel)"
        else:
            val = fmt(se)
        rows.append((f"Monte Carlo standard error, {r}",
                     f"standard error of the {r} signed error due to the finite number of draws", val))
    if R["layout"] == "overnight":
        rows.append(("Non-finite draws", "number of infinite or NaN values among the sampled outputs", fmt(R["nonfinite"])))
    elif R["layout"] == "margin_only":
        rows.append(("Dropped draws", "fraction of draws discarded for containing a non-finite value before averaging",
                     f"{R['dropped_frac']:g}%" if R.get("dropped_frac") is not None else NR))
    else:
        rows.append(("Dropped draws", "draws discarded for containing a non-finite value before averaging",
                     fmt(R["dropped"]) + (" (NaN seen)" if R.get("anynan") else "")))
    if (~fin).sum():
        rows.append(("Non-finite pixels in the effect map", "pixels of the estimated effect that are ±inf or NaN; "
                     "excluded from every average above", fmt(int((~fin).sum()))))
    return "Table 2. Performance", ("Metric", "Definition", "Result"), rows


def _curve_rows(v, t, label=""):
    b = int(np.argmin(v))
    L = len(v)
    rows = [
        (f"Validation loss at best epoch{label}",
         "negative log-likelihood of the validation images, summed over the pixel dimensions, in nats", fmt(v[b], nd=2)),
        (f"Training loss at best epoch{label}", "the negative log-likelihood of the training images at the best epoch", fmt(t[b], nd=2)),
        (f"Train–validation gap at best epoch{label}", "training loss minus validation loss; negative means the model fits training images better",
         fmt(t[b] - v[b], signed=True, nd=2)),
        (f"Validation loss at epoch 1{label}", "", fmt(v[0], nd=2)),
    ]
    for k in (50, 100):
        if L >= k:
            rows.append((f"Validation loss at epoch {k}{label}", "", fmt(v[k - 1], nd=2)))
    rows.append((f"Validation loss at last epoch{label}", "", fmt(v[-1], nd=2)))
    if b >= 30:
        rows.append((f"Improvement over the 30 epochs before the best{label}",
                     "best validation loss minus the best value seen in the 30 epochs before it",
                     fmt(v[b] - v[b - 30:b].min(), signed=True, nd=2)))
    w = min(30, L)
    rows.append((f"Validation loss spread, last {w} epochs{label}", f"standard deviation of the validation loss over the final {w} epochs",
                 fmt(v[-w:].std(), nd=2)))
    rows.append((f"Validation loss range, last {w} epochs{label}", f"lowest and highest value over the final {w} epochs",
                 f"{v[-w:].min():.2f} to {v[-w:].max():.2f}"))
    return rows


def table3(R):
    caps = f"patience ({R['patience']} epochs), epoch cap ({fmt(R['epoch_cap'])})"
    if R["wall_cap_s"]:
        caps += f", wall clock ({R['wall_cap_s'] / 3600:g} h)"
    rows = [
        ("Stopping rule", f"condition that ended training, out of: {caps}", fmt(R["termination"])),
        ("Converged", "training ended on the patience rule rather than a cap", fmt(R["converged"])),
        ("Epochs run", "number of epochs trained", fmt(R["epochs"])),
        ("Best epoch", "epoch with the lowest validation loss; its checkpoint is the one reported", fmt(R["best_epoch"])),
    ]
    if R["sep"]:
        ai = R["arm_info"]
        rows = [
            ("Stopping rule, untreated arm", f"condition that ended training, out of: {caps}", fmt(ai.get("arm0_termination"))),
            ("Stopping rule, treated arm", "", fmt(ai.get("arm1_termination"))),
            ("Converged", "both arms ended on the patience rule", fmt(R["converged"])),
            ("Epochs run, untreated arm", "number of epochs trained", fmt(ai.get("arm0_epochs_run"))),
            ("Epochs run, treated arm", "", fmt(ai.get("arm1_epochs_run"))),
            ("Best epoch, untreated arm", "epoch with the lowest validation loss; its checkpoint is the one reported", fmt(ai.get("arm0_best_epoch"))),
            ("Best epoch, treated arm", "", fmt(ai.get("arm1_best_epoch"))),
        ]
        rows += _curve_rows(R["val"], R["train"], ", untreated arm")
        rows += _curve_rows(R["val1"], R["train1"], ", treated arm")
    else:
        rows += _curve_rows(R["val"], R["train"])
    rows.append(("Training time", "wall-clock seconds", fmt(round(R["wall_s"])) if R["wall_s"] else NR))
    return "Table 3. Training and convergence", ("Metric", "Definition", "Result"), rows


def table4(R):
    trf = R["conditioner"] == "transformer"
    mlp = R["conditioner"] == "mlp"
    na = "NA"
    cond = {"mlp": "MLP", "transformer": "transformer", None: "NA (location_translation has no conditioner)"}[R["conditioner"]]
    rows = [
        ("Model", R["model"]),
        ("Margin model", R["arm"]),
        ("Margin conditioner", cond),
        ("Margin MLP hidden width", fmt(R["nn_width"]) if mlp else na),
        ("Margin MLP hidden depth", fmt(R["nn_depth"]) if mlp else na),
        ("Margin transformer channel width", fmt(R["nn_width"]) if trf else na),
        ("Margin transformer attention blocks", fmt(R["nn_depth"]) if trf else na),
        ("Margin transformer heads", fmt(R["heads"]) if trf else na),
        ("Margin transformer expansion", fmt(R["expansion"]) if trf else na),
        ("Margin flow layers", fmt(R["flow_layers"])),
        ("Margin spline knots", fmt(R["knots"])),
    ]
    if R["has_copula"]:
        note = R["cop_note"]
        rows += [
            ("Copula conditioner", "MLP"),
            ("Copula MLP hidden width", fmt(R["cop_width"]) + note),
            ("Copula MLP hidden depth", fmt(R["cop_depth"]) + note),
            ("Copula flow layers", fmt(R["cop_layers"]) + note),
            ("Copula spline knots", fmt(R["cop_knots"]) + note),
        ]
    else:
        rows += [("Copula conditioner", na), ("Copula MLP hidden width", na), ("Copula MLP hidden depth", na),
                 ("Copula flow layers", na), ("Copula spline knots", na)]
    obj = "negative log-likelihood of the training images"
    if R["has_copula"]:
        obj += ": margin term plus copula term"
    rows += [
        ("Trainable parameters", fmt(R["n_params"]) if R["n_params"] else NR + " (no parameter count in log.txt)"),
        ("Optimiser", "Adam"),
        ("Learning rate", fmt(R["lr"], nd=4) if R["lr"] else NR),
        ("Batch size", fmt(R["batch_size"]) + R["batch_note"]),
        ("Precision", R["precision"] + R["precision_note"]),
        ("Objective", obj),
        ("Checkpoint selection", "lowest validation loss"),
    ]
    return "Table 4. Architecture and optimisation", None, rows


WANDB_KEY = "tables/summary"
HTML_STYLE = """
<style>
body { font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 13px;
       color: #222; margin: 8px 12px; }
h2 { font-size: 15px; margin: 18px 0 6px; }
h1 { font-size: 15px; font-family: ui-monospace, Menlo, Consolas, monospace; margin: 4px 0 10px; }
table { border-collapse: collapse; width: 100%; margin-bottom: 6px; table-layout: auto; }
th, td { text-align: left; vertical-align: top; padding: 4px 8px; border-bottom: 1px solid #ddd;
         word-wrap: break-word; }
th { background: #f2f2f2; font-weight: 600; }
td.metric { font-weight: 600; white-space: nowrap; width: 1%; }
td.definition { color: #555; }
td.result { font-family: ui-monospace, Menlo, Consolas, monospace; white-space: nowrap;
            text-align: right; width: 1%; }
</style>
"""


def html_escape(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def html_tables(run_name: str, tables) -> str:
    """The four tables as one HTML document: full width, wrapping cells, numbers right-aligned."""
    parts = [HTML_STYLE, f"<h1>{html_escape(run_name)}</h1>"]
    for title, header, rows in tables:
        parts.append(f"<h2>{html_escape(title)}</h2><table>")
        if header:
            parts.append("<tr>" + "".join(f"<th>{html_escape(h)}</th>" for h in header) + "</tr>")
            for r in rows:
                parts.append(f"<tr><td class='metric'>{html_escape(r[0])}</td>"
                             f"<td class='definition'>{html_escape(r[1])}</td>"
                             f"<td class='result'>{html_escape(r[2])}</td></tr>")
        else:
            for r in rows:
                parts.append(f"<tr><td class='metric'>{html_escape(r[0])}</td>"
                             f"<td>{html_escape(r[1])}</td></tr>")
        parts.append("</table>")
    return "\n".join(parts)


def draw_ate_maps(d: str, R: dict) -> str:
    """Redraw plots/ate_maps.png for any layout through exp_ate_recovery.plot_ate_maps:
    estimated, true, signed error, and the two arm errors when the folder has them,
    with region averages under each error panel."""
    from exp_ate_recovery import plot_ate_maps
    os.makedirs(f"{d}/plots", exist_ok=True)
    path = f"{d}/plots/ate_maps.png"
    plot_ate_maps(R["size"], R["radius"], np.asarray(R["tau"]), np.asarray(R["ate"]), path,
                  title=R["run"], e0=R["e0"], e1=R["e1"])
    return path


def log_to_wandb(d: str, run_name: str, tables, image: str | None = None) -> str:
    """Attach the four tables to the folder's wandb run as one HTML panel (key tables/summary).

    wandb.Table panels squeeze every column to the same narrow width and cut long cells,
    which makes the Definition column unreadable; an HTML block renders at full width.
    Returns a one-line status. Skips folders whose wandb.json says there is no run.
    """
    wj = load_json(f"{d}/wandb.json")
    if not wj.get("id"):
        return "no wandb run for this folder; nothing logged"
    import wandb
    m = re.match(r"https://wandb\.ai/([^/]+)/([^/]+)/runs/", wj.get("url", ""))
    if not m:
        return f"wandb.json has no parsable url; nothing logged ({wj.get('url')!r})"
    entity, project = m.group(1), m.group(2).replace("%20", " ")
    html = html_tables(run_name, tables)
    run = wandb.init(entity=entity, project=project, id=wj["id"], resume="must")
    payload = {WANDB_KEY: wandb.Html(html, inject=False)}
    if image:
        payload["plots/ate_maps"] = wandb.Image(image)
    run.log(payload)
    run.finish(quiet=True)
    return (f"logged {', '.join(payload)} to wandb run {wj['name']} ({wj['id']}); "
            "also wrote tables.html")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None, help="output markdown path (default: <run_dir>/tables.md)")
    ap.add_argument("--wandb", action="store_true",
                    help="also log the four tables as wandb.Table panels to the run named in wandb.json")
    ap.add_argument("--quiet", action="store_true", help="do not print the tables")
    ap.add_argument("--plots", action="store_true",
                    help="redraw plots/ate_maps.png (estimated, true, signed error, arm errors, "
                         "with region averages) and, with --wandb, log it as plots/ate_maps")
    args = ap.parse_args()
    d = args.run_dir.rstrip("/")
    R = read_run(d)
    image = None
    if args.plots:
        image = draw_ate_maps(d, R)
        print(f"drew {image}", file=sys.stderr)
    tables = [table1(R), table2(R), table3(R), table4(R)]
    text = f"# {R['run']}\n\n" + "\n\n".join(
        f"**{title}**\n\n" + md_table(rows, header) for title, header, rows in tables) + "\n"
    out = args.out or f"{d}/tables.md"
    with open(out, "w") as f:
        f.write(text)
    with open(f"{d}/tables.html", "w") as f:       # the same four tables, readable in a browser
        f.write(html_tables(R["run"], tables))
    if not args.quiet:
        print(text)
    print(f"written to {out} and tables.html", file=sys.stderr)
    if args.wandb:
        print(log_to_wandb(d, R["run"], tables, image), file=sys.stderr)


if __name__ == "__main__":
    main()
