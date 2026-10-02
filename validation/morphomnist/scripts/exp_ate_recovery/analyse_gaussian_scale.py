"""Gaussian-scale overnight grid: paired comparison, decision rules, figures, W&B log (2026-10-02).

Decision rules: docs/gaussian_scale/OVERNIGHT_PREREG.md. Sources, joined on (preset, dataset k, fit seed):
  new       runs/gaussian_scale/<run>/  (metrics.json + arrays.npz)                       -> U-std, G-raw,
            G-std, G-LT, anchors (G-std-zshuf, G-LT-zshuf, G-std-copw16), U-raw-refit
  stored    Laura's folders on this branch: runs/exp_ate_recovery (U-raw, E2 datasets 1-3, 5 fit seeds) and
            runs/frengression (FR, E2 datasets 1-3, seed k). No arrays.npz there: tau_hat (and the do(0)
            variance) are recomputed once from the saved weights (load_model + the same read-out, n_mc /
            seed_mc of the run) and cached under analysis/cache/.
  wandb     Laura's W&B summaries (read-only), project proj-lb/Frugal Images, groups grid_8x8_alldigits
            (U-raw) and grid_8x8_alldigits_frengression (FR), state finished, E1/E2 datasets 1-6; duplicates
            -> latest finished. Used where no stored folder exists (scalar metrics only: no slope_imb).

Outputs in runs/gaussian_scale/analysis/: grid_gaussian_scale.md / .csv, gs_*.png, wandb_sources.csv; the
whole lot is logged to ONE W&B run (name analysis_gaussian_scale, group grid_8x8_alldigits_gaussian, fixed
id in analysis/wandb_run_id.txt, resume="allow"). Flags: --no-wandb, --cache-only (fill the cache, exit).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

MM = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, MM)
NEW_ROOT = os.path.join(MM, "runs", "gaussian_scale")
OUT = os.path.join(NEW_ROOT, "analysis")
CACHE = os.path.join(OUT, "cache")
PROJECT, ENTITY = "Frugal Images", "proj-lb"
GROUP = "grid_8x8_alldigits_gaussian"
SHORT = {"exp1_rct_homogeneous": "E1", "exp2_confounded_homogeneous": "E2"}
ARM_ORDER = ["FR", "U-raw", "U-std", "G-raw", "G-std", "G-LT"]
EXTRA = ["G-std-copw16", "G-std-zshuf", "G-LT-zshuf", "U-raw-refit"]
METRICS = ["ate_mae", "signed_disc", "mae_disc", "signed_ring", "mae_far", "slope_imb", "rho_retained",
           "gen_sd_mae_t0", "gen_sd_mae_t1", "best_epoch", "n_epochs_run", "gz_implied_ks_max",
           "gz_implied_ks_mean", "gy_ks_max", "s_per_epoch", "total_s", "design_naive_bias_mae",
           "shift_vs_crn_maxabs"]
VMAX = 0.05


# ---------------------------------------------------------------------------- sources
def code_of(c: dict) -> str:
    arm, ys = c.get("arm"), c.get("y_scaling", "none")
    zs = c.get("z_shuffle_seed") is not None
    if arm == "flexible_continuous":
        return ("U-std" if ys == "standardize" else "U-raw-refit")
    if arm == "flexible_continuous_gaussian":
        base = "G-std" if ys == "standardize" else "G-raw"
        if c.get("copula_nn_width") != 50:
            base += f"-copw{c.get('copula_nn_width')}"
        return base + ("-zshuf" if zs else "")
    if arm == "location_translation_gaussian":
        return "G-LT" + ("" if ys == "standardize" else "-raw") + ("-zshuf" if zs else "")
    return f"other:{arm}"


def new_rows():
    rows = []
    for d in sorted(glob.glob(os.path.join(NEW_ROOT, "*_d0-9_*/"))):
        if not (os.path.exists(d + "metrics.json") and os.path.exists(d + "config.json")):
            continue
        rec = json.load(open(d + "config.json"))
        c, m = rec["config"], json.load(open(d + "metrics.json"))
        if c.get("preset") not in SHORT or c.get("size") != 8:
            continue
        a = np.load(d + "arrays.npz") if os.path.exists(d + "arrays.npz") else None
        wb = json.load(open(d + "wandb.json")) if os.path.exists(d + "wandb.json") else {}
        rows.append({"source": "new", "code": code_of(c), "preset": SHORT[c["preset"]], "k": c["seed_assign"],
                     "seed": c["seed_fit"], "run": os.path.basename(d.rstrip("/")), "wandb_id": wb.get("id"),
                     "dataset_id": rec.get("dataset_id"), **{k: m.get(k, np.nan) for k in METRICS},
                     "_tau": None if a is None else np.asarray(a["tau_hat"], np.float64),
                     "_var0": None if a is None or "mc_var0" not in a.files else np.asarray(a["mc_var0"], np.float64),
                     "_dir": d})
    return rows


def _cache_path(d):
    return os.path.join(CACHE, os.path.basename(d.rstrip("/")) + ".npz")


def recompute_stored(d: str, kind: str):
    """tau_hat and the do(0) variance of a stored run without arrays.npz, from its saved weights."""
    p = _cache_path(d)
    if os.path.exists(p):
        z = np.load(p)
        return z["tau_hat"], z["var0"]
    os.makedirs(CACHE, exist_ok=True)
    if kind == "flow":
        import exp_ate_recovery as E
        rec = json.load(open(os.path.join(d, "config.json")))["config"]
        cfg = E.Config(**{k: v for k, v in rec.items() if k in {f.name for f in E.fields(E.Config)}})
        flow = E.load_model(d)
        data = E.build_data(cfg)
        tau, _, ex = E._tau_hat_flexible_continuous(cfg, flow, data, cfg.size ** 2,
                                                   ot=E.outcome_transform_for(cfg, data))
        var0 = ex["mc_var0"]
    else:
        import exp_frengression_recovery as F
        rec = json.load(open(os.path.join(d, "config.json")))["config"]
        cfg = F.Config(**{k: v for k, v in rec.items() if k in {f.name for f in F.fields(F.Config)}})
        model, inputs = F.load_model(d)
        y0, y1, _ = F.sample_margins(cfg, model, inputs)
        tau, var0 = (y1 - y0).mean(0), y0.var(0)
    np.savez(p, tau_hat=np.asarray(tau, np.float64), var0=np.asarray(var0, np.float64))
    return np.asarray(tau, np.float64), np.asarray(var0, np.float64)


def stored_rows(compute: bool = True):
    rows = []
    for root, code, kind, mfile in (("exp_ate_recovery", "U-raw", "flow", "model.eqx"),
                                    ("frengression", "FR", "freng", "model.pt")):
        for d in sorted(glob.glob(os.path.join(MM, "runs", root, "*_k64_s*_d0-9_*/"))):
            if not (os.path.exists(d + "metrics.json") and os.path.exists(d + mfile)):
                continue
            rec = json.load(open(d + "config.json"))
            c, m = rec["config"], json.load(open(d + "metrics.json"))
            if c.get("preset") not in SHORT or (root == "exp_ate_recovery" and c.get("arm") != "flexible_continuous"):
                continue
            wb = json.load(open(d + "wandb.json")) if os.path.exists(d + "wandb.json") else {}
            tau = var0 = None
            if compute or os.path.exists(_cache_path(d)):
                try:
                    tau, var0 = recompute_stored(d, kind)
                except Exception as exc:  # noqa: BLE001
                    print(f"recompute failed for {d}: {type(exc).__name__}: {exc}")
            row = {"source": "stored", "code": code, "preset": SHORT[c["preset"]], "k": c["seed_assign"],
                   "seed": c["seed_fit"], "run": os.path.basename(d.rstrip("/")), "wandb_id": wb.get("id"),
                   "dataset_id": rec.get("dataset_id"), **{k: m.get(k, np.nan) for k in METRICS},
                   "_tau": tau, "_var0": var0, "_dir": d}
            if tau is not None:
                row["ate_mae_recomputed"] = None      # filled in attach_truth
            rows.append(row)
    return rows


NAME_RE = {"U-raw": re.compile(r"^ff_(e[12])_flexcont_sa(\d+)_lr0\.001_copw16_k64_s(\d+)_d0-9_[0-9a-f]{6}$"),
           "FR": re.compile(r"^frengression_(e[12])_sa(\d+)_k64_s(\d+)_d0-9_[0-9a-f]{6}$")}


def wandb_rows():
    import wandb
    api = wandb.Api(timeout=60)
    rows, src = [], []
    for code, group in (("U-raw", "grid_8x8_alldigits"), ("FR", "grid_8x8_alldigits_frengression")):
        best = {}
        for r in api.runs(f"{ENTITY}/{PROJECT}", filters={"group": group, "state": "finished"}):
            mt = NAME_RE[code].match(r.name or "")
            if not mt:
                continue
            preset, k, seed = mt.group(1).upper(), int(mt.group(2)), int(mt.group(3))
            if k > 6:
                continue
            key = (preset, k, seed)
            src.append({"code": code, "preset": preset, "k": k, "seed": seed, "wandb_id": r.id, "name": r.name,
                        "created_at": r.created_at})
            if key not in best or r.created_at > best[key][0]:
                best[key] = (r.created_at, r)
        for (preset, k, seed), (_, r) in best.items():
            s = r.summary
            rows.append({"source": "wandb", "code": code, "preset": preset, "k": k, "seed": seed, "run": r.name,
                         "wandb_id": r.id, "dataset_id": s.get("dataset_id"),
                         **{m: (s.get(m) if isinstance(s.get(m), (int, float)) else np.nan) for m in METRICS},
                         "_tau": None, "_var0": None, "_dir": None})
    sd = pd.DataFrame(src)
    if len(sd):
        sd["duplicate"] = sd.duplicated(["code", "preset", "k", "seed"], keep=False)
    return rows, sd


# ---------------------------------------------------------------------------- truth per dataset
_TRUTH: dict = {}


def truth(preset: str, k: int):
    """ATE, naive difference, imbalance and the untreated rows' per-pixel sd of dataset (preset, k)."""
    key = (preset, k)
    if key not in _TRUTH:
        import exp_ate_recovery as E
        name = {"E1": "exp1_rct_homogeneous", "E2": "exp2_confounded_homogeneous"}[preset]
        data = E.build_data(E.Config(preset=name, size=8, digit=None, seed_data=101, seed_assign=k))
        Y = np.asarray(data["Y"], np.float64)
        T = np.asarray(data["X"])[:, 0].astype(bool)
        ate = np.asarray(data["ATE"], np.float64)
        naive = Y[T].mean(0) - Y[~T].mean(0)
        _TRUTH[key] = {"ate": ate, "naive": naive, "imb": naive - ate, "sd0": Y[~T].std(0),
                       "data_hash": data["data_hash"]}
    return _TRUTH[key]


def attach_truth(rows):
    import exp_ate_recovery as E
    disc, ring, far = E.region_masks(8, 2)
    for r in rows:
        if r["_tau"] is None:
            continue
        t = truth(r["preset"], r["k"])
        err = r["_tau"] - t["ate"]
        r["_err"] = err
        if r["source"] != "new":       # recomputed from weights: fill what the stored metrics lack
            r["ate_mae_recomputed"] = float(np.abs(err).mean())
            if not np.isfinite(r.get("slope_imb", np.nan)):
                r["slope_imb"] = float(np.polyfit(t["imb"], err, 1)[0])
            if not np.isfinite(r.get("rho_retained", np.nan)):
                r["rho_retained"] = float(np.dot(err, t["imb"]) / np.dot(t["imb"], t["imb"]))
        if r["_var0"] is not None:
            r["_ring"] = np.sqrt(r["_var0"]) / t["sd0"]


def merge(new, stored, wb):
    """One row per (code, preset, k, seed): new > stored > wandb (the W&B id is kept on stored rows)."""
    out, seen = [], {}
    for r in new + stored + wb:
        key = (r["code"], r["preset"], r["k"], r["seed"])
        if key in seen:
            if r["source"] == "wandb" and seen[key].get("wandb_id") is None:
                seen[key]["wandb_id"] = r["wandb_id"]
            continue
        seen[key] = r
        out.append(r)
    return out


# ---------------------------------------------------------------------------- statistics
def se(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def wilcoxon_p(d):
    from scipy.stats import wilcoxon
    d = np.asarray(d, float)
    if len(d) < 2 or np.all(d == 0):
        return np.nan
    try:
        return float(wilcoxon(d, alternative="two-sided").pvalue)
    except ValueError:
        return np.nan


def per_k(df, code, preset="E2", metric="ate_mae", seed_is_k=True):
    s = df[(df.code == code) & (df.preset == preset)]
    if seed_is_k:
        s = s[s.seed == s.k]
    return s.set_index("k")[metric].astype(float).dropna()


def paired(df, a, b, preset="E2", metric="ate_mae", ks=None):
    x, y = per_k(df, a, preset, metric), per_k(df, b, preset, metric)
    both = x.index.intersection(y.index)
    if ks is not None:
        both = both.intersection(pd.Index(ks))
    d = (x[both] - y[both]).sort_index()
    return {"arm": a, "vs": b, "preset": preset, "metric": metric, "datasets": list(map(int, d.index)),
            "n": len(d), "a_mean": float(x[both].mean()) if len(d) else np.nan,
            "b_mean": float(y[both].mean()) if len(d) else np.nan,
            "diff_mean": float(d.mean()) if len(d) else np.nan, "diff_se": se(d), "all_negative": bool(len(d) and (d < 0).all()),
            "wilcoxon_p": wilcoxon_p(d), "per_dataset": {int(k): float(v) for k, v in d.items()}}


def rule_beats(p, need=6):
    if p["n"] < need:
        return f"not resolved at this n (n={p['n']} of {need})"
    ok = p["all_negative"] and p["diff_mean"] < -2 * p["diff_se"]
    return "MET" if ok else "NOT MET"


def decisions(df):
    out = []
    p = paired(df, "G-std", "U-raw")
    out.append(("Beats the current flow (G-std - U-raw < 0 on every dataset 1-6, mean < -2 SE)", rule_beats(p), p))
    p = paired(df, "G-std", "FR")
    out.append(("Beats frengression (G-std - FR, same rule)", rule_beats(p), p))
    # not a standardisation artefact
    p1 = paired(df, "G-std", "U-std", ks=[1, 2, 3])
    p2 = paired(df, "G-raw", "U-raw", ks=[1, 2, 3])
    if p1["n"] < 3 or p2["n"] < 3:
        v = f"not resolved at this n (G-std/U-std n={p1['n']}, G-raw/U-raw n={p2['n']} of 3)"
    else:
        v = "MET" if (p1["all_negative"] and p2["all_negative"]) else "NOT MET"
    out.append(("Not a standardisation artefact (G-std < U-std AND G-raw < U-raw on datasets 1-3)", v, {"G-std vs U-std": p1, "G-raw vs U-raw": p2}))
    # genuinely adjusting
    r = per_k(df, "G-std-zshuf", metric="rho_retained")
    v = ("not resolved (no shuffled-Z G-std fit yet)" if len(r) == 0 else
         ("MET" if (r >= 0.8).all() else "NOT MET") + f" (rho_retained {', '.join(f'k{k}: {x:.3f}' for k, x in r.items())})")
    out.append(("Genuinely adjusting: G-std with shuffled Z returns ~naive (rho_retained >= 0.8)", v, r.to_dict()))
    r = per_k(df, "G-LT-zshuf", metric="rho_retained")
    if len(r) == 0:
        v = "not resolved (no shuffled-Z G-LT fit yet)"
    elif (r <= 0.1).all():
        v = "SHORTCUT (shuffled-Z G-LT still recovers the truth: excluded from any adjustment claim)"
    else:
        v = "no shortcut flag"
    v += f" (rho_retained {', '.join(f'k{k}: {x:.3f}' for k, x in r.items())})" if len(r) else ""
    out.append(("G-LT benchmark-shortcut check (shuffled Z, rho_retained <= 0.1)", v, r.to_dict()))
    return out


def laura_way(df, rows):
    """5-fit (or N-fit) mean map of each arm vs FR seed k, datasets 1-3 (E2)."""
    res = []
    for code in ("U-raw", "G-std"):
        for k in (1, 2, 3):
            taus = [r["_tau"] for r in rows if r["code"] == code and r["preset"] == "E2" and r["k"] == k
                    and r.get("_tau") is not None]
            if not taus:
                continue
            t = truth("E2", k)
            fr = df[(df.code == "FR") & (df.preset == "E2") & (df.k == k) & (df.seed == k)].ate_mae
            res.append({"arm": code, "k": k, "n_fits": len(taus),
                        "avg_map_ate_mae": float(np.abs(np.mean(taus, 0) - t["ate"]).mean()),
                        "FR_seed_k": float(fr.iloc[0]) if len(fr) else np.nan})
    return pd.DataFrame(res)


# ---------------------------------------------------------------------------- figures
def fig_headline(df, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    e2 = df[(df.preset == "E2") & (df.seed == df.k)]
    for ax, (metric, lab) in zip(axes, (("ate_mae", "ATE MAE"), ("abs_disc", "|signed disc bias|"),
                                        ("slope_imb", "leftover slope on imbalance"))):
        for i, code in enumerate(ARM_ORDER):
            s = e2[e2.code == code]
            v = (s.signed_disc.abs() if metric == "abs_disc" else s[metric]).astype(float).dropna()
            if not len(v):
                continue
            ks = s.loc[v.index, "k"].values
            ax.scatter(np.full(len(v), i) + (ks - 3.5) * 0.04, v, s=18, c=ks, cmap="viridis", vmin=1, vmax=6, zorder=3)
            ax.errorbar(i + 0.28, v.mean(), yerr=se(v) if len(v) > 1 else 0, fmt="s", color="k", ms=6, capsize=3)
            ax.text(i, ax.get_ylim()[1] if False else v.max(), f"n={len(v)}", ha="center", va="bottom", fontsize=7)
        if metric == "ate_mae":
            nb = e2.design_naive_bias_mae.astype(float).dropna()
            if len(nb):
                ax.axhline(nb.mean(), ls="--", color="grey", lw=1)
                ax.text(len(ARM_ORDER) - 0.5, nb.mean(), f"naive {nb.mean():.3f}", ha="right", va="bottom", fontsize=8)
            ax.set_yscale("log")
        if metric == "slope_imb":
            ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks(range(len(ARM_ORDER)), ARM_ORDER)
        ax.set_title(lab)
    fig.suptitle("E2, all digits 8x8, fit seed = dataset k (points coloured by dataset; black = mean +- SE)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_maps(rows, path, key="_err", title="mean tau_hat - ATE (E2, fit seed k)", vmin=-VMAX, vmax=VMAX,
             codes=None, cmap="RdBu_r"):
    import matplotlib.pyplot as plt
    codes = codes or (ARM_ORDER + EXTRA)
    maps = []
    for code in codes:
        m = [r[key] for r in rows if r["code"] == code and r["preset"] == "E2" and r["seed"] == r["k"]
             and r.get(key) is not None]
        if m:
            maps.append((code, np.mean(m, 0), len(m)))
    if not maps:
        return
    fig, axes = plt.subplots(1, len(maps), figsize=(2.3 * len(maps) + 1, 2.8), squeeze=False)
    for ax, (code, m, n) in zip(axes[0], maps):
        im = ax.imshow(m.reshape(8, 8), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{code} (n={n})\nMAE {np.abs(m - (1 if key == '_ring' else 0)).mean():.4f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8)
    fig.suptitle(title, fontsize=9)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_attribution(df, path):
    import matplotlib.pyplot as plt
    e2 = df[(df.preset == "E2") & (df.seed == df.k) & (df.k <= 3)]
    cells = {("U", "raw"): "U-raw", ("U", "std"): "U-std", ("G", "raw"): "G-raw", ("G", "std"): "G-std"}
    M = np.full((2, 2), np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, sc in enumerate(("U", "G")):
        for j, ys in enumerate(("raw", "std")):
            v = e2[e2.code == cells[(sc, ys)]].ate_mae.astype(float)
            M[i, j] = v.mean() if len(v) else np.nan
    im = axes[0].imshow(M, cmap="viridis")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, "n/a" if np.isnan(M[i, j]) else f"{M[i, j]:.4f}", ha="center", va="center", color="w")
    axes[0].set_xticks([0, 1], ["raw logit Y", "standardised Y"]); axes[0].set_yticks([0, 1], ["uniform scale", "Gaussian scale"])
    axes[0].set_title("mean ATE MAE, E2 datasets 1-3")
    fig.colorbar(im, ax=axes[0])
    for k in (1, 2, 3):
        ys = [e2[(e2.code == c) & (e2.k == k)].ate_mae.astype(float) for c in ("U-raw", "U-std", "G-raw", "G-std")]
        ys = [y.iloc[0] if len(y) else np.nan for y in ys]
        axes[1].plot(range(4), ys, "o-", label=f"dataset {k}")
    axes[1].set_xticks(range(4), ["U-raw", "U-std", "G-raw", "G-std"]); axes[1].legend(); axes[1].set_ylabel("ATE MAE")
    axes[1].set_title("paired by dataset (fit seed k)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def fig_anchors(df, path):
    import matplotlib.pyplot as plt
    e2 = df[(df.preset == "E2") & (df.seed == df.k)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ("rho_retained", "ate_mae")):
        labels = ["G-std", "G-std-zshuf", "G-LT", "G-LT-zshuf", "U-raw"]
        for i, c in enumerate(labels):
            v = e2[e2.code == c][metric].astype(float).dropna()
            if len(v):
                ax.scatter(np.full(len(v), i), v, c=e2[e2.code == c].loc[v.index, "k"], cmap="viridis", vmin=1, vmax=6)
                ax.errorbar(i + 0.2, v.mean(), yerr=se(v) if len(v) > 1 else 0, fmt="s", color="k", capsize=3)
        if metric == "rho_retained":
            ax.axhline(1, ls="--", color="r", lw=1); ax.text(0, 1.02, "naive", color="r", fontsize=8)
            ax.axhline(0, ls="--", color="g", lw=1); ax.text(0, 0.02, "truth", color="g", fontsize=8)
            ax.axhline(0.8, ls=":", color="grey"); ax.axhline(0.1, ls=":", color="grey")
        else:
            nb = e2.design_naive_bias_mae.astype(float).dropna()
            if len(nb):
                ax.axhline(nb.mean(), ls="--", color="r", lw=1)
            ax.set_yscale("log")
        ax.set_xticks(range(len(labels)), labels, rotation=20); ax.set_title(metric)
    fig.suptitle("Anchors (E2): full model vs placebo-shuffled covariates")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ---------------------------------------------------------------------------- report
def fmt(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{nd}f}" if isinstance(x, float) else str(x)


def md(df, floatfmt=".4f"):
    """A pandas frame as a Markdown table (no tabulate dependency)."""
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "n/a" if not np.isfinite(v) else format(float(v), floatfmt)
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "".join("| " + " | ".join(cell(v) for v in row) + " |\n" for row in df.itertuples(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--no-recompute", action="store_true", help="use cached stored-run maps only")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")

    stored = stored_rows(compute=not a.no_recompute)
    if a.cache_only:
        print(f"cache filled for {sum(r['_tau'] is not None for r in stored)} stored runs"); return
    new = new_rows()
    try:
        wb, wsrc = wandb_rows()
    except Exception as exc:  # noqa: BLE001
        print(f"W&B read failed ({type(exc).__name__}: {exc}); continuing without it")
        wb, wsrc = [], pd.DataFrame()
    rows = merge(new, stored, wb)
    attach_truth(rows)
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    df = df.sort_values(["preset", "code", "k", "seed"]).reset_index(drop=True)
    df.to_csv(os.path.join(OUT, "grid_gaussian_scale.csv"), index=False)
    if len(wsrc):
        wsrc.to_csv(os.path.join(OUT, "wandb_sources.csv"), index=False)

    # paired tables
    pairs = []
    for preset in ("E2", "E1"):
        for arm in ["U-std", "G-raw", "G-std", "G-LT", "G-std-copw16", "G-std-zshuf", "G-LT-zshuf", "U-raw-refit"]:
            for ref in ("U-raw", "FR"):
                for metric in ("ate_mae", "mae_disc", "slope_imb"):
                    p = paired(df, arm, ref, preset, metric)
                    if p["n"]:
                        pairs.append(p)
    pt = pd.DataFrame(pairs)
    dec = decisions(df)
    lw = laura_way(df, rows)

    figs = {}
    fig_headline(df, figs.setdefault("gs_headline", os.path.join(OUT, "gs_headline.png")))
    fig_maps(rows, figs.setdefault("gs_error_maps", os.path.join(OUT, "gs_error_maps.png")))
    fig_attribution(df, figs.setdefault("gs_attribution", os.path.join(OUT, "gs_attribution.png")))
    fig_anchors(df, figs.setdefault("gs_anchors", os.path.join(OUT, "gs_anchors.png")))
    fig_maps(rows, figs.setdefault("gs_ring", os.path.join(OUT, "gs_ring.png")), key="_ring",
             title="per-pixel model/data sd ratio under do(0): sqrt(model var of Y(0)) / sd of untreated images (1 = right)",
             vmin=0.5, vmax=1.5, codes=["U-raw", "U-std", "G-raw", "G-std", "FR"], cmap="PuOr_r")

    # summary table per (preset, code): mean (SE) over datasets, fit seed k
    summ = []
    for (preset, code), s in df[df.seed == df.k].groupby(["preset", "code"]):
        summ.append({"preset": preset, "arm": code, "datasets": int(s.k.nunique()),
                     **{m: f"{s[m].astype(float).mean():.4f} ({se(s[m].astype(float).dropna()):.4f})"
                        for m in ("ate_mae", "signed_disc", "mae_disc", "signed_ring", "mae_far", "slope_imb",
                                  "rho_retained", "gen_sd_mae_t0", "best_epoch", "gz_implied_ks_max")}})
    summ = pd.DataFrame(summ)
    counts = df.groupby(["source", "preset", "code"]).size().rename("fits").reset_index()
    stored_check = df[df.source == "stored"][["code", "preset", "k", "seed", "ate_mae", "ate_mae_recomputed"]] \
        if "ate_mae_recomputed" in df else pd.DataFrame()
    failed = []
    for d in sorted(glob.glob(os.path.join(NEW_ROOT, "*_d0-9_*/"))):
        if not os.path.exists(d + "metrics.json"):
            failed.append(os.path.basename(d.rstrip("/")))

    with open(os.path.join(OUT, "grid_gaussian_scale.md"), "w") as f:
        f.write("# Gaussian-scale frugal flow: overnight paper grid\n\n")
        f.write("Generated by `validation/morphomnist/scripts/exp_ate_recovery/analyse_gaussian_scale.py`. "
                "Pre-registration: `docs/gaussian_scale/OVERNIGHT_PREREG.md` (exploratory status). "
                "Paired = same dataset k, fit seed k vs fit seed k.\n\n")
        f.write("## Decision rules\n\n| Rule | Verdict |\n|---|---|\n")
        for name, verdict, _ in dec:
            f.write(f"| {name} | {verdict} |\n")
        f.write("\n### Rule details\n\n")
        for name, verdict, p in dec:
            f.write(f"- **{name}**: {verdict}\n")
            if isinstance(p, dict) and "diff_mean" in p:
                f.write(f"  - n={p['n']} datasets {p['datasets']}: mean diff {fmt(p['diff_mean'])} (SE {fmt(p['diff_se'])}), "
                        f"all negative {p['all_negative']}, Wilcoxon p {fmt(p['wilcoxon_p'], 3)}; "
                        f"per dataset {json.dumps({k: round(v, 4) for k, v in p['per_dataset'].items()})}\n")
            elif isinstance(p, dict):
                for kk, pp in p.items():
                    if isinstance(pp, dict) and "diff_mean" in pp:
                        f.write(f"  - {kk}: n={pp['n']}, mean diff {fmt(pp['diff_mean'])} (SE {fmt(pp['diff_se'])}), "
                                f"per dataset {json.dumps({k: round(v, 4) for k, v in pp['per_dataset'].items()})}\n")
        f.write("\n## Per arm (mean over datasets, SE in brackets; fit seed = dataset k)\n\n" + md(summ) + "\n")
        cols = ["preset", "metric", "arm", "vs", "n", "a_mean", "b_mean", "diff_mean", "diff_se", "all_negative", "wilcoxon_p"]
        if len(pt):
            f.write("\n## Paired differences (arm - reference)\n\n" + md(pt[cols], ".4f") + "\n")
        if len(lw):
            f.write("\n## Laura's way: N-fit mean map vs frengression seed k (E2, datasets 1-3)\n\n" + md(lw, ".4f") + "\n")
        f.write("\n## Fit counts by source\n\n" + md(counts) + "\n")
        if len(stored_check):
            f.write("\n## Stored folders: ate_mae in metrics.json vs recomputed from the saved weights\n\n" + md(stored_check, ".6f") + "\n")
        if len(wsrc):
            dups = wsrc[wsrc.duplicate]
            f.write(f"\n## W&B sources\n\n{len(wsrc)} finished runs read (`wandb_sources.csv`); "
                    f"{len(dups)} rows in duplicated cells (latest finished kept).\n")
        f.write("\n## Unfinished / failed new run folders\n\n" + ("\n".join(f"- {x}" for x in failed) if failed else "none") + "\n")
        f.write("\n## Figures\n\n" + "\n".join(f"- `{os.path.basename(p)}`" for p in figs.values() if os.path.exists(p)) + "\n")
    print(open(os.path.join(OUT, "grid_gaussian_scale.md")).read())

    if not a.no_wandb:
        try:
            import wandb
            idf = os.path.join(OUT, "wandb_run_id.txt")
            rid = open(idf).read().strip() if os.path.exists(idf) else __import__("uuid").uuid4().hex[:8]
            open(idf, "w").write(rid)
            run = wandb.init(entity=ENTITY, project=PROJECT, group=GROUP, name="analysis_gaussian_scale", id=rid,
                             resume="allow", job_type="analysis", tags=["gaussian-scale", "analysis"], reinit=True)
            def _typed(d):
                # keep numeric columns numeric so W&B can sort, filter and chart them;
                # only non-numeric columns become strings
                d = d.copy()
                for c in d.columns:
                    if not __import__("pandas").api.types.is_numeric_dtype(d[c]):
                        d[c] = d[c].astype(str)
                return d
            payload = {"table/per_run": wandb.Table(dataframe=_typed(df)),
                       "table/decisions": wandb.Table(columns=["rule", "verdict"], data=[[n, v] for n, v, _ in dec])}
            if len(pt):
                payload["table/paired"] = wandb.Table(dataframe=_typed(pt[cols]))
            for name, p in figs.items():
                if os.path.exists(p):
                    payload[f"plots/{name}"] = wandb.Image(p)
            run.log(payload)
            summ = {"n_new_fits": int((df.source == "new").sum())}
            if {"preset", "code", "ate_mae"} <= set(df.columns):
                for (pr, code), v in df.groupby(["preset", "code"])["ate_mae"].mean().items():
                    summ[f"mean_ate_mae/{pr}/{code}"] = float(v)
            run.summary.update(summ)
            run.finish()
            print(f"logged to W&B run {rid}")
        except Exception as exc:  # noqa: BLE001
            print(f"W&B log failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
