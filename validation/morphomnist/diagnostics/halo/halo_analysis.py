"""Halo ladder analysis (HALO_PREREG.md v1, "Gates" and "Analysis and wording").

    python halo_analysis.py --stage {S1,S2,S3,S4,all} --runs-root ~/work/halo-runs

Unit = seed_data (a replication on the fixed digit-0 corpus for Corpus A; a disjoint
all-digit draw for Corpus B): maps and per-cell scalars are averaged over seed_fit within
seed_data before any contrast. Three kinds of rows:
  primary      paired contrasts of a declared primary family, Holm across the family's
               (contrast x endpoint) tests, GATED: Holm p < 0.05 AND paired 95% bootstrap CI
               excludes 0 AND |mean| >= the prereg minimum effect (MIN_EFFECT);
  elevation    an arm's endpoint (already model minus reference data) vs 0, same gate,
               Holm across that family's arms x endpoints;
  exploratory  everything else: mean, CI, unadjusted p, never "resolved".
Attribution lines are produced only by resolved gates; a null prints "no resolved change at
this n (...)". Cells with > 0.1% non-finite draws or clamped base coordinates are listed and
excluded; cells from different XLA flag strings are never pooled (the script refuses).
Template coefficients are descriptive only. Writes <runs-root>/_analysis/<stage>_tables.{md,json},
<stage>_maps.png (FIXED colour scales, floor row on top) and <stage>_templates.png.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

EXCLUDE_FRAC = 1e-3
TEMPLATE_COEFS = ("b_mix", "b_sd", "b_imb", "b_thick", "b_ring")
SCALES = {"E_mu0": ("RdBu_r", -0.08, 0.08), "E_sd0": ("RdBu_r", -0.1, 0.1),
          "R_sd0": ("RdBu_r", -2, 2), "LEAK_X0": ("viridis", 0, 0.2), "E_tau": ("RdBu_r", -0.08, 0.08),
          "E_mu_diff": ("RdBu_r", -0.08, 0.08)}
# gate (iii): minimum paired effect per primary endpoint (HALO_PREREG v1 "Gates")
MIN_EFFECT = {"LEAK_X_exact": 0.02, "Rsd_exact_median": 0.5, "Etau_disc_mean": 0.03,
              "Etau_active_off_rms": 0.03, "slope_imb_offsupport": 0.05}
UNIT = {"A": "10 replications on the fixed digit-0 corpus", "B": "10 disjoint all-digit draws"}


# ------------------------------------------------------------------ loading
def load_stage(root: str, stage: str) -> list[dict]:
    recs = []
    for d in sorted(glob.glob(os.path.join(root, stage, f"{stage}_*"))):
        try:
            met = json.load(open(os.path.join(d, "metrics.json")))
            cfg = json.load(open(os.path.join(d, "config.json")))
        except Exception:
            continue
        if met.get("complete") is not True:
            continue
        z = np.load(os.path.join(d, "maps.npz"))
        recs.append({"cfg": cfg["config"], "xla": cfg["xla_flags"], "run_id": cfg["run_id"],
                     "met": met, "maps": {k: z[k] for k in z.files}})
    return recs


def label(c: dict) -> str:
    s = ("B:" if c.get("corpus", "A") == "B" else "") + f"{c['arm']}/{c['preproc']}"
    if c["task"] == "cond":
        s += f"/bs{c['base_shift']:g}/{c['preset']}"
    elif c["stage"] == "S3":
        s += "/uncond"
    return s


def check_flags(recs) -> None:
    flags = {r["xla"] for r in recs}
    if len(flags) > 1:
        sys.exit(f"REFUSING to pool cells from {len(flags)} XLA flag strings: {sorted(flags)}")


def excluded(r) -> bool:
    m = r["met"]
    return m["frac_nonfinite"] > EXCLUDE_FRAC or m["frac_clamped_coords"] > EXCLUDE_FRAC


# ------------------------------------------------------------------ endpoints
def _cm(v, mask, f="mean"):
    x = np.asarray(v)[np.asarray(mask, bool)]
    x = x[np.isfinite(x)]
    if not len(x):
        return float("nan")
    if f == "rms":
        return float(np.sqrt(np.mean(x ** 2)))
    return float({"mean": np.mean, "median": np.median}[f](x))


def endpoints(maps: dict, cells: list[dict]) -> dict:
    """Endpoints on one seed_data's seed_fit-mean maps. Primary ones are named in
    MIN_EFFECT; the rest are exploratory. Class-cross-table means go under 'xt:'."""
    c = lambda k: maps[f"cls_{k}"].astype(bool)
    e = {"LEAK_X_exact": _cm(maps["LEAK_X0"], c("exact_floor")),
         "Rsd_exact_median": _cm(maps["R_sd0"], c("exact_floor"), "median"),
         "LEAK_X_pure": _cm(maps["LEAK_X0"], c("pure_floor")),
         "Rsd_quiet_median": _cm(maps["R_sd0"], c("quiet"), "median"),
         "KS_exact_mean": _cm(maps["KS0"], c("exact_floor")),
         "Esd_mixture": _cm(maps["E_sd0"], c("mixture")),
         "FLOORMASS_mixture": _cm(maps["FLOORMASS0"], c("mixture")),
         "NBCORR_active": _cm(maps["NBCORR0"], c("active")),
         "Emu0_active_off_rms": _cm(maps["E_mu0"], c("active_off"), "rms")}
    if "E_tau" in maps:
        e.update({"Etau_disc_mean": _cm(maps["E_tau"], c("disc")),
                  "Etau_active_off_rms": _cm(maps["E_tau"], c("active_off"), "rms"),
                  "Etau_active_off_mean": _cm(maps["E_tau"], c("active_off")),
                  "tauhat_disc_mean": _cm(maps["tau_hat"], c("disc")),
                  "slope_imb_offsupport": float(np.mean([r["met"]["slope_imb_offsupport"] for r in cells]))})
        for cl in ("disc", "active_off", "quiet"):
            e[f"Emu_diff_{cl}"] = _cm(maps["E_mu_diff"], c(cl))
    for f in ("exact_floor", "pure_floor", "mixture", "ink"):
        for r in ("reg_disc", "reg_ring", "reg_far"):
            m = c(f) & c(r)
            if m.any():
                for k in ("E_mu0", "R_sd0", "LEAK_X0", "E_tau"):
                    if k in maps:
                        e[f"xt:{k}:{f}&{r}"] = _cm(maps[k], m)
    return e


def by_config(recs):
    """label -> {"cfg", "seed": {seed_data: {"maps", "cells", "ep"}}}; seed_fit averaged."""
    g = defaultdict(lambda: {"cfg": None, "seed": defaultdict(list)})
    for r in recs:
        if excluded(r):
            continue
        L = label(r["cfg"])
        g[L]["cfg"] = r["cfg"]
        g[L]["seed"][r["cfg"]["seed_data"]].append(r)
    out = {}
    for L, v in g.items():
        seeds = {}
        for sd, cells in sorted(v["seed"].items()):
            keys = [k for k in cells[0]["maps"] if all(k in c["maps"] for c in cells)]
            mm = {k: np.mean([c["maps"][k] for c in cells], axis=0) for k in keys}
            seeds[sd] = {"maps": mm, "cells": cells, "ep": endpoints(mm, cells)}
        out[L] = {"cfg": v["cfg"], "seed": seeds}
    return out


# ------------------------------------------------------------------ statistics
def boot_ci(x: np.ndarray, B: int = 10000, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def _test(d) -> dict:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return {"n": len(d), "mean": float("nan"), "ci": [float("nan")] * 2, "p": float("nan")}
    p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return {"n": len(d), "mean": float(d.mean()), "ci": list(boot_ci(d)), "p": p}


def _vals(groups, L, ep) -> dict:
    return {s: v["ep"].get(ep, np.nan) for s, v in groups[L]["seed"].items()}


def paired(groups, a: str, b: str, ep: str) -> dict:
    """Paired contrast a - b over the seed_data both configs have."""
    va, vb = _vals(groups, a, ep), _vals(groups, b, ep)
    return _test([va[k] - vb[k] for k in sorted(set(va) & set(vb))])


def holm(ps: list[float]) -> list[float]:
    idx = [i for i, p in enumerate(ps) if np.isfinite(p)]
    order = sorted(idx, key=lambda i: ps[i])
    out, run = [float("nan")] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(order) - r) * ps[i]))
        out[i] = run
    return out


def _corpus(L: str) -> str:
    return "B" if L.startswith("B:") else "A"


def gate(r: dict) -> dict:
    """Gate = (i) Holm p < 0.05 AND (ii) CI excludes 0 AND (iii) |mean| >= minimum effect."""
    lo, hi = r["ci"]
    ok_i = np.isfinite(r.get("p_holm", np.nan)) and r["p_holm"] < 0.05
    ok_ii = np.isfinite(lo) and (lo > 0 or hi < 0)
    ok_iii = np.isfinite(r["mean"]) and abs(r["mean"]) >= MIN_EFFECT.get(r["endpoint"], 0.0)
    r["resolved"] = bool(ok_i and ok_ii and ok_iii)
    unit = UNIT[_corpus(r["arm"])]
    if r["resolved"]:
        r["text"] = (f"RESOLVED: mean {r['mean']:+.4f}, paired 95% CI [{lo:+.4f}, {hi:+.4f}], "
                     f"Holm p {r['p_holm']:.4f} ({unit})")
    elif np.isfinite(lo) and lo <= 0 <= hi:
        r["text"] = f"no resolved change at this n ({unit}): paired 95% CI [{lo:+.4f}, {hi:+.4f}] includes 0"
    else:
        why = [n for n, ok in (("Holm p>=0.05", ok_i), ("CI includes 0", ok_ii), ("below minimum effect", ok_iii)) if not ok]
        r["text"] = (f"not resolved ({', '.join(why)}): mean {r['mean']:+.4f}, CI [{lo:+.4f}, {hi:+.4f}] ({unit})")
    return r


def primary_family(groups, name: str, contrasts: list[tuple[str, str]], eps: tuple) -> list[dict]:
    """Paired contrasts (a - b) x endpoints, Holm across the whole family, gated."""
    rows = [{"family": name, "kind": "primary", "endpoint": ep, "arm": a, "ref": b, **paired(groups, a, b, ep)}
            for a, b in contrasts if a in groups and b in groups for ep in eps]
    for r, ph in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = ph
        gate(r)
    return rows


def elevation_family(groups, name: str, arms: list[str], eps: tuple) -> list[dict]:
    """'Elevated' = the arm-vs-reference-data endpoint (already model minus data) differs
    from 0 under the same conjunction; Holm across arms x endpoints."""
    rows = [{"family": name, "kind": "elevation", "endpoint": ep, "arm": L, "ref": "reference data",
             **_test(list(_vals(groups, L, ep).values()))} for L in arms if L in groups for ep in eps]
    for r, ph in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = ph
        gate(r)
    return rows


def exploratory(groups, ref: str, others: list[str], eps: tuple, name: str) -> list[dict]:
    rows = []
    for L in others:
        if L == ref or L not in groups or ref not in groups:
            continue
        for ep in eps:
            r = {"family": name, "kind": "exploratory", "endpoint": ep, "arm": L, "ref": ref,
                 **paired(groups, L, ref, ep)}
            lo, hi = r["ci"]
            r["p_holm"] = float("nan")
            r["resolved"] = False
            r["text"] = f"exploratory: mean {r['mean']:+.4f}, CI [{lo:+.4f}, {hi:+.4f}], unadjusted p {r['p']:.4f}"
            rows.append(r)
    return rows


def _find(rows, kind, arm, ep, ref=None):
    for r in rows:
        if r["kind"] == kind and r["arm"] == arm and r["endpoint"] == ep and (ref is None or r["ref"] == ref):
            return r
    return None


def s1_attribution(rows, pfx: str = "") -> list[str]:
    """HALO_PREREG v1 S1 attribution rules, per primary endpoint. Positive findings only:
    a null never yields an attribution line."""
    out = []
    ref, p1, a2 = f"{pfx}A1s/P0", f"{pfx}A1s/P1", f"{pfx}A2/P1"
    tag = pfx or "A:"
    for ep in ("LEAK_X_exact", "Rsd_exact_median"):
        el = lambda a: (_find(rows, "elevation", a, ep) or {}).get("resolved", False)
        con = lambda a: (_find(rows, "primary", ref, ep, a) or {}).get("resolved", False)
        if not el(ref):
            out.append(f"{tag}{ep}: no resolved Sense-2 artefact in the unconditional margin at this n "
                       f"({UNIT['B' if pfx else 'A']})")
            continue
        fired = []
        if con(p1) and not el(p1):
            fired.append("H_sliver (preprocessing) on the current stack: P1 removes the elevation")
        if not el(a2) and con(a2):
            fired.append("the Uniform+atanh construction: A2/P1 not elevated and A1s/P0 vs A2/P1 resolves")
        if el(a2):
            fired.append("not construction-specific (A2/P1 also elevated); S3 decides library-generality")
        out += [f"{tag}{ep}: {f}" for f in fired] or [f"{tag}{ep}: A1s/P0 elevated; no attribution rule fires"]
    return out


def variance_split(cfg_group: dict, ep_fn) -> dict:
    """Two-way crossed (seed_data x seed_fit, one cell each) sums-of-squares split of a
    per-cell scalar; returns the fractions of total SS."""
    rows = [(sd, c["cfg"]["seed_fit"], ep_fn(c)) for sd, v in cfg_group["seed"].items() for c in v["cells"]]
    sds, sfs = sorted({r[0] for r in rows}), sorted({r[1] for r in rows})
    if len(rows) != len(sds) * len(sfs) or len(sds) < 2 or len(sfs) < 2:
        return {"note": "incomplete grid"}
    Y = np.full((len(sds), len(sfs)), np.nan)
    for sd, sf, v in rows:
        Y[sds.index(sd), sfs.index(sf)] = v
    g = Y.mean()
    ss_d = len(sfs) * ((Y.mean(1) - g) ** 2).sum()
    ss_f = len(sds) * ((Y.mean(0) - g) ** 2).sum()
    ss_t = ((Y - g) ** 2).sum()
    return {"seed_data": ss_d / ss_t, "seed_fit": ss_f / ss_t, "residual": (ss_t - ss_d - ss_f) / ss_t}


def template_table(groups: dict, mapname: str) -> dict:
    """DESCRIPTIVE only (prereg v1): mean [95% CI] over seed_data of the OLS coefficients."""
    out = {}
    for L, v in groups.items():
        row = {}
        for b in TEMPLATE_COEFS + ("r2",):
            x = [np.nanmean([c["met"]["templates"].get(mapname, {}).get(b, np.nan) for c in s["cells"]])
                 for s in v["seed"].values()]
            x = np.array(x, float)
            x = x[np.isfinite(x)]
            row[b] = {"mean": float(x.mean()) if len(x) else float("nan"), "ci": list(boot_ci(x))}
        out[L] = row
    return out


# ------------------------------------------------------------------ figures
def _s0_floor(root: str) -> dict:
    fs = sorted(glob.glob(os.path.join(root, "S0", "s0_sd*.npz"))) or \
        sorted(glob.glob(os.path.expanduser("~/work/halo-runs/S0/s0_sd*.npz")))
    if not fs:
        return {}
    zs = [np.load(f) for f in fs]
    return {k: np.mean([z[k] for z in zs], 0) for k in ("floor_se_mean", "floor_se_sd", "floor_se_naive")}


def panel(groups: dict, root: str, path: str, title: str) -> None:
    """Seed-mean 8x8 maps per config (rows) with a floor row on top, FIXED colour scales."""
    cols = [k for k in SCALES if any(k in s["maps"] for v in groups.values() for s in v["seed"].values())]
    labels = list(groups)
    fl = _s0_floor(root)
    a0 = next(iter(next(iter(groups.values()))["seed"].values()))["maps"]
    ref_sd = a0["gen_sd0"] - a0["E_sd0"]
    floor = {"E_mu0": (fl.get("floor_se_mean"), "floor: bootstrap SE mean (S0)"),
             "E_sd0": (fl.get("floor_se_sd"), "floor: bootstrap SE sd (S0)"),
             "R_sd0": (np.log2(1 + a0["floor_E_sd0"] / ref_sd), "floor: log2(1+split|dsd|/sd)"),
             "LEAK_X0": (a0["LEAK_DATA0"], "data: P(Y outside sliver)"),
             "E_tau": (fl.get("floor_se_naive"), "floor: bootstrap SE naive diff (S0)"),
             "E_mu_diff": (fl.get("floor_se_naive"), "floor: bootstrap SE naive diff (S0)")}
    fig, axes = plt.subplots(len(labels) + 1, len(cols), figsize=(2.3 * len(cols), 2.0 * (len(labels) + 1)),
                             squeeze=False, layout="constrained")
    for j, k in enumerate(cols):
        cmap, lo, hi = SCALES[k]
        im = None
        for i in range(len(labels) + 1):
            ax = axes[i, j]
            ax.set_xticks([]), ax.set_yticks([])
            if i == 0:
                img, t = floor[k]
                ax.set_title(f"{k}\n{t}", fontsize=6)
            else:
                ss = list(groups[labels[i - 1]]["seed"].values())
                img = np.mean([s["maps"][k] for s in ss], 0) if all(k in s["maps"] for s in ss) else None
                if j == 0:
                    ax.set_ylabel(f"{labels[i-1]}\n(n_sd={len(ss)})", fontsize=6)
            if img is None or not np.isfinite(np.asarray(img, float)).any():
                ax.imshow(np.full((8, 8), np.nan), vmin=0, vmax=1)
                ax.set_facecolor("0.9")
                continue
            im = ax.imshow(np.asarray(img, float).reshape(8, 8), cmap=cmap, vmin=lo, vmax=hi)
        if im is not None:
            fig.colorbar(im, ax=axes[-1, j], location="bottom", shrink=0.8)
    fig.suptitle(f"{title}: seed-mean maps, fixed scales (E_mu/E_tau/E_mu_diff ±0.08, E_sd ±0.1, "
                 "R_sd log2 ±2, LEAK_X 0–0.2)", fontsize=8)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def template_bars(tt: dict, mapname: str, path: str, title: str) -> None:
    labels = list(tt)
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(labels) * 1.2), 3.5))
    w = 0.8 / len(TEMPLATE_COEFS)
    for j, b in enumerate(TEMPLATE_COEFS):
        m = np.array([tt[L][b]["mean"] for L in labels])
        ci = np.array([tt[L][b]["ci"] for L in labels])
        err = np.abs(np.vstack([m - ci[:, 0], ci[:, 1] - m]))
        ax.bar(np.arange(len(labels)) + j * w, m, w, yerr=np.nan_to_num(err), label=b, capsize=2)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(np.arange(len(labels)) + 0.4, labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(f"coefficient on standardised template ({mapname})")
    ax.legend(fontsize=7, ncol=5)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ stages
S1_EP = ("LEAK_X_exact", "Rsd_exact_median")
S1_EXPL = ("LEAK_X_pure", "Rsd_quiet_median", "KS_exact_mean", "Esd_mixture", "FLOORMASS_mixture",
           "NBCORR_active", "Emu0_active_off_rms")
S2_EP = ("Etau_disc_mean", "Etau_active_off_rms")
S2_EXPL = ("Emu_diff_disc", "Emu_diff_active_off", "Emu_diff_quiet", "slope_imb_offsupport",
           "Emu0_active_off_rms", "Rsd_quiet_median", "LEAK_X_exact")


def analyse(stage: str, root: str) -> dict:
    recs = load_stage(root, stage)
    if not recs:
        print(f"{stage}: no complete cells under {root}")
        return {}
    extra = load_stage(root, "S1") + load_stage(root, "S2") if stage == "S3" else []
    check_flags(recs + extra)
    out = {"stage": stage, "n_cells": len(recs),
           "excluded": [r["run_id"] for r in recs if excluded(r)],
           "quiet_flagged": [r["run_id"] for r in recs if r["met"].get("quiet_flag")]}
    groups = by_config(recs)
    allg = {**by_config(extra), **groups}
    out["endpoints"] = {L: {s: {k: v for k, v in g["ep"].items() if not k.startswith("xt:")}
                            for s, g in G["seed"].items()} for L, G in groups.items()}
    out["cross_table"] = {L: {k: float(np.nanmean([g["ep"].get(k, np.nan) for g in G["seed"].values()]))
                              for k in next(iter(G["seed"].values()))["ep"] if k.startswith("xt:")}
                          for L, G in groups.items()}
    rows, attr, tmap = [], [], "E_sd0"
    if stage == "S1":
        for pfx in ("", "B:"):
            ref = f"{pfx}A1s/P0"
            prim = [(ref, f"{pfx}A1s/P1"), (ref, f"{pfx}A2/P1")] + ([(ref, "A1/P0")] if not pfx else [])
            fam = f"S1 primary{' (Corpus B)' if pfx else ''}"
            rows += primary_family(groups, fam, prim, S1_EP)
            rows += elevation_family(groups, f"S1 elevation{' (Corpus B)' if pfx else ''}",
                                     [ref, f"{pfx}A1s/P1", f"{pfx}A2/P1"] + (["A1/P0"] if not pfx else []), S1_EP)
            if ref in groups:
                attr += s1_attribution(rows, pfx)
        expl = [L for L in groups if not L.startswith("B:") and not L.endswith("/P4")]
        rows += exploratory(groups, "A1s/P0", expl, S1_EP + S1_EXPL, "S1 exploratory vs A1s/P0")
        rows += exploratory(groups, "Csmooth/P0", ["Czinf/P0"], ("Rsd_exact_median", "KS_exact_mean"),
                            "S1 exploratory C-zinf vs C-smooth")
    elif stage == "S2":
        tmap = "E_tau"
        for pfx in ("", "B:"):
            ff0, ff1, a2, lt = (f"{pfx}ff_cond/P0/bs1/E1", f"{pfx}ff_cond/P1/bs1/E1",
                                f"{pfx}a2_cond/P1/bs1/E1", f"{pfx}lt/P0/bs1/E1")
            rows += primary_family(groups, f"S2 tau=1 primary{' (Corpus B)' if pfx else ''}",
                                   [(ff0, ff1), (ff1, a2), (ff1, lt)], S2_EP)
            rows += elevation_family(groups, f"S2 tau=1 elevation{' (Corpus B)' if pfx else ''}",
                                     [ff0, ff1, a2, lt, f"{pfx}sep/P0/bs1/E1"], S2_EP)
        for bs in ("1", "0"):
            fam = [L for L in groups if f"/bs{bs}/" in L and not L.startswith("B:")]
            rows += exploratory(groups, f"ff_cond/P1/bs{bs}/E1", fam, S2_EP + S2_EXPL,
                                f"S2 tau={bs} exploratory vs ff_cond/P1")
        out["variance_split"] = {L: {c: variance_split(g, lambda r, c=c: _cm(r["maps"]["E_tau"], r["maps"][f"cls_{c}"]))
                                     for c in ("disc", "active_off")} for L, g in groups.items()}
        out["Emu0_corr_across_tau"] = {}
        for L in groups:
            L0 = L.replace("/bs1/", "/bs0/")
            if "/bs1/" in L and L0 in groups:
                a = np.mean([s["maps"]["E_mu0"] for s in groups[L]["seed"].values()], 0)
                b = np.mean([s["maps"]["E_mu0"] for s in groups[L0]["seed"].values()], 0)
                out["Emu0_corr_across_tau"][L] = float(np.corrcoef(a, b)[0, 1])
        for L, g in groups.items():
            if "/bs1/" in L and any(x in L for x in ("ff_cond/P1", "a2_cond/P1")):
                td = np.nanmean([v["ep"]["tauhat_disc_mean"] for v in g["seed"].values()])
                rm = np.nanmean([v["ep"]["Etau_active_off_rms"] for v in g["seed"].values()])
                ok = abs(td - 1) <= 0.05 and rm <= 0.03
                attr.append(f"{L}: disc tau_hat {td:.3f}, active_off RMS {rm:.3f} -> "
                            + ("identifies the effect (prereg statement)" if ok else
                               "identification statement NOT met"))
    elif stage == "S3":
        for L, g in groups.items():
            c = g["cfg"]
            if c["task"] == "uncond":
                rows += exploratory(allg, "A1s/P0", [L], S1_EP + S1_EXPL, "S3 exploratory vs A1s/P0")
            else:
                rows += exploratory(allg, f"ff_cond/P1/bs{c['base_shift']:g}/E1", [L], S2_EP + S2_EXPL,
                                    "S3 exploratory vs ff_cond/P1")
        rows += elevation_family(groups, "S3 elevation (exploratory)",
                                 [L for L in groups if groups[L]["cfg"]["task"] == "uncond"], S1_EP)
        rows += elevation_family(groups, "S3 elevation tau=1 (exploratory)",
                                 [L for L in groups if "/bs1/" in L], S2_EP)
    elif stage == "S4":
        tmap = "E_tau"
        rows += primary_family(groups, "S4 primary", [("ff_full/P0/bs1/E2", "ff_full/P1/bs1/E2")],
                               ("slope_imb_offsupport",))
        for a, b in (("ff_full/P0/bs1/E1", "ff_full/P1/bs1/E1"), ("ff_cond/P0/bs1/E2", "ff_cond/P1/bs1/E2"),
                     ("ff_full/P0/bs1/E2", "ff_cond/P0/bs1/E2"), ("ff_full/P1/bs1/E2", "ff_cond/P1/bs1/E2")):
            rows += exploratory(groups, b, [a], ("slope_imb_offsupport",) + S2_EP + ("Etau_active_off_mean",),
                                f"S4 exploratory {a} vs {b}")
        r = _find(rows, "primary", "ff_full/P0/bs1/E2", "slope_imb_offsupport")
        if r is not None:
            attr.append(("Sense-1 ring has a margin component of size "
                         f"{r['mean']:+.3f} (slope P0 - P1)") if r["resolved"] else
                        f"no resolved margin component of the E2 ring at this n ({UNIT['A']}); attribution stays open")
    out["contrasts"], out["attribution"] = rows, attr
    out["templates_descriptive"] = {m: template_table(groups, m) for m in ("E_mu0", "E_sd0", "E_tau")}
    out["template_corr_mean"] = {L: np.mean([c["met"]["template_corr"] for s in G["seed"].values()
                                            for c in s["cells"]], 0).round(3).tolist() for L, G in groups.items()}
    adir = os.path.join(root, "_analysis")
    os.makedirs(adir, exist_ok=True)
    panel(groups, root, os.path.join(adir, f"{stage}_maps.png"), stage)
    template_bars(out["templates_descriptive"][tmap], tmap, os.path.join(adir, f"{stage}_templates.png"),
                  f"{stage}: DESCRIPTIVE template coefficients of {tmap} (mean, 95% bootstrap CI over seed_data)")
    json.dump(out, open(os.path.join(adir, f"{stage}_tables.json"), "w"), indent=1, default=float)
    open(os.path.join(adir, f"{stage}_tables.md"), "w").write(to_md(out, tmap))
    print(f"{stage}: {len(recs)} cells, {len(groups)} configs -> {adir}/{stage}_*")
    return out


def to_md(out: dict, tmap: str) -> str:
    L = [f"# {out['stage']} tables", "", f"cells: {out['n_cells']}; excluded (>0.1% non-finite/clamped): "
         f"{len(out['excluded'])} {out['excluded']}; Corpus-A quiet-count flagged: {len(out['quiet_flagged'])}", "",
         "## Gate outcomes and attribution (positive findings only)", ""]
    L += [f"- {a}" for a in out["attribution"]] or ["- (none)"]
    eps = sorted({k for g in out["endpoints"].values() for s in g.values() for k in s})
    L += ["", "## Endpoints (mean over seed_data of seed_fit-averaged values)", "",
          "| config | n_sd | " + " | ".join(eps) + " |", "|---|---|" + "---|" * len(eps)]
    for cfg, g in out["endpoints"].items():
        vals = [np.nanmean([s.get(e, np.nan) for s in g.values()]) for e in eps]
        L.append(f"| {cfg} | {len(g)} | " + " | ".join(f"{v:+.4f}" for v in vals) + " |")
    L += ["", "## Contrasts (primary = gated; elevation = endpoint vs 0; exploratory = CI only)", "",
          "| kind | family | endpoint | arm | ref | n | mean | 95% CI | p | p_Holm | reading |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in out["contrasts"]:
        L.append(f"| {r['kind']} | {r['family']} | {r['endpoint']} | {r['arm']} | {r['ref']} | {r['n']} | "
                 f"{r['mean']:+.4f} | [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] | {r['p']:.4f} | {r['p_holm']:.4f} | {r['text']} |")
    L += ["", f"## Template coefficients of {tmap}: DESCRIPTIVE ONLY (mean [95% CI] over seed_data)", "",
          "| config | " + " | ".join(TEMPLATE_COEFS) + " | r2 |", "|---|" + "---|" * (len(TEMPLATE_COEFS) + 1)]
    for cfg, row in out["templates_descriptive"][tmap].items():
        L.append(f"| {cfg} | " + " | ".join(f"{row[b]['mean']:+.3f} [{row[b]['ci'][0]:+.3f}, {row[b]['ci'][1]:+.3f}]"
                                           for b in TEMPLATE_COEFS + ("r2",)) + " |")
    for k in ("cross_table", "template_corr_mean", "variance_split", "Emu0_corr_across_tau"):
        if k in out:
            L += ["", f"## {k}", "", "```", json.dumps(out[k], indent=1, default=float), "```"]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["S1", "S2", "S3", "S4", "all"])
    ap.add_argument("--runs-root", default=os.path.expanduser("~/work/halo-runs"))
    a = ap.parse_args(argv)
    for s in (["S1", "S2", "S3", "S4"] if a.stage == "all" else [a.stage]):
        analyse(s, os.path.expanduser(a.runs_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
