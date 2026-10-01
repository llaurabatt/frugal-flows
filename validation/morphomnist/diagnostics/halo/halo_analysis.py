"""Halo ladder analysis (HALO_PREREG.md v1, "Gates" and "Analysis and wording").

    python halo_analysis.py --stage {S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,all} --runs-root ~/work/halo-runs
                            [--compare-root ~/work/halo-runs]   (S5/S6: where S0/S1/S2/S4 live)

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
    if c.get("copula_rank_rule") is not None:          # S7 (Amendment A3) cells only
        s += f"/R{c['copula_rank_rule']}/W{c['copula_width']}" + ("/paper" if c.get("paper_setting") else "")
    elif c.get("paper_setting"):                       # S9 (Amendment A4) paper-setting cells
        s += "/paper"
    return s


def check_flags(recs) -> None:
    flags = {r["xla"] for r in recs}
    if len(flags) > 1:
        sys.exit(f"REFUSING to pool cells from {len(flags)} XLA flag strings: {sorted(flags)}")


def excluded(r) -> bool:
    """>0.1% non-finite rows or clamped coordinates, or diverged (max |E_mu| > 10, v1.1)."""
    m = r["met"]
    return (m["frac_nonfinite"] > EXCLUDE_FRAC or m["frac_clamped_coords"] > EXCLUDE_FRAC
            or bool(m.get("diverged", False)))


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
    # a row may carry its family's own minimum effect (S6, Amendment A2); else the v1.1 table
    ok_iii = np.isfinite(r["mean"]) and abs(r["mean"]) >= r.get("min_effect", MIN_EFFECT.get(r["endpoint"], 0.0))
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
    if len(sfs) < 2:
        return {"note": "single seed_fit (exploratory config): no variance split"}
    if len(rows) != len(sds) * len(sfs) or len(sds) < 2:
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


# ------------------------------------------------------------------ S5 (Amendment A1)
S5_PANEL = ("A1s/P0", "A1s/P1", "A1s/P5", "ff_cond/P0/bs1/E1", "ff_cond/P1/bs1/E1", "ff_cond/P5/bs1/E1",
            "ff_full/P0/bs1/E2", "ff_full/P1/bs1/E2", "ff_full/P5/bs1/E2")
S5_COMPARE = {"A1s/P0", "A1s/P1", "A2/P1", "A1/P0", "B:A1s/P0", "B:A1s/P1",
              "ff_cond/P0/bs1/E1", "ff_cond/P1/bs1/E1",
              "ff_full/P0/bs1/E1", "ff_full/P1/bs1/E1", "ff_full/P0/bs1/E2", "ff_full/P1/bs1/E2"}
S5_S2_EP = ("Etau_disc_mean", "Etau_active_off_rms", "Etau_active_off_rms_floorref")


def _add_s5_derived(groups: dict, croot: str) -> None:
    """Derived per-seed endpoints, written into each seed's ``ep``:
    Etau_active_off_rms_floorref = RMS of the seed_fit-mean E_tau on active_off minus the RMS
        of S0's model-free null naive T-difference on Y0 (``floor_null_naive_Y0``) on the same
        pixels (S0 Corpus-A file of that seed_data);
    Etau_disc_E2mE1 (ff_full E2 configs) = Etau_disc_mean(E2) - Etau_disc_mean(E1), same
        preproc, same seed_data."""
    s0 = {}
    for L, G in groups.items():
        if L.startswith("B:"):
            continue
        for sd, v in G["seed"].items():
            if "E_tau" not in v["maps"]:
                continue
            if sd not in s0:
                f = os.path.join(croot, "S0", f"s0_sd{sd}.npz")
                s0[sd] = np.load(f)["floor_null_naive_Y0"] if os.path.exists(f) else None
            m = v["maps"]["cls_active_off"].astype(bool)
            v["ep"]["Etau_active_off_rms_floorref"] = (
                float("nan") if s0[sd] is None else
                v["ep"]["Etau_active_off_rms"] - _cm(s0[sd], m, "rms"))
    for L, G in groups.items():
        if L.startswith("ff_full/") and L.endswith("/E2"):
            L1 = L[:-2] + "E1"
            if L1 not in groups:
                continue
            for sd, v in G["seed"].items():
                v1 = groups[L1]["seed"].get(sd)
                v["ep"]["Etau_disc_E2mE1"] = (float("nan") if v1 is None else
                                              v["ep"]["Etau_disc_mean"] - v1["ep"]["Etau_disc_mean"])


def _level(groups, L: str, ep: str, name: str) -> list[dict]:
    """Exploratory one-sample row: the config's endpoint vs 0 (mean, CI, unadjusted p)."""
    if L not in groups:
        return []
    r = {"family": name, "kind": "exploratory", "endpoint": ep, "arm": L, "ref": "0",
         **_test(list(_vals(groups, L, ep).values())), "p_holm": float("nan"), "resolved": False}
    lo, hi = r["ci"]
    r["text"] = f"exploratory level: mean {r['mean']:+.4f}, CI [{lo:+.4f}, {hi:+.4f}], unadjusted p {r['p']:.4f}"
    return [r]


def s5_rows(allg: dict, croot: str) -> tuple[list[dict], list[str]]:
    """Amendment A1. Primary family (Holm across 3 contrasts x 2 endpoints = 6 tests):
    A1s/P5 - A1s/P0, A1s/P5 - A1s/P1 (paired by seed_data, S1 cells as comparators) and the
    elevation of A1s/P5 vs reference data, gated as v1.1. P0/P1 elevation status is read
    from the S1 elevation family recomputed exactly as S1's analysis does. Everything else
    exploratory (CIs only)."""
    _add_s5_derived(allg, croot)
    p5, p0, p1 = "A1s/P5", "A1s/P0", "A1s/P1"
    fam = "S5 primary (Amendment A1)"
    rows = [{"family": fam, "kind": "primary", "endpoint": ep, "arm": p5, "ref": ref, **paired(allg, p5, ref, ep)}
            for ref in (p0, p1) if p5 in allg and ref in allg for ep in S1_EP]
    rows += [{"family": fam, "kind": "elevation", "endpoint": ep, "arm": p5, "ref": "reference data",
              **_test(list(_vals(allg, p5, ep).values()))} for ep in S1_EP if p5 in allg]
    for r, ph in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = ph
        gate(r)
    s1_el = elevation_family(allg, "S1 elevation (recomputed, as S1)", [p0, p1, "A2/P1", "A1/P0"], S1_EP)
    rows += s1_el
    attr = []
    for ep in S1_EP:
        rem = _find(rows, "primary", p5, ep, p0) or {}
        dif = _find(rows, "primary", p5, ep, p1) or {}
        el5 = (_find(rows, "elevation", p5, ep) or {})
        el0 = (_find(s1_el, "elevation", p0, ep) or {}).get("resolved", False)
        el1 = (_find(s1_el, "elevation", p1, ep) or {}).get("resolved", False)
        fired = []
        if rem.get("resolved") and not el5.get("resolved") and not dif.get("resolved"):
            fired.append("floored scaling suffices: P5 removes the A1s/P0 elevation (P5 vs P0 resolved, "
                         "P5 not elevated) and is not resolvedly different from P1")
        if el5.get("resolved") and not el1:
            fired.append("the floor leaves part of the artefact: A1s/P5 elevated while A1s/P1 is not")
        if fired:
            attr += [f"A:{ep}: {f}" for f in fired]
        else:
            attr.append(f"A:{ep}: no attribution rule fires. A1s/P0 elevated (S1): {el0}; A1s/P1 elevated: {el1}. "
                        f"P5 vs P0 -> {rem.get('text', 'n/a')}; P5 vs P1 -> {dif.get('text', 'n/a')}; "
                        f"P5 elevation -> {el5.get('text', 'n/a')}")
    rows += exploratory(allg, "B:A1s/P0", ["B:A1s/P5"], S1_EP + S1_EXPL, "S5 exploratory Corpus B vs B:A1s/P0")
    rows += exploratory(allg, "B:A1s/P1", ["B:A1s/P5"], S1_EP + S1_EXPL, "S5 exploratory Corpus B vs B:A1s/P1")
    for ep in S1_EP:
        rows += _level(allg, "B:A1s/P5", ep, "S5 exploratory Corpus B elevation")
    rows += exploratory(allg, p0, [p5], S1_EXPL, "S5 exploratory secondary vs A1s/P0")
    f5 = "ff_cond/P5/bs1/E1"
    for ref in ("ff_cond/P1/bs1/E1", "ff_cond/P0/bs1/E1"):
        rows += exploratory(allg, ref, [f5], S5_S2_EP + ("tauhat_disc_mean", "Emu0_active_off_rms"),
                            f"S5 exploratory S2-type vs {ref}")
    for ep in S5_S2_EP:
        rows += _level(allg, f5, ep, "S5 exploratory S2-type level")
    e2 = "ff_full/P5/bs1/E2"
    for ref in ("ff_full/P0/bs1/E2", "ff_full/P1/bs1/E2"):
        rows += exploratory(allg, ref, [e2], ("slope_imb_offsupport", "Etau_disc_E2mE1") + S2_EP,
                            f"S4-type exploratory vs {ref}")
    rows += exploratory(allg, "ff_full/P0/bs1/E1", ["ff_full/P5/bs1/E1"], S2_EP,
                        "S4-type exploratory E1 vs ff_full/P0/bs1/E1")
    for ep in ("slope_imb_offsupport", "Etau_disc_E2mE1"):
        rows += _level(allg, e2, ep, "S4-type exploratory level under P5")
    return rows, attr


# ------------------------------------------------------------------ S6 (Amendment A2)
S6_ARMS = {"U": "ff_cond/P1/bs1/E1", "N": "n_cond/P1/bs1/E1", "A2": "a2_cond/P1/bs1/E1",
           "LT": "lt/P0/bs1/E1", "LT/P1": "lt/P1/bs1/E1", "LT-N": "lt_n/P1/bs1/E1"}
S6_COMPARE = {S6_ARMS["U"], S6_ARMS["A2"], S6_ARMS["LT"]}       # from S2
S6_TAU0 = ("n_cond/P1/bs0/E1", "lt_n/P1/bs0/E1", "lt/P1/bs0/E1")
S6_EP = ("ate_mae", "Etau_disc_mean")
S6_MIN = {"ate_mae": 0.003, "Etau_disc_mean": 0.01}            # A2 minimum effects
S6_SEC = ("ate_mae_off", "KS_active_mean", "Rsd_quiet_median")


def _s6_cell(maps: dict) -> dict:
    """Per-CELL S6 endpoints (Amendment A2), later averaged over seed_fit within seed_data:
    ate_mae = mean |E_tau| over all 64 px; ate_mae_off = the same over ATE == 0 px;
    KS_active_mean = mean over active px of (KS0 + KS1) / 2; Rsd_quiet_median = median
    log2 R_sd0 (do(0) arm) over quiet px; ate_mae_sampled = mean |tau_hat_crn - ATE| (equal
    to ate_mae except for the LT arms, whose primary tau_hat is the fitted ate vector)."""
    et, ate = np.asarray(maps["E_tau"], float), np.asarray(maps["ATE"], float)
    act, quiet = maps["cls_active"].astype(bool), maps["cls_quiet"].astype(bool)
    ks = (np.asarray(maps["KS0"], float) + np.asarray(maps["KS1"], float)) / 2
    return {"ate_mae": float(np.mean(np.abs(et))), "ate_mae_off": _cm(np.abs(et), ate == 0),
            "KS_active_mean": _cm(ks, act), "Rsd_quiet_median": _cm(maps["R_sd0"], quiet, "median"),
            "ate_mae_sampled": float(np.mean(np.abs(np.asarray(maps["tau_hat_crn"], float) - ate)))}


def _add_s6_derived(groups: dict) -> None:
    """Write the per-cell S6 endpoints (seed_fit-averaged) into every seed's ``ep``. This
    replaces the seed-mean-map ``Rsd_quiet_median`` with the per-cell one for S6 tables."""
    for G in groups.values():
        for v in G["seed"].values():
            if "E_tau" not in v["maps"]:
                continue
            per = [_s6_cell(c["maps"]) for c in v["cells"]]
            for k in per[0]:
                v["ep"][k] = float(np.nanmean([p[k] for p in per]))


def _naive_s6(croot: str) -> dict:
    """Model-free reference: per seed_data, mean |naive T-difference - ATE| over 64 px on
    the E1 tau=1 data (S0 ``imb_E1_bs1``)."""
    vals = {}
    for sd in range(31, 41):
        f = os.path.join(croot, "S0", f"s0_sd{sd}.npz")
        if os.path.exists(f):
            vals[sd] = float(np.mean(np.abs(np.load(f)["imb_E1_bs1"])))
    x = np.array(list(vals.values()))
    return {"per_seed": vals, "mean": float(x.mean()) if len(x) else float("nan"), "ci": list(boot_ci(x))}


def s6_rows(allg: dict) -> tuple[list[dict], list[str], list[str]]:
    """Amendment A2. Primary family: Holm over 3 contrasts x 2 endpoints (ate_mae,
    Etau_disc_mean): U - N, N - A2, N - LT-N, gated with the A2 minimum effects. Exploratory
    rows carry CIs only. Returns (rows, gate lines, prediction-vs-outcome lines)."""
    _add_s6_derived(allg)
    A = S6_ARMS
    fam = "S6 primary (Amendment A2)"
    prim = [(A["U"], A["N"]), (A["N"], A["A2"]), (A["N"], A["LT-N"])]
    rows = [{"family": fam, "kind": "primary", "endpoint": ep, "arm": a, "ref": b, "min_effect": S6_MIN[ep],
             **paired(allg, a, b, ep)} for a, b in prim if a in allg and b in allg for ep in S6_EP]
    for r, ph in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = ph
        gate(r)
    gates = [f"{r['arm']} - {r['ref']} [{r['endpoint']}]: {r['text']}" for r in rows]
    expl = [(A["LT/P1"], A["LT"], "LT(P0) - LT/P1"), (A["LT-N"], A["LT/P1"], "LT/P1 - LT-N"),
            (A["LT-N"], A["U"], "U - LT-N"), (A["A2"], A["U"], "U - A2")]
    for ref, arm, name in expl:
        rows += exploratory(allg, ref, [arm], S6_EP + S6_SEC + ("ate_mae_sampled",), f"S6 exploratory {name}")
    for a, b in prim:
        rows += exploratory(allg, b, [a], S6_SEC, f"S6 exploratory secondary {a} - {b}")
    for L in S6_TAU0:
        for ep in ("Etau_disc_mean", "Etau_active_off_mean", "ate_mae"):
            rows += _level(allg, L, ep, "S6 exploratory tau=0 level vs 0")
    # predictions stated in A2 before running, printed beside outcomes; gates are NOT changed
    un = _find(rows, "primary", A["U"], "ate_mae", A["N"]) or {}
    ua2 = _find(rows, "exploratory", A["U"], "ate_mae", A["A2"]) or {}
    preds = []
    m_un, m_ua2 = un.get("mean", float("nan")), ua2.get("mean", float("nan"))
    preds.append(f"PREDICTION U - N > 0 and >= half of U - A2 (ate_mae) | OUTCOME: U - N mean {m_un:+.4f} "
                 f"({un.get('text', 'n/a')}); U - A2 mean {m_ua2:+.4f} (exploratory); ratio (U-N)/(U-A2) = "
                 f"{(m_un / m_ua2) if np.isfinite(m_ua2) and m_ua2 != 0 else float('nan'):.2f}; "
                 f"direction {'undetermined (nan)' if not (np.isfinite(m_un) and np.isfinite(m_ua2)) else 'as predicted' if m_un > 0 and m_un >= 0.5 * m_ua2 else 'NOT as predicted'} "
                 f"(descriptive; only the gate line above resolves anything)")
    for ep in S6_EP:
        r = _find(rows, "primary", A["N"], ep, A["A2"]) or {}
        preds.append(f"PREDICTION N - A2 not resolved [{ep}] | OUTCOME: resolved={r.get('resolved')}; {r.get('text', 'n/a')}")
    r = _find(rows, "primary", A["N"], "ate_mae", A["LT-N"]) or {}
    preds.append(f"PREDICTION N - LT-N > 0 (ate_mae) | OUTCOME: mean {r.get('mean', float('nan')):+.4f}; {r.get('text', 'n/a')}")
    return rows, gates, preds


def s6_figure(allg: dict, naive: dict, path: str) -> None:
    """Top: seed-mean E_tau maps (fixed +-0.08) for the six tau=1 arms. Bottom: per-cell ATE
    MAE (seed_fit-averaged), mean +- 95% bootstrap CI over seed_data, naive-difference line."""
    names = [k for k in S6_ARMS if S6_ARMS[k] in allg]
    fig = plt.figure(figsize=(2.2 * max(len(names), 1), 6.0), layout="constrained")
    gs = fig.add_gridspec(2, max(len(names), 1), height_ratios=[1, 1.2])
    im = None
    for j, k in enumerate(names):
        ss = list(allg[S6_ARMS[k]]["seed"].values())
        ax = fig.add_subplot(gs[0, j])
        ax.set_xticks([]), ax.set_yticks([])
        im = ax.imshow(np.mean([s["maps"]["E_tau"] for s in ss], 0).reshape(8, 8), cmap="RdBu_r",
                       vmin=-0.08, vmax=0.08)
        ax.set_title(f"{k}\n{S6_ARMS[k]}\n(n_sd={len(ss)})", fontsize=7)
    if im is not None:
        fig.colorbar(im, ax=fig.axes[:len(names)], location="right", shrink=0.8, label="E_tau (logit)")
    ax = fig.add_subplot(gs[1, :])
    ms, lo, hi = [], [], []
    for k in names:
        x = np.array([v for v in _vals(allg, S6_ARMS[k], "ate_mae").values() if np.isfinite(v)])
        m = float(x.mean()) if len(x) else float("nan")
        c = boot_ci(x)
        ms.append(m), lo.append(m - c[0] if np.isfinite(c[0]) else 0), hi.append(c[1] - m if np.isfinite(c[1]) else 0)
    ax.bar(range(len(names)), ms, yerr=[lo, hi], capsize=4, color="0.55")
    if np.isfinite(naive.get("mean", np.nan)):
        ax.axhline(naive["mean"], color="C3", ls="--", lw=1, label=f"naive difference {naive['mean']:.4f}")
        ax.legend(fontsize=7)
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel("ATE MAE (mean |E_tau| over 64 px)")
    ax.set_title("S6: per-cell ATE MAE, mean and 95% bootstrap CI over seed_data", fontsize=8)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ S7 (Amendment A3)
S7_CFG = {("old", 50): "Rold/W50", ("new", 50): "Rnew/W50", ("old", 16): "Rold/W16", ("new", 16): "Rnew/W16"}
S7_BASE = "ff_full/P0/bs1/{e}/{c}"
S7_EP = ("Etau_disc_mean", "ate_mae", "slope_imb_offsupport")
S7_MIN = {"Etau_disc_mean": 0.01, "ate_mae": 0.003, "slope_imb_offsupport": 0.05}    # A3 minimum effects
S7_S4_LABELS = {"ff_full/P0/bs1/E2", "ff_full/P0/bs1/E1"}       # the (W50, new) comparators


def s7_label(e: str, rule: str, w: int, paper: bool = False) -> str:
    return S7_BASE.format(e=e, c=S7_CFG[(rule, w)]) + ("/paper" if paper else "")


def s7_relabel_s4(recs: list[dict]) -> list[dict]:
    """S4 ff_full/P0 cells ARE the (W50, new) S7 cells (A3): give them the S7 keys so they group
    under the S7 label. Only the in-memory cfg copy changes; nothing on disk."""
    out = []
    for r in recs:
        if label(r["cfg"]) in S7_S4_LABELS:
            out.append({**r, "cfg": {**r["cfg"], "copula_rank_rule": "new", "copula_width": 50}})
    return out


def _s7_cell(r: dict) -> dict:
    """Per-CELL S7 endpoints: E_tau disc-class mean, ATE MAE = mean |E_tau| over 64 px, and the
    cell's off-support slope of E_tau on the naive-bias map (halo_fit ``slope_imb_offsupport``)."""
    m = r["maps"]
    et = np.asarray(m["E_tau"], float)
    return {"Etau_disc_mean": _cm(et, m["cls_disc"]), "ate_mae": float(np.mean(np.abs(et))),
            "slope_imb_offsupport": float(r["met"]["slope_imb_offsupport"])}


def _add_s7_derived(groups: dict) -> None:
    """Per-cell endpoints on every cell (``c['s7']``) and their seed_fit means in each seed's ``ep``."""
    for G in groups.values():
        for v in G["seed"].values():
            if "E_tau" not in v["maps"]:
                continue
            for c in v["cells"]:
                c["s7"] = _s7_cell(c)
            for k in S7_EP:
                v["ep"][k] = float(np.nanmean([c["s7"][k] for c in v["cells"]]))


def paired_cells(groups, a: str, b: str, ep: str, sf: int | None = None) -> dict:
    """a - b paired by (seed_data, seed_fit), then averaged within seed_data (A3); ``sf``
    restricts to one seed_fit. Units = seed_data with at least one matched pair."""
    if a not in groups or b not in groups:
        return _test([])
    d = []
    for sd in sorted(set(groups[a]["seed"]) & set(groups[b]["seed"])):
        ca = {c["cfg"]["seed_fit"]: c["s7"][ep] for c in groups[a]["seed"][sd]["cells"]}
        cb = {c["cfg"]["seed_fit"]: c["s7"][ep] for c in groups[b]["seed"][sd]["cells"]}
        sfs = [f for f in sorted(set(ca) & set(cb)) if sf is None or f == sf]
        x = [ca[f] - cb[f] for f in sfs if np.isfinite(ca[f]) and np.isfinite(cb[f])]
        if x:
            d.append(float(np.mean(x)))
    return _test(d)


def _s7_expl(groups, a, b, eps, name, sf=None) -> list[dict]:
    rows = []
    for ep in eps:
        r = {"family": name, "kind": "exploratory", "endpoint": ep, "arm": a, "ref": b,
             **paired_cells(groups, a, b, ep, sf), "p_holm": float("nan"), "resolved": False}
        lo, hi = r["ci"]
        r["text"] = f"exploratory: mean {r['mean']:+.4f}, CI [{lo:+.4f}, {hi:+.4f}], unadjusted p {r['p']:.4f}"
        rows.append(r)
    return rows


def s7_rows(allg: dict) -> tuple[list[dict], list[str], list[str]]:
    """Amendment A3. Primary family: Holm over 2 widths x 3 endpoints, old - new at W50 (new =
    the S4 cells) and at W16 on E2, paired by (seed_data, seed_fit) then averaged within
    seed_data, gated with the A3 minimum effects. Everything else exploratory with CIs."""
    _add_s7_derived(allg)
    fam = "S7 primary (Amendment A3)"
    rows = []
    for w in (50, 16):
        a, b = s7_label("E2", "old", w), s7_label("E2", "new", w)
        rows += [{"family": fam, "kind": "primary", "endpoint": ep, "arm": a, "ref": b, "min_effect": S7_MIN[ep],
                  **paired_cells(allg, a, b, ep)} for ep in S7_EP if a in allg and b in allg]
    for r, ph in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = ph
        gate(r)
    gates = [f"{r['arm']} - {r['ref']} [{r['endpoint']}]: {r['text']}" for r in rows]
    for rule in ("old", "new"):
        for w in (50, 16):
            rows += _s7_expl(allg, s7_label("E2", rule, w), s7_label("E1", rule, w), ("Etau_disc_mean",),
                             f"S7 exploratory E2 - E1 at seed_fit 41 ({rule}, W{w})", sf=41)
    for w in (50, 16):
        rows += _s7_expl(allg, s7_label("E1", "old", w), s7_label("E1", "new", w), ("ate_mae", "Etau_disc_mean"),
                         f"S7 exploratory E1 old - new (W{w})")
    rows += _s7_expl(allg, s7_label("E2", "new", 16), s7_label("E2", "new", 50), S7_EP,
                     "S7 exploratory W16/new - W50/new (E2)")
    rows += _s7_expl(allg, s7_label("E2", "old", 16, True), s7_label("E2", "new", 16, True), S7_EP,
                     "S7 exploratory paper setting old - new (E2, lr 1e-3, <=1000 ep, W16; n=5)")
    for L in [s7_label(e, r, w) for e in ("E2", "E1") for r in ("old", "new") for w in (50, 16)] + \
            [s7_label("E2", r, 16, True) for r in ("old", "new")]:
        for ep in S7_EP:
            rows += _level(allg, L, ep, "S7 exploratory level vs 0")
    # predictions stated in A3 before running, printed beside outcomes; gates are NOT changed
    preds = []
    for w, claim in ((16, "old - new > 0 on all three endpoints (the copula cannot adjust for thickness "
                          "through 48 unseen pixels)"),
                     (50, "old - new small (the 14 unseen pixels are mostly the quiet bottom rows)")):
        parts = []
        for ep in S7_EP:
            r = _find(rows, "primary", s7_label("E2", "old", w), ep, s7_label("E2", "new", w)) or {}
            m = r.get("mean", float("nan"))
            parts.append(f"{ep}: mean {m:+.4f} (sign {'>0' if m > 0 else '<=0' if np.isfinite(m) else 'nan'}); "
                         f"{r.get('text', 'n/a')}")
        preds.append(f"PREDICTION W{w}: {claim} | OUTCOME: " + " || ".join(parts)
                     + " (descriptive; only the gate lines resolve anything)")
    return rows, gates, preds


def s7_fit_summary(allg: dict) -> dict:
    """Per config: epochs run / best epoch medians, and the copula reachability the cells
    recorded from their fitted masks (proof of the rule; the S4 comparators predate it)."""
    out = {}
    for L, G in allg.items():
        cells = [c for v in G["seed"].values() for c in v["cells"]]
        ep = [i["n_epochs"] for c in cells for i in c["met"]["fit_info"]]
        be = [i["best_epoch"] for c in cells for i in c["met"]["fit_info"]]
        reach = [c["met"]["copula_reachable"] for c in cells if "copula_reachable" in c["met"]]
        rules = sorted({c["met"].get("copula_rank_rule", "n/a (S4: package rule)") for c in cells})
        out[L] = {"n_cells": len(cells), "epochs_run_median": float(np.median(ep)) if ep else float("nan"),
                  "best_epoch_median": float(np.median(be)) if be else float("nan"),
                  "epochs_run_range": [int(min(ep)), int(max(ep))] if ep else None,
                  "copula_reachable": sorted(set(reach)) or "not recorded", "rules_recorded": rules,
                  "wall_s_median": float(np.median([c["met"].get("wall_s", np.nan) for c in cells]))}
    return out


def s7_figure(allg: dict, path: str) -> None:
    """Top: seed-mean naive-bias map then seed-mean E_tau maps on E2 (fixed +-0.15) for
    old/new x W50/W16. Bottom: E2 disc bias and ATE MAE, mean +- 95% bootstrap CI over seed_data."""
    names = [(r, w) for w in (50, 16) for r in ("old", "new") if s7_label("E2", r, w) in allg]
    if not names:
        return
    fig = plt.figure(figsize=(2.3 * (len(names) + 1), 6.4), layout="constrained")
    gs = fig.add_gridspec(2, 2 * (len(names) + 1), height_ratios=[1, 1.1])
    ss0 = list(allg[s7_label("E2", *names[0])]["seed"].values())
    panels = [("naive bias (naive diff - ATE), clipped", np.mean([s["maps"]["imb"] for s in ss0], 0), len(ss0))]
    for r, w in names:
        ss = list(allg[s7_label("E2", r, w)]["seed"].values())
        panels.append((f"E_tau {r} / W{w}" + (" (S4)" if (r, w) == ("new", 50) else ""),
                       np.mean([s["maps"]["E_tau"] for s in ss], 0), len(ss)))
    im = None
    for j, (t, img, n) in enumerate(panels):
        ax = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
        ax.set_xticks([]), ax.set_yticks([])
        im = ax.imshow(np.asarray(img, float).reshape(8, 8), cmap="RdBu_r", vmin=-0.15, vmax=0.15)
        ax.set_title(f"{t}\nE2, n_sd={n}", fontsize=7)
    fig.colorbar(im, ax=fig.axes[:len(panels)], location="right", shrink=0.8, label="logit (fixed ±0.15)")
    half = len(names) + 1
    for k, (ep, yl) in enumerate((("Etau_disc_mean", "E2 disc bias (E_tau disc mean)"),
                                  ("ate_mae", "E2 ATE MAE (mean |E_tau|, 64 px)"))):
        ax = fig.add_subplot(gs[1, k * half:(k + 1) * half])
        ms, lo, hi = [], [], []
        for r, w in names:
            x = np.array([v for v in _vals(allg, s7_label("E2", r, w), ep).values() if np.isfinite(v)])
            m = float(x.mean()) if len(x) else float("nan")
            c = boot_ci(x)
            ms.append(m), lo.append(m - c[0] if np.isfinite(c[0]) else 0), hi.append(c[1] - m if np.isfinite(c[1]) else 0)
        ax.bar(range(len(names)), ms, yerr=[lo, hi], capsize=4,
               color=["C3" if r == "old" else "C0" for r, _ in names])
        ax.axhline(0, color="0.3", lw=0.6)
        ax.set_xticks(range(len(names)), [f"{r}\nW{w}" for r, w in names], fontsize=7)
        ax.set_ylabel(yl, fontsize=7)
        ax.set_title("mean, 95% bootstrap CI over seed_data", fontsize=7)
    fig.suptitle("S7 (Amendment A3): copula hidden ranks old vs new; E2, P0, harness settings", fontsize=8)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ S8/S9 (Amendment A4)
S8_ARMS = {"U": "ff_cond/P1/bs1/E1", "N": "n_cond/P1/bs1/E1", "LT-N": "lt_n/P1/bs1/E1"}
S8_STATS = ("latent_ks_val_mean", "latent_ks_all_mean", "latent_ks_val_max", "latent_ks_all_max",
            "latent_offdiag_corr_all", "latent_offdiag_corr_val")


def _median(x) -> float:
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    return float(np.median(x)) if len(x) else float("nan")


def s8_summary(groups: dict) -> dict:
    """Amendment A4, S8 (descriptive, no gate): per arm, medians over cells of the latent
    calibration stats, best epoch and epochs run, plus the per-cell ATE MAE (mean |E_tau|)."""
    out = {}
    for k, L in S8_ARMS.items():
        if L not in groups:
            continue
        cells = [c for v in groups[L]["seed"].values() for c in v["cells"]]
        cal = [c["met"].get("calibration", {}) for c in cells]
        row = {"label": L, "n_cells": len(cells), "latent_kind": sorted({c.get("latent_kind", "?") for c in cal})}
        for st in S8_STATS:
            row[st] = _median([c.get(st, np.nan) for c in cal])
        row["best_epoch_median"] = _median([i["best_epoch"] for c in cells for i in c["met"]["fit_info"]])
        row["epochs_run_median"] = _median([i["n_epochs"] for c in cells for i in c["met"]["fit_info"]])
        row["ate_mae_median"] = _median([float(np.mean(np.abs(c["maps"]["E_tau"]))) for c in cells])
        # per-pixel KS (all rows), median over cells, for the KS map
        row["latent_ks_all_pixel_median"] = np.median([c["latent_ks_all"] for c in cal if "latent_ks_all" in c],
                                                      0).tolist() if any("latent_ks_all" in c for c in cal) else []
        out[k] = row
    return out


S9_FF = "ff_full/P1/bs1/{e}"                 # FF-uniform comparator (S4 cells)
S9_FLEX, S9_SHIFT = "gff_flex/P1/bs1/{e}", "gff_shift/P1/bs1/{e}"
S9_NAMES = {"FF-uniform": S9_FF, "GFF-flex": S9_FLEX, "GFF-shift": S9_SHIFT}
S9_EP = S7_EP
S9_MIN = S7_MIN                              # A4 minimum effects: 0.01 disc bias, 0.003 ATE MAE, 0.05 slope
S9_S7_PAPER = "ff_full/P0/bs1/E2/Rnew/W16/paper"
S9_CAL = ("gY_ks_val_mean", "gY_ks_all_mean", "gY_ks_all_max", "gY_offdiag_corr_all", "n_nonfinite_joint")
S9_CAL_VEC = ("gZ_implied_ks", "gZ_resid_ks_all", "gZ_resid_ks_val", "gZ_data_ks")


def s9_label(name: str, e: str, paper: bool = False) -> str:
    return S9_NAMES[name].format(e=e) + ("/paper" if paper else "")


def s9_rows(allg: dict) -> tuple[list[dict], list[str], list[str]]:
    """Amendment A4. Primary family (Holm over 3 contrasts x 3 endpoints, E2): GFF-flex - FF-uniform,
    GFF-shift - FF-uniform, GFF-shift - GFF-flex; paired by (seed_data, seed_fit), averaged within
    seed_data; gated with the v1.1 conjunction and the A4 minimum effects. Everything else exploratory."""
    _add_s7_derived(allg)
    fam = "S9 primary (Amendment A4)"
    prim = [("GFF-flex", "FF-uniform"), ("GFF-shift", "FF-uniform"), ("GFF-shift", "GFF-flex")]
    rows = []
    for a, b in prim:
        A, B = s9_label(a, "E2"), s9_label(b, "E2")
        rows += [{"family": fam, "kind": "primary", "endpoint": ep, "arm": A, "ref": B, "min_effect": S9_MIN[ep],
                  **paired_cells(allg, A, B, ep)} for ep in S9_EP if A in allg and B in allg]
    for r, ph in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = ph
        gate(r)
    gates = [f"{r['arm']} - {r['ref']} [{r['endpoint']}]: {r['text']}" for r in rows]
    for n in S9_NAMES:
        rows += _s7_expl(allg, s9_label(n, "E2"), s9_label(n, "E1"), ("Etau_disc_mean",),
                         f"S9 exploratory E2 - E1 at seed_fit 41 ({n})", sf=41)
    for n in ("GFF-flex", "GFF-shift"):
        rows += _s7_expl(allg, s9_label(n, "E1"), s9_label("FF-uniform", "E1"), ("ate_mae", "Etau_disc_mean"),
                         f"S9 exploratory E1 {n} - FF-uniform")
    rows += _s7_expl(allg, s9_label("GFF-shift", "E1"), s9_label("GFF-flex", "E1"), ("ate_mae", "Etau_disc_mean"),
                     "S9 exploratory E1 GFF-shift - GFF-flex")
    for n in ("GFF-flex", "GFF-shift"):
        rows += _s7_expl(allg, s9_label(n, "E2", True), S9_S7_PAPER, S9_EP,
                         f"S9 exploratory paper setting {n} (P1) - S7 new-rank W16 (P0); different preproc "
                         "and copula width: descriptive (E2, lr 1e-3, <=1000 ep, n=5)")
    for L in [s9_label(n, e) for n in S9_NAMES for e in ("E2", "E1")] + \
            [s9_label(n, "E2", True) for n in ("GFF-flex", "GFF-shift")]:
        for ep in S9_EP:
            rows += _level(allg, L, ep, "S9 exploratory level vs 0")
    # predictions stated in A4 before running, printed beside outcomes; gates are NOT changed
    preds = []
    r = _find(rows, "primary", s9_label("GFF-flex", "E2"), "ate_mae", s9_label("FF-uniform", "E2")) or {}
    m = r.get("mean", float("nan"))
    preds.append(f"PREDICTION GFF-flex lowers E2 ATE MAE relative to FF-uniform (GFF-flex - FF-uniform < 0) | "
                 f"OUTCOME: mean {m:+.4f} ({'direction as predicted' if m < 0 else 'direction NOT as predicted' if np.isfinite(m) else 'nan'}); "
                 f"{r.get('text', 'n/a')} (descriptive; only the gate lines resolve anything)")
    e1 = {n: float(np.nanmean(list(_vals(allg, s9_label(n, "E1"), "ate_mae").values())))
          if s9_label(n, "E1") in allg else float("nan") for n in S9_NAMES}
    fin = {k: v for k, v in e1.items() if np.isfinite(v)}
    low = min(fin, key=fin.get) if fin else "n/a"
    preds.append("PREDICTION GFF-shift has the lowest E1 ATE MAE | OUTCOME: seed-mean E1 ATE MAE "
                 + ", ".join(f"{k} {v:.4f}" for k, v in e1.items()) + f"; lowest: {low} "
                 "(descriptive; the E1 contrasts are exploratory rows with CIs only)")
    for a in ("GFF-flex", "GFF-shift"):
        r = _find(rows, "primary", s9_label(a, "E2"), "Etau_disc_mean", s9_label("FF-uniform", "E2")) or {}
        preds.append(f"NOT PREDICTED: effect of {a} on the E2 disc bias | OUTCOME: mean {r.get('mean', float('nan')):+.4f}; "
                     f"{r.get('text', 'n/a')}")
    return rows, gates, preds


def s9_calibration(allg: dict) -> dict:
    """Per config: medians over cells of the A4 latent calibration stats, best epoch, epochs run,
    non-finite draws, and |shift-vector tau_hat - sampled tau_hat| for GFF-shift."""
    out = {}
    for L, G in allg.items():
        cells = [c for v in G["seed"].values() for c in v["cells"]]
        cal = [c["met"].get("calibration") for c in cells if c["met"].get("calibration")]
        row = {"n_cells": len(cells),
               "best_epoch_median": _median([i["best_epoch"] for c in cells for i in c["met"]["fit_info"]]),
               "epochs_run_median": _median([i["n_epochs"] for c in cells for i in c["met"]["fit_info"]]),
               "n_nonfinite_draws_total": int(sum(c["met"].get("n_nonfinite", 0) for c in cells)),
               "wall_s_median": _median([c["met"].get("wall_s", np.nan) for c in cells])}
        if cal:
            for st in S9_CAL:
                row[st + "_median"] = _median([c.get(st, np.nan) for c in cal])
            for st in S9_CAL_VEC:
                row[st + "_median"] = _median([float(np.mean(c[st])) for c in cal if st in c])
        if any("lt_ate_minus_crn_meanabs" in c["met"] for c in cells):
            row["shift_vs_sampled_meanabs_median"] = _median([c["met"].get("lt_ate_minus_crn_meanabs", np.nan)
                                                              for c in cells])
        out[L] = row
    return out


def s9_figure(allg: dict, path: str) -> None:
    """Top: seed-mean E_tau maps on E2 (fixed +-0.15) and E1 (fixed +-0.08) for FF-uniform, GFF-flex,
    GFF-shift. Bottom: E2 disc bias, E2 ATE MAE, E1 ATE MAE, mean +- 95% bootstrap CI over seed_data."""
    names = list(S9_NAMES)
    fig = plt.figure(figsize=(11, 8.2), layout="constrained")
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 1.1])
    for i, (e, lim) in enumerate((("E2", 0.15), ("E1", 0.08))):
        axs, im = [], None
        for j, n in enumerate(names):
            ax = fig.add_subplot(gs[i, 2 * j:2 * j + 2])
            ax.set_xticks([]), ax.set_yticks([])
            axs.append(ax)
            L = s9_label(n, e)
            if L not in allg:
                ax.set_title(f"{n} {e}: no cells", fontsize=7)
                continue
            ss = list(allg[L]["seed"].values())
            im = ax.imshow(np.mean([s["maps"]["E_tau"] for s in ss], 0).reshape(8, 8), cmap="RdBu_r",
                           vmin=-lim, vmax=lim)
            ax.set_title(f"E_tau {n}\n{e}, n_sd={len(ss)}", fontsize=7)
        if im is not None:
            fig.colorbar(im, ax=axs, location="right", shrink=0.8, label=f"logit (fixed ±{lim})")
    for k, (e, ep, yl) in enumerate((("E2", "Etau_disc_mean", "E2 disc bias (E_tau disc mean)"),
                                     ("E2", "ate_mae", "E2 ATE MAE (mean |E_tau|, 64 px)"),
                                     ("E1", "ate_mae", "E1 ATE MAE (mean |E_tau|, 64 px)"))):
        ax = fig.add_subplot(gs[2, 2 * k:2 * k + 2])
        ms, lo, hi = [], [], []
        for n in names:
            L = s9_label(n, e)
            x = np.array([v for v in _vals(allg, L, ep).values() if np.isfinite(v)]) if L in allg else np.array([])
            m = float(x.mean()) if len(x) else float("nan")
            c = boot_ci(x)
            ms.append(m), lo.append(m - c[0] if np.isfinite(c[0]) else 0), hi.append(c[1] - m if np.isfinite(c[1]) else 0)
        ax.bar(range(len(names)), ms, yerr=[lo, hi], capsize=4, color=["0.55", "C0", "C2"])
        ax.axhline(0, color="0.3", lw=0.6)
        ax.set_xticks(range(len(names)), names, fontsize=7)
        ax.set_ylabel(yl, fontsize=7)
        ax.set_title("mean, 95% bootstrap CI over seed_data", fontsize=7)
    fig.suptitle("S9 (Amendment A4): Gaussian-scale frugal flow vs FF-uniform (S4 ff_full/P1); P1, harness settings",
                 fontsize=8)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ S10 (Amendment A5)
S10_TRUTHS = (1.0, 0.0, -1.0)
S10_INITS = ("zero", "naive", "plus2")
S10_EP = ("tauhat_disc_mean", "Etau_disc_mean", "rho", "ate_mae", "Etau_active_off_mean", "Etau_quiet_mean",
          "dist_from_init", "dist_from_truth", "dist_naive_from_truth")
S10_ANCHOR_A = ("lt_n", 1.0, "zero", 1.2, False)
S10_ANCHOR_B = ("gff_shift", 1.0, "zero", 1.2, True)
S10_SEEDS = tuple(range(31, 36))


def s10_key(c: dict) -> tuple:
    """(arm, base_shift, shift_init, ps_slope, placebo). S9 cells (no S10 keys) = init zero, no placebo."""
    return (c["arm"], float(c["base_shift"]), c.get("shift_init", "zero"), float(c["ps_slope"]),
            bool(c.get("placebo_covariate", False)))


def s10_name(k: tuple) -> str:
    arm, bs, init, ps, pl = k
    if k == S10_ANCHOR_A:
        return "Anchor A (lt_n, no copula)"
    if k == S10_ANCHOR_B:
        return "Anchor B (placebo covariate)"
    return f"{arm} truth {bs:+g} init {init} ps {ps:g}"


def s10_cell_endpoints(r: dict) -> dict:
    """S10 endpoints of one halo cell from its saved maps (tau_hat, ATE, imb = naive - ATE, classes);
    the start vector from metrics (S10) or zero (S9 cells: LocCond(ate=0))."""
    import halo_metrics as hmx
    m = r["maps"]
    truth = np.asarray(m["ATE"], float)
    naive = np.asarray(m["imb"], float) + truth
    s10 = r["met"].get("s10", {})
    init = np.asarray(s10["init_vector"], float) if "init_vector" in s10 else np.zeros_like(truth)
    e = hmx.s10_endpoints(m["tau_hat"], truth, naive, m["cls_disc"], m["cls_active_off"], m["cls_quiet"], init)
    fi = r["met"]["fit_info"][0]
    e.update(epochs_run=float(fi["n_epochs"]), best_epoch=float(fi["best_epoch"]))
    if s10:
        e["stored_vs_recomputed_maxabs"] = float(max(abs(s10[k] - e[k]) for k in S10_EP if k in s10))
    return e


def _s10_group(recs: list[dict]) -> dict:
    """key -> {seed_data: endpoints averaged over that seed's cells}; plus per-seed maps."""
    g: dict = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if excluded(r):
            continue
        r["s10ep"] = s10_cell_endpoints(r)
        g[s10_key(r["cfg"])][r["cfg"]["seed_data"]].append(r)
    out = {}
    for k, seeds in g.items():
        out[k] = {}
        for sd, cells in sorted(seeds.items()):
            ep = {e: float(np.mean([c["s10ep"][e] for c in cells])) for e in cells[0]["s10ep"]}
            mp = {n: np.mean([np.asarray(c["maps"][n], float) for c in cells], 0) for n in ("tau_hat", "ATE", "imb")}
            out[k][sd] = {"ep": ep, "maps": mp, "n_cells": len(cells),
                          "seed_fits": sorted(c["cfg"]["seed_fit"] for c in cells)}
    return out


def _s10_stat(G: dict, ep: str, seeds=None) -> dict:
    x = np.array([v["ep"][ep] for sd, v in sorted(G.items()) if (seeds is None or sd in seeds)
                  and np.isfinite(v["ep"].get(ep, np.nan))])
    lo, hi = boot_ci(x)
    return {"n": int(len(x)), "mean": float(x.mean()) if len(x) else float("nan"), "ci": [lo, hi],
            "per_seed": x.tolist()}


def s10_clauses(T: dict) -> tuple[list[dict], bool]:
    """Amendment A5 pass criteria, evaluated EXACTLY as written. ``T`` = key -> endpoint -> stat."""
    get = lambda k, ep: T.get(k, {}).get(ep, {}).get("mean", float("nan"))
    main = {(bs, i): ("gff_shift", bs, i, 1.2, False) for bs in S10_TRUTHS for i in S10_INITS}
    cl = []
    # (i) every (truth, init) at ps 1.2: |seed-mean disc bias| <= 0.03 AND seed-mean rho <= 0.10
    det, ok = [], True
    for (bs, i), k in main.items():
        b, rho = get(k, "Etau_disc_mean"), get(k, "rho")
        good = bool(np.isfinite(b) and np.isfinite(rho) and abs(b) <= 0.03 and rho <= 0.10)
        ok &= good
        det.append(f"truth {bs:+g} init {i}: disc bias {b:+.4f}, rho {rho:+.4f} -> {'ok' if good else 'FAILS'}")
    cl.append({"clause": "(i) every (truth, init) at ps_slope 1.2: |seed-mean disc bias| <= 0.03 AND seed-mean rho <= 0.10",
               "pass": ok, "detail": det})
    # (ii) per truth: max pairwise difference of seed-mean disc tau_hat over the three inits <= 0.02
    det, ok = [], True
    for bs in S10_TRUTHS:
        v = [get(main[(bs, i)], "tauhat_disc_mean") for i in S10_INITS]
        d = float(max(abs(a - b) for a in v for b in v)) if all(np.isfinite(v)) else float("nan")
        good = bool(np.isfinite(d) and d <= 0.02)
        ok &= good
        det.append(f"truth {bs:+g}: disc tau_hat " + ", ".join(f"{i} {x:+.4f}" for i, x in zip(S10_INITS, v))
                   + f"; max pairwise diff {d:.4f} -> {'ok' if good else 'FAILS'}")
    cl.append({"clause": "(ii) per truth, max pairwise init difference in seed-mean disc tau_hat <= 0.02", "pass": ok,
               "detail": det})
    # (iii) naive start: seed-mean rho <= 0.10 per truth
    det, ok = [], True
    for bs in S10_TRUTHS:
        rho = get(main[(bs, "naive")], "rho")
        good = bool(np.isfinite(rho) and rho <= 0.10)
        ok &= good
        det.append(f"truth {bs:+g} init naive: rho {rho:+.4f} -> {'ok' if good else 'FAILS'}")
    cl.append({"clause": "(iii) from the naive start, seed-mean rho <= 0.10 for each truth", "pass": ok, "detail": det})
    # (iv) both anchors: seed-mean rho >= 0.80
    det, ok = [], True
    for k in (S10_ANCHOR_A, S10_ANCHOR_B):
        rho = get(k, "rho")
        good = bool(np.isfinite(rho) and rho >= 0.80)
        ok &= good
        det.append(f"{s10_name(k)}: rho {rho:+.4f} -> {'ok' if good else 'FAILS'}")
    cl.append({"clause": "(iv) both anchors return the naive answer: seed-mean rho >= 0.80", "pass": ok, "detail": det})
    return cl, all(c["pass"] for c in cl)


def _fr_endpoints(froot: str) -> dict:
    """(preset, seed_data) -> S10 endpoints of the frengression cell (halo_s10.json from halo_frengression post)."""
    out = {}
    for f in glob.glob(os.path.join(froot, "fr_*", "halo_s10.json")):
        j = json.load(open(f))
        out[(j["preset"], j["seed_data"])] = {**j["endpoints"], "identical": j["dataset_check"]["identical"],
                                              "tau_hat": np.asarray(j["tau_hat"])}
    return out


def s10_frengression(froot: str, comp: dict) -> dict:
    """Exploratory paired block: frengression - comparator by seed_data, comparators seed_fit-averaged."""
    fr = _fr_endpoints(froot)
    chk = {}
    cf = os.path.join(froot, "_dataset_check.json")
    if os.path.exists(cf):
        c = json.load(open(cf))
        chk = {"n": c["n"], "n_not_identical_or_missing": c["n_not_identical_or_missing"]}
    rows = []
    for e, eps in (("E2", ("ate_mae", "Etau_disc_mean", "rho")), ("E1", ("ate_mae",))):
        for name, G in comp.get(e, {}).items():
            for ep in eps:
                f = {sd: v[ep] for (p, sd), v in fr.items() if p == e}
                o = {sd: v["ep"][ep] for sd, v in G.items()}
                sds = sorted(set(f) & set(o))
                t = _test([f[sd] - o[sd] for sd in sds])
                rows.append({"preset": e, "endpoint": ep, "comparator": name, "n": t["n"], "seeds": sds,
                             "fr_mean": float(np.mean([f[s] for s in sds])) if sds else float("nan"),
                             "comp_mean": float(np.mean([o[s] for s in sds])) if sds else float("nan"),
                             "diff_mean": t["mean"], "ci": t["ci"], "p": t["p"]})
    levels = {e: {ep: _s10_stat({sd: {"ep": v} for (p, sd), v in fr.items() if p == e}, ep)
                  for ep in ("ate_mae", "Etau_disc_mean", "rho", "Etau_active_off_mean")} for e in ("E2", "E1")}
    return {"dataset_check": chk, "rows": rows, "fr_levels": levels, "n_cells": len(fr),
            "all_identical": bool(fr) and all(v["identical"] for v in fr.values())}


def s10_figure(G: dict, path: str) -> None:
    keys = {(bs, i): ("gff_shift", bs, i, 1.2, False) for bs in S10_TRUTHS for i in S10_INITS}
    seedmean = lambda k, n: np.mean([v["maps"][n] for v in G[k].values()], 0) if k in G else None
    fig = plt.figure(figsize=(11, 17), layout="constrained")
    gsp = fig.add_gridspec(7, 5)
    lev = err = None
    for b, bs in enumerate(S10_TRUTHS):
        k0 = next((keys[(bs, i)] for i in S10_INITS if keys[(bs, i)] in G), None)
        truth = seedmean(k0, "ATE") if k0 else None
        naive = (seedmean(k0, "imb") + truth) if k0 else None
        panels = [("truth", truth), ("naive (T=1 - T=0)", naive)] + \
                 [(f"tau_hat init {i}", seedmean(keys[(bs, i)], "tau_hat")) for i in S10_INITS]
        for j, (t, v) in enumerate(panels):
            ax = fig.add_subplot(gsp[2 * b, j])
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"truth {bs:+g}: {t}", fontsize=7)
            if v is not None:
                lev = ax.imshow(v.reshape(8, 8), cmap="RdBu_r", vmin=-1.6, vmax=1.6)
        for j in range(5):
            ax = fig.add_subplot(gsp[2 * b + 1, j])
            ax.set_xticks([]), ax.set_yticks([])
            if j == 0:
                ax.axis("off")
                continue
            if j == 1:
                ax.set_title(f"naive - truth (±0.8)", fontsize=7)
                if naive is not None:
                    ax.imshow((naive - truth).reshape(8, 8), cmap="RdBu_r", vmin=-0.8, vmax=0.8)
                continue
            i = S10_INITS[j - 2]
            v = seedmean(keys[(bs, i)], "tau_hat")
            ax.set_title(f"tau_hat - truth, init {i} (±0.15)", fontsize=7)
            if v is not None:
                err = ax.imshow((v - truth).reshape(8, 8), cmap="RdBu_r", vmin=-0.15, vmax=0.15)
    for j, k in enumerate((S10_ANCHOR_A, S10_ANCHOR_B)):
        ax = fig.add_subplot(gsp[6, j])
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f"{s10_name(k)}\ntau_hat - truth (±0.8)", fontsize=7)
        if k in G:
            ax.imshow((seedmean(k, "tau_hat") - seedmean(k, "ATE")).reshape(8, 8), cmap="RdBu_r", vmin=-0.8, vmax=0.8)
    ax = fig.add_subplot(gsp[6, 2:])
    ks = sorted(G, key=lambda k: (k[4], k[0] != "gff_shift", k[3], -k[1], S10_INITS.index(k[2])))
    ms, lo, hi = [], [], []
    for k in ks:
        st = _s10_stat(G[k], "rho")
        ms.append(st["mean"]), lo.append(st["mean"] - st["ci"][0]), hi.append(st["ci"][1] - st["mean"])
    ax.bar(range(len(ks)), ms, yerr=[np.nan_to_num(lo), np.nan_to_num(hi)], capsize=2,
           color=["C3" if k in (S10_ANCHOR_A, S10_ANCHOR_B) else "C1" if k[3] != 1.2 else "C0" for k in ks])
    for y in (0.10, 0.80):
        ax.axhline(y, color="0.4", lw=0.6, ls="--")
    ax.axhline(0, color="0.2", lw=0.6)
    ax.set_xticks(range(len(ks)), [s10_name(k).replace("gff_shift ", "").replace(" ps 1.2", "") for k in ks],
                  rotation=70, fontsize=5.5, ha="right")
    ax.set_ylabel("rho (0 = truth, 1 = naive)", fontsize=7)
    ax.set_title("retained-confounding fraction, seed mean ± 95% bootstrap CI (dashed: 0.10, 0.80)", fontsize=7)
    if lev is not None:
        fig.colorbar(lev, ax=fig.axes[:5], location="right", shrink=0.6, label="logit (±1.6)")
    fig.suptitle("S10 (Amendment A5): is the confounded recovery genuine? GFF-shift, E2, P1, seeds 31-35 x sf41", fontsize=9)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def analyse_s10(root: str, croot: str) -> dict:
    recs = load_stage(root, "S10")
    s9 = load_stage(croot, "S9")
    reuse = [r for r in s9 if label(r["cfg"]) == "gff_shift/P1/bs1/E2" and r["cfg"]["seed_fit"] == 41
             and r["cfg"]["seed_data"] in S10_SEEDS]
    check_flags(recs + reuse)
    out = {"stage": "S10", "n_cells": len(recs), "n_reused_s9": len(reuse),
           "reused_s9": [r["run_id"] for r in reuse],
           "excluded": [r["run_id"] for r in recs + reuse if excluded(r)],
           "diverged": [r["run_id"] for r in recs + reuse if r["met"].get("diverged")]}
    try:
        out["n_expected"] = json.load(open(os.path.join(root, "S10", "_stage.json")))["n_cells"]
    except Exception:
        out["n_expected"] = None
    out["partial"] = out["n_expected"] is not None and len(recs) < out["n_expected"]
    G = _s10_group(recs + reuse)
    T = {k: {ep: _s10_stat(G[k], ep) for ep in S10_EP + ("epochs_run", "best_epoch")} for k in G}
    for k in G:
        T[k]["epochs_run_median"] = float(np.median([c for v in G[k].values() for c in [v["ep"]["epochs_run"]]]))
        T[k]["best_epoch_median"] = float(np.median([v["ep"]["best_epoch"] for v in G[k].values()]))
    out["consistency_stored_vs_recomputed_maxabs"] = float(max(
        [r["s10ep"].get("stored_vs_recomputed_maxabs", 0.0) for r in recs if "s10ep" in r] or [float("nan")]))
    out["table"] = {s10_name(k): {"key": list(k), **T[k]} for k in sorted(G, key=str)}
    out["clauses"], out["verdict_pass"] = s10_clauses(T)
    # frengression paired block (exploratory): comparators from S9 (both seed_fits) and S4 ff_full/P1
    comp_recs = [r for r in s9 if not r["cfg"].get("paper_setting") and r["cfg"]["arm"] in ("gff_shift", "gff_flex")]
    comp_recs += [r for r in load_stage(croot, "S4") if label(r["cfg"]) in ("ff_full/P1/bs1/E2", "ff_full/P1/bs1/E1")]
    check_flags(comp_recs)
    comp: dict = {"E2": {}, "E1": {}}
    for name, arm in (("GFF-shift (S9)", "gff_shift"), ("GFF-flex (S9)", "gff_flex"), ("FF-uniform (S4 ff_full/P1)", "ff_full")):
        for e in ("E2", "E1"):
            rr = [r for r in comp_recs if r["cfg"]["arm"] == arm and r["cfg"]["preset"] == e]
            gg = _s10_group(rr)
            if gg:
                comp[e][name] = next(iter(gg.values())) if len(gg) == 1 else {}
    out["frengression"] = s10_frengression(os.path.join(root, "S10", "frengression"), comp)
    adir = os.path.join(root, "_analysis")
    os.makedirs(adir, exist_ok=True)
    s10_figure(G, os.path.join(adir, "S10_genuine.png"))
    json.dump(out, open(os.path.join(adir, "S10_tables.json"), "w"), indent=1, default=float)
    open(os.path.join(adir, "S10_tables.md"), "w").write(s10_md(out))
    print(s10_md(out))
    return out


def s10_md(out: dict) -> str:
    f = lambda st: (f"{st['mean']:+.4f} [{st['ci'][0]:+.4f}, {st['ci'][1]:+.4f}]" if st["n"] else "n/a")
    L = [f"# S10 tables (Amendment A5)" + (f" — PARTIAL ({out['n_cells']} of {out['n_expected']} cells)"
                                            if out.get("partial") else ""), "",
         f"S10 cells: {out['n_cells']} (expected {out['n_expected']}); reused S9 gff_shift E2 sf41 cells "
         f"(truth +1, init zero): {out['n_reused_s9']}; excluded: {out['excluded']}; diverged: {out['diverged']}; "
         f"stored-vs-recomputed endpoint max |diff|: {out['consistency_stored_vs_recomputed_maxabs']:.2e}", "",
         f"## Pass criteria (Amendment A5, evaluated as written) — OVERALL: "
         f"{'PASS' if out['verdict_pass'] else 'FAIL'}", ""]
    for c in out["clauses"]:
        L.append(f"- **{'PASS' if c['pass'] else 'FAIL'}** {c['clause']}")
        L += [f"    - {d}" for d in c["detail"]]
    L += ["", "## Per configuration: seed mean [95% bootstrap CI over seed_data]", "",
          "| config | n_sd | disc tau_hat | disc bias | rho | ATE MAE | active-off E_tau | quiet E_tau | "
          "dist from init | dist from truth | dist naive-truth | epochs run (median) | best epoch (median) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, t in out["table"].items():
        L.append(f"| {name} | {t['rho']['n']} | " + " | ".join(f(t[e]) for e in S10_EP)
                 + f" | {t['epochs_run_median']:.0f} | {t['best_epoch_median']:.0f} |")
    fr = out["frengression"]
    L += ["", "## Paired frengression block (EXPLORATORY; frengression - comparator, paired by seed_data)", "",
          f"dataset identity: {fr['dataset_check']}; all frengression datasets identical to the halo build: "
          f"{fr['all_identical']} ({fr['n_cells']} cells)", "",
          "| preset | endpoint | comparator | n | frengression | comparator | diff | 95% CI | Wilcoxon p |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in fr["rows"]:
        L.append(f"| {r['preset']} | {r['endpoint']} | {r['comparator']} | {r['n']} | {r['fr_mean']:+.4f} | "
                 f"{r['comp_mean']:+.4f} | {r['diff_mean']:+.4f} | [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] | {r['p']:.4f} |")
    L += ["", "frengression levels: " + json.dumps({e: {k: round(v["mean"], 4) for k, v in d.items()}
                                                    for e, d in fr["fr_levels"].items()})]
    return "\n".join(L) + "\n"


def analyse(stage: str, root: str, compare_root: str | None = None) -> dict:
    croot = compare_root or root
    if stage == "S10":                              # Amendment A5: own grouping (shift_init / placebo keys)
        return analyse_s10(root, croot)
    recs = load_stage(root, stage)
    if not recs:
        print(f"{stage}: no complete cells under {root}")
        return {}
    extra = load_stage(croot, "S1") + load_stage(croot, "S2") if stage == "S3" else []
    if stage == "S5":                               # comparison arms from S1/S2/S4 (Amendment A1)
        extra = [r for st in ("S1", "S2", "S4") for r in load_stage(croot, st) if label(r["cfg"]) in S5_COMPARE]
    if stage == "S6":                               # comparison arms U, A2, LT(P0) from S2 (Amendment A2)
        extra = [r for r in load_stage(croot, "S2") if label(r["cfg"]) in S6_COMPARE]
    if stage == "S7":                               # (W50, new) comparators = S4 ff_full/P0 (Amendment A3)
        extra = s7_relabel_s4(load_stage(croot, "S4"))
    if stage == "S9":                               # FF-uniform = S4 ff_full/P1; S7 paper new-rank (Amendment A4)
        extra = [r for r in load_stage(croot, "S4") if label(r["cfg"]) in {S9_FF.format(e=e) for e in ("E2", "E1")}]
        extra += [r for r in load_stage(croot, "S7") if label(r["cfg"]) == S9_S7_PAPER]
    check_flags(recs + extra)
    out = {"stage": stage, "n_cells": len(recs),
           "excluded": [r["run_id"] for r in recs if excluded(r)],
           "diverged": [r["run_id"] for r in recs if r["met"].get("diverged")],
           "quiet_flagged": [r["run_id"] for r in recs if r["met"].get("quiet_flag")]}
    try:
        expected = json.load(open(os.path.join(root, stage, "_stage.json")))["n_cells"]
    except Exception:
        expected = None
    out["n_expected"] = expected
    out["partial"] = expected is not None and len(recs) < expected
    groups = by_config(recs)
    allg = {**by_config(extra), **groups}
    adir = os.path.join(root, "_analysis")
    os.makedirs(adir, exist_ok=True)
    if not groups:                                  # every cell excluded: report, no figures
        json.dump(out, open(os.path.join(adir, f"{stage}_tables.json"), "w"), indent=1, default=float)
        open(os.path.join(adir, f"{stage}_tables.md"), "w").write(
            f"# {stage} tables\n\nALL {len(recs)} cells excluded. diverged: {out['diverged']}; "
            f"excluded: {out['excluded']}\n")
        print(f"{stage}: all {len(recs)} cells excluded")
        return out
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
    elif stage == "S5":
        tmap = "E_mu0"
        out["compare_root"] = croot
        out["compare_cells"] = {L: sum(len(sv["cells"]) for sv in G["seed"].values())
                                for L, G in allg.items() if L not in groups}
        r5, a5 = s5_rows(allg, croot)
        rows += r5
        attr += a5
        out["endpoints"] = {L: {sd: {k: v for k, v in g["ep"].items() if not k.startswith("xt:")}
                                for sd, g in G["seed"].items()} for L, G in allg.items()}
        out["p5_fit"] = {L: {k: float(np.mean([c["met"].get("preproc", {}).get(k, np.nan)
                                                for sv in G["seed"].values() for c in sv["cells"]]))
                             for k in ("floor_value", "y_sd_global", "n_floored")}
                         for L, G in groups.items()}
        groups = {L: allg[L] for L in S5_PANEL if L in allg}      # figure + templates: the comparison rows
    elif stage == "S6":
        tmap = "E_tau"
        out["compare_root"] = croot
        out["compare_cells"] = {L: sum(len(sv["cells"]) for sv in G["seed"].values())
                                for L, G in allg.items() if L not in groups}
        r6, gates, preds = s6_rows(allg)
        rows += r6
        attr += gates
        out["predictions"] = preds
        out["naive_ate_mae"] = _naive_s6(croot)
        out["endpoints"] = {L: {sd: {k: v for k, v in g["ep"].items() if not k.startswith("xt:")}
                                for sd, g in G["seed"].items()} for L, G in allg.items()}
        out["lt_ate_vs_sampled_meanabs"] = {L: float(np.mean([c["met"]["lt_ate_minus_crn_meanabs"]
                                                              for sv in G["seed"].values() for c in sv["cells"]]))
                                            for L, G in allg.items()
                                            if all("lt_ate_minus_crn_meanabs" in c["met"]
                                                   for sv in G["seed"].values() for c in sv["cells"])}
        s6_figure(allg, out["naive_ate_mae"], os.path.join(adir, "S6_ate_mae.png"))
        groups = {S6_ARMS[k]: allg[S6_ARMS[k]] for k in S6_ARMS if S6_ARMS[k] in allg}
    elif stage == "S7":
        tmap = "E_tau"
        out["compare_root"] = croot
        out["compare_cells"] = {L: sum(len(sv["cells"]) for sv in G["seed"].values())
                                for L, G in allg.items() if L not in groups}
        r7, gates, preds = s7_rows(allg)
        rows += r7
        attr += gates
        out["predictions"] = preds
        out["predictions_source"] = "Amendment A3"
        out["fit_summary"] = s7_fit_summary(allg)
        cf = os.path.join(root, "S7", "connectivity.json")
        if os.path.exists(cf):
            out["connectivity"] = json.load(open(cf))["rows"]
        out["endpoints"] = {L: {sd: {k: v for k, v in g["ep"].items() if not k.startswith("xt:")}
                                for sd, g in G["seed"].items()} for L, G in allg.items()}
        s7_figure(allg, os.path.join(adir, "S7_rankfix.png"))
        groups = {L: allg[L] for L in [s7_label("E2", r, w) for w in (50, 16) for r in ("old", "new")] if L in allg}
    elif stage == "S8":
        tmap = "E_tau"
        out["calibration"] = s8_summary(groups)
        out["predictions"] = ["S8 is descriptive (Amendment A4): no gate, no prediction."]
        out["predictions_source"] = "Amendment A4"
    elif stage == "S9":
        tmap = "E_tau"
        out["compare_root"] = croot
        out["compare_cells"] = {L: sum(len(sv["cells"]) for sv in G["seed"].values())
                                for L, G in allg.items() if L not in groups}
        r9, gates, preds = s9_rows(allg)
        rows += r9
        attr += gates
        out["predictions"] = preds
        out["predictions_source"] = "Amendment A4"
        out["calibration"] = s9_calibration(allg)
        out["endpoints"] = {L: {sd: {k: v for k, v in g["ep"].items() if not k.startswith("xt:")}
                                for sd, g in G["seed"].items()} for L, G in allg.items()}
        s9_figure(allg, os.path.join(adir, "S9_gaussian.png"))
        groups = {L: allg[L] for L in [s9_label(n, e) for e in ("E2", "E1") for n in S9_NAMES] if L in allg}
    out["contrasts"], out["attribution"] = rows, attr
    out["templates_descriptive"] = {m: template_table(groups, m) for m in ("E_mu0", "E_sd0", "E_tau")}
    out["template_corr_mean"] = {L: np.mean([c["met"]["template_corr"] for s in G["seed"].values()
                                            for c in s["cells"]], 0).round(3).tolist() for L, G in groups.items()}
    panel(groups, croot, os.path.join(adir, f"{stage}_maps.png"), stage)
    template_bars(out["templates_descriptive"][tmap], tmap, os.path.join(adir, f"{stage}_templates.png"),
                  f"{stage}: DESCRIPTIVE template coefficients of {tmap} (mean, 95% bootstrap CI over seed_data)")
    json.dump(out, open(os.path.join(adir, f"{stage}_tables.json"), "w"), indent=1, default=float)
    open(os.path.join(adir, f"{stage}_tables.md"), "w").write(to_md(out, tmap))
    print(f"{stage}: {len(recs)} cells, {len(groups)} configs -> {adir}/{stage}_*")
    return out


def to_md(out: dict, tmap: str) -> str:
    L = [f"# {out['stage']} tables" + (f" — PARTIAL ({out['n_cells']} of {out['n_expected']} cells)"
                                         if out.get("partial") else ""), "",
         f"cells: {out['n_cells']}; diverged (max |E_mu| > 10): {len(out['diverged'])} {out['diverged']}; "
         f"excluded (>0.1% non-finite/clamped, or diverged): {len(out['excluded'])} {out['excluded']}; Corpus-A quiet-count flagged: {len(out['quiet_flagged'])}", "",
         "## Gate outcomes and attribution (positive findings only)", ""]
    L += [f"- {a}" for a in out["attribution"]] or ["- (none)"]
    if out.get("predictions"):
        L += ["", f"## Predictions stated before running ({out.get('predictions_source', 'Amendment A2')}) beside their outcomes", "",
              "Descriptive only: the gate lines above are the only resolved/unresolved statements.", ""]
        L += [f"- {p}" for p in out["predictions"]]
    if out.get("naive_ate_mae"):
        nv = out["naive_ate_mae"]
        L += ["", f"Naive-difference ATE MAE (S0 imb_E1_bs1, mean |.| over 64 px): {nv['mean']:.4f} "
              f"[{nv['ci'][0]:.4f}, {nv['ci'][1]:.4f}] over {len(nv['per_seed'])} seed_data"]
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
    for k in ("cross_table", "template_corr_mean", "variance_split", "Emu0_corr_across_tau",
              "lt_ate_vs_sampled_meanabs", "compare_cells", "fit_summary", "connectivity", "calibration"):
        if k in out:
            L += ["", f"## {k}", "", "```", json.dumps(out[k], indent=1, default=float), "```"]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "all"])
    ap.add_argument("--runs-root", default=os.path.expanduser("~/work/halo-runs"))
    ap.add_argument("--compare-root", default=None,
                    help="root holding S0/S1/S2/S4 comparison cells (default: --runs-root); S7 reads S4 from it")
    a = ap.parse_args(argv)
    cr = os.path.expanduser(a.compare_root) if a.compare_root else None
    for s in (["S1", "S2", "S3", "S4"] if a.stage == "all" else [a.stage]):
        analyse(s, os.path.expanduser(a.runs_root), cr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
