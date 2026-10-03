"""Figures and tables for the paper's MorphoMNIST experiments, one topic at a time.

    python scripts/paper/make_paper_outputs.py setup      # experimental set-up (appendix)

Each topic writes PDFs, LaTeX snippets and a compiled preview to runs/paper/<topic>/. Nothing is
written into the paper repository; the snippets are copied there by hand once checked.

Topic "setup" (no fits needed; everything comes from the data generator):
  table_presets.tex       one row per preset: assignment rule and the individual effect
  fig_assignment.pdf      thickness of treated and untreated units (E1; E2-E6, which share one
                          assignment) and the assignment probability against thickness
  fig_treated_<s>x<s>.pdf one figure per resolution: five units along the thickness range, untreated
                          and treated under E2 and under E6, as pixel intensities
  fig_truth_<s>x<s>.pdf   one figure per resolution, one row per preset: true ATE, average individual
                          effect of the thinnest and of the thickest 10 % of units, naive estimate, and
                          naive estimate minus true ATE (dataset 1, named in the caption)
Datasets: all ten digits (n = 60000), seed_data 101, assignment seeds 1-10, as in the 8x8 paper grid
(scripts/exp_ate_recovery/grid_8x8_alldigits_v2.sh); the same seeds at 16x16. Units, potential
outcomes and individual effects do not depend on the assignment draw; only the assignment, and
so the observed images and the naive estimate, differ.
"""
import argparse
import os
import shutil
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

MM = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, MM)
import exp_ate_recovery as E  # noqa: E402
import prepare_morphomnist_exps as P  # noqa: E402

OUT = os.path.join(MM, "runs", "paper")

SIZES, SEED_DATA, ASSIGN_SEEDS, SHOW_DATASET = (8, 16), 101, range(1, 11), 1
PRESETS = ["exp1_rct_homogeneous", "exp2_confounded_homogeneous", "exp3_confounded_heterogeneous",
           "exp4_covariate_cate", "exp5_quantile_effect", "exp6_spatial_cate"]
LABEL = {p: f"E{i + 1}" for i, p in enumerate(PRESETS)}
TAIL = 0.10                     # "thin" / "thick" = the lowest / highest 10 % of thickness

TEXTWIDTH = 6.75                # AISTATS \textwidth, inches (aistats2027.sty)
plt.rcParams.update({
    "text.usetex": True, "font.family": "serif", "font.size": 8, "axes.titlesize": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "text.latex.preamble": r"\usepackage{amsmath,amssymb}", "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def build(preset, size, seed_assign):
    cfg = E.Config(preset=preset, size=size, digit=None, seed_data=SEED_DATA, seed_assign=seed_assign,
                   arm="flexible_continuous")
    return E.build_data(cfg)


def naive(d):
    T = np.asarray(d["X"])[:, 0].astype(bool)
    Y = np.asarray(d["Y"])
    return Y[T].mean(0) - Y[~T].mean(0)


def disc_mask(size):
    return E.region_masks(size, E.Config(size=size).effective_radius)[0].reshape(size, size)


def outline_disc(ax, D):
    """Black outline around the effect's support (the disc), D a (size, size) boolean mask."""
    size = D.shape[0]
    for i in range(size):
        for j in range(size):
            if not D[i, j]:
                continue
            for di, dj, xy, w, h in ((-1, 0, (j - .5, i - .5), 1, 0), (1, 0, (j - .5, i + .5), 1, 0),
                                     (0, -1, (j - .5, i - .5), 0, 1), (0, 1, (j + .5, i - .5), 0, 1)):
                ii, jj = i + di, j + dj
                if not (0 <= ii < size and 0 <= jj < size and D[ii, jj]):
                    ax.add_patch(Rectangle(xy, w, h, fill=False, lw=0.6, ec="k"))


def bare(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4)


def write_figure_tex(out, name, caption, label, width):
    with open(os.path.join(out, name + ".tex"), "w") as f:
        f.write("\\begin{figure}[t]\n\\centering\n"
                f"\\includegraphics[width={width}]{{figures/experiments/{name}.pdf}}\n"
                f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{figure}}\n")


# --------------------------------------------------------------------------------------------- #
# topic: setup
# --------------------------------------------------------------------------------------------- #
def effect_formula(c):
    """The factor in tau_ik = m_k (1 + ...) for one preset's generator config, as LaTeX."""
    terms = []
    if c.effect_mode == "outcome_coupled":
        if c.a_cov:
            terms.append(f"{c.a_cov:g}\\,h_i")
        if c.b_quant:
            terms.append(f"{c.b_quant:g}\\,g_{{ik}}")
    elif c.effect_mode == "covariate_only":
        cov = f"({c.a_cov:g} + {c.a_spatial:g}\\,\\psi_k)" if c.a_spatial else f"{c.a_cov:g}"
        terms += [f"{cov}\\,h_i", f"{c.a_bright:g}\\,b_i", f"{c.a_inter:g}\\,(hb)_i"]
    else:                                                       # quantile_primitive
        terms.append(f"{c.b_quant:g}\\,g_{{ik}}")
    return "$m_k$" if not terms else "$m_k\\,(1 + " + " + ".join(terms) + ")$"


def assignment_text(c):
    if not c.ps_slope:
        return "random, $\\tfrac12$"
    icpt = f"{c.ps_intercept:g} + " if c.ps_intercept else ""
    return f"$\\sigma({icpt}{c.ps_slope:g}\\,\\tilde t_i)$"


def truth_figure(out, size, data):
    """One row per preset: true ATE, thinnest / thickest 10 %, naive, naive - ATE; one colour scale."""
    D = disc_mask(size)
    cols = ["True ATE", "Thinnest\n10\\,\\%", "Thickest\n10\\,\\%", "Naive\nestimate", "Naive $-$\ntrue ATE"]
    maps = {}
    for p in PRESETS:
        d = data[p]
        ite, t, ate = np.asarray(d["ITE"]), d["THICKNESS"], np.asarray(d["ATE"])
        nv = naive(d)
        maps[p] = [ate, ite[t <= np.quantile(t, TAIL)].mean(0), ite[t >= np.quantile(t, 1 - TAIL)].mean(0),
                   nv, nv - ate]
    vmax = max(np.abs(np.asarray(m)).max() for p in PRESETS for m in maps[p])
    w = TEXTWIDTH * 0.62
    fig, axes = plt.subplots(len(PRESETS), 5, figsize=(w, w * 6 / 5 * 1.02),
                             gridspec_kw=dict(wspace=0.06, hspace=0.06))
    for r, p in enumerate(PRESETS):
        for j in range(5):
            ax = axes[r, j]
            h = ax.imshow(np.asarray(maps[p][j]).reshape(size, size), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            outline_disc(ax, D); bare(ax)
            if r == 0:
                ax.set_title(cols[j], fontsize=7, pad=3)
            if j == 0:
                ax.set_ylabel(LABEL[p], rotation=0, ha="right", va="center")
    b0, b4 = axes[-1, 0].get_position(), axes[-1, 4].get_position()
    cax = fig.add_axes([b0.x0, b0.y0 - 0.035, b4.x1 - b0.x0, 0.012])
    fig.colorbar(h, cax=cax, orientation="horizontal").ax.tick_params(labelsize=6)
    name = f"fig_truth_{size}x{size}"
    fig.savefig(os.path.join(out, name + ".pdf")); plt.close(fig)
    cap = (f"True and naive effects at ${size}\\times{size}$ for the six presets (rows), on the logit scale "
           r"and one colour scale. Columns 1--3: the true ATE (the disc, black outline) and the average "
           r"individual effect of the 10\,\% thinnest and of the 10\,\% thickest units. Columns 4--5: the "
           r"naive estimate (mean treated image minus mean untreated image) and its difference from the true "
           r"ATE, from a single simulated dataset (one draw of the treatment assignment).")
    write_figure_tex(out, name, cap, f"fig:truth{size}", "0.62\\textwidth")
    return name


def assignment_figure(out, data):
    """Thickness of treated and untreated units (E1; E2-E6, which share one assignment) and the
    assignment probability against thickness. Resolution-independent: assignment uses thickness only."""
    from scipy.stats import gaussian_kde
    e1, e2 = data["exp1_rct_homogeneous"], data["exp2_confounded_homogeneous"]
    for p in PRESETS[2:]:
        assert np.array_equal(np.asarray(data[p]["X"]), np.asarray(e2["X"])), f"{p} assignment differs from E2"
    t = e2["THICKNESS"]
    grid = np.linspace(np.quantile(t, 0.001), np.quantile(t, 0.999), 300)
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 1.65), gridspec_kw=dict(wspace=0.3))
    for ax, d, title in ((axes[0], e1, "(a) E1: random assignment"), (axes[1], e2, "(b) E2--E6: confounded")):
        T = np.asarray(d["X"])[:, 0].astype(bool)
        for sel, lab, col in ((~T, "untreated", "C0"), (T, "treated", "C3")):
            ax.plot(grid, gaussian_kde(t[sel])(grid), color=col, lw=1, label=lab)
            ax.axvline(t[sel].mean(), color=col, lw=0.6, ls="--")
        ax.set_title(title, pad=3); ax.set_xlabel("thickness $t_i$"); ax.set_ylabel("density")
        ax.legend(frameon=False, loc="upper right")
    z = (grid - t.mean()) / t.std()
    c2, c1 = P.PRESETS["exp2_confounded_homogeneous"], P.PRESETS["exp1_rct_homogeneous"]
    axes[2].plot(grid, 1 / (1 + np.exp(-(c2.ps_intercept + c2.ps_slope * z))), color="k", lw=1, label="E2--E6")
    axes[2].plot(grid, np.full_like(grid, 1 / (1 + np.exp(-c1.ps_intercept))), color="k", lw=1, ls=":", label="E1")
    axes[2].set_ylim(0, 1); axes[2].set_title("(c) assignment probability", pad=3)
    axes[2].set_xlabel("thickness $t_i$"); axes[2].set_ylabel("$P(T_i = 1 \\mid t_i)$")
    axes[2].legend(frameon=False, loc="upper left")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(os.path.join(out, "fig_assignment.pdf")); plt.close(fig)
    cap = (r"Treatment assignment. (a, b) Density of thickness $t_i$ (MorphoMNIST's thickness attribute, "
           r"rescaled to $[-1, 1]$) among untreated and treated units, kernel density estimates over the "
           r"$n = 60\,000$ units, for one draw of the treatment assignment; dashed lines are the group "
           r"means. E2--E6 assign treatment by the same rule, so one panel covers all five. (c) The assignment probability: $\tfrac12$ in E1 and "
           f"$\\sigma({c2.ps_slope:g}\\,\\tilde t_i)$ in E2--E6, with $\\tilde t_i$ the standardised thickness. "
           r"Thickness is the only variable that enters the assignment, so it is the only confounder; "
           r"assignment does not depend on the image resolution.")
    write_figure_tex(out, "fig_assignment", cap, "fig:assignment", "\\textwidth")
    return "fig_assignment"


def treated_figure(out, size, data):
    """Five units along the thickness range: untreated image, and the same unit treated under E2 and
    under E6, as pixel intensities in [0, 1]."""
    from prepare_data import inverse_logit
    D = disc_mask(size)
    e2, e6 = data["exp2_confounded_homogeneous"], data["exp6_spatial_cate"]
    assert np.array_equal(np.asarray(e2["Y0"]), np.asarray(e6["Y0"])), "E2 and E6 untreated images differ"
    t = e2["THICKNESS"]
    qs = (0.1, 0.3, 0.5, 0.7, 0.9)
    units = [int(np.argmin(np.abs(t - np.quantile(t, q)))) for q in qs]
    rows = [("Untreated\n$Y_i(0)$", e2["Y0"]), ("Treated, E2\n$Y_i(1)$", e2["Y1"]),
            ("Treated, E6\n$Y_i(1)$", e6["Y1"])]
    w = TEXTWIDTH * 0.62
    fig, axes = plt.subplots(len(rows), len(units), figsize=(w, w * len(rows) / len(units) * 1.04),
                             gridspec_kw=dict(wspace=0.06, hspace=0.06))
    for r, (lab, Y) in enumerate(rows):
        pix = inverse_logit(np.asarray(Y)[units])
        for j, u in enumerate(units):
            ax = axes[r, j]
            h = ax.imshow(pix[j].reshape(size, size), cmap="gray", vmin=0, vmax=1)
            bare(ax)
            if r > 0:
                for patch in list(ax.patches):
                    patch.remove()
                outline_disc(ax, D)
                for patch in ax.patches:
                    patch.set_edgecolor("C3"); patch.set_linewidth(0.5)
            if r == 0:
                ax.set_title(f"{int(qs[j] * 100)}th pct.\n$t_i = {t[u]:+.2f}$", fontsize=7, pad=3)
            if j == 0:
                ax.set_ylabel(lab, fontsize=7)
    b0, b4 = axes[-1, 0].get_position(), axes[-1, -1].get_position()
    cax = fig.add_axes([b0.x0, b0.y0 - 0.05, b4.x1 - b0.x0, 0.018])
    fig.colorbar(h, cax=cax, orientation="horizontal", label="pixel intensity").ax.tick_params(labelsize=6)
    name = f"fig_treated_{size}x{size}"
    fig.savefig(os.path.join(out, name + ".pdf")); plt.close(fig)
    cap = (f"What treatment does to a digit, at ${size}\\times{size}$. Columns: five units at the 10th, "
           r"30th, 50th, 70th and 90th percentile of thickness $t_i$. Row 1: the untreated image. Rows 2--3: "
           r"the same unit treated, under E2 (every unit gets the same effect, the disc) and under E6 (the "
           r"effect grows with thickness and brightness and, for thick digits, is shifted towards the bottom "
           r"of the disc). The red outline marks the disc. Images are shown as pixel intensities. The "
           r"effect is added on the logit scale, so in intensity it is largest for mid-grey pixels and "
           r"smallest for black and nearly white pixels.")
    write_figure_tex(out, name, cap, f"fig:treated{size}", "0.62\\textwidth")
    return name


def setup(out):
    """Describes how the data are generated only; how many datasets are drawn, and the size of the
    confounding in them, belong to the inference topic. The figures use one draw of the assignment."""
    data = {s: {p: build(p, s, SHOW_DATASET) for p in PRESETS} for s in SIZES}

    # ---------------- table ----------------
    gen = {p: P.PRESETS[p] for p in PRESETS}
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\begin{tabular}{lll}", r"\toprule",
             r"Preset & $P(T_i = 1 \mid Z_i)$ & Individual effect $\tau_{ik}$ \\", r"\midrule"]
    for p in PRESETS:
        lines.append(f"{LABEL[p]} & {assignment_text(gen[p])} & {effect_formula(gen[p])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{The six presets. Second column: the probability of treatment "
              r"(Section~\ref{app:setup-assignment}), with $\tilde t_i$ the standardised thickness. Third "
              r"column: the individual effect at pixel $k$, with $m$ the disc and the modulators $h_i$, "
              r"$b_i$, $(hb)_i$, $g_{ik}$ and $\psi_k$ defined in Section~\ref{app:setup-effect}.}",
              r"\label{tab:presets}", r"\end{table}"]
    with open(os.path.join(out, "table_presets.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    return (["table_presets", assignment_figure(out, data[SIZES[0]])]
            + [treated_figure(out, s, data[s]) for s in SIZES]
            + [truth_figure(out, s, data[s]) for s in SIZES])


# --------------------------------------------------------------------------------------------- #
def preview(out, parts):
    """Compile the snippets on a page of AISTATS text width (one column, as in the appendix)."""
    for p in parts:   # snippets reference figures/experiments/<name>.pdf; the preview reads them locally
        s = open(os.path.join(out, p + ".tex")).read().replace("figures/experiments/", "")
        open(os.path.join(out, "_preview_" + p + ".tex"), "w").write(s)
    body = "\n\\clearpage\n".join(f"\\input{{_preview_{p}.tex}}" for p in parts)
    tex = ("\\documentclass{article}\n\\usepackage[paperwidth=8.5in,paperheight=11in,textwidth=6.75in,"
           "textheight=9.25in]{geometry}\n\\usepackage{amsmath,amssymb,booktabs,graphicx}\n"
           "\\newcommand{\\doo}[1]{\\mathrm{do}(#1)}\n"
           "\\begin{document}\n" + body + "\n\\end{document}\n")
    open(os.path.join(out, "preview.tex"), "w").write(tex)
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "preview.tex"], cwd=out,
                       capture_output=True, text=True)
    print("preview:", os.path.join(out, "preview.pdf") if r.returncode == 0 else "pdflatex failed:\n" + r.stdout[-2000:])


# --------------------------------------------------------------------------------------------- #
# topic: inference
# --------------------------------------------------------------------------------------------- #
INF_GRID = {   # resolution -> (presets, assignment seeds), as run by the grid launchers
    8: (PRESETS, range(1, 11)),                                               # grid_8x8_alldigits_v2.sh
    16: (["exp1_rct_homogeneous", "exp2_confounded_homogeneous", "exp4_covariate_cate",
          "exp6_spatial_cate"], range(1, 6)),                                 # grid_16x16_alldigits.sh
}
READOUT_MC = 5000             # paired draws for every effect read-out (both frugal models)
FIT_SEEDS = lambda k: [k, 1001, 1002, 1003, 1004]   # noqa: E731  (IFF; Frengression uses seed k)
BASELINES = [("naive", "Naive difference"), ("ipw", "IPW"), ("ols", "OLS"), ("aipw", "AIPW"),
             ("oracle_ipw", "Oracle IPW")]
METHOD_ORDER = [m for _, m in BASELINES] + ["Frengression", "IFF, single fit", "IFF, 5-fit average"]


def _runs(root, size, pattern):
    """{(preset short, assignment seed, fit seed): run dir} for finished runs with saved weights."""
    import glob, json, re
    rx = re.compile(pattern)
    model = "model.eqx" if root == "exp_ate_recovery" else "model.pt"
    out = {}
    for d in glob.glob(os.path.join(MM, "runs", root, f"*_k{size ** 2}_s*_d0-9_*/")):
        if not (os.path.exists(d + "metrics.json") and os.path.exists(d + model)):
            continue
        m = rx.fullmatch(json.load(open(d + "config.json")).get("wandb_name", ""))
        if m:
            out[(m["p"], int(m["k"]), int(m["s"]))] = d
    return out


def collect_inference(size):
    """One row per (preset, dataset, method) with the scores; plus, per (preset, dataset), the
    effect estimates of dataset SHOW_DATASET for the error-map figure, and completeness counts."""
    import json
    import pandas as pd
    import dataset_store as DS
    K = size ** 2
    ff = _runs("exp_ate_recovery", size,
               rf"ff_(?P<p>e[1-6])_flexcont_sa(?P<k>\d+)_lr0\.001_copw16_k{K}_s(?P<s>\d+)_d0-9_[0-9a-f]{{6}}")
    fr = _runs("frengression", size, rf"frengression_(?P<p>e[1-6])_sa(?P<k>\d+)_k{K}_s(?P<s>\d+)_d0-9_[0-9a-f]{{6}}")
    bidx = pd.read_csv(os.path.join(MM, "runs", "baselines", "index.csv"))
    presets, seeds = INF_GRID[size]
    D = disc_mask(size).ravel()
    disc, ring, far = E.region_masks(size, E.Config(size=size).effective_radius)
    rows, shown, status = [], {}, []
    for p in presets:
        ps = PRESETS.index(p); short = f"e{ps + 1}"
        for k in seeds:
            fits = [ff[(short, k, s)] for s in FIT_SEEDS(k) if (short, k, s) in ff]
            fr_run = fr.get((short, k, k))
            ref = (fits or [fr_run])[0] if (fits or fr_run) else None
            status.append({"preset": LABEL[p], "dataset": k, "iff_fits": len(fits), "frengression": int(fr_run is not None)})
            if ref is None:
                continue
            data = DS.run_arrays(ref, need=("Y",))
            ate = np.asarray(data["ATE"]); T = np.asarray(data["X"])[:, 0].astype(bool); Y = np.asarray(data["Y"])
            imb = Y[T].mean(0) - Y[~T].mean(0) - ate
            did = json.load(open(os.path.join(ref, "config.json")))["dataset_id"]

            def score(tau, method, extra=None):
                err = np.asarray(tau) - ate
                rows.append({"preset": LABEL[p], "dataset": k, "method": method, "mae": np.abs(err).mean(),
                             "disc": err[disc].mean(), "ring": err[ring].mean(), "far": err[far].mean(),
                             "slope": np.polyfit(imb, err, 1)[0], **(extra or {})})
            ests = {}
            b = bidx[(bidx.dataset_id == did) & (bidx.method == "ols")]
            if len(b):
                with np.load(os.path.join(MM, "runs", "baselines", b.iloc[0].run_id, "arrays.npz")) as z:
                    for key, name in BASELINES:
                        ests[name] = np.asarray(z[f"tau_hat_{key}"]); score(ests[name], name)
            if fr_run is not None:
                ests["Frengression"] = DS.effect_map(fr_run, READOUT_MC); score(ests["Frengression"], "Frengression")
            if fits:
                taus = [DS.effect_map(r, READOUT_MC) for r in fits]
                single = pd.DataFrame([{"mae": np.abs(t - ate).mean(), "disc": (t - ate)[disc].mean(),
                                        "ring": (t - ate)[ring].mean(), "far": (t - ate)[far].mean(),
                                        "slope": np.polyfit(imb, t - ate, 1)[0]} for t in taus]).mean()
                rows.append({"preset": LABEL[p], "dataset": k, "method": "IFF, single fit", **single.to_dict()})
                ests["IFF, single fit"] = taus[0]
                if len(taus) == 5:
                    ests["IFF, 5-fit average"] = np.mean(taus, 0); score(ests["IFF, 5-fit average"], "IFF, 5-fit average")
            if k == SHOW_DATASET:
                shown[LABEL[p]] = {"ate": ate, **ests}
    return pd.DataFrame(rows), shown, pd.DataFrame(status)


def _cell(s, scale):
    if len(s) == 0:
        return "--"
    se = s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else np.nan
    return f"{s.mean() * scale:.1f}" + (f" ({se * scale:.1f})" if np.isfinite(se) else "")


def inference_tables(out, size, x, status):
    presets = [LABEL[p] for p in INF_GRID[size][0]]
    ndata = len(INF_GRID[size][1])
    # ---- error over all pixels: methods x presets ----
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\begin{tabular}{l" + "c" * len(presets) + "}", r"\toprule",
             "Method & " + " & ".join(presets) + r" \\", r"\midrule"]
    for m in METHOD_ORDER:
        if m == "Frengression":
            lines.append(r"\midrule")
        cells = [_cell(x[(x.preset == pr) & (x.method == m)].mae, 1e3) for pr in presets]
        lines.append(f"{m} & " + " & ".join(cells) + r" \\")
    counts = " & ".join(str(int(((status.preset == pr) & status.included).sum())) for pr in presets)
    lines += [r"\midrule", f"Datasets & {counts} \\\\", r"\bottomrule", r"\end{tabular}",
              f"\\caption{{Error of the estimated effect map at ${size}\\times{size}$: mean absolute difference "
              r"between estimated and true ATE over the $K$ pixels, $\times 10^{3}$, averaged over datasets, "
              r"with its standard error over datasets in brackets. IFF, single fit: the error of one fit, "
              r"averaged over the five fits of each dataset; IFF, 5-fit average: the error of the average of the "
              r"five fits' effect maps. The naive difference measures the size of the confounding; oracle IPW "
              r"uses the true treatment probabilities. Last row: datasets (assignment draws) included, "
              f"out of {ndata}.}}",
              f"\\label{{tab:errors{size}}}", r"\end{table}"]
    open(os.path.join(out, f"table_errors_{size}x{size}.tex"), "w").write("\n".join(lines) + "\n")
    # ---- where the error sits: preset x method rows ----
    meths = ["OLS", "AIPW", "Frengression", "IFF, 5-fit average"]
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{llcccc}", r"\toprule",
             r"Preset & Method & Disc & Ring & Background & Leftover slope \\", r"\midrule"]
    for i, pr in enumerate(presets):
        for j, m in enumerate(meths):
            s = x[(x.preset == pr) & (x.method == m)]
            cells = [_cell(s[c], 1e3) for c in ("disc", "ring", "far")]
            sl = "--" if pr == "E1" or len(s) == 0 else (f"{s.slope.mean():.3f}" + (f" ({s.slope.std(ddof=1) / np.sqrt(len(s)):.3f})" if len(s) > 1 else ""))
            lines.append(f"{pr if j == 0 else ''} & {m} & " + " & ".join(cells) + f" & {sl} \\\\")
        if i < len(presets) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}",
              f"\\caption{{Where the error sits at ${size}\\times{size}$. Signed error (estimate minus true "
              r"ATE, $\times 10^{3}$) averaged over the disc, over the ring of pixels bordering the disc, and over "
              r"the remaining background, and the leftover slope: the slope of a least-squares fit of the error "
              r"map on the confounding map (naive estimate minus true ATE) across pixels, which is 0 when no "
              r"confounding is left in the estimate and 1 when none was removed; not defined for E1, which has "
              r"no confounding. Means over datasets, standard errors in brackets. These quantities are averages of "
              r"the signed error, so for IFF they are the same for a single fit (averaged over the five fits) as "
              r"for the 5-fit average; only the absolute error in Table~\ref{tab:errors" + str(size) + r"} differs.}",
              f"\\label{{tab:regions{size}}}", r"\end{table}"]
    open(os.path.join(out, f"table_regions_{size}x{size}.tex"), "w").write("\n".join(lines) + "\n")
    return [f"table_errors_{size}x{size}", f"table_regions_{size}x{size}"]


def inference_error_figure(out, size, shown):
    presets = [LABEL[p] for p in INF_GRID[size][0] if LABEL[p] in shown]
    cols = ["OLS", "AIPW", "Frengression", "IFF, single fit", "IFF, 5-fit average"]
    titles = ["OLS", "AIPW", "Frengression", "IFF\nsingle fit", "IFF\n5-fit average"]
    D = disc_mask(size)
    errs = {(pr, c): (shown[pr][c] - shown[pr]["ate"]) if c in shown[pr] else None for pr in presets for c in cols}
    vmax = max(np.abs(e).max() for e in errs.values() if e is not None)
    w = TEXTWIDTH * 0.62
    fig, axes = plt.subplots(len(presets), len(cols), figsize=(w, w * len(presets) / len(cols) * 1.02),
                             gridspec_kw=dict(wspace=0.06, hspace=0.06), squeeze=False)
    h = None
    for r, pr in enumerate(presets):
        for j, c in enumerate(cols):
            ax = axes[r, j]; bare(ax)
            if errs[(pr, c)] is not None:
                h = ax.imshow(errs[(pr, c)].reshape(size, size), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
                outline_disc(ax, D)
            if r == 0:
                ax.set_title(titles[j], fontsize=7, pad=3)
            if j == 0:
                ax.set_ylabel(pr, rotation=0, ha="right", va="center")
    b0, b4 = axes[-1, 0].get_position(), axes[-1, -1].get_position()
    cax = fig.add_axes([b0.x0, b0.y0 - 0.035 * 6 / len(presets), b4.x1 - b0.x0, 0.012 * 6 / len(presets)])
    fig.colorbar(h, cax=cax, orientation="horizontal").ax.tick_params(labelsize=6)
    name = f"fig_errors_{size}x{size}"
    fig.savefig(os.path.join(out, name + ".pdf")); plt.close(fig)
    cap = (f"Error maps at ${size}\\times{size}$: estimated minus true ATE, on the logit scale and one colour "
           r"scale, for a single simulated dataset (one draw of the treatment assignment). IFF single fit: one "
           r"of the five fits; IFF 5-fit average: the average of the five fits' effect maps. Black outline: "
           r"the disc where the true ATE is non-zero.")
    write_figure_tex(out, name, cap, f"fig:errors{size}", "0.62\\textwidth")
    return name


def hparams_table(out):
    import glob, json
    f = json.load(open(glob.glob(os.path.join(MM, "runs", "exp_ate_recovery",
                                               "*_ff_e2_flexcont_sa1_lr0.001_copw16_k64_s1_d0-9_*/config.json"))[0]))["config"]
    g = json.load(open(glob.glob(os.path.join(MM, "runs", "frengression",
                                              "*_frengression_e2_sa1_k64_s1_d0-9_*/config.json"))[0]))["config"]
    rows = [
        ("Causal margin", f"{f['flow_layers']} autoregressive spline layers, width {f['nn_width']}, "
                          f"{f['rqs_knots']} knots, permutation between layers",
         f"generator, {g['num_layer']} layers of width {g['hidden_dim']}, noise dimension {g['noise_dim']}"),
        ("Dependence component", f"copula flow, {f['copula_flow_layers']} layers, width {f['copula_nn_width']}, "
                                 f"{f['copula_rqs_knots']} knots",
         f"generator, {g['num_layer']} layers of width {g['hidden_dim']}"),
        ("Covariates", "ranks: spline flow per continuous covariate, empirical CDF for the digit class",
         "standardised"),
        ("Outcome", "logit image as is", "standardised per pixel"),
        ("Objective", "likelihood", "energy score"),
        ("Optimiser", f"Adam, learning rate {f['learning_rate']:g}, batch {f['batch_size']}",
         f"Adam, learning rate {g['lr']:g}, full batch"),
        ("Stopping", f"30 epochs without improvement of the held-out likelihood (10\\,\\% of units), "
                     f"at most {f['max_epochs']} epochs", f"{g['num_iters']} iterations"),
        ("Effect read-out", f"{READOUT_MC} paired draws under $\\doo{{0}}$ and $\\doo{{1}}$",
         f"{READOUT_MC} paired draws under $\\doo{{0}}$ and $\\doo{{1}}$"),
        ("Fits per dataset", "5 (fit seeds), effect maps averaged", "1"),
    ]
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{p{0.2\linewidth}p{0.38\linewidth}p{0.32\linewidth}}",
             r"\toprule", r" & IFF & Frengression \\", r"\midrule"]
    lines += [f"{a} & {b} & {c} \\\\" for a, b, c in rows]
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Settings of the two frugal models, identical at $8\times8$ and $16\times16$ and across "
              r"presets.}", r"\label{tab:hparams}", r"\end{table}"]
    open(os.path.join(out, "table_hparams.tex"), "w").write("\n".join(lines) + "\n")
    return "table_hparams"


def inference(out):
    parts = [hparams_table(out)]
    for size in INF_GRID:
        x, shown, status = collect_inference(size)
        x.to_csv(os.path.join(out, f"scores_{size}x{size}.csv"), index=False)
        status.to_csv(os.path.join(out, f"status_{size}x{size}.csv"), index=False)
        # every method on the same datasets: those with all five IFF fits, the Frengression fit and baselines
        ok = status[(status.iff_fits == 5) & (status.frengression == 1)][["preset", "dataset"]]
        has_b = x[x.method == "OLS"][["preset", "dataset"]]
        ok = ok.merge(has_b, on=["preset", "dataset"])
        x = x.merge(ok, on=["preset", "dataset"])
        status["included"] = status.set_index(["preset", "dataset"]).index.isin(ok.set_index(["preset", "dataset"]).index)
        status.to_csv(os.path.join(out, f"status_{size}x{size}.csv"), index=False)
        print(f"  {size}x{size}: datasets included per preset: {ok.groupby('preset').size().to_dict()}")
        parts += inference_tables(out, size, x, status) + [inference_error_figure(out, size, shown)]
    return parts


TOPICS = {"setup": setup, "inference": inference}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("topic", choices=sorted(TOPICS))
    a = ap.parse_args()
    out = os.path.join(OUT, a.topic)
    if os.path.isdir(out):                      # outputs are regenerated in full each time
        shutil.rmtree(out)
    os.makedirs(out)
    parts = TOPICS[a.topic](out)
    preview(out, parts)
    for junk in ("preview.aux", "preview.log"):
        if os.path.exists(os.path.join(out, junk)):
            os.remove(os.path.join(out, junk))
