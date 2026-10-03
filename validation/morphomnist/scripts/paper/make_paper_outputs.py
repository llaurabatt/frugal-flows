"""Figures and tables for the paper's MorphoMNIST experiments, one topic at a time.

    python scripts/paper/make_paper_outputs.py setup      # experimental set-up (appendix)

Each topic writes PDFs, LaTeX snippets and a compiled preview to runs/paper/<topic>/. Nothing is
written into the paper repository; the snippets are copied there by hand once checked.

Topic "setup" (no fits needed; everything comes from the data generator):
  table_presets.tex       one row per preset: assignment, the individual effect, and how large the
                          confounding is at each resolution (mean over the pixels of |naive estimate -
                          true ATE|; mean and range over the 10 datasets)
  fig_assignment.pdf      thickness of treated and untreated units (E1; E2-E6, which share one
                          assignment) and the assignment probability against thickness
  fig_treated_<s>x<s>.pdf one figure per resolution: five units along the thickness range, untreated
                          and treated under E2 and under E6, as pixel intensities
  fig_truth_<s>x<s>.pdf   one figure per resolution, one row per preset: true ATE, average individual
                          effect of the thinnest and of the thickest 10 % of units, naive estimate, and
                          naive estimate minus true ATE (dataset 1, named in the caption)
Datasets: all ten digits (n = 60000), seed_data 101, assignment seeds 1-10, as in the 8x8 paper grid
(scripts/exp_ate_recovery/grid_8x8_alldigits_v2.sh); the same seeds at 16x16. Units, potential
outcomes and individual effects are the same in all ten datasets of a preset; only the assignment, and
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
    radius = E.Config(size=size).effective_radius
    cap = (f"Ground truth at ${size}\\times{size}$ for the six presets (rows), on the logit scale. "
           r"Columns 1--3: the true ATE, which is the same in every preset (a disc of radius "
           f"{radius} pixels, black outline), and the average individual effect over the 10\\,\\% "
           r"thinnest and the 10\,\% thickest units. They coincide with the ATE in E1 and E2. In E3 "
           r"and E5 they differ partly or wholly because the effect grows with the pixel's own untreated "
           r"value, and thick digits have brighter pixels; in E4 and E6 thickness enters the effect "
           r"directly, and in E6 it also moves the effect towards the bottom (thick) or the top (thin) "
           r"of the disc. Columns 4--5: the naive estimate (mean treated image minus mean untreated "
           r"image) and its difference from the true ATE, for one of the ten datasets; columns 1--3 are "
           r"the same in all ten. All panels "
           r"share one colour scale.")
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
           r"$n = 60\,000$ units of one of the ten datasets; dashed lines are the group means. E2--E6 "
           r"assign treatment by the same rule, so one panel covers all five. (c) The assignment probability: $\tfrac12$ in E1 and "
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
           r"of the disc). Images are shown as pixel intensities; the effect is added on the logit scale, "
           r"so it mainly brightens dark pixels inside the disc (red outline) and changes white pixels little.")
    write_figure_tex(out, name, cap, f"fig:treated{size}", "0.62\\textwidth")
    return name


def setup(out):
    data = {s: {} for s in SIZES}
    conf = {s: {p: [] for p in PRESETS} for s in SIZES}
    for s in SIZES:
        for p in PRESETS:
            for k in ASSIGN_SEEDS:
                d = build(p, s, k)
                if k == SHOW_DATASET:
                    data[s][p] = d
                conf[s][p].append(np.abs(naive(d) - np.asarray(d["ATE"])).mean())
            c = conf[s][p]
            print(f"  {s}x{s} {LABEL[p]}: confounding {np.mean(c):.4f} [{min(c):.4f}, {max(c):.4f}]")

    # ---------------- table ----------------
    gen = {p: P.PRESETS[p] for p in PRESETS}
    res_cols = " & ".join(f"${s}\\times{s}$" for s in SIZES)
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\begin{tabular}{lll" + "c" * len(SIZES) + "}", r"\toprule",
             r" & & & \multicolumn{" + str(len(SIZES)) + r"}{c}{Confounding} \\",
             r"\cmidrule(l){4-" + str(3 + len(SIZES)) + "}",
             r"Preset & $P(T_i = 1 \mid Z_i)$ & Individual effect $\tau_{ik}$ & " + res_cols + r" \\",
             r"\midrule"]
    for p in PRESETS:
        cells = " & ".join(f"{np.mean(conf[s][p]):.3f} [{min(conf[s][p]):.3f}, {max(conf[s][p]):.3f}]"
                           for s in SIZES)
        lines.append(f"{LABEL[p]} & {assignment_text(gen[p])} & {effect_formula(gen[p])} & {cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{The six MorphoMNIST presets. Each unit $i$ is an MNIST training image "
              r"($n = 60\,000$, all ten digits), average-pooled to $8\times8$ or $16\times16$ "
              r"($K = 64$ or $256$ pixels), dequantised and mapped to logits. The treated image is "
              r"$Y_i(1) = Y_i(0) + \tau_i$, with $m$ a disc at the image centre (value 1 inside, 0 "
              r"outside; radius 2 pixels at $8\times8$, 4 at $16\times16$). $\tilde t_i$ is the "
              r"standardised thickness. $h_i$, $b_i$ are the thickness and brightness ranks and $g_{ik}$ "
              r"the rank of $Y_{ik}(0)$ among all units, each rescaled to $[-1, 1]$ and centred to sample "
              r"mean exactly zero; $(hb)_i$ is their centred product and $\psi_k$ a top-to-bottom gradient "
              r"over the disc. Because every modulating term has mean zero, the true ATE equals $m$ exactly "
              r"in every preset. Brightness is included in the covariates $Z_i$ in E4 and E6, where the "
              r"effect depends on it. Confounding: mean absolute difference between the naive estimate "
              r"(mean treated image minus mean untreated image) and the true ATE over the $K$ pixels; mean "
              f"over the {len(ASSIGN_SEEDS)} datasets (assignment seeds) and, in brackets, the smallest "
              r"and largest. In E1 it reflects only the randomness of the assignment.}",
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
           "\\begin{document}\n" + body + "\n\\end{document}\n")
    open(os.path.join(out, "preview.tex"), "w").write(tex)
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "preview.tex"], cwd=out,
                       capture_output=True, text=True)
    print("preview:", os.path.join(out, "preview.pdf") if r.returncode == 0 else "pdflatex failed:\n" + r.stdout[-2000:])


TOPICS = {"setup": setup}

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
