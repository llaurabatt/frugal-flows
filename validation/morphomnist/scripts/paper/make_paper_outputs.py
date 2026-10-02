"""Figures and tables for the paper's MorphoMNIST experiments, one topic at a time.

    python scripts/paper/make_paper_outputs.py setup      # experimental set-up (appendix)

Each topic writes PDFs, LaTeX snippets and a compiled preview to runs/paper/<topic>/. Nothing is
written into the paper repository; the snippets are copied there by hand once checked.

Topic "setup" (no fits needed; everything comes from the data generator):
  table_presets.tex   one row per preset: assignment, the individual effect, and how large the
                      confounding is (mean over the 64 pixels of |naive estimate - true ATE|, mean and
                      range over the 10 datasets)
  fig_data.pdf        (a) MNIST digits at 28x28 and at 8x8; (b) two units of E6, thin and thick:
                      untreated image, treated image, individual effect (logit scale)
  fig_truth.pdf       one row per preset: true ATE, average individual effect of the thinnest and of
                      the thickest 10 % of units, naive estimate and confounding map on dataset 1
Datasets are those of the 8x8 paper grid: all ten digits (n = 60000), seed_data 101, assignment
seeds 1-10. Units, images, potential outcomes and individual effects are the same in all ten
datasets of a preset; only the assignment, and so the observed images, differ.
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

# the 8x8 paper grid (scripts/exp_ate_recovery/grid_8x8_alldigits_v2.sh)
SIZE, SEED_DATA, ASSIGN_SEEDS, SHOW_DATASET = 8, 101, range(1, 11), 1
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


def build(preset, seed_assign):
    cfg = E.Config(preset=preset, size=SIZE, digit=None, seed_data=SEED_DATA, seed_assign=seed_assign,
                   arm="flexible_continuous")
    return E.build_data(cfg)


def naive(d):
    T = np.asarray(d["X"])[:, 0].astype(bool)
    Y = np.asarray(d["Y"])
    return Y[T].mean(0) - Y[~T].mean(0)


def img(v):
    return np.asarray(v).reshape(SIZE, SIZE)


def outline_disc(ax, disc):
    """Black outline around the effect's support (the disc)."""
    D = img(disc)
    for i in range(SIZE):
        for j in range(SIZE):
            if not D[i, j]:
                continue
            for di, dj, xy, w, h in ((-1, 0, (j - .5, i - .5), 1, 0), (1, 0, (j - .5, i + .5), 1, 0),
                                     (0, -1, (j - .5, i - .5), 0, 1), (0, 1, (j + .5, i - .5), 0, 1)):
                ii, jj = i + di, j + dj
                if not (0 <= ii < SIZE and 0 <= jj < SIZE and D[ii, jj]):
                    ax.add_patch(Rectangle(xy, w, h, fill=False, lw=0.6, ec="k"))


def bare(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4)


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


def setup(out):
    disc, _, _ = E.region_masks(SIZE, E.Config(size=SIZE).effective_radius)
    data, conf = {}, {p: [] for p in PRESETS}
    treated = {p: [] for p in PRESETS}
    for p in PRESETS:
        for k in ASSIGN_SEEDS:
            d = build(p, k)
            if k == SHOW_DATASET:
                data[p] = d
            conf[p].append(np.abs(naive(d) - np.asarray(d["ATE"])).mean())
            treated[p].append(float(np.asarray(d["X"]).mean()))
        print(f"  {LABEL[p]}: confounding {np.mean(conf[p]):.4f} [{min(conf[p]):.4f}, {max(conf[p]):.4f}]")

    # ---------------- table ----------------
    gen = {p: P.PRESETS[p] for p in PRESETS}
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\begin{tabular}{lllc}", r"\toprule",
             r"Preset & $P(T_i = 1 \mid Z_i)$ & Individual effect $\tau_{ik}$ & "
             r"Confounding $\tfrac{1}{K}\sum_k |\hat\tau^{\mathrm{naive}}_k - \tau_k|$ \\", r"\midrule"]
    for p in PRESETS:
        c = conf[p]
        lines.append(f"{LABEL[p]} & {assignment_text(gen[p])} & {effect_formula(gen[p])} & "
                     f"{np.mean(c):.3f} \\; [{min(c):.3f}, {max(c):.3f}] \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{The six MorphoMNIST presets. Each unit $i$ is an MNIST training image "
              r"($n = 60\,000$, all ten digits), padded to $32\times32$, average-pooled to $8\times8$, "
              r"dequantised and mapped to logits, "
              r"$K = 64$ pixels. The treated image is $Y_i(1) = Y_i(0) + \tau_i$, with "
              r"$m$ the disc of radius 2 at the image centre (value 1 inside, 0 outside). "
              r"$\tilde t_i$ is the standardised thickness. "
              r"$h_i$, $b_i$ are the thickness and brightness ranks and $g_{ik}$ the rank of $Y_{ik}(0)$ "
              r"among all units, each rescaled to $[-1, 1]$ and centred to sample mean exactly zero; "
              r"$(hb)_i$ is their centred product and $\psi_k$ a top-to-bottom gradient over the disc. "
              r"Because every modulating term has mean zero, the true ATE equals $m$ exactly in every "
              r"preset. Brightness is included in the covariates $Z_i$ in E4 and E6, where the effect "
              r"depends on it. Last column: mean absolute difference between the naive estimate "
              r"(treated minus untreated mean image) and the true ATE over the 64 pixels; mean over the "
              f"{len(ASSIGN_SEEDS)} datasets (assignment seeds) and, in brackets, the smallest and largest. "
              r"In E1 it reflects only the randomness of the assignment.}",
              r"\label{tab:presets}", r"\end{table}"]
    with open(os.path.join(out, "table_presets.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---------------- figure: data ----------------
    ds = P.MorphoMNIST(os.path.join(MM, "data"), split="train")
    pool = np.random.default_rng(SEED_DATA).permutation(len(ds))   # build_experiment's unit order
    d6 = data["exp6_spatial_cate"]
    assert np.allclose(ds.thickness[pool].numpy(), d6["THICKNESS"]), "unit order does not match the generator"
    labels = ds.digit.numpy().argmax(1)[pool]
    show = [int(np.where(labels == c)[0][0]) for c in (0, 2, 3, 5, 7, 9)]    # first unit of six classes
    big = ds.images[pool[show]].numpy()[:, 0]                          # (6, 32, 32): 28x28 padded, in [-1, 1]
    small = P.downsample(ds.images[pool[show]], size=SIZE).numpy()[:, 0]
    th = d6["THICKNESS"]
    thin = int(np.argmin(np.abs(th - np.quantile(th, 0.05))))
    thick = int(np.argmin(np.abs(th - np.quantile(th, 0.95))))

    fig = plt.figure(figsize=(TEXTWIDTH, 2.55))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], wspace=0.22)
    ga = gs[:, 0].subgridspec(2, len(show), wspace=0.06, hspace=0.06)
    for j, u in enumerate(show):
        for r, (im, title) in enumerate(((big[j], "32$\\times$32"), (small[j], "8$\\times$8"))):
            ax = fig.add_subplot(ga[r, j]); ax.imshow(im, cmap="gray", vmin=-1, vmax=1)
            bare(ax)
            if j == 0:
                ax.set_ylabel(title)
    fig.text(0.125, 0.94, "(a) MNIST digits and the resolution we fit", ha="left")
    gb = gs[:, 1].subgridspec(2, 3, wspace=0.06, hspace=0.18)
    y0, y1 = np.asarray(d6["Y0"]), np.asarray(d6["Y1"])
    lo, hi = np.quantile(np.r_[y0[[thin, thick]].ravel(), y1[[thin, thick]].ravel()], [0.0, 1.0])
    vt = np.abs(np.asarray(d6["ITE"])[[thin, thick]]).max()
    for r, (u, name) in enumerate(((thin, "thin"), (thick, "thick"))):
        for j, (v, title, kw) in enumerate(((y0[u], "$Y_i(0)$", dict(cmap="gray", vmin=lo, vmax=hi)),
                                            (y1[u], "$Y_i(1)$", dict(cmap="gray", vmin=lo, vmax=hi)),
                                            (y1[u] - y0[u], "$\\tau_i$",
                                             dict(cmap="RdBu_r", vmin=-vt, vmax=vt)))):
            ax = fig.add_subplot(gb[r, j]); h = ax.imshow(img(v), **kw); bare(ax)
            if j == 2:
                outline_disc(ax, disc)
            if r == 0:
                ax.set_title(title, pad=3)
            if j == 0:
                ax.set_ylabel(f"{name}\n$t_i = {th[u]:+.2f}$")
    fig.text(0.635, 0.94, "(b) E6: a thin and a thick unit (logit scale)", ha="left")
    cax = fig.add_axes([0.915, 0.16, 0.008, 0.62]); fig.colorbar(h, cax=cax).ax.tick_params(labelsize=6)
    fig.savefig(os.path.join(out, "fig_data.pdf")); plt.close(fig)

    # ---------------- figure: truth panels ----------------
    cols = ["True ATE\n$m$", "Thinnest\n10\\,\\%", "Thickest\n10\\,\\%",
            f"Naive\n(dataset {SHOW_DATASET})", f"Confounding\n(dataset {SHOW_DATASET})"]
    maps = {}
    for p in PRESETS:
        d = data[p]
        ite, t = np.asarray(d["ITE"]), d["THICKNESS"]
        lo_m, hi_m = t <= np.quantile(t, TAIL), t >= np.quantile(t, 1 - TAIL)
        nv = naive(d)
        maps[p] = [d["ATE"], ite[lo_m].mean(0), ite[hi_m].mean(0), nv, nv - d["ATE"]]
    v_eff = max(np.abs(np.asarray(maps[p][i])).max() for p in PRESETS for i in range(4))
    v_conf = max(np.abs(np.asarray(maps[p][4])).max() for p in PRESETS)
    fig, axes = plt.subplots(len(PRESETS), 5, figsize=(TEXTWIDTH * 0.62, TEXTWIDTH * 0.62 * 6 / 5 * 1.02),
                             gridspec_kw=dict(wspace=0.06, hspace=0.06))
    for r, p in enumerate(PRESETS):
        for j in range(5):
            ax = axes[r, j]
            vm = v_conf if j == 4 else v_eff
            h = ax.imshow(img(maps[p][j]), cmap="RdBu_r", vmin=-vm, vmax=vm)
            outline_disc(ax, disc); bare(ax)
            if j == 3:
                h_eff = h
            if j == 4:
                h_conf = h
            if r == 0:
                ax.set_title(cols[j], fontsize=7, pad=3)
            if j == 0:
                ax.set_ylabel(LABEL[p], rotation=0, ha="right", va="center")
    b0, b3 = axes[-1, 0].get_position(), axes[-1, 3].get_position()
    cax = fig.add_axes([b0.x0, b0.y0 - 0.035, b3.x1 - b0.x0, 0.012])
    fig.colorbar(h_eff, cax=cax, orientation="horizontal").ax.tick_params(labelsize=6)
    b4 = axes[-1, 4].get_position()
    cax = fig.add_axes([b4.x0, b4.y0 - 0.035, b4.x1 - b4.x0, 0.012])
    cb = fig.colorbar(h_conf, cax=cax, orientation="horizontal"); cb.ax.tick_params(labelsize=6)
    cb.set_ticks([-round(v_conf, 2), 0, round(v_conf, 2)])
    fig.savefig(os.path.join(out, "fig_truth.pdf")); plt.close(fig)

    figs = {
        "fig_data": (r"(a) MNIST training digits, padded from $28\times28$ to $32\times32$, and the same digits "
                     r"average-pooled to $8\times8$, the resolution used in the experiments. "
                     r"(b) Two units of E6, at the 5th and 95th percentile of thickness $t_i$ (rescaled to "
                     r"$[-1,1]$): untreated and treated image on the logit scale, and the unit's individual "
                     r"effect $\tau_i = Y_i(1) - Y_i(0)$. The black outline marks the disc where the ATE is non-zero. In E6 the effect of "
                     r"thick digits is shifted towards the bottom of the disc and that of thin digits towards "
                     r"the top.", "fig:data"),
        "fig_truth": (r"Ground truth for the six presets (rows). Columns 1--3: the true ATE (identical in "
                      r"every preset) and the average individual effect over the 10\,\% thinnest and the "
                      r"10\,\% thickest units. They coincide with the ATE in E1 and E2. In E3 and E5 they "
                      r"differ partly or wholly because the effect grows with the pixel's own untreated value, "
                      r"and thick digits have brighter pixels; in E4 and E6 thickness enters the effect "
                      r"directly. Columns 4--5: the naive estimate (treated minus untreated mean "
                      r"image) and its difference from the true ATE, on dataset 1 (assignment seed 1); "
                      r"columns 1--3 are the same for all ten datasets of a preset. Columns 1--4 share one "
                      r"colour scale, column 5 has its own. Black outline: the disc.", "fig:truth"),
    }
    for name, (cap, lab) in figs.items():
        width = "\\textwidth" if name == "fig_data" else "0.62\\textwidth"
        with open(os.path.join(out, name + ".tex"), "w") as f:
            f.write("\\begin{figure}[t]\n\\centering\n"
                    f"\\includegraphics[width={width}]{{figures/experiments/{name}.pdf}}\n"
                    f"\\caption{{{cap}}}\n\\label{{{lab}}}\n\\end{{figure}}\n")
    return ["table_presets", "fig_data", "fig_truth"]


# --------------------------------------------------------------------------------------------- #
def preview(out, parts):
    """Compile the snippets on a page of AISTATS text width (one column, as in the appendix)."""
    body = "\n\\clearpage\n".join(f"\\input{{{p}.tex}}" for p in parts)
    tex = ("\\documentclass{article}\n\\usepackage[paperwidth=8.5in,paperheight=11in,textwidth=6.75in,"
           "textheight=9.25in]{geometry}\n\\usepackage{amsmath,amssymb,booktabs,graphicx}\n"
           "\\graphicspath{{./}}\n\\begin{document}\n" + body + "\n\\end{document}\n")
    tex = tex.replace("figures/experiments/", "")
    with open(os.path.join(out, "preview.tex"), "w") as f:
        f.write(tex)
    for p in parts:   # snippets reference figures/experiments/<name>.pdf; the preview reads them locally
        src = os.path.join(out, p + ".tex")
        s = open(src).read().replace("figures/experiments/", "")
        open(os.path.join(out, "_preview_" + p + ".tex"), "w").write(s)
    tex = tex.replace("\\input{", "\\input{_preview_")
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
    os.makedirs(out, exist_ok=True)
    parts = TOPICS[a.topic](out)
    preview(out, parts)
    for junk in ("preview.aux", "preview.log"):
        if os.path.exists(os.path.join(out, junk)):
            os.remove(os.path.join(out, junk))
    shutil.rmtree(os.path.join(out, "__pycache__"), ignore_errors=True)
