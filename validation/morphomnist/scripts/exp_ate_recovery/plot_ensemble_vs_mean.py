"""Average-of-maps vs mean-of-MAEs, E2 all digits, datasets 1-3.

For each arm with several fits per dataset: ATE MAE of each fit, the mean of those MAEs, and the MAE of the
averaged tau_hat map (an ensemble). Frengression has one fit per dataset, so both numbers coincide.
Laura's stored fits are read from analysis/cache (recomputed from her saved weights by
analyse_gaussian_scale.py); new fits from their arrays.npz. Truth = ATE from the new dataset-k run
(same dataset hash as Laura's).
"""
import glob, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.abspath(os.path.join(HERE, "..", "..", "runs"))
GS = os.path.join(RUNS, "gaussian_scale"); OUT = os.path.join(GS, "analysis"); CACHE = os.path.join(OUT, "cache")

def seed_of(name): return int(re.search(r"_k64_s(\d+)_", name).group(1))

def collect(k):
    truth = None; arms = {"FR": {}, "U-raw": {}, "G-std": {}, "G-LT": {}}
    for f in glob.glob(os.path.join(CACHE, f"*_e2_*sa{k}_*.npz")):
        n = os.path.basename(f)
        code = "FR" if "frengression" in n else "U-raw"
        arms[code][seed_of(n)] = np.load(f)["tau_hat"]
    for d in glob.glob(os.path.join(GS, f"*_ff_e2_*_sa{k}_*")):
        n = os.path.basename(d)
        if "zshuf" in n or "copw16" in n or not os.path.exists(os.path.join(d, "arrays.npz")):
            continue
        a = np.load(os.path.join(d, "arrays.npz"))
        if "flexgauss" in n and "_ystd_" in n: code = "G-std"
        elif "loctransgauss" in n: code = "G-LT"
        else: continue
        arms[code][seed_of(n)] = a["tau_hat"]
        truth = a["ATE"]
    return truth, arms

def main(log_wandb=True):
    ks = [1, 2, 3]; order = ["FR", "U-raw", "G-std", "G-LT"]
    data = {k: collect(k) for k in ks}
    rows = []
    for k in ks:
        truth, arms = data[k]
        for code in order:
            taus = arms[code]
            if not taus: continue
            maes = {s: float(np.abs(t - truth).mean()) for s, t in sorted(taus.items())}
            avg = np.mean(list(taus.values()), 0)
            rows.append(dict(k=k, code=code, n=len(taus), maes=maes, mean_mae=float(np.mean(list(maes.values()))),
                             avg_mae=float(np.abs(avg - truth).mean()), single=taus.get(k), avg=avg, truth=truth))
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 1 + 2 * 3, width_ratios=[2.2] + [1] * 6, wspace=0.35, hspace=0.45)
    cols = {"FR": "#444444", "U-raw": "#1f77b4", "G-std": "#d62728", "G-LT": "#9467bd"}
    for i, k in enumerate(ks):
        ax = fig.add_subplot(gs[i, 0]); rk = [r for r in rows if r["k"] == k]
        for j, r in enumerate(rk):
            ax.scatter([j] * r["n"], list(r["maes"].values()), color=cols[r["code"]], alpha=0.45, s=22)
            ax.scatter(j - 0.18, r["mean_mae"], marker="_", s=300, color=cols[r["code"]], lw=2.5)
            ax.scatter(j + 0.18, r["avg_mae"], marker="*", s=140, color=cols[r["code"]], edgecolor="k")
        ax.set_xticks(range(len(rk))); ax.set_xticklabels([f"{r['code']}\n({r['n']} fit{'s' if r['n']>1 else ''})" for r in rk], fontsize=8)
        ax.set_ylabel(f"dataset {k}\nATE MAE"); ax.set_ylim(0, None); ax.grid(axis="y", alpha=0.3)
        if i == 0:
            ax.set_title("dots = single fits;  bar = mean of MAEs;  star = MAE of averaged map", fontsize=9)
        for c, code in enumerate(["FR", "U-raw", "G-std"]):
            r = next((r for r in rk if r["code"] == code), None)
            for m, (kind, img) in enumerate([("single fit", r and r["single"]), ("averaged map", r and r["avg"])]):
                if code == "FR" and m == 1: kind, img = "(1 fit only)", None
                axm = fig.add_subplot(gs[i, 1 + 2 * c + m]); axm.set_xticks([]); axm.set_yticks([])
                if img is None or r is None:
                    axm.axis("off"); axm.set_title(f"{code}\n{kind}", fontsize=8); continue
                err = (img - r["truth"]).reshape(8, 8)
                axm.imshow(err, cmap="RdBu_r", vmin=-0.05, vmax=0.05)
                axm.set_title(f"{code} {kind}\nMAE {np.abs(err).mean():.4f}", fontsize=8)
    fig.suptitle("E2, all digits 8x8: single fits vs mean of MAEs vs MAE of the averaged map (error maps: tau_hat - ATE, ±0.05)")
    p = os.path.join(OUT, "gs_ensemble_vs_mean.png"); fig.savefig(p, dpi=120, bbox_inches="tight")
    lines = ["| dataset | arm | fits | per-fit MAE | mean of MAEs | MAE of averaged map |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['k']} | {r['code']} | {r['n']} | " + ", ".join(f"{v:.4f}" for v in r["maes"].values())
                     + f" | {r['mean_mae']:.4f} | {r['avg_mae']:.4f} |")
    open(os.path.join(OUT, "gs_ensemble_vs_mean.md"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
    if log_wandb:
        import wandb
        rid = open(os.path.join(OUT, "wandb_run_id.txt")).read().strip()
        run = wandb.init(entity="proj-lb", project="Frugal Images", id=rid, resume="must")
        tbl = wandb.Table(columns=["dataset", "arm", "fits", "mean_of_maes", "mae_of_averaged_map"],
                          data=[[r["k"], r["code"], r["n"], r["mean_mae"], r["avg_mae"]] for r in rows])
        run.log({"plots/gs_ensemble_vs_mean": wandb.Image(p), "table/ensemble_vs_mean": tbl}); run.finish()
        print("logged to", rid)

if __name__ == "__main__":
    main(log_wandb="--no-wandb" not in sys.argv)
