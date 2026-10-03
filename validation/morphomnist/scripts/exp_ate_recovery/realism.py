"""Counterfactual realism of a fitted frugal flow, benchmark-style (2026-10-03).

Adapts the realism metric of the counterfactual-image benchmark (Melistas et al., NeurIPS D&B 2024,
github gulnazaki/counterfactual-benchmark, ``evaluation/metrics/fid.py``): Frechet Inception Distance
with torchmetrics' Inception v3 (``normalize=True``, greyscale repeated to 3 channels, [0, 1]). Our
intervention is the binary treatment T in logit space on 8x8 images, so the numbers are NOT
comparable with the benchmark's 32x32 thickness/intensity/digit table.

The simulator knows every unit's true counterfactual (the effect is added in logit space to the same
noisy image: Y(1) = Y(0) + ITE), so realism is scored against the TRUE counterfactual set, and the
unit-level counterfactual error is scored directly. Per run folder:

  CF   abduct-act-predict counterfactuals of all n units under 1 - T (rank-preserving margin
       transport: ``counterfactual_flexible`` for the uniform arm, ``counterfactual_gaussian`` for the
       Gaussian arm), vs the true counterfactuals Y + (1 - 2T) ITE
  INT  do(T=0) / do(T=1) draws (``interventional_samples``), vs the true Y(0) / Y(1) of all units

Metrics: FID (CF; INT do0, do1), an FID floor (true set, two random halves),
Reading FID here: CF vs true CF is a PAIRED comparison (both sets come from the same n images), so its
floor is 0 and its scale is set by the do-nothing baseline (factual images vs true CF). INT draws are
independent of the true sets, so their floor is sampling noise: the half-split floor is at n/2, and FID
bias scales ~1/n, so half of it estimates the full-n floor. unit-level CF MAE in
logit and pixel space (with the do-nothing baseline: factual image as its own counterfactual), and
the out-of-fold real-vs-generated classifier AUC on CF pixels. Inception features run on the Apple
GPU (MPS) when available, float32; the Frechet distance is computed in float64 as torchmetrics does.
True-set features are cached per dataset (shared across arms) in runs/realism_cache/.

  python realism.py RUN_DIR [RUN_DIR ...] [--wandb] [--group realism_sub5k]
Writes RUN_DIR/realism.json and RUN_DIR/plots/realism_gallery.png.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MM = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, MM)
CACHE = os.path.join(MM, "runs", "realism_cache")
N_GALLERY = 10


# ------------------------------------------------------------------ FID
_INCEPTION = None


def _device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def inception_features(pix: np.ndarray, size: int, batch: int = 250) -> np.ndarray:
    """(n, K) pixels in [0, 1] -> (n, 2048) Inception pool features, as torchmetrics' FID computes them
    (normalize=True: uint8 = (x * 255).byte(); greyscale repeated to 3 channels; resize inside)."""
    import torch
    from torchmetrics.image.fid import FrechetInceptionDistance
    global _INCEPTION
    if _INCEPTION is None:
        _INCEPTION = FrechetInceptionDistance(normalize=True).inception.to(_device()).eval()
    x = torch.as_tensor(np.clip(pix, 0, 1), dtype=torch.float32).reshape(-1, 1, size, size).repeat(1, 3, 1, 1)
    x = (x * 255).byte()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(_INCEPTION(x[i:i + batch].to(_device())).cpu().double())
    return torch.cat(out).numpy()


def frechet(fa: np.ndarray, fb: np.ndarray) -> float:
    """Frechet distance between Gaussians fitted to two feature sets (float64)."""
    from scipy import linalg
    mu1, mu2 = fa.mean(0), fb.mean(0)
    s1, s2 = np.cov(fa, rowvar=False), np.cov(fb, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2)
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def cached_features(name: str, dataset_id: str, pix: np.ndarray, size: int) -> np.ndarray:
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"{dataset_id}_{name}.npy")
    if os.path.exists(p):
        return np.load(p)
    f = inception_features(pix, size)
    np.save(p, f)
    return f


# ------------------------------------------------------------------ one run
def score_run(run_dir: str, seed: int = 0) -> dict:
    import exp_ate_recovery as E
    import jax.numpy as jnp
    import jax.random as jr
    import dataset_store
    from frugal_flows.gaussian_scale import counterfactual_gaussian
    from frugal_flows.interventions import counterfactual_flexible, interventional_samples
    from prepare_data import inverse_logit
    from sample_diagnostics import _classifier_auc

    t0 = time.time()
    rec = json.load(open(os.path.join(run_dir, "config.json")))
    stored = rec["config"]
    cfg = E.Config(**{k: v for k, v in stored.items() if k in {f.name for f in E.fields(E.Config)}})
    data = dataset_store.build_for_run(run_dir)
    flow = E.load_model(run_dir)
    ot = E.outcome_transform_for(cfg, data)
    size, K = cfg.size, cfg.size ** 2
    Y = np.asarray(data["Y"], np.float64)
    T = np.asarray(data["X"], np.float64).reshape(len(Y), -1)
    ITE = np.asarray(data["ITE"], np.float64)
    Y0, Y1 = Y - T * ITE, Y - T * ITE + ITE
    Ycf_true = Y + (1 - 2 * T) * ITE
    did = str(data["dataset_id"])

    # counterfactuals on the fitting scale, back to logit Y
    y_fit = np.asarray(ot.forward(jnp.asarray(Y))) if ot is not None else Y
    inv = (lambda a: np.asarray(ot.inverse(jnp.asarray(a)), np.float64)) if ot is not None else (lambda a: np.asarray(a, np.float64))
    if cfg.arm == "flexible_continuous_gaussian":
        cf = counterfactual_gaussian(flow, y_fit, T, 1 - T)
    elif cfg.arm == "flexible_continuous":
        u_z = np.load(os.path.join(run_dir, "arrays.npz"))["u_z"]
        cf = counterfactual_flexible(flow, y_fit, u_z, T, 1 - T)
    else:
        raise ValueError(f"no counterfactual for arm {cfg.arm}")
    Ycf = inv(cf)
    ok_cf = np.isfinite(Ycf).all(1)

    # interventional draws, same n as the data
    s = interventional_samples(jr.key(seed), flow, T.shape[1], len(Y), outcome_transform=ot, dim_y=K)
    g0, g1 = np.asarray(s["y0"], np.float64), np.asarray(s["y1"], np.float64)
    ok0, ok1 = np.isfinite(g0).all(1), np.isfinite(g1).all(1)

    px = lambda a: inverse_logit(a)
    f_cf_true = cached_features("cf_true", did, px(Ycf_true), size)
    f_y0 = cached_features("y0_true", did, px(Y0), size)
    f_y1 = cached_features("y1_true", did, px(Y1), size)
    f_fact = cached_features("factual", did, px(Y), size)
    f_cf = inception_features(px(Ycf[ok_cf]), size)
    f_g0, f_g1 = inception_features(px(g0[ok0]), size), inception_features(px(g1[ok1]), size)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Y))
    h = len(Y) // 2

    err = np.abs(Ycf[ok_cf] - Ycf_true[ok_cf])
    perr = np.abs(px(Ycf[ok_cf]) - px(Ycf_true[ok_cf]))
    auc, auc_sd, _, _ = _classifier_auc(px(Ycf_true[ok_cf]), px(Ycf[ok_cf]), seed)
    out = {
        "run": os.path.basename(run_dir.rstrip("/")), "preset": cfg.preset, "arm": cfg.arm,
        "y_scaling": cfg.y_scaling, "k": int(cfg.seed_assign), "fit_seed": int(cfg.seed_fit), "n": len(Y),
        "dataset_id": did,
        "fid_cf": frechet(f_cf, f_cf_true),
        "fid_int0": frechet(f_g0, f_y0), "fid_int1": frechet(f_g1, f_y1),
        "fid_floor_half": (fl := frechet(f_cf_true[perm[:h]], f_cf_true[perm[h:]])),
        "fid_floor_fulln_est": fl / 2,      # FID bias ~ 1/n: the n/2-vs-n/2 floor halved
        "fid_factual_vs_cf_true": frechet(f_fact, f_cf_true),       # do-nothing baseline
        "cf_mae_logit": float(err.mean()), "cf_mae_pixel": float(perr.mean()),
        "cf_mae_logit_donothing": float(np.abs(Y - Ycf_true).mean()),
        "cf_mae_pixel_donothing": float(np.abs(px(Y) - px(Ycf_true)).mean()),
        "cf_auc": auc, "cf_auc_sd": auc_sd,
        "n_nonfinite_cf": int((~ok_cf).sum()), "n_nonfinite_int": int((~ok0).sum() + (~ok1).sum()),
        "device": _device(), "wall_s": time.time() - t0,
    }
    json.dump(out, open(os.path.join(run_dir, "realism.json"), "w"), indent=1)
    gallery(Y, Ycf, Ycf_true, T, size, os.path.join(run_dir, "plots", "realism_gallery.png"),
            f"{out['run']}\nFID(CF) {out['fid_cf']:.2f} (floor {out['fid_floor_half']:.2f}), CF MAE logit {out['cf_mae_logit']:.3f}")
    return out


def gallery(Y, Ycf, Ycf_true, T, size, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from prepare_data import inverse_logit
    os.makedirs(os.path.dirname(path), exist_ok=True)
    idx = np.r_[np.where(T[:, 0] == 0)[0][:N_GALLERY // 2], np.where(T[:, 0] == 1)[0][:N_GALLERY // 2]]
    fig, axes = plt.subplots(3, len(idx), figsize=(1.5 * len(idx), 5))
    for r, (lab, arr) in enumerate((("factual", Y), ("generated CF", Ycf), ("true CF", Ycf_true))):
        for c, i in enumerate(idx):
            ax = axes[r, c]
            ax.set_xticks([]), ax.set_yticks([])
            ax.imshow(inverse_logit(arr[i]).reshape(size, size), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            if r == 0:
                ax.set_title(f"T={int(T[i, 0])}->{1 - int(T[i, 0])}", fontsize=8)
        axes[r, 0].set_ylabel(lab, fontsize=9)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def log_wandb(out: dict, run_dir: str, group: str):
    import wandb
    fit = json.load(open(os.path.join(run_dir, "wandb.json"))) if os.path.exists(os.path.join(run_dir, "wandb.json")) else {}
    run = wandb.init(project="Frugal Images", group=group, name=f"realism_{out['run']}", job_type="realism",
                     tags=["realism", out["arm"], out["preset"]],
                     config={**{k: out[k] for k in ("preset", "arm", "y_scaling", "k", "fit_seed", "n", "run")},
                             "fit_wandb_id": fit.get("id"), "fit_wandb_url": fit.get("url")}, reinit=True)
    run.log({k: v for k, v in out.items() if isinstance(v, (int, float)) and k not in ("k", "fit_seed", "n")}
            | {"plots/realism_gallery": wandb.Image(os.path.join(run_dir, "plots", "realism_gallery.png"))})
    json.dump({"id": run.id, "url": run.url, "group": group}, open(os.path.join(run_dir, "realism_wandb.json"), "w"), indent=1)
    run.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--group", default="realism_sub5k")
    ap.add_argument("--skip-done", action="store_true")
    a = ap.parse_args()
    for d in a.runs:
        if a.skip_done and os.path.exists(os.path.join(d, "realism.json")):
            continue
        out = score_run(d)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}), flush=True)
        if a.wandb:
            log_wandb(out, d, a.group)


if __name__ == "__main__":
    main()
