"""Toy MorphoMNIST ATE-map recovery: known circle effect on digit images.

Builds the toy dataset (a single digit class, randomly assigned treatment, a
fixed circle-shaped per-pixel effect painted in logit space), fits the
multivariate ``location_translation`` frugal flow, reads the per-pixel
``tau_hat`` off the ``LocCond`` blocks, and scores it against the known truth.

Every run writes a self-contained folder under ``runs/toy_ate_recovery/``:

    <UTC-stamp>_s<seed_fit>_k<K>_<random-suffix>/     # the folder name is the run_id
        config.json    run_id + every knob + git commit/dirty flag + library versions
        log.txt        live training output (`tail -f` it to watch progress)
        metrics.json   recovery scores (MAE/RMSE/corr vs truth) + best val loss
        arrays.npz     raw arrays: tau_hat, truth, data, losses (replottable)
        plots/         every figure, saved as PNG

    config.json and log.txt are written at launch, so the folder existing with
    only those two files means the run is still training; metrics.json,
    arrays.npz and plots/ appear when it completes.

Usage:
    python toy_ate_recovery.py                      # notebook-equivalent defaults
    python toy_ate_recovery.py --size 8 --seed-fit 7 --max-epochs 200
    python toy_ate_recovery.py --replot runs/toy_ate_recovery/<run-id>

The default configuration reproduces ``validation/ff_morphomnist_toy.ipynb``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import secrets
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)  # prepare_data does `from dataset import ...`

import equinox
import flowjax
import frugal_flows
import jax
import jax.numpy as jnp
import jax.random as jr
import paramax
from frugal_flows.causal_flows import get_independent_quantiles, train_frugal_flow
from prepare_data import inverse_logit, prepare_flow_data, unflatten


@dataclass
class Config:
    # data
    size: int = 8            # image side; K = size^2 outcome dims
    digit: int = 0           # which digit class to subset
    n: int | None = None     # cap on subset size (None = all)
    seed_prepare: int = 0    # prepare_flow_data RNG
    seed_data: int = 123     # treatment-assignment RNG
    data_dir: str = "data"   # MorphoMNIST data, relative to this script
    # effect (the known truth)
    radius: int = 2          # circle radius, centred
    base_shift: float = 1.0  # effect magnitude in logit space
    # model / training
    ate_init: float = 0.5    # recorded: tau_hat is sensitive to its start
    rqs_knots: int = 8
    nn_depth: int = 1
    nn_width: int = 50
    flow_layers: int = 4
    learning_rate: float = 1e-2
    max_epochs: int = 100
    max_patience: int = 30
    batch_size: int = 100
    marginal_max_epochs: int = 70
    marginal_max_patience: int = 10
    seed_fit: int = 34
    x64: bool = False        # precision the flow trains at (recorded in config.json)


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def build_toy_data(cfg: Config) -> dict:
    """Digit subset + random treatment + known circle effect in logit space."""
    data = prepare_flow_data(
        os.path.join(SCRIPT_DIR, cfg.data_dir), size=cfg.size, seed=cfg.seed_prepare
    )
    Z = np.asarray(data["Z"])                 # (n, 11): [thickness, digit onehot]
    idx = np.where(Z[:, 1 + cfg.digit] == 1)[0]
    rng = np.random.default_rng(cfg.seed_data)
    if cfg.n is not None:
        idx = rng.choice(idx, cfg.n, replace=False)

    Y0 = np.asarray(data["Y0"])[idx]          # untreated images in logit space
    thickness = Z[idx, 0]
    X = (rng.uniform(size=len(Y0)) < 0.5).astype(np.float64)[:, None]

    xx, yy = np.meshgrid(np.arange(cfg.size), np.arange(cfg.size), indexing="ij")
    c = (cfg.size - 1) / 2
    circle = (((xx - c) ** 2 + (yy - c) ** 2) <= cfg.radius**2).astype(np.float64)
    ite = cfg.base_shift * circle.reshape(1, -1)          # (1, K)

    Y1 = Y0 + ite
    Y = np.where(X == 1, Y1, Y0)
    return {
        "Y": Y, "X": X, "thickness": thickness, "Y0": Y0, "Y1": Y1,
        "ite": ite, "ate_true": ite.mean(axis=0), "circle": circle.reshape(-1),
    }


def fit_flow(cfg: Config, toy: dict):
    """Stage-1 marginal quantiles for Z = thickness, then the frugal flow."""
    key = jr.PRNGKey(cfg.seed_fit)
    key, subkey = jr.split(key)
    z_res = get_independent_quantiles(
        key=subkey,
        z_cont=jnp.asarray(toy["thickness"])[:, None],
        max_epochs=cfg.marginal_max_epochs,
        max_patience=cfg.marginal_max_patience,
        return_z_cont_flow=True,
    )
    u_z = np.asarray(z_res["u_z_cont"])

    key, subkey = jr.split(key)
    flow, losses = train_frugal_flow(
        causal_model="location_translation",
        key=subkey,
        y=jnp.asarray(toy["Y"]),
        u_z=jnp.asarray(u_z),
        condition=jnp.asarray(toy["X"]),
        learning_rate=cfg.learning_rate,
        max_epochs=cfg.max_epochs,
        max_patience=cfg.max_patience,
        batch_size=cfg.batch_size,
        causal_model_args={
            "ate": cfg.ate_init,
            "RQS_knots": cfg.rqs_knots,
            "nn_depth": cfg.nn_depth,
            "nn_width": cfg.nn_width,
            "flow_layers": cfg.flow_layers,
        },
    )
    return flow, losses, u_z


def evaluate(cfg: Config, flow, toy: dict, losses: dict, wall_time_s: float) -> tuple[np.ndarray, dict]:
    """Per-pixel tau_hat off the LocCond blocks, scored against the known truth."""
    K = cfg.size**2
    loccond_block = flow.bijection.bijections[5]
    tau_hat = np.array(
        [float(paramax.unwrap(loccond_block.bijections[k]).ate) for k in range(K)]
    )
    truth = np.asarray(toy["ate_true"])
    inside = toy["circle"].astype(bool)
    err = tau_hat - truth
    metrics = {
        "ate_mae": float(np.abs(err).mean()),
        "ate_rmse": float(np.sqrt((err**2).mean())),
        "ate_corr": float(np.corrcoef(tau_hat, truth)[0, 1]),
        "ate_max_abs_err": float(np.abs(err).max()),
        "tau_hat_mean_inside_circle": float(tau_hat[inside].mean()),
        "tau_hat_mean_outside_circle": float(tau_hat[~inside].mean()),
        "true_effect_inside_circle": float(cfg.base_shift),
        "best_val_loss": float(np.min(losses["val"])),
        "n_epochs_run": int(len(losses["train"])),
        "n_units": int(toy["Y"].shape[0]),
        "n_pixels": int(K),
        "wall_time_s": float(wall_time_s),
    }
    return tau_hat, metrics


# --------------------------------------------------------------------------- #
# plots (all read from plain arrays, so --replot needs no refit)
# --------------------------------------------------------------------------- #
def make_plots(cfg: Config, toy: dict, losses: dict, tau_hat: np.ndarray, u_z: np.ndarray, plots_dir: str):
    os.makedirs(plots_dir, exist_ok=True)
    size = cfg.size

    def save(fig, name):
        fig.savefig(os.path.join(plots_dir, name), dpi=120, bbox_inches="tight")
        plt.close(fig)

    # 1. sample images: factual / Y(1) / Y(0)
    fig, axes = plt.subplots(3, 8, figsize=(14, 5))
    for row, (arr, label) in enumerate(
        [(toy["Y"], "factual"), (toy["Y1"], "Y(1)"), (toy["Y0"], "Y(0)")]
    ):
        imgs = inverse_logit(np.asarray(arr[:8]))
        for i, ax in enumerate(axes[row]):
            ax.imshow(unflatten(imgs[i], size)[0], cmap="gray")
            ax.axis("off")
        axes[row, 0].set_title(label, loc="left")
    fig.suptitle("Sample images (rows: factual, treated, untreated)")
    save(fig, "sample_images.png")

    # 2. Y(1) - Y(0) in pixel space, first 8 treated units
    T = np.asarray(toy["X"]).flatten().astype(bool)
    diff = inverse_logit(np.asarray(toy["Y1"])) - inverse_logit(np.asarray(toy["Y0"]))
    idx = np.where(T)[0][:8]
    fig, axes = plt.subplots(1, 8, figsize=(14, 2))
    for i, ax in zip(idx, axes):
        im = ax.imshow(diff[i].reshape(size, size), cmap="RdBu_r", vmin=-0.2, vmax=0.2)
        ax.axis("off")
    fig.suptitle("Y(1) − Y(0) (pixel space)")
    fig.colorbar(im, ax=axes, shrink=0.8)
    save(fig, "diff_maps.png")

    # 3. balance check: thickness and mean brightness by arm
    thickness = np.asarray(toy["thickness"])
    mean_bright = inverse_logit(np.asarray(toy["Y"])).mean(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].hist([thickness[~T], thickness[T]], bins=40, label=["T=0", "T=1"],
                 density=True, histtype="step")
    axes[0].legend()
    axes[0].set_title("Thickness by arm (balance check)")
    axes[1].hist([mean_bright[~T], mean_bright[T]], bins=40, label=["T=0", "T=1"],
                 density=True, histtype="step")
    axes[1].legend()
    axes[1].set_title("Mean brightness by arm")
    save(fig, "balance_check.png")

    # 4. the known truth: ATE map + heterogeneity summaries
    ate_true = np.asarray(toy["ate_true"])
    ite = np.asarray(toy["ite"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    im = axes[0].imshow(ate_true.reshape(size, size))
    fig.colorbar(im, ax=axes[0])
    axes[0].set_title("True per-pixel ATE")
    axes[1].hist(ite.sum(axis=1), bins=50)
    axes[1].set_title("Per-unit total ITE (heterogeneity)")
    im = axes[2].imshow(ite.std(axis=0).reshape(size, size))
    fig.colorbar(im, ax=axes[2])
    axes[2].set_title("ITE std across units")
    save(fig, "truth_panels.png")

    # 5. stage-1 quantiles sanity: U_Z should be ~Uniform
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(np.asarray(u_z)[:, 0], bins=30)
    ax.set_title(r"Histogram of $U_{Z}$ (thickness quantiles)")
    save(fig, "uz_hist.png")

    # 6. training curves
    train, val = np.array(losses["train"]), np.array(losses["val"])
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = np.arange(1, len(train) + 1)
    ax.plot(epochs, train, label="train", marker="o", markersize=3)
    ax.plot(epochs, val, label="val", marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    ax.set_title("Training / validation loss")
    save(fig, "loss_curves.png")

    # 7. the headline: estimated vs true ATE map, and their difference
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    vmax = float(max(np.abs(tau_hat).max(), np.abs(ate_true).max()))
    im0 = axes[0].imshow(tau_hat.reshape(size, size), vmin=-vmax, vmax=vmax)
    axes[0].set_title(r"Estimated $\hat{\tau}$ per pixel")
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(ate_true.reshape(size, size), vmin=-vmax, vmax=vmax)
    axes[1].set_title("True ATE per pixel")
    fig.colorbar(im1, ax=axes[1])
    err = (tau_hat - ate_true).reshape(size, size)
    lim = float(np.abs(err).max()) or 1.0
    im2 = axes[2].imshow(err, cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[2].set_title(r"Error $\hat{\tau} -$ truth")
    fig.colorbar(im2, ax=axes[2])
    save(fig, "ate_maps.png")


# --------------------------------------------------------------------------- #
# run folder + metadata
# --------------------------------------------------------------------------- #
def _git_info() -> dict:
    def run(*args):
        try:
            return subprocess.run(
                ["git", *args], cwd=SCRIPT_DIR, capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except Exception:
            return "unavailable"

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


class _Tee:
    """Duplicate a stream's writes into a file, so training progress shows on
    the terminal AND lands in the run folder's log.txt."""

    def __init__(self, stream, fileobj):
        self._stream = stream
        self._file = fileobj

    def write(self, s):
        self._stream.write(s)
        self._file.write(s)
        self._file.flush()
        return len(s)

    def flush(self):
        self._stream.flush()
        self._file.flush()


def write_config(cfg: Config, run_id: str, run_dir: str):
    """Create the run folder and record the full configuration at launch.

    Refuses an existing folder: a run-id collision must fail at launch rather
    than let two runs silently write into the same directory.
    """
    os.makedirs(run_dir, exist_ok=False)
    config_record = {
        "run_id": run_id,
        "config": asdict(cfg),
        "git": _git_info(),
        "versions": {
            "jax": jax.__version__,
            "flowjax": flowjax.__version__,
            "equinox": equinox.__version__,
            "numpy": np.__version__,
            "frugal_flows": getattr(frugal_flows, "__version__", "unversioned"),
        },
        "x64_active": bool(jax.config.jax_enable_x64),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config_record, f, indent=2)


def save_run(cfg: Config, toy: dict, losses: dict, tau_hat: np.ndarray, u_z: np.ndarray,
             metrics: dict, run_dir: str):
    metrics = {"run_id": os.path.basename(run_dir), **metrics}
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez(
        os.path.join(run_dir, "arrays.npz"),
        tau_hat=tau_hat,
        ate_true=np.asarray(toy["ate_true"]),
        ite=np.asarray(toy["ite"]),
        circle=np.asarray(toy["circle"]),
        Y0=np.asarray(toy["Y0"]),
        X=np.asarray(toy["X"]),
        thickness=np.asarray(toy["thickness"]),
        u_z=np.asarray(u_z),
        loss_train=np.asarray(losses["train"]),
        loss_val=np.asarray(losses["val"]),
    )
    make_plots(cfg, toy, losses, tau_hat, u_z, os.path.join(run_dir, "plots"))


def replot(run_dir: str):
    """Regenerate every plot for an existing run from its saved arrays."""
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = Config(**json.load(f)["config"])
    a = np.load(os.path.join(run_dir, "arrays.npz"))
    Y1 = a["Y0"] + a["ite"]
    toy = {
        "Y": np.where(a["X"] == 1, Y1, a["Y0"]), "X": a["X"], "Y0": a["Y0"], "Y1": Y1,
        "thickness": a["thickness"], "ite": a["ite"], "ate_true": a["ate_true"],
        "circle": a["circle"],
    }
    losses = {"train": a["loss_train"], "val": a["loss_val"]}
    make_plots(cfg, toy, losses, a["tau_hat"], a["u_z"], os.path.join(run_dir, "plots"))
    print(f"replotted: {os.path.join(run_dir, 'plots')}")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(Config):
        flag = "--" + f.name.replace("_", "-")
        if f.type == "bool":
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        elif f.type == "int | None":
            parser.add_argument(flag, type=int, default=f.default)
        else:
            parser.add_argument(flag, type=type(f.default), default=f.default)
    parser.add_argument("--replot", metavar="RUN_DIR", default=None,
                        help="regenerate plots for an existing run, no refit")
    args = parser.parse_args(argv)

    if args.replot is not None:
        replot(args.replot)
        return

    cfg = Config(**{f.name: getattr(args, f.name) for f in fields(Config)})
    jax.config.update("jax_enable_x64", cfg.x64)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_id = f"{stamp}_s{cfg.seed_fit}_k{cfg.size**2}_{secrets.token_hex(3)}"
    run_dir = os.path.join(SCRIPT_DIR, "runs", "toy_ate_recovery", run_id)
    write_config(cfg, run_id, run_dir)
    print(f"run dir: {run_dir}")

    with open(os.path.join(run_dir, "log.txt"), "w") as lf:
        out, err = _Tee(sys.stdout, lf), _Tee(sys.stderr, lf)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            print(f"started {stamp}; watch progress with: tail -f {run_dir}/log.txt")
            toy = build_toy_data(cfg)
            print(f"data built: {toy['Y'].shape[0]} units x {toy['Y'].shape[1]} pixels")
            t0 = time.monotonic()
            flow, losses, u_z = fit_flow(cfg, toy)
            wall = time.monotonic() - t0
            n_ep = len(losses["train"])
            print(f"fit finished: {n_ep} epochs in {wall:.0f}s (~{wall / n_ep:.1f}s/epoch)")
            tau_hat, metrics = evaluate(cfg, flow, toy, losses, wall)
            save_run(cfg, toy, losses, tau_hat, u_z, metrics, run_dir)
            for k, v in metrics.items():
                print(f"  {k}: {v:.4g}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"run dir: {run_dir}")


if __name__ == "__main__":
    main()
