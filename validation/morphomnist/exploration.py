"""exploration.py — MorphoMNIST data exploration and causal diagnostics.

Run from validation/morphomnist/:
    python exploration.py --data_dir data/ --out_dir exploration_outputs/

Produces:
    sample_grid.png           — random image samples arranged in a grid
    samples_by_digit.png      — one row per digit class
    samples_by_thickness.png  — images sorted by thickness quintile
    distributions.png         — marginal distributions of thickness and intensity
    causal_scatter.png        — thickness vs intensity scatter (empirical DAG edge)
    ate_thickness.png         — estimated effect of thickness on intensity
    dataset_stats.txt         — summary statistics
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

from dataset import MorphoMNIST, unnormalize_attr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_uint8(tensor):
    """(C, H, W) in [-1, 1] -> (H, W, 3) uint8 for matplotlib."""
    img = (tensor.squeeze().numpy() + 1.0) / 2.0   # [0, 1]
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def save_grid(images, path, nrow=16, title=None):
    """Save a list of (1,H,W) tensors as a single image grid."""
    grid = make_grid(torch.stack(images), nrow=nrow, normalize=True, value_range=(-1, 1), pad_value=1)
    arr  = grid.permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(figsize=(nrow * 0.6, len(images) / nrow * 0.6 + 0.5))
    ax.imshow(arr, cmap="gray")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Image grids
# ---------------------------------------------------------------------------

def save_random_grid(dataset, out_dir, n=128, nrow=16):
    idx = torch.randperm(len(dataset))[:n]
    images = [dataset[i][0] for i in idx]
    save_grid(images, os.path.join(out_dir, "sample_grid.png"),
              nrow=nrow, title="Random samples")


def save_samples_by_digit(dataset, out_dir, n_per_class=16):
    """One row per digit class (0–9)."""
    digit_labels = dataset.digit.argmax(dim=1)   # (N,)
    rows = []
    for cls in range(10):
        idx = (digit_labels == cls).nonzero(as_tuple=True)[0]
        chosen = idx[torch.randperm(len(idx))[:n_per_class]]
        rows.extend([dataset.images[i] for i in chosen])
    save_grid(rows, os.path.join(out_dir, "samples_by_digit.png"),
              nrow=n_per_class, title="Samples by digit class (rows 0–9)")


def save_samples_by_thickness(dataset, out_dir, n_quintiles=5, n_per_bin=16):
    """Images sorted into thickness quintiles, one row per quintile."""
    t = dataset.thickness.numpy()
    quantiles = np.quantile(t, np.linspace(0, 1, n_quintiles + 1))
    rows = []
    for i in range(n_quintiles):
        lo, hi = quantiles[i], quantiles[i + 1]
        idx = np.where((t >= lo) & (t <= hi))[0]
        chosen = np.random.choice(idx, size=min(n_per_bin, len(idx)), replace=False)
        rows.extend([dataset.images[j] for j in chosen])
    save_grid(rows, os.path.join(out_dir, "samples_by_thickness.png"),
              nrow=n_per_bin,
              title="Samples by thickness quintile (thin → thick, top to bottom)")


# ---------------------------------------------------------------------------
# Distribution plots
# ---------------------------------------------------------------------------

def plot_distributions(dataset, out_dir):
    t = unnormalize_attr(dataset.thickness.numpy(), "thickness")
    s = unnormalize_attr(dataset.intensity.numpy(), "intensity")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(t, bins=60, color="steelblue", edgecolor="white", linewidth=0.3)
    axes[0].set_title("Thickness distribution")
    axes[0].set_xlabel("Thickness (raw units)")
    axes[0].set_ylabel("Count")

    axes[1].hist(s, bins=60, color="tomato", edgecolor="white", linewidth=0.3)
    axes[1].set_title("Intensity distribution")
    axes[1].set_xlabel("Intensity (raw units)")

    fig.tight_layout()
    path = os.path.join(out_dir, "distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot_causal_scatter(dataset, out_dir, n_points=5000):
    """Scatter plot visualising the causal edge thickness -> intensity."""
    t = unnormalize_attr(dataset.thickness.numpy(), "thickness")
    s = unnormalize_attr(dataset.intensity.numpy(), "intensity")

    idx = np.random.choice(len(t), size=min(n_points, len(t)), replace=False)
    digit_labels = dataset.digit.argmax(dim=1).numpy()

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(t[idx], s[idx], c=digit_labels[idx],
                         cmap="tab10", alpha=0.4, s=6, rasterized=True)
    cbar = fig.colorbar(scatter, ax=ax, ticks=range(10))
    cbar.set_label("Digit class")
    ax.set_xlabel("Thickness")
    ax.set_ylabel("Intensity")
    ax.set_title("Thickness → Intensity (colour = digit class)")

    # Simple linear fit
    coeffs = np.polyfit(t[idx], s[idx], 1)
    x_line = np.linspace(t.min(), t.max(), 100)
    ax.plot(x_line, np.polyval(coeffs, x_line), "k--", linewidth=1.5,
            label=f"OLS slope = {coeffs[0]:.2f}")
    ax.legend()

    fig.tight_layout()
    path = os.path.join(out_dir, "causal_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Causal / ATE diagnostics (observational, no model required)
# ---------------------------------------------------------------------------

def estimate_ate_thickness_on_intensity(dataset, out_dir, n_bins=10):
    """
    Empirical E[intensity | thickness bin] — a proxy for the causal effect
    of thickness on intensity using the observational data.

    Computes the bin-wise conditional mean and variance, and plots them.
    This is only observational (not interventional) but illustrates the
    causal edge in the data.
    """
    t = unnormalize_attr(dataset.thickness.numpy(), "thickness")
    s = unnormalize_attr(dataset.intensity.numpy(), "intensity")

    bin_edges  = np.quantile(t, np.linspace(0, 1, n_bins + 1))
    bin_mids   = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_means  = []
    bin_stds   = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (t >= lo) & (t <= hi)
        vals = s[mask]
        bin_means.append(vals.mean())
        bin_stds.append(vals.std())

    bin_means = np.array(bin_means)
    bin_stds  = np.array(bin_stds)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(bin_mids, bin_means, "o-", color="steelblue", label="E[intensity | thickness bin]")
    ax.fill_between(bin_mids, bin_means - bin_stds, bin_means + bin_stds,
                    alpha=0.2, color="steelblue", label="±1 std")
    ax.set_xlabel("Thickness (bin midpoint)")
    ax.set_ylabel("Intensity")
    ax.set_title("Empirical effect of thickness on intensity\n(observational, binned conditional mean)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "ate_thickness.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")

    # Print numeric summary
    naive_ate = bin_means[-1] - bin_means[0]
    print(f"\n  Naive observational ATE (top bin - bottom bin): {naive_ate:.2f} intensity units")
    return bin_mids, bin_means, bin_stds


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def save_stats(train_set, test_set, out_dir):
    lines = []
    for name, ds in [("train", train_set), ("test", test_set)]:
        t = unnormalize_attr(ds.thickness.numpy(), "thickness")
        s = unnormalize_attr(ds.intensity.numpy(), "intensity")
        digit_counts = ds.digit.argmax(dim=1).bincount(minlength=10).numpy()
        lines.append(f"=== {name} set ({len(ds)} samples) ===")
        lines.append(f"  thickness  : mean={t.mean():.3f}  std={t.std():.3f}  min={t.min():.3f}  max={t.max():.3f}")
        lines.append(f"  intensity  : mean={s.mean():.3f}  std={s.std():.3f}  min={s.min():.3f}  max={s.max():.3f}")
        lines.append(f"  digit dist : { {i: int(c) for i, c in enumerate(digit_counts)} }")
        lines.append(f"  image shape: {ds.images.shape}")
        lines.append("")

    text = "\n".join(lines)
    print("\n" + text)
    path = os.path.join(out_dir, "dataset_stats.txt")
    with open(path, "w") as f:
        f.write(text)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.path.join(os.path.dirname(__file__), "data"))
    parser.add_argument("--out_dir",  default=os.path.join(os.path.dirname(__file__), "exploration_outputs"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(0)

    print("Loading datasets …")
    train_set = MorphoMNIST(args.data_dir, split="train")
    test_set  = MorphoMNIST(args.data_dir, split="test")
    print(f"  train: {len(train_set)} samples   test: {len(test_set)} samples")

    print("\nSaving image grids …")
    save_random_grid(train_set, args.out_dir)
    save_samples_by_digit(train_set, args.out_dir)
    save_samples_by_thickness(train_set, args.out_dir)

    print("\nPlotting distributions …")
    plot_distributions(train_set, args.out_dir)
    plot_causal_scatter(train_set, args.out_dir)

    print("\nCausal / ATE diagnostics …")
    estimate_ate_thickness_on_intensity(train_set, args.out_dir)

    print("\nDataset statistics …")
    save_stats(train_set, test_set, args.out_dir)

    print("\nDone. All outputs in:", args.out_dir)


if __name__ == "__main__":
    main()