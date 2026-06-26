"""Prepare MorphoMNIST images as flat (n, K) arrays for frugal-flow training.

This is the single boundary between image-shaped data and the frugal_flows
library: everything downstream of `prepare_flow_data` sees only 2-D arrays.


Then the experiment notebook/script does:
data = prepare_flow_data("data/", size=8, n_samples=10000)
data["Y"].shape == (10000, 64) — frugal_flows never sees an image shape
np.savez("morphomnist_flow_data_8x8.npz", **{k: np.asarray(v) for k, v in data.items()})
"""

import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F

from dataset import MorphoMNIST


def downsample(images, size=8):
    """(n, 1, 32, 32) in [-1, 1] -> (n, 1, size, size), via adaptive average pooling."""
    return F.adaptive_avg_pool2d(images, output_size=(size, size))

def unflatten(Y_flat, size=8):
    """(n, K) -> (n, size, size), for plotting only."""
    return np.asarray(Y_flat).reshape(-1, size, size)

def dequantize_and_logit(images_flat, rng, alpha=0.05):
    """Map pixel values to unbounded reals suitable for a continuous flow.

    images_flat : (n, K) numpy array in [-1, 1]
    Steps:
      1. [-1, 1] -> [0, 1]
      2. add uniform dequantisation noise at the original 256-level granularity
      3. squeeze into (alpha/2, 1 - alpha/2) to avoid logit blow-up at 0/1
      4. logit transform -> reals
    Returns (y, ) and the forward is invertible via `inverse_logit`.
    """
    x = (images_flat + 1.0) / 2.0
    x = x + rng.uniform(0.0, 1.0 / 256.0, size=x.shape)
    x = np.clip(x, 0.0, 1.0)
    x = alpha / 2 + (1 - alpha) * x
    return np.log(x) - np.log1p(-x)


def inverse_logit(y, alpha=0.05):
    """Map flow-space values back to [0, 1] pixel intensities."""
    x = 1.0 / (1.0 + np.exp(-y))
    x = (x - alpha / 2) / (1 - alpha)
    return np.clip(x, 0.0, 1.0)


def make_ite(Y0_flat, Z, base_shift=0.5, size=8):
    """Per-unit treatment effect tau_i, shape (n, K).

    Example of a heterogeneous effect: brightness shift applied only where
    the (untreated) image has ink, scaled by a covariate. Replace with
    whatever effect your toy demands.
    """
    ink_mask = (Y0_flat > Y0_flat.mean(axis=1, keepdims=True)).astype(np.float64)
    scale = 1.0 + 0.5 * (Z[:, [0]] - Z[:, [0]].mean())   # covariate-modulated
    return base_shift * ink_mask * scale                  # (n, K)


def prepare_flow_data(data_dir, split="train", size=8, n_samples=None, seed=0):
    ds = MorphoMNIST(data_dir, split=split)
    rng = np.random.default_rng(seed)
    n = len(ds) if n_samples is None else min(n_samples, len(ds))
    idx = rng.permutation(len(ds))[:n]

    # --- covariates and confounded treatment assignment ---
    thickness = ds.thickness[idx].numpy()
    Z = np.hstack([thickness[:, None], ds.digit[idx].numpy()])
    z_score = (thickness - thickness.mean()) / thickness.std()
    intercept = 0.0        # 0 → ~50% treated; -1.0 → ~27%; +1.0 → ~73%
    slope     = 1.5        # confounding strength (0 = RCT, no confounding)
    p = 1.0 / (1.0 + np.exp(-(intercept + slope * z_score)))
    T = (rng.uniform(size=n) < p).astype(np.float64)[:, None]

    # --- potential outcomes in flow (logit) space ---
    images = downsample(ds.images[idx], size=size)
    Y0 = dequantize_and_logit(images.reshape(n, -1).numpy(), rng)   # untreated
    ITE = make_ite(Y0, Z)                                            # (n, K)
    Y1 = Y0 + ITE
    Y_obs = np.where(T == 1, Y1, Y0)        # factual outcome — the ONLY Y the model sees

    return {
        # ---- model inputs (training may touch only these) ----
        "Y": jnp.asarray(Y_obs),
        "X": jnp.asarray(T),                 # treatment
        "Z": jnp.asarray(Z),                 # covariates/confounders
        # ---- ground truth (evaluation only) ----
        "Y0": jnp.asarray(Y0),
        "Y1": jnp.asarray(Y1),
        "ITE": jnp.asarray(ITE),             # (n, K) per-unit, per-pixel
        "ATE": jnp.asarray(ITE.mean(axis=0)),  # (K,) — save this
        "image_size": size,
        "seed": seed,
    }