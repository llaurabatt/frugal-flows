"""Halo ladder S3: the same tasks in a different NF library (PyTorch, zuko 1.6).

Arms (Normal base, 4 transforms, one hidden layer of 48, as the flowjax arms):
    zuko_nsf           ``zuko.flows.NSF``: autoregressive monotone RQS, 8 bins, domain [-5, 5]
                       (identity outside)
    zuko_maf           ``zuko.flows.MAF``: autoregressive affine
    zuko_nsf_coupling  ``zuko.flows.NSF(passes=2)``: the same RQS as coupling transforms
                       (zuko's documented way to get coupling NSF; there is no separate class)
Training mirrors ``fit_to_data``: Adam, batch 100 (last truncated batch dropped), a seeded
10% validation split, per-epoch shuffles, best-validation checkpoint, patience counted as
epochs since the minimum. Differences: the validation loss is the mean over the WHOLE
validation set (not over batches) and torch's RNG, not JAX's.
Sampling uses common random numbers exactly: one Normal base draw z, mapped through
``flow(c).transform.inv(z)`` under c = 0 and c = 1.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import zuko

ARMS = ("zuko_nsf", "zuko_maf", "zuko_nsf_coupling")


def build(arm: str, K: int, cond_dim: int, cfg: dict) -> zuko.flows.Flow:
    kw = dict(features=K, context=cond_dim, transforms=cfg["layers"],
              hidden_features=(cfg["width"],) * cfg["depth"])
    if arm == "zuko_nsf":
        return zuko.flows.NSF(bins=cfg["knots"], **kw)
    if arm == "zuko_maf":
        return zuko.flows.MAF(**kw)
    if arm == "zuko_nsf_coupling":
        return zuko.flows.NSF(bins=cfg["knots"], passes=2, **kw)
    raise ValueError(arm)


def _nll(flow, y, c):
    return -(flow(c).log_prob(y) if c is not None else flow().log_prob(y)).mean()


def fit(cfg: dict, Y: np.ndarray, X: np.ndarray | None, lr: float):
    """Train one zuko flow. Returns (flow, losses dict with train/val/info)."""
    torch.set_num_threads(1)
    torch.manual_seed(cfg["seed_fit"])
    K = Y.shape[1]
    cond_dim = X.shape[1] if X is not None else 0
    flow = build(cfg["arm"], K, cond_dim, cfg)
    g = torch.Generator().manual_seed(cfg["seed_fit"])
    n = len(Y)
    perm = torch.randperm(n, generator=g)
    n_val = round(0.1 * n)
    tr, va = perm[: n - n_val], perm[n - n_val:]
    y = torch.as_tensor(Y, dtype=torch.float32)
    c = torch.as_tensor(X, dtype=torch.float32) if X is not None else None
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    losses = {"train": [], "val": []}
    best, best_state, best_epoch, term = np.inf, copy.deepcopy(flow.state_dict()), 0, "epoch_cap"
    t0 = time.monotonic()
    B = cfg["batch"]
    for epoch in range(cfg["max_epochs"]):
        flow.train()
        order = tr[torch.randperm(len(tr), generator=g)]
        bl = []
        for i in range(0, len(order) - B + 1, B):
            idx = order[i: i + B]
            loss = _nll(flow, y[idx], None if c is None else c[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            bl.append(loss.item())
        flow.eval()
        with torch.no_grad():
            v = _nll(flow, y[va], None if c is None else c[va]).item()
        losses["train"].append(float(np.mean(bl)))
        losses["val"].append(v)
        if v <= best:          # ties become the best, as in fit_to_data
            best, best_state, best_epoch = v, copy.deepcopy(flow.state_dict()), epoch + 1
        elif epoch + 1 - best_epoch > cfg["patience"]:
            term = "patience"
            break
        if not np.isfinite(v):
            term = "nonfinite"
            break
    flow.load_state_dict(best_state)
    losses["info"] = {"best_epoch": best_epoch, "n_epochs": len(losses["train"]),
                      "termination": term, "wall_s": time.monotonic() - t0, "lr": lr,
                      "val_idx": va.numpy(), "train_idx": tr.numpy()}
    return flow, losses


def fit_with_fallback(cfg: dict, Y, X):
    """lr = cfg["lr"] first; if the best validation loss is non-finite, refit at 1e-3
    (recorded in losses["info"]["lr"] and ["lr_fallback"])."""
    flow, losses = fit(cfg, Y, X, cfg["lr"])
    if not np.isfinite(min(losses["val"], default=np.inf)):
        flow, losses = fit(cfg, Y, X, 1e-3)
        losses["info"]["lr_fallback"] = True
    return flow, losses


@torch.no_grad()
def sample_arms(flow, n_mc: int, seed_mc: int, task: str, K: int):
    """(y0, y1 or None) from ONE base draw z (common random numbers)."""
    z = torch.randn(n_mc, K, generator=torch.Generator().manual_seed(seed_mc))
    if task == "uncond":
        return flow().transform.inv(z).numpy(), None
    y0 = flow(torch.zeros(n_mc, 1)).transform.inv(z).numpy()
    y1 = flow(torch.ones(n_mc, 1)).transform.inv(z).numpy()
    return y0, y1
