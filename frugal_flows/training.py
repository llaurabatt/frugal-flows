"""A drop-in replacement for ``flowjax.train.fit_to_data`` that records what the library
keeps to itself.

Called with the library's arguments only, it reproduces ``flowjax.train.fit_to_data`` bit
for bit: the same random-key sequence (one split for the train/validation split, three per
epoch for the two shuffles, one per training batch and one per validation batch), the same
``step``, the same loss, the same batching (last truncated batch dropped), the same
best-parameter rule (an epoch whose validation loss equals the running minimum becomes the
best; training stops when more than ``max_patience`` epochs have passed since the minimum).
The reproduction is asserted by ``tests/test_training_loop.py`` against the library.

On top of that it returns, inside ``losses["info"]``:

* ``train_idx`` / ``val_idx``: the row indices the split put in each set;
* ``best_epoch`` (1-based), ``termination`` (``patience`` / ``epoch_cap`` / ``wall_cap``),
  ``n_epochs``, ``wall_s``;

and takes four optional extras:

* ``wall_cap_s``: stop after the first epoch that ends past this many seconds;
* ``select_fn(params, static, *val_arrays) -> float``: a criterion, lower is better,
  evaluated on the WHOLE validation set after every epoch; when given, the best parameters
  and the patience count follow it instead of the batched validation loss (which is still
  recorded in ``losses["val"]``); its series is returned in ``losses["select"]``;
* ``on_epoch(epoch, params, static) -> dict | None``: called after every epoch; whatever
  it returns (a dict) is appended, with the epoch number, to ``losses["track"]``.
* ``ema_decay``: keep an exponential moving average of the parameters, updated after every
  optimisation step (``ema = d * ema + (1 - d) * params``, started at the initial
  parameters). When given, the validation loss, ``select_fn``, ``on_epoch``, the choice of
  the best epoch and the returned distribution all use the AVERAGED parameters; the
  optimiser itself still steps the raw ones. Motivation (2026-09-27): the effect a flow
  implies is a tiny part of its likelihood, so the raw parameters drift along directions
  the loss barely sees, and the kept epoch inherits that drift (fit-seed control: the
  margin's own error was entirely fit-seed noise). ``None`` (default) keeps the library's
  behaviour bit for bit; ``0.0`` gives the raw parameters exactly.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Iterable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import paramax
from flowjax.train.losses import MaximumLikelihoodLoss
from flowjax.train.train_utils import (
    count_fruitless,
    get_batches,
    step,
    train_val_split,
)
from jaxtyping import ArrayLike, PRNGKeyArray, PyTree
from tqdm import tqdm


def fit_to_data(
    key: PRNGKeyArray,
    dist: PyTree,
    data: ArrayLike | Iterable[ArrayLike] = (),
    *,
    loss_fn: Callable | None = None,
    learning_rate: float = 5e-4,
    optimizer: optax.GradientTransformation | None = None,
    max_epochs: int = 100,
    max_patience: int = 5,
    batch_size: int = 100,
    val_prop: float = 0.1,
    return_best: bool = True,
    show_progress: bool = True,
    wall_cap_s: float | None = None,
    select_fn: Callable | None = None,
    on_epoch: Callable | None = None,
    ema_decay: float | None = None,
):
    """See the module docstring. Returns ``(dist, losses)`` like the library."""
    t_start = time.monotonic()
    data = (data,) if isinstance(data, ArrayLike) else data
    data = tuple(jnp.asarray(a) for a in data)

    if loss_fn is None:
        loss_fn = MaximumLikelihoodLoss()
    if optimizer is None:
        optimizer = optax.adam(learning_rate)

    params, static = eqx.partition(
        dist,
        eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
    )
    best_params = params
    opt_state = optimizer.init(params)
    if ema_decay is not None and not 0.0 <= ema_decay < 1.0:
        raise ValueError(f"ema_decay must be in [0, 1), got {ema_decay}")
    ema = params
    ema_update = eqx.filter_jit(
        lambda e, p: jax.tree_util.tree_map(lambda a, b: ema_decay * a + (1.0 - ema_decay) * b, e, p)
    )

    # train / validation split: the library permutes each array with ``subkey``; the
    # permutation of ``arange(n)`` under the same key is the row order it used
    key, subkey = jr.split(key)
    train_data, val_data = train_val_split(subkey, data, val_prop=val_prop)
    n = int(data[0].shape[0])
    perm = np.asarray(jr.permutation(subkey, jnp.arange(n)))
    n_train = n - round(val_prop * n)
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    val_full = tuple(a[val_idx] for a in data)          # unshuffled, for select_fn

    losses: dict = {"train": [], "val": []}
    if select_fn is not None:
        losses["select"] = []
    if on_epoch is not None:
        losses["track"] = []
    termination = "epoch_cap"
    best_epoch = 0

    loop = tqdm(range(max_epochs), disable=not show_progress)
    for epoch in loop:
        key, *subkeys = jr.split(key, 3)
        train_data = [jr.permutation(subkeys[0], a) for a in train_data]
        val_data = [jr.permutation(subkeys[1], a) for a in val_data]

        batch_losses = []
        for batch in zip(*get_batches(train_data, batch_size), strict=True):
            key, subkey = jr.split(key)
            params, opt_state, loss_i = step(
                params, static, *batch, optimizer=optimizer, opt_state=opt_state,
                loss_fn=loss_fn, key=subkey,
            )
            batch_losses.append(loss_i)
            if ema_decay is not None:
                ema = ema_update(ema, params)
        losses["train"].append((sum(batch_losses) / len(batch_losses)).item())
        # the parameters everything below is judged on: the running average when it is kept
        eval_params = params if ema_decay is None else ema

        batch_losses = []
        for batch in zip(*get_batches(val_data, batch_size), strict=True):
            key, subkey = jr.split(key)
            loss_i = eqx.filter_jit(loss_fn)(eval_params, static, *batch, key=subkey)
            batch_losses.append(loss_i)
        losses["val"].append((sum(batch_losses) / len(batch_losses)).item())

        if select_fn is not None:
            losses["select"].append(float(select_fn(eval_params, static, *val_full)))
            criterion = losses["select"]
        else:
            criterion = losses["val"]
        if on_epoch is not None:
            row = on_epoch(epoch + 1, eval_params, static)
            if row:
                losses["track"].append({"epoch": epoch + 1, **row})

        loop.set_postfix({k: v[-1] for k, v in losses.items() if k in ("train", "val", "select")})
        if criterion[-1] == min(criterion):
            best_params = eval_params
            best_epoch = epoch + 1
        elif count_fruitless(criterion) > max_patience:
            loop.set_postfix_str(f"{loop.postfix} (Max patience reached)")
            termination = "patience"
            break
        if wall_cap_s is not None and time.monotonic() - t_start > wall_cap_s:
            termination = "wall_cap"
            break

    params = best_params if return_best else (params if ema_decay is None else ema)
    dist = eqx.combine(params, static)
    losses["info"] = {
        "train_idx": train_idx, "val_idx": val_idx, "n_train": int(n_train), "n_val": int(n - n_train),
        "best_epoch": int(best_epoch), "n_epochs": len(losses["train"]),
        "termination": termination, "wall_s": float(time.monotonic() - t_start),
        "selected_on": "select_fn" if select_fn is not None else "val_loss",
        "ema_decay": ema_decay,
    }
    return dist, losses
