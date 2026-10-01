"""Halo ladder: the flowjax arms, their fit and their paired read-out.

Arms (all K=64, logit pixels, float32):
    ff_stack     the package margin as a distribution: Uniform[-1,1]^K -> Invert(Scan(L x
                 [MAF layer with RQS(knots, interval 1), Permute])) -> Invert(Tanh) per pixel.
                 rank_mode "modulo" = flowjax ``MaskedAutoregressive`` (A1); "spread" =
                 Laura's ``MaskedAutoregressiveSpread`` (the package's current margin).
                 Built line-for-line like ``exp_ate_recovery._margin_dist`` (asserted in
                 ``test_halo``), so the same key gives the same weights.
    fj_normal    flowjax ``masked_autoregressive_flow`` on a Normal base, RQS(knots,
                 interval 5; identity outside) or flowjax's default affine transformer.
    fj_coupling  flowjax ``coupling_flow``, Normal base, RQS(knots, interval 5).
    loctrans     the package's LT margin (T masked from the MAF) followed by
                 ``LocCond(ate=0)``: y = margin + ate*t; tau_hat is the fitted ``ate``.
    normal_spread  (S6 "N", Amendment A2) Normal(0, I_K) -> Invert(Scan(L x
                 [``MaskedAutoregressiveSpread`` RQS(knots, interval 5), Permute])), no tanh;
                 T as the unmasked conditioner: fj_normal with Laura's spread ranks.
    loctrans_normal (S6 "LT-N") the same Normal-base stack with T MASKED, then the package
                 ``LocCond(ate=0)``, mirroring ``loctrans_margin``: each layer is the package's
                 ``MaskedAutoregressiveMaskedCond(cond_dim_mask=1)`` (T fed, given rank
                 ``dim`` so no output sees it; hidden ranks ``autoregressive_hidden_ranks(lo=0)``,
                 identical to ``MaskedAutoregressiveSpread`` unconditional). A plain
                 ``cond_dim=None`` MAF cannot be used: flowjax's MAF hstacks any condition the
                 Chain passes into its MLP input.
Training is ``frugal_flows.training.fit_to_data`` (bit-identical to flowjax's; no EMA, no
wall cap), with the key sequence of ``exp_ate_recovery._fit_margin_only``.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import paramax
from flowjax.bijections import (Invert, MaskedAutoregressive, RationalQuadraticSpline, Scan,
                                Stack, Tanh)
from flowjax.distributions import Normal, Transformed, Uniform
from flowjax.flows import _add_default_permute, coupling_flow, masked_autoregressive_flow
from frugal_flows.basic_flows import masked_autoregressive_bijection_masked_condition
from frugal_flows.bijections.loc_cond import LocCond
from frugal_flows.bijections.masked_autoregressive_masked_cond import MaskedAutoregressiveMaskedCond
from frugal_flows.bijections.masked_autoregressive_spread import MaskedAutoregressiveSpread
from frugal_flows.interventions import sample_clamped
from frugal_flows.training import fit_to_data

RANK_CLASSES = {"modulo": MaskedAutoregressive, "spread": MaskedAutoregressiveSpread}


def ff_stack(key, K: int, cond_dim: int | None = None, width: int = 48, depth: int = 1,
             layers: int = 4, knots: int = 8, rank_mode: str = "modulo") -> Transformed:
    """The FF margin as a stand-alone distribution (see module docstring)."""
    cls = RANK_CLASSES[rank_mode]
    transformer = RationalQuadraticSpline(knots=knots, interval=1)

    def make_layer(k):
        bk, pk = jr.split(k)
        b = cls(key=bk, transformer=transformer, dim=K, cond_dim=cond_dim or None,
                nn_width=width, nn_depth=depth)
        return _add_default_permute(b, K, pk)

    margin = Invert(Scan(eqx.filter_vmap(make_layer)(jr.split(key, layers))))
    dist = Transformed(Uniform(-jnp.ones(K), jnp.ones(K)), margin)
    return Transformed(dist, Stack([Invert(Tanh(()))] * K)).merge_transforms()


def fj_normal(key, K: int, cond_dim: int | None = None, transformer: str = "rqs",
              interval: float = 5.0, width: int = 48, depth: int = 1, layers: int = 4,
              knots: int = 8) -> Transformed:
    """flowjax MAF on a standard Normal base. ``transformer="affine"`` passes None, i.e.
    flowjax's own default (affine with a minimum scale)."""
    tr = RationalQuadraticSpline(knots=knots, interval=interval) if transformer == "rqs" else None
    return masked_autoregressive_flow(key, base_dist=Normal(jnp.zeros(K)), transformer=tr,
                                      cond_dim=cond_dim or None, flow_layers=layers,
                                      nn_width=width, nn_depth=depth)


def fj_coupling(key, K: int, cond_dim: int | None = None, interval: float = 5.0,
                width: int = 48, depth: int = 1, layers: int = 4, knots: int = 8) -> Transformed:
    """flowjax RQS coupling flow on a standard Normal base."""
    return coupling_flow(key, base_dist=Normal(jnp.zeros(K)),
                         transformer=RationalQuadraticSpline(knots=knots, interval=interval),
                         cond_dim=cond_dim or None, flow_layers=layers, nn_width=width, nn_depth=depth)


def loctrans_margin(key, K: int, width: int = 48, depth: int = 1, layers: int = 4,
                    knots: int = 8) -> Transformed:
    """The package's location-translation margin without the copula: Uniform[-1,1]^K ->
    ``masked_autoregressive_bijection_masked_condition`` (T is fed but MASKED from every
    output, spread ranks; ``causal_flows.train_frugal_flow_location_translation``) ->
    Invert(Tanh) -> ``LocCond(ate=0)``, i.e. y = margin(u) + ate * t per pixel."""
    margin = masked_autoregressive_bijection_masked_condition(
        key=key, dim=K, condition=jnp.zeros((1, 1)), RQS_knots=knots, nn_depth=depth,
        nn_width=width, flow_layers=layers)
    dist = Transformed(Uniform(-jnp.ones(K), jnp.ones(K)), margin)
    dist = Transformed(dist, Stack([Invert(Tanh(()))] * K))
    return Transformed(dist, LocCond(ate=jnp.zeros(K), cond_dim=1)).merge_transforms()


def _spread_normal_bijection(key, K: int, cond_dim: int | None, width: int, depth: int,
                             layers: int, knots: int, interval: float, mask_cond: bool = False):
    """Invert(Scan(L x [MAF layer RQS(knots, interval), Permute])), built with flowjax
    ``masked_autoregressive_flow``'s key split (bij_key, perm_key) per layer. Layer =
    ``MaskedAutoregressiveSpread`` (T unmasked when ``cond_dim``) or, with ``mask_cond``,
    the package's ``MaskedAutoregressiveMaskedCond(cond_dim_mask=1)`` (T masked)."""
    transformer = RationalQuadraticSpline(knots=knots, interval=interval)

    def make_layer(k):
        bk, pk = jr.split(k)
        if mask_cond:
            b = MaskedAutoregressiveMaskedCond(key=bk, transformer=transformer, dim=K, cond_dim_mask=1,
                                               nn_width=width, nn_depth=depth)
        else:
            b = MaskedAutoregressiveSpread(key=bk, transformer=transformer, dim=K, cond_dim=cond_dim or None,
                                           nn_width=width, nn_depth=depth)
        return _add_default_permute(b, K, pk)

    return Invert(Scan(eqx.filter_vmap(make_layer)(jr.split(key, layers))))


def normal_spread(key, K: int, cond_dim: int | None = None, width: int = 48, depth: int = 1,
                  layers: int = 4, knots: int = 8, interval: float = 5.0) -> Transformed:
    """S6 arm N (see module docstring)."""
    return Transformed(Normal(jnp.zeros(K)),
                       _spread_normal_bijection(key, K, cond_dim, width, depth, layers, knots, interval))


def loctrans_normal(key, K: int, width: int = 48, depth: int = 1, layers: int = 4,
                    knots: int = 8, interval: float = 5.0) -> Transformed:
    """S6 arm LT-N: Normal-base stack with T masked, then ``LocCond(ate=0)`` composed exactly
    as in ``loctrans_margin`` (y = margin(z) + ate * t per pixel)."""
    dist = Transformed(Normal(jnp.zeros(K)),
                       _spread_normal_bijection(key, K, None, width, depth, layers, knots, interval,
                                                mask_cond=True))
    return Transformed(dist, LocCond(ate=jnp.zeros(K), cond_dim=1)).merge_transforms()


def loccond_ate(dist) -> np.ndarray:
    """The fitted ``ate`` vector of the (single) LocCond in a loctrans distribution."""
    hits = [x for x in jax.tree_util.tree_leaves(
        paramax.unwrap(dist), is_leaf=lambda x: isinstance(x, LocCond)) if isinstance(x, LocCond)]
    assert len(hits) == 1, len(hits)
    return np.asarray(hits[0].ate, dtype=np.float64)


def build(arm: str, key, K: int, cond_dim: int | None, cfg: dict):
    """Dispatch an arm name to its builder with the cell's hyperparameters."""
    hp = dict(width=cfg["width"], depth=cfg["depth"], layers=cfg["layers"], knots=cfg["knots"])
    if arm in ("A1", "A1s", "A1w", "Csmooth", "Czinf", "ff_cond", "sep"):
        return ff_stack(key, K, cond_dim, rank_mode=cfg["rank_mode"], **hp)
    if arm in ("A2", "a2_cond"):
        return fj_normal(key, K, cond_dim, "rqs", **hp)
    if arm == "A3":
        return fj_normal(key, K, cond_dim, "affine", **hp)
    if arm == "A4":
        return fj_coupling(key, K, cond_dim, **hp)
    if arm == "lt":
        return loctrans_margin(key, K, **hp)
    if arm == "n_cond":
        return normal_spread(key, K, cond_dim, **hp)
    if arm == "lt_n":
        return loctrans_normal(key, K, **hp)
    raise ValueError(f"unknown jax arm {arm!r}")


def fit(key, dist, Y, cond, cfg: dict):
    """``frugal_flows.training.fit_to_data`` with the ladder's common settings."""
    data = (jnp.asarray(Y), jnp.asarray(cond)) if cond is not None else jnp.asarray(Y)
    return fit_to_data(key, dist, data=data, learning_rate=cfg["lr"], max_epochs=cfg["max_epochs"],
                       max_patience=cfg["patience"], batch_size=cfg["batch"], val_prop=0.1,
                       return_best=True, show_progress=False, ema_decay=None, wall_cap_s=None)


def fit_arm(cfg: dict, Y: np.ndarray, X: np.ndarray):
    """Build and fit the cell's jax arm with Laura's margin-only key sequence.
    Returns (dist or (dist0, dist1) for SEP, list of loss dicts)."""
    K = Y.shape[1]
    key = jr.PRNGKey(cfg["seed_fit"])
    key, _ = jr.split(key)                      # stage-1 slot, as in _fit_margin_only
    arm = cfg["arm"]
    if arm == "sep":
        t = X[:, 0].astype(bool)
        out, losses = [], []
        for rows in (~t, t):
            key, bkey, fkey = jr.split(key, 3)
            d, l = fit(fkey, build(arm, bkey, K, None, cfg), Y[rows], None, cfg)
            out.append(d), losses.append(l)
        return tuple(out), losses
    cond_dim = 1 if cfg["task"] == "cond" else None
    key, bkey, fkey = jr.split(key, 3)
    d, l = fit(fkey, build(arm, bkey, K, cond_dim, cfg), Y, X if cond_dim else None, cfg)
    return d, [l]


def sample_arms(seed_mc: int, dist, n_mc: int, task: str):
    """Paired draws with common random numbers (one typed key for both arms).
    Returns (y0, y1 or None, n_clamped). Normal-base flows bypass the clamp (n_clamped 0
    there does not mean 'no boundary issue')."""
    key = jr.key(seed_mc)
    if isinstance(dist, tuple):                 # SEP: same key -> same base draws
        s0, c0 = sample_clamped(key, dist[0], n_mc)
        s1, c1 = sample_clamped(key, dist[1], n_mc)
        return np.asarray(s0), np.asarray(s1), c0 + c1
    if task == "uncond":
        s, c = sample_clamped(key, dist, n_mc)
        return np.asarray(s), None, c
    s0, c0 = sample_clamped(key, dist, n_mc, jnp.zeros((n_mc, 1)))
    s1, c1 = sample_clamped(key, dist, n_mc, jnp.ones((n_mc, 1)))
    return np.asarray(s0), np.asarray(s1), c0 + c1
