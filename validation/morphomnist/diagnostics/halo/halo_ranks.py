"""Stage S7 (HALO_PREREG Amendment A3): the copula's hidden-rank rule, old vs new.

Laura's fix 1b2510a replaced, in ``frugal_flows/bijections/masked_autoregressive_first_uniform.py``,
``hidden_ranks = jnp.arange(nn_width) % dim`` with
``h_ranks = autoregressive_hidden_ranks(nn_width, dim, lo=max(cond_u_y_dim - 1, 0))``.

``copula_rank_rule("old")`` restores the pre-fix rule for the COPULA ONLY, by rebinding the
NAME ``autoregressive_hidden_ranks`` inside that one module for the duration of the block
(the masks are built in ``MaskedAutoregressiveFirstUniform.__init__``, which looks the name
up in its module globals at call time). The outcome margin's ``MaskedAutoregressiveSpread``
imported the function into ITS OWN module, so it is untouched (``test_halo`` checks this,
and checks the patched masks equal those of the true pre-fix source).

``python halo_ranks.py [--out ~/work/halo-runs/S7/connectivity.json]`` prints and saves the
connectivity table: how many of the K outcome-rank inputs have a path to the covariate
output through the copula layer's masks, for rule {old, new} x width {16, 50} x K {64, 256}.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os

import numpy as np

COPULA_MODULE = "frugal_flows.bijections.masked_autoregressive_first_uniform"
RULES = ("old", "new")


def old_hidden_ranks(nn_width: int, dim: int, lo: int = 0):
    """The pre-1b2510a rule (flowjax's convention): ``arange(nn_width) % dim``; ``lo`` ignored."""
    import jax.numpy as jnp
    return jnp.arange(nn_width) % dim


@contextlib.contextmanager
def copula_rank_rule(rule: str):
    """Within the block, copulas are BUILT with ``rule`` ("old" = pre-fix, "new" = the fix,
    i.e. no change). Restores the module's original binding on exit, even on error."""
    if rule not in RULES:
        raise ValueError(f"copula_rank_rule: {rule!r} not in {RULES}")
    mod = importlib.import_module(COPULA_MODULE)
    orig = mod.autoregressive_hidden_ranks
    if rule == "old":
        mod.autoregressive_hidden_ranks = old_hidden_ranks
    try:
        yield
    finally:
        mod.autoregressive_hidden_ranks = orig


# ------------------------------------------------------------------ masks and reachability
def mlp_masks(layer) -> list[np.ndarray]:
    """The boolean mask arrays of a masked-autoregressive layer's MLP, in layer order, as
    stored in the ``Parameterize(jnp.where, mask, weight, 0)`` wrappers. Each is (out, in),
    or (n_flow_layers, out, in) for a vmapped (Scan) stack."""
    return [np.asarray(lin.weight.args[0]).astype(bool) for lin in layer.masked_autoregressive_mlp.layers]


def _reach(masks: list[np.ndarray]) -> np.ndarray:
    P = None
    for M in masks:
        M = M.astype(int)
        P = M if P is None else ((M @ P) > 0).astype(int)
    return P > 0                                      # (out, in)


def covariate_reach(masks: list[np.ndarray], K: int, nvars: int = 1) -> np.ndarray:
    """(nvars, K) boolean: can covariate output v's transformer parameters depend on outcome
    rank j? Output rows are grouped by coordinate (``out_ranks = repeat(arange(dim), p)``)."""
    P = _reach(masks)
    dim = K + nvars
    per = P.reshape(dim, -1, P.shape[1]).any(axis=1)
    return per[K:, :K]


def n_reachable(masks: list[np.ndarray], K: int, nvars: int = 1) -> int:
    """Outcome ranks reachable by ANY covariate output (nvars=1: by the covariate)."""
    return int(covariate_reach(masks, K, nvars).any(axis=0).sum())


def build_copula_layer(K: int, width: int, nvars: int = 1, knots: int = 8, depth: int = 1, seed: int = 0):
    """One copula layer exactly as ``masked_autoregressive_flow_first_uniform`` builds it inside
    ``train_frugal_flow_flexible_continuous`` (dim K + nvars, masked treatment condition of size
    1, cond_u_y_dim = K, RQS(knots, interval 1)), under whatever rank rule is in force."""
    import jax.random as jr
    from flowjax.bijections import RationalQuadraticSpline
    mod = importlib.import_module(COPULA_MODULE)
    return mod.MaskedAutoregressiveFirstUniform(
        jr.PRNGKey(seed), transformer=RationalQuadraticSpline(knots=knots, interval=1), dim=K + nvars,
        cond_dim_mask=1, nn_width=width, nn_depth=depth, cond_u_y_dim=K)


def fitted_copula_reach(flow, K: int) -> dict:
    """From a FITTED frugal flow: find every MaskedAutoregressiveFirstUniform (the copula; its
    flow layers are vmapped, so masks carry a leading layer axis) and count, per flow layer,
    the outcome ranks reachable by the covariate output(s). Also returns the distinct hidden
    ranks implied by the first mask (rank of hidden unit h = max input rank it reads)."""
    import jax
    mod = importlib.import_module(COPULA_MODULE)
    found = [x for x in jax.tree_util.tree_leaves(
        flow, is_leaf=lambda x: isinstance(x, mod.MaskedAutoregressiveFirstUniform))
        if isinstance(x, mod.MaskedAutoregressiveFirstUniform)]
    if len(found) != 1:
        raise RuntimeError(f"expected exactly one copula stack, found {len(found)}")
    cop = found[0]
    dim = cop.shape[-1]
    nvars = dim - K
    ms = mlp_masks(cop)
    stacked = ms[0].ndim == 3
    per_layer = []
    for i in range(ms[0].shape[0] if stacked else 1):
        mi = [m[i] for m in ms] if stacked else ms
        per_layer.append(n_reachable(mi, K, nvars))
    m0 = ms[0][0] if stacked else ms[0]
    width = int(m0.shape[0])
    return {"copula_reachable_per_layer": per_layer, "copula_reachable": int(min(per_layer)),
            "copula_blind": int(K - min(per_layer)), "copula_mask_width": width,
            "copula_dim": int(dim), "copula_nvars": int(nvars)}


def connectivity_table(widths=(16, 50), Ks=(64, 256), nvars: int = 1) -> list[dict]:
    rows = []
    for rule in RULES:
        for W in widths:
            for K in Ks:
                with copula_rank_rule(rule):
                    layer = build_copula_layer(K, W, nvars)
                n = n_reachable(mlp_masks(layer), K, nvars)
                rows.append({"rule": rule, "width": W, "K": K, "nvars": nvars, "reachable": n, "blind": K - n})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/work/halo-runs/S7/connectivity.json"))
    a = ap.parse_args(argv)
    rows = connectivity_table()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"rows": rows, "note": "outcome ranks (of K) with a path to the covariate output "
               "through one copula layer's masks; nvars=1, depth 1, masked treatment"}, open(a.out, "w"), indent=1)
    print(f"{'rule':5} {'W':>3} {'K':>4} {'reachable':>9} {'blind':>5}")
    for r in rows:
        print(f"{r['rule']:5} {r['width']:>3} {r['K']:>4} {r['reachable']:>9} {r['blind']:>5}")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
