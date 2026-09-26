"""Every masked autoregressive layer in the package must let each used output depend on
every earlier input, whatever its width, and on an unmasked condition. Until 2026-09-25 the
hidden ranks were ``arange(nn_width) % dim``, which cut inputs >= nn_width off entirely when
the layer was narrower than its dimension (the copula at 64 pixels and width 50 ignored
pixels 50-63; flowjax's own layer, used by the outcome margin, has the same rule).

The checks read the connections from the masked weights (a path input -> hidden unit ->
output exists iff every weight on it is unmasked), so nothing is compiled: an earlier
version differentiated each layer, and the extra compilations pushed the test suite's
single process past the kernel's limit on memory mappings (vm.max_map_count 65530), which
made a later test fail with "LLVM compilation error: Cannot allocate memory".
"""
import jax.random as jr
import numpy as np
import paramax
import pytest
from flowjax.bijections import MaskedAutoregressive, RationalQuadraticSpline
from frugal_flows.bijections import (
    MaskedAutoregressiveFirstUniform,
    MaskedAutoregressiveHeterogeneous,
    MaskedAutoregressiveMaskedCond,
    MaskedAutoregressiveSpread,
)
from frugal_flows.bijections.ranks import autoregressive_hidden_ranks

RQS = RationalQuadraticSpline(knots=4, interval=1)


def _paths(layer, dim):
    """(dim, n_inputs) boolean: can coordinate i's transformer parameters depend on input j?
    Inputs are the dim coordinates followed by any conditioning inputs."""
    mlp = paramax.unwrap(layer.masked_autoregressive_mlp)
    P = None
    for lin in mlp.layers:
        M = (np.asarray(lin.weight) != 0).astype(int)          # (out, in)
        P = M if P is None else (M @ P > 0).astype(int)
    per_coord = P.reshape(dim, -1, P.shape[1]).max(axis=1)     # rows grouped by coordinate
    return per_coord > 0


@pytest.mark.parametrize("width,lo,dim", [(50, 63, 66), (96, 63, 66), (8, 0, 30), (100, 0, 30),
                                          (3, 0, 3), (5, 0, 1), (6, -1, 12), (20, -1, 12), (4, -1, 1)])
def test_ranks_cover_the_range(width, lo, dim):
    r = np.asarray(autoregressive_hidden_ranks(width, dim, lo))
    assert len(r) == width
    hi = dim - 2
    if hi >= lo:
        assert r.min() == lo and r.max() == hi
        if width >= hi - lo + 1:
            assert set(r.tolist()) == set(range(lo, hi + 1))
    else:
        assert (r == lo).all()


@pytest.mark.parametrize("width", [8, 50, 96])
def test_copula_covariates_see_every_outcome_rank(width):
    K, nvars = 64, 2
    layer = MaskedAutoregressiveFirstUniform(jr.PRNGKey(0), transformer=RQS, dim=K + nvars,
                                             cond_dim_mask=1, nn_width=width, nn_depth=1, cond_u_y_dim=K)
    D = _paths(layer, K + nvars)
    assert D[K, :K].all(), f"first covariate ignores outcome ranks {np.flatnonzero(~D[K, :K]).tolist()}"
    assert D[K + 1, :K + 1].all(), "second covariate must read every outcome rank and the first covariate"
    assert not D[K, K + 1:].any(), "autoregressive order violated (or the masked condition leaks)"


@pytest.mark.parametrize("width", [8, 48])
def test_masked_cond_reaches_every_input_and_the_condition(width):
    dim = 64
    layer = MaskedAutoregressiveMaskedCond(jr.PRNGKey(1), transformer=RQS, dim=dim, cond_dim_nomask=1,
                                           nn_width=width, nn_depth=1)
    D = _paths(layer, dim)
    assert D[dim - 1, :dim - 1].all()
    assert not np.triu(D[:, :dim]).any(), "autoregressive order violated"
    assert D[0, dim], "output 0 must depend on the unmasked condition"


def test_heterogeneous_reaches_every_input_and_the_condition():
    dim = 20
    layer = MaskedAutoregressiveHeterogeneous(jr.PRNGKey(2), transformer=RQS, dim=dim, cond_dim_nomask=2,
                                              nn_width=6, nn_depth=1, identity_idx=5)
    D = _paths(layer, dim)
    assert D[dim - 1, :dim - 1].all()
    assert D[0, dim:].any(), "output 0 must depend on the unmasked condition"


@pytest.mark.parametrize("cond_dim", [None, 1])
def test_spread_margin_equals_flowjax_when_width_covers_the_ranks(cond_dim):
    """With enough hidden units the ranks are flowjax's own: identical masks and weights."""
    dim = 10
    width = dim if cond_dim else dim - 1
    for w in (width, 3 * width):
        a = MaskedAutoregressive(jr.PRNGKey(3), transformer=RQS, dim=dim, cond_dim=cond_dim, nn_width=w, nn_depth=1)
        b = MaskedAutoregressiveSpread(jr.PRNGKey(3), transformer=RQS, dim=dim, cond_dim=cond_dim, nn_width=w, nn_depth=1)
        for la, lb in zip(paramax.unwrap(a.masked_autoregressive_mlp).layers,
                          paramax.unwrap(b.masked_autoregressive_mlp).layers, strict=True):
            assert np.array_equal(np.asarray(la.weight), np.asarray(lb.weight))
            assert np.array_equal(np.asarray(la.bias), np.asarray(lb.bias))


@pytest.mark.parametrize("width", [8, 48])
def test_spread_margin_reaches_every_input_and_the_condition(width):
    dim = 64
    b = MaskedAutoregressiveSpread(jr.PRNGKey(4), transformer=RQS, dim=dim, cond_dim=1, nn_width=width, nn_depth=1)
    f = MaskedAutoregressive(jr.PRNGKey(4), transformer=RQS, dim=dim, cond_dim=1, nn_width=width, nn_depth=1)
    D, Df = _paths(b, dim), _paths(f, dim)
    assert D[dim - 1, :dim - 1].all(), "last output must read every earlier input"
    assert not Df[dim - 1, :dim - 1].all(), "flowjax's layer misses some at this width (the defect)"
    assert not np.triu(D[:, :dim]).any(), "autoregressive order violated"
    assert D[0, dim], "output 0 must depend on the condition"
