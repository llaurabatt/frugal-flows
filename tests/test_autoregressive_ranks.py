"""Every masked autoregressive layer in the package must let each used output depend on
every earlier input, whatever its width. Until 2026-09-25 the hidden ranks were
``arange(nn_width) % dim``, which cut inputs >= nn_width off entirely when the layer was
narrower than its dimension (the copula at 64 pixels and width 50 ignored pixels 50-63)."""
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from flowjax.bijections import RationalQuadraticSpline
from frugal_flows.bijections import (
    MaskedAutoregressiveFirstUniform,
    MaskedAutoregressiveHeterogeneous,
    MaskedAutoregressiveMaskedCond,
)
from frugal_flows.bijections.ranks import autoregressive_hidden_ranks

RQS = RationalQuadraticSpline(knots=4, interval=1)


def _dependence(fn, dim, n=16, seed=0):
    """(out, in) boolean matrix: does output i move when input j moves, at any of n points."""
    X = jnp.asarray(np.random.default_rng(seed).uniform(-0.9, 0.9, (n, dim)))
    return np.abs(np.asarray(jax.vmap(jax.jacfwd(fn))(X))).max(0) > 0


@pytest.mark.parametrize("width,lo,dim", [(50, 63, 66), (96, 63, 66), (8, 0, 30), (100, 0, 30), (3, 0, 3), (5, 0, 1)])
def test_ranks_cover_the_top(width, lo, dim):
    r = np.asarray(autoregressive_hidden_ranks(width, dim, lo))
    assert len(r) == width
    if dim >= 2:
        assert r.min() >= lo and r.max() == dim - 2 or dim - 2 < lo
        if width >= dim - 1 - lo:
            assert set(r.tolist()) == set(range(lo, dim - 1))


@pytest.mark.parametrize("width", [8, 50, 96])
def test_copula_covariates_see_every_outcome_rank(width):
    K, nvars = 64, 2
    layer = MaskedAutoregressiveFirstUniform(jr.PRNGKey(0), transformer=RQS, dim=K + nvars,
                                             cond_dim_mask=1, nn_width=width, nn_depth=1, cond_u_y_dim=K)
    c = jnp.zeros((1,))
    D = _dependence(lambda x: layer.transform(x, c), K + nvars)
    assert D[K, :K].all(), f"first covariate ignores outcome ranks {np.flatnonzero(~D[K, :K]).tolist()}"
    assert D[K + 1, :K + 1].all(), "second covariate must read every outcome rank and the first covariate"
    assert not D[K, K + 1:].any(), "autoregressive order violated"


@pytest.mark.parametrize("width", [8, 48])
def test_masked_cond_last_output_sees_all_earlier_inputs(width):
    dim = 64
    layer = MaskedAutoregressiveMaskedCond(jr.PRNGKey(1), transformer=RQS, dim=dim, cond_dim_nomask=1,
                                           nn_width=width, nn_depth=1)
    c = jnp.ones((1,))
    D = _dependence(lambda x: layer.transform(x, c), dim)
    assert D[dim - 1, :dim - 1].all()
    assert not np.triu(D, 1).any(), "autoregressive order violated"


def test_heterogeneous_last_output_sees_all_earlier_inputs():
    dim = 20
    layer = MaskedAutoregressiveHeterogeneous(jr.PRNGKey(2), transformer=RQS, dim=dim, cond_dim_nomask=2,
                                              nn_width=6, nn_depth=1)
    c = jnp.ones((2,))
    D = _dependence(lambda x: layer.transform(x, c), dim)
    assert D[dim - 1, :dim - 1].all()
