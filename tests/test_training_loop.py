"""frugal_flows.training.fit_to_data must reproduce flowjax.train.fit_to_data bit for bit
when called with the library's arguments, and record the split it used."""
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from flowjax.distributions import Normal
from flowjax.flows import masked_autoregressive_flow
from flowjax.train import fit_to_data as fit_lib
from frugal_flows.training import fit_to_data as fit_ours


def _data(n=300, d=3, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, d))
    x = np.column_stack([z[:, 0], z[:, 0] + 0.5 * z[:, 1], z[:, 2] ** 2])
    c = rng.integers(0, 2, size=(n, 1)).astype(float)
    return jnp.asarray(x), jnp.asarray(c)


def _flow(key, cond_dim=None):
    return masked_autoregressive_flow(key, base_dist=Normal(jnp.zeros(3)), cond_dim=cond_dim,
                                      flow_layers=2, nn_width=8)


def _leaves(dist):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(dist) if hasattr(x, "shape")]


@pytest.mark.parametrize("conditional", [False, True])
def test_reproduces_library(conditional):
    x, c = _data()
    data = (x, c) if conditional else x
    kw = dict(learning_rate=1e-2, max_epochs=6, max_patience=2, batch_size=50, show_progress=False)
    d_lib, l_lib = fit_lib(jr.PRNGKey(3), _flow(jr.PRNGKey(1), 1 if conditional else None), data, **kw)
    d_our, l_our = fit_ours(jr.PRNGKey(3), _flow(jr.PRNGKey(1), 1 if conditional else None), data, **kw)
    assert l_lib["train"] == l_our["train"]
    assert l_lib["val"] == l_our["val"]
    for a, b in zip(_leaves(d_lib), _leaves(d_our), strict=True):
        assert np.array_equal(a, b)


def test_records_split_and_info():
    x, c = _data()
    _, l = fit_ours(jr.PRNGKey(3), _flow(jr.PRNGKey(1), 1), (x, c), learning_rate=1e-2,
                    max_epochs=4, max_patience=2, batch_size=50, show_progress=False)
    info = l["info"]
    n = x.shape[0]
    assert info["n_val"] == round(0.1 * n) and info["n_train"] == n - info["n_val"]
    assert sorted(np.r_[info["train_idx"], info["val_idx"]].tolist()) == list(range(n))
    assert info["termination"] in ("patience", "epoch_cap")
    assert 1 <= info["best_epoch"] <= info["n_epochs"]
    assert info["selected_on"] == "val_loss"
    # the recorded indices are the rows the library's split would have used
    from flowjax.train.train_utils import train_val_split
    key, subkey = jr.split(jr.PRNGKey(3))
    tr, va = train_val_split(subkey, (x, c), val_prop=0.1)
    assert np.array_equal(np.asarray(va[0]), np.asarray(x)[info["val_idx"]])
    assert np.array_equal(np.asarray(tr[0]), np.asarray(x)[info["train_idx"]])


def test_select_fn_and_hooks():
    x, c = _data()
    calls = []
    n_select = [0]

    def select_fn(params, static, xv, cv):        # a criterion that is not the val loss: rises
        n_select[0] += 1                           # every epoch, so epoch 1 stays the minimum
        return float(n_select[0])

    def on_epoch(epoch, params, static):
        calls.append(epoch)
        return {"probe": float(epoch)}

    _, l = fit_ours(jr.PRNGKey(3), _flow(jr.PRNGKey(1), 1), (x, c), learning_rate=1e-2,
                    max_epochs=5, max_patience=1, batch_size=50, show_progress=False,
                    select_fn=select_fn, on_epoch=on_epoch)
    assert l["info"]["selected_on"] == "select_fn"
    assert len(l["select"]) == l["info"]["n_epochs"]
    # rising criterion: epoch 1 is the minimum; patience 1 (library rule: stop when MORE than
    # max_patience epochs have passed since the minimum) stops after epoch 3
    assert l["info"]["best_epoch"] == 1 and l["info"]["termination"] == "patience" and l["info"]["n_epochs"] == 3
    assert [r["epoch"] for r in l["track"]] == calls == [1, 2, 3]


def test_ema_zero_is_the_raw_loop_and_ema_changes_the_result():
    """ema_decay=0.0 averages nothing, so it must reproduce the default loop exactly; a real
    decay must return different parameters and record itself."""
    x, c = _data()
    kw = dict(learning_rate=1e-2, max_epochs=5, max_patience=2, batch_size=50, show_progress=False)
    d0, l0 = fit_ours(jr.PRNGKey(3), _flow(jr.PRNGKey(1), 1), (x, c), **kw)
    de, le = fit_ours(jr.PRNGKey(3), _flow(jr.PRNGKey(1), 1), (x, c), ema_decay=0.0, **kw)
    assert l0["train"] == le["train"] and l0["val"] == le["val"]
    for a, b in zip(_leaves(d0), _leaves(de), strict=True):
        assert np.array_equal(a, b)
    d9, l9 = fit_ours(jr.PRNGKey(3), _flow(jr.PRNGKey(1), 1), (x, c), ema_decay=0.9, **kw)
    assert l9["info"]["ema_decay"] == 0.9 and l0["info"]["ema_decay"] is None
    assert l9["train"] == l0["train"][:len(l9["train"])], "the optimiser still steps the raw parameters"
    assert any(not np.array_equal(a, b) for a, b in zip(_leaves(d0), _leaves(d9), strict=True))
    with pytest.raises(ValueError):
        fit_ours(jr.PRNGKey(3), _flow(jr.PRNGKey(1), 1), (x, c), ema_decay=1.0, **kw)


def test_wall_cap():
    x, c = _data()
    _, l = fit_ours(jr.PRNGKey(3), _flow(jr.PRNGKey(1), 1), (x, c), learning_rate=1e-2,
                    max_epochs=50, max_patience=50, batch_size=50, show_progress=False, wall_cap_s=0.0)
    assert l["info"]["termination"] == "wall_cap" and l["info"]["n_epochs"] == 1
