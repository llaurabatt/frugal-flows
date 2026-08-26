"""Fix candidate 1 for bug #14: give LocCond.ate its own (larger) learning rate.

Bug #14: tau converges more slowly than the copula overfits. Validation loss bottoms
out around epoch 91 and, because fit_to_data defaults to return_best=True, the flow
returned is the one from that epoch -- with tau only ~77% of the way to the truth.
Training longer does not help (150 and 300 epochs give identical output).

If the mechanism is right, letting tau travel faster should let it arrive before the
copula overfits. Here optax.multi_transform gives the K LocCond ``ate`` leaves a
larger step size and leaves every other parameter on the original one.

Run order:
  Step A verifies the label pytree really selects exactly the K ate leaves and
         nothing else -- if this is wrong the whole experiment is meaningless.
  Step B is the baseline staircase (equal learning rates), reproducing bug #14.
  Step C repeats it for each candidate ate learning rate.
"""
import equinox as eqx
import jax, jax.numpy as jnp, jax.random as jr, optax, paramax
jax.config.update("jax_enable_x64", True)
from frugal_flows.causal_flows import train_frugal_flow, get_independent_quantiles

TAU = jnp.array([2.0, -1.0])
N, K = 4000, 2
BASE_LR = 1e-3
ATE_LRS = [1e-2, 5e-2]
BUDGETS = [20, 40, 80, 150, 300]
CMA = dict(ate=jnp.zeros(K), RQS_knots=8, nn_depth=1, nn_width=40, flow_layers=4)
FLOW = dict(flow_layers=4, nn_width=40, nn_depth=1, RQS_knots=8)


def make_data(key):
    k = jr.split(key, 3)
    Z = jr.normal(k[0], (N, 2))
    T = jr.bernoulli(k[1], jax.nn.sigmoid(0.8 * Z[:, 0] - 0.6 * Z[:, 1])).astype(float)
    L = jnp.array([[1.0, 0.0], [0.7, 0.7]])
    noise = jr.normal(k[2], (N, 2)) @ L.T
    eta = noise + jnp.stack([Z[:, 0] + Z[:, 1], Z[:, 0] - 0.5 * Z[:, 1]], axis=1)
    return Z, T, TAU * T[:, None] + eta


def ate_leaves(tree):
    """The K LocCond.ate leaves, at bijections[5] of the merged chain."""
    return [tree.bijection.bijections[5].bijections[k].ate for k in range(K)]


def read_tau(flow):
    lc = flow.bijection.bijections[5]
    return [round(float(paramax.unwrap(lc.bijections[k]).ate), 3) for k in range(K)]


def partition_like_flowjax(flow):
    """Exactly how fit_to_data splits the flow before handing params to optax."""
    return eqx.partition(
        flow, eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
    )[0]


_SENTINEL = object()


def label_fn(params):
    """Label every param leaf 'rest' except the K ate leaves, labelled 'ate'."""
    marked = eqx.tree_at(ate_leaves, params, replace=[_SENTINEL] * K)
    return jax.tree.map(lambda x: "ate" if x is _SENTINEL else "rest",
                        marked, is_leaf=lambda x: x is _SENTINEL)


def fit(key, Y, u_z, T, epochs, optimizer=None):
    flow, losses = train_frugal_flow(
        key=key, y=Y, u_z=u_z, condition=T[:, None],
        causal_model="location_translation", causal_model_args=CMA,
        optimizer=optimizer, learning_rate=BASE_LR,
        max_epochs=epochs, max_patience=epochs + 1,  # early stopping disabled
        show_progress=False, **FLOW)
    val = jnp.asarray(losses["val"])
    return flow, read_tau(flow), int(val.argmin()) + 1, float(val.min()), float(val[-1])


Z, T, Y = make_data(jr.key(100))
u_z = get_independent_quantiles(key=jr.key(300), z_cont=Z, max_epochs=40, max_patience=8,
                                return_z_cont_flow=True, show_progress=False)["u_z_cont"]

# --- Step A: verify the labelling before trusting anything downstream ---------
probe, _, _, _, _ = fit(jr.key(200), Y, u_z, T, 1)
params = partition_like_flowjax(probe)
labels = label_fn(params)
flat = jax.tree.leaves(labels)
n_ate = sum(1 for x in flat if x == "ate")
print("=== Step A: label check ===", flush=True)
print(f"param leaves: {len(flat)}  labelled 'ate': {n_ate}  (expected {K})", flush=True)
print("ate leaf shapes:", [jnp.shape(a) for a in ate_leaves(params)], flush=True)
assert n_ate == K, f"expected {K} ate leaves, labelled {n_ate}"
print("OK\n", flush=True)

# --- Step B: baseline, equal learning rates ----------------------------------
print("=== Step B: baseline (single lr=1e-3) — reproduces bug #14 ===", flush=True)
print(f"{'budget':>7} {'tau_hat':>20} {'best_ep':>8} {'best_val':>10} {'last_val':>10}", flush=True)
for ep in BUDGETS:
    _, th, bep, bv, lv = fit(jr.key(200), Y, u_z, T, ep)
    print(f"{ep:>7} {str(th):>20} {bep:>8} {bv:>10.5f} {lv:>10.5f}", flush=True)

# --- Step C: larger step size on ate only ------------------------------------
for alr in ATE_LRS:
    print(f"\n=== Step C: ate lr={alr:g}, rest lr={BASE_LR:g} ===", flush=True)
    print(f"{'budget':>7} {'tau_hat':>20} {'best_ep':>8} {'best_val':>10} {'last_val':>10}",
          flush=True)
    opt = optax.multi_transform(
        {"ate": optax.adam(alr), "rest": optax.adam(BASE_LR)}, label_fn)
    for ep in BUDGETS:
        _, th, bep, bv, lv = fit(jr.key(200), Y, u_z, T, ep, optimizer=opt)
        print(f"{ep:>7} {str(th):>20} {bep:>8} {bv:>10.5f} {lv:>10.5f}", flush=True)

print(f"\ntrue tau: {[float(x) for x in TAU]}  (start [0.0, 0.0])", flush=True)
print("DONE", flush=True)
