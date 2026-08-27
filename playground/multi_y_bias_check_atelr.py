"""Confirmation for the bug #14 fix: does a larger ate step size remove the anchoring?

Identical to multi_y_bias_check.py -- same DGP, seeds, stage-1 quantiles, fitting keys,
capacity, max_epochs and max_patience -- with exactly one change: the K LocCond ``ate``
leaves get their own larger learning rate via optax.multi_transform. Any difference in
the results is therefore attributable to that step size alone.

The falsifiable claim: with the fix the three starting values (truth / zero / naive)
should CONVERGE TO EACH OTHER, since Evans & Didelez Thm 5.1 says the MLE is consistent
and the anchoring was a failure to reach it. Recorded no-fix results for comparison:

    truth [2.051, -0.918]   zero [1.581, -1.107]   naive [2.258, -0.566]
"""
import equinox as eqx
import jax, jax.numpy as jnp, jax.random as jr, optax, paramax
jax.config.update("jax_enable_x64", True)
from frugal_flows.causal_flows import train_frugal_flow, get_independent_quantiles

TAU = jnp.array([2.0, -1.0])
N, K = 4000, 2
BASE_LR, ATE_LR = 1e-3, 1e-2
SEEDS = range(8)
ARMS = ("truth", "zero", "naive")
_SENTINEL = object()


def make_data(key):
    k = jr.split(key, 3)
    Z = jr.normal(k[0], (N, 2))
    T = jr.bernoulli(k[1], jax.nn.sigmoid(0.8 * Z[:, 0] - 0.6 * Z[:, 1])).astype(float)
    L = jnp.array([[1.0, 0.0], [0.7, 0.7]])
    noise = jr.normal(k[2], (N, 2)) @ L.T
    eta = noise + jnp.stack([Z[:, 0] + Z[:, 1], Z[:, 0] - 0.5 * Z[:, 1]], axis=1)
    return Z, T, TAU * T[:, None] + eta


def naive(T, Y):
    return jnp.array([float(Y[T == 1, k].mean() - Y[T == 0, k].mean()) for k in range(K)])


def label_fn(params):
    """Label the K LocCond.ate leaves 'ate', everything else 'rest'."""
    marked = eqx.tree_at(
        lambda t: [t.bijection.bijections[5].bijections[k].ate for k in range(K)],
        params, replace=[_SENTINEL] * K)
    return jax.tree.map(lambda x: "ate" if x is _SENTINEL else "rest",
                        marked, is_leaf=lambda x: x is _SENTINEL)


OPT = optax.multi_transform({"ate": optax.adam(ATE_LR), "rest": optax.adam(BASE_LR)},
                            label_fn)


def fit_tau(key, u_z, T, Y, ate_init):
    flow, losses = train_frugal_flow(
        key=key, y=Y, u_z=u_z, condition=T[:, None],
        causal_model="location_translation",
        causal_model_args=dict(ate=ate_init, RQS_knots=8, nn_depth=1, nn_width=40,
                               flow_layers=4),
        optimizer=OPT, learning_rate=BASE_LR, max_epochs=150, max_patience=15,
        flow_layers=4, nn_width=40, nn_depth=1, RQS_knots=8, show_progress=False)
    lc = flow.bijection.bijections[5]
    tau = jnp.array([float(paramax.unwrap(lc.bijections[k]).ate) for k in range(K)])
    return tau, float(jnp.asarray(losses["val"]).min())


tau_hats = {a: [] for a in ARMS}
vals = {a: [] for a in ARMS}
for s in SEEDS:
    Z, T, Y = make_data(jr.key(100 + s))
    u_z = get_independent_quantiles(key=jr.key(300 + s), z_cont=Z, max_epochs=40,
                                    max_patience=8, return_z_cont_flow=True,
                                    show_progress=False)["u_z_cont"]
    starts = {"truth": TAU, "zero": jnp.zeros(K), "naive": naive(T, Y)}
    for a in ARMS:
        th, bv = fit_tau(jr.key(200 + s), u_z, T, Y, starts[a])
        tau_hats[a].append(th)
        vals[a].append(bv)
        print(f"seed {s} [{a:>5}] tau_hat={[round(float(x),3) for x in th]}"
              f"  best_val={bv:.5f}", flush=True)

print(f"\n=== SUMMARY over {len(list(SEEDS))} seeds (ate lr={ATE_LR:g}) ===", flush=True)
print("tau_true      :", [float(x) for x in TAU], flush=True)
for a in ARMS:
    TH = jnp.stack(tau_hats[a])
    print(f"[{a:>5}] mean:", [round(float(x), 4) for x in TH.mean(0)],
          " std:", [round(float(x), 4) for x in TH.std(0)],
          " bias:", [round(float(x), 4) for x in (TH.mean(0) - TAU)],
          f" mean_best_val: {sum(vals[a])/len(vals[a]):.5f}", flush=True)

means = jnp.stack([jnp.stack(tau_hats[a]).mean(0) for a in ARMS])
print("\nspread across arms (max-min per dim):",
      [round(float(x), 4) for x in (means.max(0) - means.min(0))], flush=True)
print("no-fix spread was: [0.677, 0.541]", flush=True)
print("DONE", flush=True)
