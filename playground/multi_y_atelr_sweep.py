"""Bug #14 fix, step 2b: sweep the ate learning rate.

ATE_LR=1e-2 was picked from a single seed; 5e-2 overshot on that seed, and the truth
arm drifted up to 2.146 in the 8-seed run, which looks like mild overshoot. This sweeps
the ate step size over a few values on 4 seeds, from the zero start (the honest start,
and the one nearest the MLE so far). Reports mean tau_hat, bias and best_val per rate so
the best-generalising step size can be chosen before any package API is decided.
"""
import equinox as eqx
import jax, jax.numpy as jnp, jax.random as jr, optax, paramax
jax.config.update("jax_enable_x64", True)
from frugal_flows.causal_flows import train_frugal_flow, get_independent_quantiles

TAU = jnp.array([2.0, -1.0])
N, K = 4000, 2
BASE_LR = 1e-3
ATE_LRS = [3e-3, 5e-3, 1e-2, 2e-2, 3e-2]
SEEDS = range(4)
_SENTINEL = object()


def make_data(key):
    k = jr.split(key, 3)
    Z = jr.normal(k[0], (N, 2))
    T = jr.bernoulli(k[1], jax.nn.sigmoid(0.8 * Z[:, 0] - 0.6 * Z[:, 1])).astype(float)
    L = jnp.array([[1.0, 0.0], [0.7, 0.7]])
    noise = jr.normal(k[2], (N, 2)) @ L.T
    eta = noise + jnp.stack([Z[:, 0] + Z[:, 1], Z[:, 0] - 0.5 * Z[:, 1]], axis=1)
    return Z, T, TAU * T[:, None] + eta


def label_fn(params):
    marked = eqx.tree_at(
        lambda t: [t.bijection.bijections[5].bijections[k].ate for k in range(K)],
        params, replace=[_SENTINEL] * K)
    return jax.tree.map(lambda x: "ate" if x is _SENTINEL else "rest",
                        marked, is_leaf=lambda x: x is _SENTINEL)


def fit_tau(key, u_z, T, Y, opt):
    flow, losses = train_frugal_flow(
        key=key, y=Y, u_z=u_z, condition=T[:, None],
        causal_model="location_translation",
        causal_model_args=dict(ate=jnp.zeros(K), RQS_knots=8, nn_depth=1, nn_width=40,
                               flow_layers=4),
        optimizer=opt, learning_rate=BASE_LR, max_epochs=150, max_patience=15,
        flow_layers=4, nn_width=40, nn_depth=1, RQS_knots=8, show_progress=False)
    lc = flow.bijection.bijections[5]
    tau = jnp.array([float(paramax.unwrap(lc.bijections[k]).ate) for k in range(K)])
    return tau, float(jnp.asarray(losses["val"]).min())


# Precompute data + quantiles once per seed (shared across all rates).
prep = []
for s in SEEDS:
    Z, T, Y = make_data(jr.key(100 + s))
    u_z = get_independent_quantiles(key=jr.key(300 + s), z_cont=Z, max_epochs=40,
                                    max_patience=8, return_z_cont_flow=True,
                                    show_progress=False)["u_z_cont"]
    prep.append((u_z, T, Y))

print("true tau: [2.0, -1.0]  (zero start, 4 seeds)\n", flush=True)
print(f"{'ate_lr':>8} {'mean tau_hat':>20} {'bias':>18} {'mean_best_val':>14}", flush=True)
for alr in ATE_LRS:
    opt = optax.multi_transform(
        {"ate": optax.adam(alr), "rest": optax.adam(BASE_LR)}, label_fn)
    ths, bvs = [], []
    for (u_z, T, Y), s in zip(prep, SEEDS):
        th, bv = fit_tau(jr.key(200 + s), u_z, T, Y, opt)
        ths.append(th); bvs.append(bv)
    TH = jnp.stack(ths)
    m = TH.mean(0)
    print(f"{alr:>8g} {str([round(float(x),3) for x in m]):>20}"
          f" {str([round(float(x),3) for x in (m - TAU)]):>18}"
          f" {sum(bvs)/len(bvs):>14.5f}", flush=True)

print("\nDONE", flush=True)
