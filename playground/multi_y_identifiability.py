"""Is tau identified by the frugal likelihood, or just anchored to its start?

multi_y_bias_check.py showed that tau_hat depends strongly on where ``ate`` is
initialised, in a correctly-specified DGP (pure additive homogeneous location shift,
n=4000). Two explanations remain, and the achieved validation loss separates them:

  - all three arms reach the SAME loss  -> tau is not identified; the likelihood is
    flat along tau and the data cannot distinguish these values. Structural.
  - the truth arm reaches a LOWER loss  -> the likelihood does prefer the truth and
    the other arms are stuck / not travelling. An optimisation problem, fixable.

Reports, per seed and per arm, the best validation loss alongside tau_hat, at both
the standard and the larger capacity.
"""
import jax, jax.numpy as jnp, jax.random as jr, paramax
jax.config.update("jax_enable_x64", True)
from frugal_flows.causal_flows import train_frugal_flow, get_independent_quantiles

TAU = jnp.array([2.0, -1.0])
N = 4000
SEEDS = range(3)
ARMS = ("truth", "zero", "naive")


def make_data(key):
    k = jr.split(key, 3)
    Z = jr.normal(k[0], (N, 2))
    T = jr.bernoulli(k[1], jax.nn.sigmoid(0.8 * Z[:, 0] - 0.6 * Z[:, 1])).astype(float)
    L = jnp.array([[1.0, 0.0], [0.7, 0.7]])
    noise = jr.normal(k[2], (N, 2)) @ L.T
    eta = noise + jnp.stack([Z[:, 0] + Z[:, 1], Z[:, 0] - 0.5 * Z[:, 1]], axis=1)
    return Z, T, TAU * T[:, None] + eta


def naive(T, Y):
    return jnp.array([float(Y[T == 1, k].mean() - Y[T == 0, k].mean()) for k in range(2)])


def u_z_of(key, Z):
    return get_independent_quantiles(key=key, z_cont=Z, max_epochs=40, max_patience=8,
                                     return_z_cont_flow=True, show_progress=False)["u_z_cont"]


def fit(key, Y, u_z, T, ate, layers, width, epochs):
    """Returns (tau_hat, best validation loss, final validation loss)."""
    flow, losses = train_frugal_flow(
        key=key, y=Y, u_z=u_z, condition=T[:, None], causal_model="location_translation",
        causal_model_args=dict(ate=ate, RQS_knots=8, nn_depth=1, nn_width=width,
                               flow_layers=layers),
        learning_rate=1e-3, max_epochs=epochs, max_patience=15, flow_layers=layers,
        nn_width=width, nn_depth=1, RQS_knots=8, show_progress=False)
    lc = flow.bijection.bijections[5]
    K = Y.shape[1]
    tau_hat = jnp.array([float(paramax.unwrap(lc.bijections[k]).ate) for k in range(K)])
    val = jnp.asarray(losses["val"])
    return tau_hat, float(val.min()), float(val[-1])


def run(title, layers, width, epochs, fitkey_base):
    print(f"\n=== {title} ===", flush=True)
    best = {a: [] for a in ARMS}
    for s in SEEDS:
        Z, T, Y = make_data(jr.key(100 + s))
        uz = u_z_of(jr.key(300 + s), Z)
        starts = {"truth": TAU, "zero": jnp.zeros(2), "naive": naive(T, Y)}
        for a in ARMS:
            th, vmin, vlast = fit(jr.key(fitkey_base + s), Y, uz, T, starts[a],
                                  layers, width, epochs)
            best[a].append(vmin)
            print(f"seed {s} [{a:>5}] tau_hat={[round(float(x),3) for x in th]}"
                  f"  best_val={vmin:.5f}  final_val={vlast:.5f}", flush=True)
        # Within a seed the arms share data, u_z and fitting key, so these losses are
        # directly comparable: only the starting value of ate differs.
        spread = max(best[a][-1] for a in ARMS) - min(best[a][-1] for a in ARMS)
        print(f"seed {s} -> best_val spread across arms: {spread:.5f}", flush=True)
    print(f"--- {title}: mean best_val ---", flush=True)
    for a in ARMS:
        print(f"[{a:>5}] {sum(best[a])/len(best[a]):.5f}", flush=True)


run("standard capacity (4 layers, width 40, 150 epochs)", 4, 40, 150, 200)
run("larger capacity (8 layers, width 80, 300 epochs)", 8, 80, 300, 600)
print("\nDONE", flush=True)
