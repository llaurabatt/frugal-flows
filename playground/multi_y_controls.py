"""Two controls for the multivariate-Y location_translation residual bias, across ATE starts.

1. K=1 control: fit the SCALAR location_translation on each outcome dim separately,
   same DGP. If the scalar model shows the same residual, the bias is inherent to the
   frugal-flow estimator, not introduced by the multivariate code.
2. Capacity sweep: fit the K=2 model with a larger copula + more epochs. If the bias
   shrinks toward 0, it is capacity/optimization-limited, not structural.

Both controls are run at three starting values of the trainable ``ate`` parameter --
the true TAU (as earlier runs did), 0 (LocCond's default), and the per-dim confounded
difference in means -- so the conclusions can be checked for dependence on starting
the optimiser at the answer. Data seed, stage-1 quantiles and fitting key are shared
across the three arms, so the arms are paired.
"""
import jax, jax.numpy as jnp, jax.random as jr, paramax
jax.config.update("jax_enable_x64", True)
from frugal_flows.causal_flows import train_frugal_flow, get_independent_quantiles

TAU = jnp.array([2.0, -1.0])
N = 4000
SEEDS_SCALAR = range(5)
SEEDS_BIG = range(3)
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
    """Per-dim confounded difference in means, E[Y|T=1] - E[Y|T=0]."""
    return jnp.array([float(Y[T == 1, k].mean() - Y[T == 0, k].mean()) for k in range(2)])


def u_z_of(key, Z):
    return get_independent_quantiles(key=key, z_cont=Z, max_epochs=40, max_patience=8,
                                     return_z_cont_flow=True, show_progress=False)["u_z_cont"]


def fit(key, Y, u_z, T, ate, layers, width, epochs):
    flow, _ = train_frugal_flow(key=key, y=Y, u_z=u_z, condition=T[:, None],
                                causal_model="location_translation",
                                causal_model_args=dict(ate=ate, RQS_knots=8, nn_depth=1,
                                                       nn_width=width, flow_layers=layers),
                                learning_rate=1e-3, max_epochs=epochs, max_patience=15,
                                flow_layers=layers, nn_width=width, nn_depth=1, RQS_knots=8,
                                show_progress=False)
    lc = flow.bijection.bijections[5]
    K = Y.shape[1]
    return jnp.array([float(paramax.unwrap(lc.bijections[k]).ate) for k in range(K)])


def summarise(title, hats, n_seeds):
    print(f"\n--- {title} (over {n_seeds} seeds) ---", flush=True)
    print("tau_true:", [float(x) for x in TAU], flush=True)
    for a in ARMS:
        H = jnp.stack(hats[a])
        print(f"[{a:>5}] mean:", [round(float(x), 4) for x in H.mean(0)],
              " std:", [round(float(x), 4) for x in H.std(0)],
              " bias:", [round(float(x), 4) for x in (H.mean(0) - TAU)], flush=True)


# ---- 1. K=1 control: scalar fit on each dim --------------------------------
print("=== K=1 control (scalar model per dim) ===", flush=True)
scal = {a: [] for a in ARMS}
for s in SEEDS_SCALAR:
    Z, T, Y = make_data(jr.key(100 + s))
    uz = u_z_of(jr.key(300 + s), Z)
    nv = naive(T, Y)
    starts = {"truth": TAU, "zero": jnp.zeros(2), "naive": nv}
    for a in ARMS:
        st = starts[a]
        # Each dim fitted on its own with the matching scalar start.
        th0 = fit(jr.key(400 + s), Y[:, 0:1], uz, T, st[0:1], 4, 40, 150)
        th1 = fit(jr.key(500 + s), Y[:, 1:2], uz, T, st[1:2], 4, 40, 150)
        row = jnp.array([float(th0[0]), float(th1[0])])
        scal[a].append(row)
        print(f"seed {s} [{a:>5}] start={[round(float(x),3) for x in st]}"
              f" -> scalar tau_hat={[round(float(x),3) for x in row]}", flush=True)
summarise("K=1 scalar control", scal, len(list(SEEDS_SCALAR)))

# ---- 2. Capacity sweep: bigger K=2 model -----------------------------------
print("\n=== capacity sweep (K=2, flow_layers=8 nn_width=80 epochs=300) ===", flush=True)
big = {a: [] for a in ARMS}
for s in SEEDS_BIG:
    Z, T, Y = make_data(jr.key(100 + s))
    uz = u_z_of(jr.key(300 + s), Z)
    nv = naive(T, Y)
    starts = {"truth": TAU, "zero": jnp.zeros(2), "naive": nv}
    for a in ARMS:
        th = fit(jr.key(600 + s), Y, uz, T, starts[a], 8, 80, 300)
        big[a].append(th)
        print(f"seed {s} [{a:>5}] start={[round(float(x),3) for x in starts[a]]}"
              f" -> big tau_hat={[round(float(x),3) for x in th]}", flush=True)
summarise("capacity sweep", big, len(list(SEEDS_BIG)))

print("\nbaseline K=2 bias (prev run, truth-start only): [0.0509, 0.0902]", flush=True)
print("DONE", flush=True)
