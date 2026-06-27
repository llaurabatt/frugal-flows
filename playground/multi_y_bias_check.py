"""Multi-seed bias check for multivariate-Y location_translation.

Same confounded DGP across several seeds. Reports, per dim:
  - frugal-flow ATE estimate (read off the trained LocCond block),
  - naive confounded difference-in-means E[Y|T=1]-E[Y|T=0],
so we can see whether tau_hat is centred on the truth (variance) or biased.
"""
import jax, jax.numpy as jnp, jax.random as jr, paramax
jax.config.update("jax_enable_x64", True)
from frugal_flows.causal_flows import train_frugal_flow, get_independent_quantiles

TAU = jnp.array([2.0, -1.0])
N = 4000
SEEDS = range(8)


def make_data(key):
    k = jr.split(key, 3)
    Z = jr.normal(k[0], (N, 2))
    T = jr.bernoulli(k[1], jax.nn.sigmoid(0.8 * Z[:, 0] - 0.6 * Z[:, 1])).astype(float)
    L = jnp.array([[1.0, 0.0], [0.7, 0.7]])
    noise = jr.normal(k[2], (N, 2)) @ L.T
    eta = noise + jnp.stack([Z[:, 0] + Z[:, 1], Z[:, 0] - 0.5 * Z[:, 1]], axis=1)
    Y = TAU * T[:, None] + eta
    return Z, T, Y


def fit_tau(key, Z, T, Y):
    ka, kb = jr.split(key)
    u_z = get_independent_quantiles(key=ka, z_cont=Z, max_epochs=40, max_patience=8,
                                    return_z_cont_flow=True, show_progress=False)["u_z_cont"]
    cma = dict(ate=TAU, RQS_knots=8, nn_depth=1, nn_width=40, flow_layers=4)
    flow, _ = train_frugal_flow(key=kb, y=Y, u_z=u_z, condition=T[:, None],
                                causal_model="location_translation", causal_model_args=cma,
                                learning_rate=1e-3, max_epochs=150, max_patience=15,
                                flow_layers=4, nn_width=40, nn_depth=1, RQS_knots=8,
                                show_progress=False)
    lc = flow.bijection.bijections[5]
    return jnp.array([float(paramax.unwrap(lc.bijections[k]).ate) for k in range(2)])


def naive(T, Y):
    return jnp.array([float(Y[T == 1, k].mean() - Y[T == 0, k].mean()) for k in range(2)])


tau_hats, naives = [], []
for s in SEEDS:
    Z, T, Y = make_data(jr.key(100 + s))
    th = fit_tau(jr.key(200 + s), Z, T, Y)
    nv = naive(T, Y)
    tau_hats.append(th); naives.append(nv)
    print(f"seed {s}: tau_hat={[round(float(x),3) for x in th]}  "
          f"naive={[round(float(x),3) for x in nv]}", flush=True)

TH = jnp.stack(tau_hats); NV = jnp.stack(naives)
print("\n=== SUMMARY over", len(SEEDS), "seeds ===", flush=True)
print("tau_true      :", [float(x) for x in TAU], flush=True)
print("tau_hat  mean :", [round(float(x), 4) for x in TH.mean(0)],
      " std:", [round(float(x), 4) for x in TH.std(0)], flush=True)
print("naive    mean :", [round(float(x), 4) for x in NV.mean(0)],
      " std:", [round(float(x), 4) for x in NV.std(0)], flush=True)
bias = TH.mean(0) - TAU
print("flow bias     :", [round(float(x), 4) for x in bias], flush=True)
print("DONE", flush=True)
