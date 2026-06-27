"""Two controls for the multivariate-Y location_translation residual bias.

1. K=1 control: fit the SCALAR location_translation on each outcome dim separately,
   same DGP. If the scalar model shows the same ~+0.05/+0.09 residual, the bias is
   inherent to the frugal-flow estimator, not introduced by the multivariate code.
2. Capacity sweep: fit the K=2 model with a larger copula + more epochs. If the bias
   shrinks toward 0, it is capacity/optimization-limited, not structural.
"""
import jax, jax.numpy as jnp, jax.random as jr, paramax
jax.config.update("jax_enable_x64", True)
from frugal_flows.causal_flows import train_frugal_flow, get_independent_quantiles

TAU = jnp.array([2.0, -1.0])
N = 4000


def make_data(key):
    k = jr.split(key, 3)
    Z = jr.normal(k[0], (N, 2))
    T = jr.bernoulli(k[1], jax.nn.sigmoid(0.8 * Z[:, 0] - 0.6 * Z[:, 1])).astype(float)
    L = jnp.array([[1.0, 0.0], [0.7, 0.7]])
    noise = jr.normal(k[2], (N, 2)) @ L.T
    eta = noise + jnp.stack([Z[:, 0] + Z[:, 1], Z[:, 0] - 0.5 * Z[:, 1]], axis=1)
    return Z, T, TAU * T[:, None] + eta


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


# ---- 1. K=1 control: scalar fit on each dim --------------------------------
print("=== K=1 control (scalar model per dim) ===", flush=True)
scal = []
for s in range(5):
    Z, T, Y = make_data(jr.key(100 + s))
    uz = u_z_of(jr.key(300 + s), Z)
    th0 = fit(jr.key(400 + s), Y[:, 0:1], uz, T, TAU[0:1], 4, 40, 150)
    th1 = fit(jr.key(500 + s), Y[:, 1:2], uz, T, TAU[1:2], 4, 40, 150)
    row = jnp.array([float(th0[0]), float(th1[0])])
    scal.append(row)
    print(f"seed {s}: scalar tau_hat={[round(float(x),3) for x in row]}", flush=True)
SC = jnp.stack(scal)
print("scalar mean:", [round(float(x), 4) for x in SC.mean(0)],
      " bias:", [round(float(x), 4) for x in (SC.mean(0) - TAU)], flush=True)

# ---- 2. Capacity sweep: bigger K=2 model -----------------------------------
print("\n=== capacity sweep (K=2, flow_layers=8 nn_width=80 epochs=300) ===", flush=True)
big = []
for s in range(3):
    Z, T, Y = make_data(jr.key(100 + s))
    uz = u_z_of(jr.key(300 + s), Z)
    th = fit(jr.key(600 + s), Y, uz, T, TAU, 8, 80, 300)
    big.append(th)
    print(f"seed {s}: big tau_hat={[round(float(x),3) for x in th]}", flush=True)
BG = jnp.stack(big)
print("big mean:", [round(float(x), 4) for x in BG.mean(0)],
      " bias:", [round(float(x), 4) for x in (BG.mean(0) - TAU)], flush=True)

print("\ntau_true:", [float(x) for x in TAU], flush=True)
print("baseline K=2 bias (prev run): [0.0509, 0.0902]", flush=True)
print("DONE", flush=True)
