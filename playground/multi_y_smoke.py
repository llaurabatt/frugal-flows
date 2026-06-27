"""K=2 multivariate-Y smoke test for location_translation (Architecture B)."""
import jax, jax.numpy as jnp, jax.random as jr, paramax
jax.config.update("jax_enable_x64", True)
from frugal_flows.causal_flows import train_frugal_flow, get_independent_quantiles

keys = jr.split(jr.key(0), 10)
N = 3000
# Confounders Z (2-dim), binary treatment T confounded by Z
Z = jr.normal(keys[0], (N, 2))
logit = 0.8 * Z[:, 0] - 0.6 * Z[:, 1]
T = jr.bernoulli(keys[1], jax.nn.sigmoid(logit)).astype(float)

# Treatment-free residual eta: correlated across the 2 Y-dims AND depends on Z (confounding)
L = jnp.array([[1.0, 0.0], [0.7, 0.7]])           # correlate the 2 outcome dims
noise = jr.normal(keys[2], (N, 2)) @ L.T
eta = noise + jnp.stack([Z[:, 0] + Z[:, 1], Z[:, 0] - 0.5 * Z[:, 1]], axis=1)

tau_true = jnp.array([2.0, -1.0])                  # per-dim ATE
Y = tau_true * T[:, None] + eta                    # Y = tau*T + eta  (mu0 = 0)
print("Y shape:", Y.shape, " std per dim:", [float(Y[:, k].std()) for k in range(2)], flush=True)

# Stage 1: marginal quantiles for Z
res = get_independent_quantiles(key=keys[3], z_cont=Z, max_epochs=60, max_patience=10,
                                return_z_cont_flow=True, show_progress=False)
u_z = res["u_z_cont"]

# Stage 2: multivariate-Y frugal flow (location_translation, K=2)
cma = dict(ate=tau_true, RQS_knots=8, nn_depth=1, nn_width=40, flow_layers=4)
flow, losses = train_frugal_flow(
    key=keys[4], y=Y, u_z=u_z, condition=T[:, None],
    causal_model="location_translation", causal_model_args=cma,
    learning_rate=1e-3, max_epochs=80, max_patience=8, flow_layers=4,
    nn_width=40, nn_depth=1, RQS_knots=8, show_progress=False,
)
tr = jnp.asarray(losses["train"]); va = jnp.asarray(losses["val"])
print("train loss first/last:", float(tr[0]), float(tr[-1]),
      " finite:", bool(jnp.isfinite(tr).all() and jnp.isfinite(va).all()), flush=True)

# Read off the trained per-dim tau from the LocCond block (bijections[5])
loccond_block = flow.bijection.bijections[5]
tau_hat = jnp.array([float(paramax.unwrap(loccond_block.bijections[k]).ate) for k in range(2)])
print("tau_true:", tau_true, " tau_hat:", tau_hat, flush=True)

# Rosenblatt property: the first K dims of the data, pushed to base, should be ~Uniform[-1,1].
# Push the data (Y,u_z) through bijection.inverse to the base; check first K coords.
data = jnp.hstack([Y, u_z])
base = jax.vmap(flow.bijection.inverse)(data, T[:, None])
for k in range(2):
    col = base[:, k]
    print(f"base dim {k}: min={float(col.min()):.3f} max={float(col.max()):.3f} "
          f"mean={float(col.mean()):.3f} (Uniform[-1,1] -> ~0)", flush=True)
print("OK", flush=True)
