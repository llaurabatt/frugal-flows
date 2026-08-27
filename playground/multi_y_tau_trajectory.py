"""Does tau_hat converge, or is training stopped/rewound before it arrives?

Theorem 5.1 of Evans & Didelez says the MLE of the frugal parameterisation is
consistent, so a tau_hat anchored near its starting value means we are not finding
the MLE. Two candidate causes, which need opposite fixes:

  - tau is still travelling when the run ends  -> budget / stopping rule,
  - tau converges early to a non-truth value   -> objective shape.

flowjax's fit_to_data exposes no per-epoch hook, and train_frugal_flow does not
forward ``return_best`` (so it defaults to True: the flow returned is the one from
the LOWEST-VALIDATION-LOSS epoch, not the final epoch). Two consequences are probed
here from a single zero start, seed 0:

  1. a staircase over max_epochs, with early stopping disabled, showing how tau_hat
     moves as the budget grows;
  2. the argmin of the validation-loss list, i.e. which epoch's parameters were
     actually returned. If that epoch is early while tau is still moving, tau_hat is
     being rewound to a point before it converged.
"""
import jax, jax.numpy as jnp, jax.random as jr, paramax
jax.config.update("jax_enable_x64", True)
from frugal_flows.causal_flows import train_frugal_flow, get_independent_quantiles

TAU = jnp.array([2.0, -1.0])
N = 4000
BUDGETS = [2, 5, 10, 20, 40, 80, 150, 300]


def make_data(key):
    k = jr.split(key, 3)
    Z = jr.normal(k[0], (N, 2))
    T = jr.bernoulli(k[1], jax.nn.sigmoid(0.8 * Z[:, 0] - 0.6 * Z[:, 1])).astype(float)
    L = jnp.array([[1.0, 0.0], [0.7, 0.7]])
    noise = jr.normal(k[2], (N, 2)) @ L.T
    eta = noise + jnp.stack([Z[:, 0] + Z[:, 1], Z[:, 0] - 0.5 * Z[:, 1]], axis=1)
    return Z, T, TAU * T[:, None] + eta


Z, T, Y = make_data(jr.key(100))
u_z = get_independent_quantiles(key=jr.key(300), z_cont=Z, max_epochs=40, max_patience=8,
                                return_z_cont_flow=True, show_progress=False)["u_z_cont"]

print("true tau:", [float(x) for x in TAU], flush=True)
print("start   : [0.0, 0.0]  (zero arm)\n", flush=True)
print(f"{'budget':>7} {'tau_hat':>22} {'best_ep':>8} {'n_ep':>6} {'best_val':>10} {'last_val':>10}",
      flush=True)

for ep in BUDGETS:
    # max_patience > max_epochs disables early stopping, so the run always uses the
    # full budget and best_ep reflects the validation curve, not a patience cutoff.
    flow, losses = train_frugal_flow(
        key=jr.key(200), y=Y, u_z=u_z, condition=T[:, None],
        causal_model="location_translation",
        causal_model_args=dict(ate=jnp.zeros(2), RQS_knots=8, nn_depth=1,
                               nn_width=40, flow_layers=4),
        learning_rate=1e-3, max_epochs=ep, max_patience=ep + 1,
        flow_layers=4, nn_width=40, nn_depth=1, RQS_knots=8, show_progress=False)
    lc = flow.bijection.bijections[5]
    tau_hat = [round(float(paramax.unwrap(lc.bijections[k]).ate), 3) for k in range(2)]
    val = jnp.asarray(losses["val"])
    best_ep = int(val.argmin()) + 1
    print(f"{ep:>7} {str(tau_hat):>22} {best_ep:>8} {len(val):>6}"
          f" {float(val.min()):>10.5f} {float(val[-1]):>10.5f}", flush=True)

print("\nDONE", flush=True)
