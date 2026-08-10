"""Multi-seed bias check for multivariate-Y location_translation, across ATE starts.

``ate`` is the *initial* value of a trainable parameter (``LocCond.ate``, default 0),
so where the optimiser starts can flatter the result. Earlier runs started it at the
true TAU -- i.e. at the answer -- which is never the situation on real data. Here the
same confounded DGP is fitted three times per seed, differing ONLY in that start:

  - "truth" : ate = TAU, reproducing the earlier runs for comparability,
  - "zero"  : ate = 0, LocCond's own default and an uninformative start,
  - "naive" : ate = the per-dim confounded difference in means, i.e. the biased
              estimate a practitioner would actually have to hand.

The data seed and the fitting key are shared across the three arms, so the arms are
paired and the only varying input is the start. Reports per dim and per arm the mean
and sd of the estimate and its bias, alongside the naive difference in means.
"""
import jax, jax.numpy as jnp, jax.random as jr, paramax
jax.config.update("jax_enable_x64", True)
from frugal_flows.causal_flows import train_frugal_flow, get_independent_quantiles

TAU = jnp.array([2.0, -1.0])
N = 4000
SEEDS = range(8)
ARMS = ("truth", "zero", "naive")


def make_data(key):
    k = jr.split(key, 3)
    Z = jr.normal(k[0], (N, 2))
    T = jr.bernoulli(k[1], jax.nn.sigmoid(0.8 * Z[:, 0] - 0.6 * Z[:, 1])).astype(float)
    L = jnp.array([[1.0, 0.0], [0.7, 0.7]])
    noise = jr.normal(k[2], (N, 2)) @ L.T
    eta = noise + jnp.stack([Z[:, 0] + Z[:, 1], Z[:, 0] - 0.5 * Z[:, 1]], axis=1)
    Y = TAU * T[:, None] + eta
    return Z, T, Y


def naive(T, Y):
    """Per-dim confounded difference in means, E[Y|T=1] - E[Y|T=0]."""
    return jnp.array([float(Y[T == 1, k].mean() - Y[T == 0, k].mean()) for k in range(2)])


def quantiles(key, Z):
    """Stage-1 marginal CDFs for Z. Fitted once per seed and reused by all arms."""
    return get_independent_quantiles(key=key, z_cont=Z, max_epochs=40, max_patience=8,
                                     return_z_cont_flow=True, show_progress=False)["u_z_cont"]


def fit_tau(key, u_z, T, Y, ate_init):
    cma = dict(ate=ate_init, RQS_knots=8, nn_depth=1, nn_width=40, flow_layers=4)
    flow, _ = train_frugal_flow(key=key, y=Y, u_z=u_z, condition=T[:, None],
                                causal_model="location_translation", causal_model_args=cma,
                                learning_rate=1e-3, max_epochs=150, max_patience=15,
                                flow_layers=4, nn_width=40, nn_depth=1, RQS_knots=8,
                                show_progress=False)
    lc = flow.bijection.bijections[5]
    return jnp.array([float(paramax.unwrap(lc.bijections[k]).ate) for k in range(2)])


tau_hats = {a: [] for a in ARMS}
naives = []
for s in SEEDS:
    Z, T, Y = make_data(jr.key(100 + s))
    u_z = quantiles(jr.key(300 + s), Z)
    nv = naive(T, Y)
    naives.append(nv)
    starts = {"truth": TAU, "zero": jnp.zeros(2), "naive": nv}
    for a in ARMS:
        # Same fitting key across arms: the start is the only thing that differs.
        th = fit_tau(jr.key(200 + s), u_z, T, Y, starts[a])
        tau_hats[a].append(th)
        print(f"seed {s} [{a:>5}] start={[round(float(x),3) for x in jnp.atleast_1d(starts[a])]}"
              f" -> tau_hat={[round(float(x),3) for x in th]}", flush=True)
    print(f"seed {s} naive={[round(float(x),3) for x in nv]}", flush=True)

NV = jnp.stack(naives)
print(f"\n=== SUMMARY over {len(list(SEEDS))} seeds ===", flush=True)
print("tau_true      :", [float(x) for x in TAU], flush=True)
print("naive    mean :", [round(float(x), 4) for x in NV.mean(0)],
      " std:", [round(float(x), 4) for x in NV.std(0)], flush=True)
for a in ARMS:
    TH = jnp.stack(tau_hats[a])
    print(f"[{a:>5}] mean:", [round(float(x), 4) for x in TH.mean(0)],
          " std:", [round(float(x), 4) for x in TH.std(0)],
          " bias:", [round(float(x), 4) for x in (TH.mean(0) - TAU)], flush=True)
print("DONE", flush=True)
