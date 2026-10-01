"""Fit the treatment-conditioned image margin ALONE on E1, no thickness copula.

Valid as a diagnostic on E1 only: assignment is random, so p(Y | T=t) = p(Y | do(T=t))
and the conditional this fits IS the interventional law.

Model: Uniform[-1,1]^K  ->  causal margin (conditioned on T)  ->  Invert(Tanh)  ->  Y.
Those are the same two blocks the joint chain puts after the copula, built by the same
_build_flexible_margin call with the same causal_model_args, so architecture, spline
interval and conditioner are unchanged.

Training mirrors flowjax's fit_to_data: same train_val_split, same batching, Adam at the
same rate, same patience, checkpoint chosen by VALIDATION LIKELIHOOD. The loop is written
out here only so the potential outcomes can be scored every epoch. They are never used for
selection.
"""
import os, sys, time, json, numpy as np, jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx, optax, paramax
sys.path.insert(0, "/home/llaurabat/ff-project/frugal-flows/validation/morphomnist")
jax.config.update("jax_enable_x64", False)
from flowjax.bijections import Invert, Stack, Tanh
from flowjax.distributions import Transformed, Uniform
from flowjax.train.train_utils import train_val_split, get_batches
from frugal_flows.causal_flows import _build_flexible_margin
from frugal_flows.interventions import interventional_samples
import exp_ate_recovery as E

preset   = sys.argv[1] if len(sys.argv) > 1 else "exp1_rct_homogeneous"
MAXEP    = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
PATIENCE = int(sys.argv[3]) if len(sys.argv) > 3 else 30
NMC_TRACK, NMC_FINAL = 2000, 5000
OUT = f"/tmp/claude-2002/-home-llaurabat-ff-project/7ae089eb-8ecf-45ac-98e7-c90c357166d0/scratchpad/margin_only/{preset[:4]}"
os.makedirs(OUT, exist_ok=True)

cfg = E.Config(preset=preset, arm="flexible_continuous", size=8, seed_data=101, seed_fit=101)
data = E.build_data(cfg)
K = cfg.size ** 2
Y  = jnp.asarray(np.asarray(data["Y"]), jnp.float32)
T  = jnp.asarray(np.asarray(data["X"]), jnp.float32)
ate = np.asarray(data["ATE"])
ITE = np.asarray(data["ITE"]); Xv = np.asarray(data["X"])[:, 0]
Y0 = np.asarray(data["Y"]) - Xv[:, None] * ITE; Y1 = Y0 + ITE
t0_target, t1_target = Y0.mean(0), Y1.mean(0)
assert np.abs(Y1 - Y0 - ate).max() == 0

# pixel groups
s = cfg.size; sup = (ate != 0).reshape(s, s); ring = np.zeros_like(sup)
for i in range(s):
    for j in range(s):
        if sup[i, j]: continue
        for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
            ii, jj = i+di, j+dj
            if 0 <= ii < s and 0 <= jj < s and sup[ii, jj]: ring[i, j] = True
far = ~sup & ~ring

# ---- the margin alone, same blocks the joint chain uses after the copula
key = jr.PRNGKey(cfg.seed_fit)
key, sub = jr.split(key); key, sub = jr.split(key)          # match fit_flow's key walk
cma = {"RQS_knots": cfg.rqs_knots, "nn_depth": cfg.nn_depth,
       "nn_width": cfg.nn_width, "flow_layers": cfg.flow_layers, "conditioner": cfg.conditioner}
key, sk = jr.split(key)
margin = _build_flexible_margin(key=sk, dim=K, condition=T, causal_model_args=cma)
dist = Transformed(Uniform(-jnp.ones(K), jnp.ones(K)), margin)
dist = Transformed(dist, Stack([Invert(Tanh(()))] * K)).merge_transforms()
nparam = sum(int(x.size) for x in jax.tree_util.tree_leaves(eqx.filter(dist, eqx.is_inexact_array)))
print(f"margin-only model: {nparam:,} trainable params, chain of {len(dist.bijection.bijections)} blocks", flush=True)

# ---- training, mirroring fit_to_data
params, static = eqx.partition(dist, eqx.is_inexact_array,
                               is_leaf=lambda l: isinstance(l, paramax.NonTrainable))
opt = optax.adam(cfg.learning_rate); opt_state = opt.init(params)
key, sk = jr.split(key)
train_d, val_d = train_val_split(sk, (Y, T), val_prop=0.1)
print(f"train {train_d[0].shape[0]} rows, validation {val_d[0].shape[0]} rows", flush=True)

def nll(p, st, x, c):
    d = paramax.unwrap(eqx.combine(p, st))
    return -d.log_prob(x, c).mean()
@eqx.filter_jit
def step(p, st, x, c, os_):
    l, g = eqx.filter_value_and_grad(nll)(p, st, x, c)
    up, os_ = opt.update(g, os_, p)
    return eqx.apply_updates(p, up), os_, l
@eqx.filter_jit
def batch_nll(p, st, x, c):
    return nll(p, st, x, c)

def score(p, n_mc, seed=0):
    d = eqx.combine(p, static)
    r = interventional_samples(jr.key(seed), d, cond_dim=1, n_mc=n_mc, dim_y=K)
    y0, y1 = np.asarray(r["y0"]), np.asarray(r["y1"])
    keep = np.isfinite(y0).all(1) & np.isfinite(y1).all(1)
    y0, y1 = y0[keep], y1[keep]
    tau = (y1 - y0).mean(0); err = (tau - ate).reshape(s, s)
    e0 = y0.mean(0) - t0_target; e1 = y1.mean(0) - t1_target
    return dict(ate_mae=float(np.abs(tau - ate).mean()),
                err_inside=float(err[sup].mean()), err_ring=float(err[ring].mean()),
                err_far=float(err[far].mean()),
                e0_ring=float(e0.reshape(s,s)[ring].mean()), e1_ring=float(e1.reshape(s,s)[ring].mean()),
                e0_inside=float(e0.reshape(s,s)[sup].mean()), e1_inside=float(e1.reshape(s,s)[sup].mean()),
                frac_dropped=float(1 - keep.mean())), tau

import wandb
run = wandb.init(entity="proj-lb", project="Frugal Images", group="margin_only",
                 name=f"margin_only_{preset[:4]}_k{K}_s101", reinit=True,
                 tags=["margin_only", preset[:4], f"k{K}"],
                 config={**{k: v for k, v in cfg.__dict__.items()},
                         "model": "image margin alone, no copula", "n_params": nparam,
                         "max_epochs": MAXEP, "patience": PATIENCE, "nmc_track": NMC_TRACK})

best_val, best_params, best_ep, hist = np.inf, params, 0, []
t_start = time.monotonic()
for ep in range(1, MAXEP + 1):
    key, sk = jr.split(key)
    tr = [jr.permutation(sk, a) for a in train_d]
    losses = []
    for xb, cb in zip(*get_batches(tr, cfg.batch_size)):
        params, opt_state, l = step(params, static, xb, cb, opt_state)
        losses.append(float(l))
    vl = float(np.mean([float(batch_nll(params, static, xb, cb))
                        for xb, cb in zip(*get_batches(val_d, cfg.batch_size))]))
    m, _ = score(params, NMC_TRACK)
    row = dict(epoch=ep, train_loss=float(np.mean(losses)), val_loss=vl, **m)
    hist.append(row); run.log(row, step=ep)
    if vl < best_val:
        best_val, best_params, best_ep = vl, params, ep
        eqx.tree_serialise_leaves(f"{OUT}/best.eqx", eqx.combine(params, static))
    if ep % 10 == 0 or ep <= 3:
        print(f"  ep {ep:>4} val {vl:8.3f} (best {best_val:8.3f} @ {best_ep})  "
              f"ate_mae {m['ate_mae']:.4f}  ring {m['err_ring']:+.4f}  inside {m['err_inside']:+.4f}", flush=True)
    if ep - best_ep >= PATIENCE:
        print(f"  early stop at {ep}: {PATIENCE} epochs since the best", flush=True); break
eqx.tree_serialise_leaves(f"{OUT}/last.eqx", eqx.combine(params, static))

mf, tau = score(best_params, NMC_FINAL)
print(f"\n##### {preset}  margin alone, no copula")
print(f"  epochs run {len(hist)}   best validation loss {best_val:.3f} at epoch {best_ep}   "
      f"{'ENDED ON CAP' if len(hist) >= MAXEP else 'ended on the patience rule'}   wall {time.monotonic()-t_start:.0f}s")
print(f"  at the best-validation checkpoint, {NMC_FINAL} paired draws:")
print(f"    ATE error   inside {mf['err_inside']:+.4f}   ring {mf['err_ring']:+.4f}   far {mf['err_far']:+.4f}   mae {mf['ate_mae']:.4f}")
print(f"    arm errors  inside  e0 {mf['e0_inside']:+.4f}  e1 {mf['e1_inside']:+.4f}")
print(f"                ring    e0 {mf['e0_ring']:+.4f}  e1 {mf['e1_ring']:+.4f}")
print(f"    non-finite draws dropped: {mf['frac_dropped']:.3%}")
print(f"\n  joint fit on the same data for comparison: ring +0.0115  inside -0.0339  far -0.0000  mae 0.0228")
print(f"                                             ring e0 +0.0683  e1 +0.0798")
run.summary.update({f"final_{k}": v for k, v in mf.items()})
run.summary.update({"best_val_loss": best_val, "best_epoch": best_ep, "epochs_run": len(hist)})
np.savez(f"{OUT}/history.npz", **{k: np.array([h[k] for h in hist]) for k in hist[0]}, tau_hat=tau, ATE=ate)
json.dump(hist, open(f"{OUT}/history.json", "w"), indent=1)
run.finish(quiet=True)
